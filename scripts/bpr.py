"""
BPR-MF baseline for implicit feedback.
"""
import time
from argparse import ArgumentParser

import polars as pl
import torch
import torch.nn as nn
from common import (
    N_ITEMS,
    N_USERS,
    TrainingLogger,
    add_common_args,
    build_interaction_matrix,
    evaluate,
    load_split,
    set_seed,
)
from tqdm import tqdm


class BPRMF(nn.Module):
    def __init__(self, n_users, n_items, dim):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        return (u * i).sum(dim=-1)


def train_epoch(args, model, opt, users, items, n_items, n_candidates=10):
    n = users.shape[0]
    perm = torch.randperm(n)
    total_loss = 0.0
    n_batches = 0

    model.train()

    for start in tqdm(range(0, n, args.batch)):
        idx = perm[start : start + args.batch]
        u = users[idx].to(args.device, non_blocking=True).long()
        pos = items[idx].to(args.device, non_blocking=True).long()

        bsz = u.shape[0]
        neg_candidates = torch.randint(0, n_items, (bsz, n_candidates), device=args.device)
        with torch.no_grad():
            u_rep = u.unsqueeze(1).expand(-1, n_candidates)
            cand_scores = model.score(u_rep.reshape(-1), neg_candidates.reshape(-1))
            cand_scores = cand_scores.view(bsz, n_candidates)
            hardest_idx = cand_scores.argmax(dim=1)
        neg = neg_candidates[torch.arange(bsz, device=args.device), hardest_idx]

        pos_score = model.score(u, pos)
        neg_score = model.score(u, neg)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main(args):
    set_seed(args.seed)
    print(f"Device: {args.device}")

    print("Loading data...")
    train_users, train_items = load_split("./data/train.parquet")
    val_users, val_items = load_split("./data/val.parquet")

    print("Building interaction matrix...")
    interaction_matrix = build_interaction_matrix([
        pl.read_parquet("./data/train.parquet", columns=["user_id", "book_id"]),
        pl.read_parquet("./data/val.parquet", columns=["user_id", "book_id"]),
        pl.read_parquet("./data/test.parquet", columns=["user_id", "book_id"]),
    ])

    eval_sample_size = min(5000, val_users.shape[0])
    eval_idx = torch.randperm(val_users.shape[0])[:eval_sample_size]
    eval_users = val_users[eval_idx]
    eval_items = val_items[eval_idx]

    model = BPRMF(N_USERS, N_ITEMS, args.embed_dim).to(args.device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    print("Baseline eval")
    recall, ndcg = evaluate(args, model, eval_users, eval_items, interaction_matrix)
    print(f"recall@{args.k} {recall:.4f}  NDCG@{args.k} {ndcg:.4f}")
    if args.eval_only:
        return

    logger = TrainingLogger(args.output)
    logger.log(0, 0.0, recall, ndcg)

    torch.save(model.state_dict(), f"{args.output}/bpr_best.pt")
    torch.save(model.state_dict(), f"{args.output}/bpr_last.pt")

    print("Starting training")
    for epoch in range(args.epoch):
        t0 = time.time()
        loss = train_epoch(args, model, opt, train_users, train_items, N_ITEMS)
        recall, ndcg = evaluate(args, model, eval_users, eval_items, interaction_matrix)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch+1}/{args.epoch}  loss {loss:.4f}  "
            f"recall@{args.k} {recall:.4f}  ndcg@{args.k} {ndcg:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if recall > logger.metrics["best_recall"]:
            torch.save(model.state_dict(), f"{args.output}/bpr_best.pt")
        torch.save(model.state_dict(), f"{args.output}/bpr_last.pt")

        logger.log(epoch + 1, elapsed, recall, ndcg)

    logger.close()
    print("Done...")


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    main(args)
