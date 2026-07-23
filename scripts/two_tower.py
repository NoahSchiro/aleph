"""
Two-tower neural network for implicit feedback.

Trained with in-batch sampled softmax: for a batch of (user, pos_item) pairs, every other user's
positive item in the batch acts as a free negative (one matmul instead of explicit negative
sampling). This is the standard two-tower training setup (I believe it matches YouTube's retrieval
models).

YouTube's retrieval model (at least from a decade ago):
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf

Note: with 2.36M items and batch_size=8192, the expected number of accidental same-item collisions
per batch (a real positive mislabeled as someone else's negative) is small (~batch^2 / (2*n_items) ~
14 per batch, ~0.2% of the batch). Negligible and not corrected for here.
"""
import time
from argparse import ArgumentParser

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
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


class Tower(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

        self.user_tower = Tower(embed_dim, hidden_dim, output_dim, dropout)
        self.item_tower = Tower(embed_dim, hidden_dim, output_dim, dropout)

    def user_repr(self, user_ids):
        return F.normalize(self.user_tower(self.user_emb(user_ids)))

    def item_repr(self, item_ids):
        return F.normalize(self.item_tower(self.item_emb(item_ids)))

    def score(self, user_ids, item_ids):
        u = self.user_repr(user_ids)
        i = self.item_repr(item_ids)
        return (u * i).sum(dim=-1)


def train_epoch(args, model, opt, users, items):
    n = users.shape[0]
    perm = torch.randperm(n)
    total_loss = 0.0
    n_batches = 0

    model.train()

    for start in tqdm(range(0, n, args.batch)):
        idx = perm[start : start + args.batch]
        u   = users[idx].to(args.device, non_blocking=True).long()
        pos = items[idx].to(args.device, non_blocking=True).long()

        u_vec = model.user_repr(u)   # (B, output_dim)
        i_vec = model.item_repr(pos) # (B, output_dim)

        # (B, B), row i's positive is column i
        logits = (u_vec @ i_vec.T) / args.temperature
        labels = torch.arange(u_vec.shape[0], device=args.device)
        loss = F.cross_entropy(logits, labels)

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

    model = TwoTower(
        N_USERS, N_ITEMS, args.embed_dim, args.hidden_dim, args.output_dim, args.dropout
    ).to(args.device)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    print("Baseline eval")
    recall, ndcg = evaluate(args, model, eval_users, eval_items, interaction_matrix)
    print(f"recall@{args.k} {recall:.4f}  NDCG@{args.k} {ndcg:.4f}")
    if args.eval_only:
        return

    logger = TrainingLogger(args.output, prefix="tt_")
    logger.log(0, 0.0, recall, ndcg)

    torch.save(model.state_dict(), f"{args.output}tt_best.pt")
    torch.save(model.state_dict(), f"{args.output}tt_last.pt")

    print("Starting training")
    for epoch in range(args.epoch):
        t0 = time.time()
        loss = train_epoch(args, model, opt, train_users, train_items)
        recall, ndcg = evaluate(args, model, eval_users, eval_items, interaction_matrix)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch+1}/{args.epoch}  loss {loss:.4f}  "
            f"recall@{args.k} {recall:.4f}  ndcg@{args.k} {ndcg:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if recall > logger.metrics["best_recall"]:
            torch.save(model.state_dict(), f"{args.output}tt_best.pt")
        torch.save(model.state_dict(), f"{args.output}tt_last.pt")

        logger.log(epoch + 1, elapsed, recall, ndcg)

    logger.close()

    print("Done...")


if __name__ == "__main__":
    parser = ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--hidden_dim", default=128, type=int, help="Tower hidden layer dimension")
    parser.add_argument("--output_dim", default=64, type=int, help="Tower output dimension")
    parser.add_argument("--dropout", default=0.1, type=float, help="Tower dropout")
    parser.add_argument("--temperature", default=0.1, type=float, help="Softmax temperature")
    args = parser.parse_args()
    main(args)
