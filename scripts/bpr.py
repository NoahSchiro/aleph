"""
BPR-MF baseline for implicit feedback.
"""
import csv
import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Gleaned from dataset analysis
N_USERS = 876_146
N_ITEMS = 2_360_651


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


def load_split(path):
    """Given a train/test split, load the users and books"""
    df = pl.read_parquet(path, columns=["user_id", "book_id"])
    users = torch.from_numpy(df["user_id"].to_numpy().astype(np.int64))
    items = torch.from_numpy(df["book_id"].to_numpy().astype(np.int64))
    return users, items


def build_interaction_matrix(datasets):
    rows, cols = [], []
    for df in datasets:
        rows.append(df["user_id"].to_numpy())
        cols.append(df["book_id"].to_numpy())
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(N_USERS, N_ITEMS))


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


def evaluate(args, model, eval_users, eval_items, interaction_matrix):
    model.eval()
    recalls, ndcgs = [], []

    eval_users_np = eval_users.cpu().numpy()
    eval_items_np = eval_items.cpu().numpy()

    for u, pos_item in zip(eval_users_np, eval_items_np):
        user_history = set(interaction_matrix.getrow(u).indices.tolist())

        negs = []
        while len(negs) < args.negatives:
            cand = np.random.randint(0, N_ITEMS, size=args.negatives - len(negs))
            cand = [c for c in cand if c not in user_history and c != pos_item]
            negs.extend(cand)
        negs = negs[:args.negatives]

        candidates = torch.tensor([pos_item] + negs, dtype=torch.long, device=args.device)
        u_tensor = torch.full((len(candidates),), u, dtype=torch.long, device=args.device)
        scores = model.score(u_tensor, candidates).cpu().detach().numpy()

        ranking = np.argsort(-scores)
        rank_of_pos = np.where(ranking == 0)[0][0]

        recalls.append(1.0 if rank_of_pos < args.k else 0.0)
        ndcgs.append(1.0 / np.log2(rank_of_pos + 2) if rank_of_pos < args.k else 0.0)

    return np.mean(recalls), np.mean(ndcgs)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

    # subsample val users for faster per-epoch eval; full val every epoch
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

    # Logging set up
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if not os.path.exists(args.output+"/log.csv"):
        os.mknod(args.output+"/log.csv")

    torch.save(model.state_dict(), args.output + "/bpr_best.pt")
    torch.save(model.state_dict(), args.output + "/bpr_last.pt")

    logged_metrics = {
        "epoch": 0,
        "epoch_time_sec": 0.0,
        "recall": recall,
        "best_recall": recall,
        "ndcg": ndcg,
        "best_ndcg": ndcg
    }
    log_csv = open(args.output + "/log.csv", "w")
    csv_writer = csv.DictWriter(log_csv, fieldnames=logged_metrics.keys())
    csv_writer.writeheader()
    csv_writer.writerow(logged_metrics)

    print("Starting training")
    for epoch in range(args.epoch):
        t0 = time.time()
        loss = train_epoch(args, model, opt, train_users, train_items, N_ITEMS)
        recall, ndcg = evaluate(args, model, eval_users, eval_items, interaction_matrix)
        elapsed = time.time()-t0
        print(
            f"epoch {epoch+1}/{args.epoch}  loss {loss:.4f}  "
            f"recall@{args.k} {recall:.4f}  ndcg@{args.k} {ndcg:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if recall > logged_metrics["best_recall"]:
            torch.save(model.state_dict(), args.output + "/bpr_best.pt")
        torch.save(model.state_dict(), args.output + "/bpr_last.pt")

        logged_metrics["epoch"] = epoch+1
        logged_metrics["epoch_time_sec"] = elapsed
        logged_metrics["recall"] = recall
        logged_metrics["best_recall"] = max(recall, logged_metrics["best_recall"])
        logged_metrics["ndcg"] = ndcg
        logged_metrics["best_ndcg"] = max(ndcg, logged_metrics["best_ndcg"])
        csv_writer.writerow(logged_metrics)

    log_csv.close()
    print("Done...")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", default=44, type=int,
        help="Random number seed"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="./output", type=str,
        help="Where to output model weights, logs, etc."
    )
    # BPR MF converges fairly fast, no need for more epochs.
    # In a pinch you could probably do 1 epoch
    parser.add_argument("-e", "--epoch", default=2, type=int,
        help="Epochs"
    )
    parser.add_argument("--lr", default=1e-2, type=float,
        help="Learning rate"
    )
    parser.add_argument("--decay", default=1e-5, type=float,
        help="Weight decay"
    )
    parser.add_argument("--batch", default=8192, type=int,
        help="Batch size"
    )
    parser.add_argument("--embed_dim", default=64, type=int,
        help="Embedding dimension"
    )
    parser.add_argument("--negatives", default=100, type=int,
        help="Number of negative examples to show per batch"
    )
    parser.add_argument("-k", default=10, type=int,
        help="When computing recall and NDCG, top k results will count as valid"
    )
    parser.add_argument("--eval-only", action="store_true",
        help="Only run an evaluation"
    )
    parser.add_argument("--ckpt", type=str,
        help="Load a checkpoint"
    )
    args = parser.parse_args()
    main(args)
