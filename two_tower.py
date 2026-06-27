"""
Two-tower neural network for implicit feedback.

Trained with in-batch sampled softmax: for a batch of (user, pos_item) pairs, every other user's
positive item in the batch acts as a free negative (one matmul instead of explicit negative sampling).
This is the standard two-tower training setup (I believe it matches YouTube's retrieval models).

YouTube's retrieval model (at least from a decade ago):
https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf

Note: with 2.36M items and batch_size=8192, the expected number of accidental same-item collisions
per batch (a real positive mislabeled as someone else's negative) is small (~batch^2 / (2*n_items) ~
14 per batch, ~0.2% of the batch). Negligible and not corrected for here.
"""
from argparse import ArgumentParser
import csv
import os
import time
import random

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.sparse import csr_matrix

# Gleaned from dataset analysis
N_USERS = 876_146
N_ITEMS = 2_360_651


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
        return self.user_tower(self.user_emb(user_ids))

    def item_repr(self, item_ids):
        return self.item_tower(self.item_emb(item_ids))

    def score(self, user_ids, item_ids):
        u = self.user_repr(user_ids)
        i = self.item_repr(item_ids)
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


def evaluate(args, model, eval_users, eval_items, interaction_matrix):
    model.eval()
    recalls, ndcgs = [], []

    eval_users_np = eval_users.cpu().numpy()
    eval_items_np = eval_items.cpu().numpy()

    with torch.no_grad():
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
            scores = model.score(u_tensor, candidates).cpu().numpy()

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

    # Logging set up
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if not os.path.exists(args.output+"/tt_log.csv"):
        os.mknod(args.output+"/tt_log.csv")
    
    torch.save(model.state_dict(), args.output + "/tt_best.pt")
    torch.save(model.state_dict(), args.output + "/tt_last.pt")

    logged_metrics = {
        "epoch": 0,
        "epoch_time_sec": 0.0,
        "recall": recall,
        "best_recall": recall,
        "ndcg": ndcg,
        "best_ndcg": ndcg
    }
    log_csv = open(args.output + "/tt_log.csv", "w")
    csv_writer = csv.DictWriter(log_csv, fieldnames=logged_metrics.keys())
    csv_writer.writeheader()
    csv_writer.writerow(logged_metrics)

    print("Starting training")
    for epoch in range(args.epoch):
        t0 = time.time()
        loss = train_epoch(args, model, opt, train_users, train_items)
        recall, ndcg = evaluate(args, model, eval_users, eval_items, interaction_matrix)
        elapsed = time.time()-t0
        print(
            f"epoch {epoch+1}/{args.epoch}  loss {loss:.4f}  "
            f"recall@{args.k} {recall:.4f}  ndcg@{args.k} {ndcg:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if recall > logged_metrics["best_recall"]:
            torch.save(model.state_dict(), args.output + "/tt_best.pt")
        torch.save(model.state_dict(), args.output + "/tt_last.pt")

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
    parser.add_argument("-e", "--epoch", default=4, type=int,
        help="Epochs"
    )
    parser.add_argument("--lr", default=1e-3, type=float,
        help="Learning rate"
    )
    parser.add_argument("--decay", default=1e-6, type=float,
        help="Weight decay"
    )
    parser.add_argument("--batch", default=8192, type=int,
        help="Batch size"
    )
    parser.add_argument("--embed_dim", default=64, type=int,
        help="Embedding table dimension"
    )
    parser.add_argument("--hidden_dim", default=128, type=int,
        help="Tower hidden layer dimension"
    )
    parser.add_argument("--output_dim", default=64, type=int,
        help="Tower output (scoring) dimension"
    )
    parser.add_argument("--dropout", default=0.1, type=float,
        help="Tower dropout"
    )
    parser.add_argument("--temperature", default=0.1, type=float,
        help="Softmax temperature for in-batch loss"
    )
    parser.add_argument("--negatives", default=100, type=int,
        help="Number of negative examples per eval user"
    )
    parser.add_argument("-k", default=10, type=int,
        help="Top-k for recall/NDCG"
    )
    parser.add_argument("--eval-only", action="store_true",
        help="Only run an evaluation"
    )
    parser.add_argument("--ckpt", type=str,
        help="Load a checkpoint"
    )
    args = parser.parse_args()
    main(args)
