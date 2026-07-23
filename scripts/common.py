import csv
import os
import random

import numpy as np
import polars as pl
import torch
from scipy.sparse import csr_matrix

N_USERS = 876_146
N_ITEMS = 2_360_651


def load_split(path):
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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TrainingLogger:
    def __init__(self, output_path, prefix=""):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        log_path = f"{output_path}{prefix}log.csv"
        self.log_csv = open(log_path, "w")
        self.metrics = {
            "epoch": 0,
            "epoch_time_sec": 0.0,
            "recall": 0.0,
            "best_recall": 0.0,
            "ndcg": 0.0,
            "best_ndcg": 0.0,
        }
        self.writer = csv.DictWriter(self.log_csv, fieldnames=self.metrics.keys())
        self.writer.writeheader()

    def log(self, epoch, elapsed, recall, ndcg):
        self.metrics["epoch"] = epoch
        self.metrics["epoch_time_sec"] = elapsed
        self.metrics["recall"] = recall
        self.metrics["best_recall"] = max(recall, self.metrics["best_recall"])
        self.metrics["ndcg"] = ndcg
        self.metrics["best_ndcg"] = max(ndcg, self.metrics["best_ndcg"])
        self.writer.writerow(self.metrics)
        self.log_csv.flush()

    def close(self):
        self.log_csv.close()


def add_common_args(parser):
    parser.add_argument("--seed", default=44, type=int, help="Random number seed")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="./output", type=str, help="Output directory")
    parser.add_argument("-e", "--epoch", default=2, type=int, help="Epochs")
    parser.add_argument("--lr", default=1e-2, type=float, help="Learning rate")
    parser.add_argument("--decay", default=1e-5, type=float, help="Weight decay")
    parser.add_argument("--batch", default=8192, type=int, help="Batch size")
    parser.add_argument("--embed_dim", default=64, type=int, help="Embedding dimension")
    parser.add_argument("--negatives",
        default=100,
        type=int,
        help="Negative examples per eval user"
    )
    parser.add_argument("-k", default=10, type=int, help="Top-k for recall/NDCG")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation")
    parser.add_argument("--ckpt", type=str, help="Load a checkpoint")
