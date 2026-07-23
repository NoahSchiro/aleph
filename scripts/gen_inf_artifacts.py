"""
Precompute assets needed to serve the model. Needs to be re-run on every retrain of the model.

item_embeddings.npy: every book run through the trained item tower once, so serving doesn't need the
GPU or the full model at request time.

book_lookup.parquet: book_id_csv -> title (+ first author), built by joining book_id_map.csv (csv
index -> raw goodreads book_id) against goodreads_books.json.gz (raw book_id -> title/author_id) and
goodreads_book_authors.json.gz (author_id -> name).
"""
import json
import os
from argparse import ArgumentParser

import numpy as np
import polars as pl
import torch
from two_tower import N_ITEMS, N_USERS, TwoTower


def build_item_embeddings(args):
    """For every book in the dataset, compute the embedding."""
    model = TwoTower(
        N_USERS,
        N_ITEMS,
        args.embed_dim,
        args.hidden_dim,
        args.output_dim,
        args.dropout
    )
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model.eval()

    embeddings = np.zeros((N_ITEMS, args.output_dim), dtype=np.float32)
    batch = 100_000
    with torch.no_grad():
        for start in range(0, N_ITEMS, batch):
            end = min(start + batch, N_ITEMS)
            ids = torch.arange(start, end, dtype=torch.long)
            embeddings[start:end] = model.item_repr(ids).numpy()

    np.save(args.output+"item_embeddings.npy", embeddings)
    print("Saved item_embeddings.npy")


def build_book_lookup(args):
    """Get a mapping from book_id -> (title, author)"""
    print("Loading book_id_map.csv...")
    id_map = pl.read_csv("./data/book_id_map.csv")  # columns: book_id_csv, book_id (raw)
    raw_to_csv = dict(zip(id_map["book_id"].cast(pl.Utf8), id_map["book_id_csv"]))

    print("Loading goodreads_book_authors.json...")
    author_name = {}
    with open("./data/goodreads_book_authors.json", "r") as f:
        for line in f:
            rec = json.loads(line)
            author_name[rec["author_id"]] = rec["name"]

    print("Loading goodreads_books.json...")
    rows = []
    with open("./data/goodreads_books.json", "r") as f:
        for line in f:
            rec = json.loads(line)
            csv_id = raw_to_csv.get(rec["book_id"])
            # If the book is not present in interaction id mapping
            if csv_id is None:
                continue
            authors = rec.get("authors", [])
            author = author_name.get(authors[0]["author_id"], "") if authors else ""
            lang = rec.get("language_code", "")
            if "-" in lang: # en-GB, en-CA, en-US -> eng
                lang = "eng"
            rows.append((csv_id, rec.get("title", ""), author, lang))

    df = pl.DataFrame(rows, schema=["book_id_csv", "title", "author", "language"], orient="row")
    df.write_parquet(args.output+"book_lookup.parquet")
    print(f"Saved book_lookup.parquet ({df.height:,} rows)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str,
        help="Path to the trained model"
    )
    parser.add_argument("--output", default="./inference_artifacts/", type=str,
        help="Output path of the inference artifacts"
    )
    parser.add_argument("--embed_dim", default=64, type=int,
        help="Embedding dimension of the trained model (must match training parameters)!"
    )
    parser.add_argument("--hidden_dim", default=128, type=int,
        help="Hidden dimension of the trained model (must match training parameters)!"
    )
    parser.add_argument("--output_dim", default=64, type=int,
        help="Output dimension of the trained model (must match training parameters)!"
    )
    parser.add_argument("--dropout", default=0.1, type=float,
        help="Dropout of the model"
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    build_item_embeddings(args)
    build_book_lookup(args)
