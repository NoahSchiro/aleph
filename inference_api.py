"""
Recommendation API.

Local: uv run uvicorn inference_api:app --reload

Lambda: this module exports `handler` (via Mangum) for the Lambda runtime. Assets are pulled from S3
to /tmp on cold start and cached there for the life of the container. Warm invocations will skip the
download.
"""
import os

import boto3
import numpy as np
import polars as pl
from fastapi import FastAPI, HTTPException
from mangum import Mangum
from pydantic import BaseModel

app = FastAPI(title="Book Recommender")

# This will be set in Lambda env vars
# You may need to change the ASSET_DIR locally if you output the inference artifacts elsewhere
S3_BUCKET = os.environ.get("MODEL_BUCKET", "")
ASSET_DIR = "/tmp/assets" if S3_BUCKET else "./inference_artifacts/"

# Same confidence mapping used during weighted training.
RATING_WEIGHT = {5: 1.3, 4: 1.0, 3: 0.7, 0: 0.6}
EXCLUDE_RATINGS = {1, 2}


def _ensure_local(filename):
    """
    This logic lets us test the API server both locally and in AWS
    """
    path = os.path.join(ASSET_DIR, filename)
    if S3_BUCKET and not os.path.exists(path):
        os.makedirs(ASSET_DIR, exist_ok=True)
        boto3.client("s3").download_file(S3_BUCKET, filename, path)
    return path


@app.on_event("startup")
def load_assets():
    app.state.item_embeddings = np.load(_ensure_local("item_embeddings.npy"))
    app.state.book_lookup = pl.read_parquet(_ensure_local("book_lookup.parquet"))
    app.state.book_lookup_dict = dict(
        zip(app.state.book_lookup["book_id_csv"], zip(
            app.state.book_lookup["title"], app.state.book_lookup["author"]
        ))
    )
    print(f"Loaded {app.state.item_embeddings.shape[0]:,} item embeddings")
    print(f"Loaded {app.state.book_lookup.height:,} book titles")


class RecommendRequest(BaseModel):
    book_ids: list[int]
    ratings: list[int] | None = None
    top_k: int = 10


class BookResult(BaseModel):
    book_id: int
    title: str
    author: str
    score: float


@app.get("/search")
def search(q: str, limit: int = 10):
    if len(q) < 2:
        raise HTTPException(400, "query too short")
    matches = app.state.book_lookup.filter(
        pl.col("title").str.to_lowercase().str.contains(q.lower(), literal=True)
    ).head(limit)
    return [
        {"book_id": r["book_id_csv"], "title": r["title"], "author": r["author"]}
        for r in matches.iter_rows(named=True)
    ]


@app.post("/recommend", response_model=list[BookResult])
def recommend(req: RecommendRequest):
    if not req.book_ids:
        raise HTTPException(400, "book_ids must be non-empty")
    if req.ratings is not None and len(req.ratings) != len(req.book_ids):
        raise HTTPException(400, "ratings must match book_ids length")

    embeddings = app.state.item_embeddings
    n_items = embeddings.shape[0]

    vecs, weights = [], []
    for i, book_id in enumerate(req.book_ids):
        if book_id < 0 or book_id >= n_items:
            raise HTTPException(400, f"unknown book_id {book_id}")
        rating = req.ratings[i] if req.ratings else 0
        if rating in EXCLUDE_RATINGS:
            continue
        weights.append(RATING_WEIGHT.get(rating, 0.6))
        vecs.append(embeddings[book_id])

    if not vecs:
        raise HTTPException(400, "no usable books after filtering disliked ratings")

    vecs = np.stack(vecs)
    weights = np.array(weights, dtype=np.float32)
    pseudo_user_vec = (vecs * weights[:, None]).sum(axis=0) / weights.sum()
    
    # Inference: comes up with a score against every book
    scores = embeddings @ pseudo_user_vec  
    scores[req.book_ids] = -np.inf  # don't recommend books already in the input

    top_idx = np.argpartition(-scores, req.top_k)[: req.top_k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for idx in top_idx:
        title, author = app.state.book_lookup_dict.get(int(idx), ("Unknown", ""))
        results.append(BookResult(book_id=int(idx), title=title, author=author, score=float(scores[idx])))
    return results


# Lambda entry point: API Gateway/Lambda Function URL invokes `handler`.
# Locally, uvicorn just imports `app` directly and ignores this.
handler = Mangum(app)
