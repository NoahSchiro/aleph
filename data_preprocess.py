"""
Convert goodreads_interactions.csv -> parquet, produce a test train split
leave-one(/two)-out train/val/test split for implicit-feedback recsys.

Test train split will be leave N out for test/val per user, and the rest
goes to train.

user_id / book_id in goodreads_interactions.csv are ALREADY the
dense 0-indexed ids from user_id_map.csv / book_id_map.csv. No remapping
needed for training. Keep book_id_map.csv around to join back to raw
Goodreads book_id -> metadata (title/author/shelves) later for the
two-tower model.
"""

import polars as pl

DATA_DIR = "./data/"
RAW_CSV = DATA_DIR + "goodreads_interactions.csv"

print("Reading CSV...")
df = pl.read_csv(
    RAW_CSV,
    dtypes={
        "user_id"    : pl.Int32,
        "book_id"    : pl.Int32,
        "is_read"    : pl.Int8,
        "rating"     : pl.Int8,
        "is_reviewed": pl.Int8,
    },
)
print(f"Loaded {df.height:,} rows")

df.write_parquet(f"{DATA_DIR}/interactions_full.parquet", compression="zstd")
print("Wrote interactions_full.parquet")

print("Shuffling...")

# global shuffle
df = df.sample(fraction=1.0, shuffle=True, seed=16)
df = df.with_columns(
    # determine rank so that rank==0 -> test, rank==1 -> val, rest -> train
    pl.int_range(pl.len()).over("user_id").alias("rank"),
    # determine number of interactions per user so that we can determine
    # how to split later
    pl.len().over("user_id").alias("n_interactions"),
)

print("Splitting...")

# single interaction from each user with > 1 interaction
test = df.filter((pl.col("n_interactions") >= 2) & (pl.col("rank") == 0))
# single interaction from each user with > 2 interactions
val = df.filter((pl.col("n_interactions") >= 3) & (pl.col("rank") == 1))
train = df.filter(
    (pl.col("n_interactions") == 1)
    | ((pl.col("n_interactions") == 2) & (pl.col("rank") == 1))
    | ((pl.col("n_interactions") >= 3) & (pl.col("rank") >= 2))
)

# Only needed these for filtering
drop_cols = ["rank", "n_interactions"]
train = train.drop(drop_cols)
val   = val.drop(drop_cols)
test  = test.drop(drop_cols)

print(f"train: {train.height}")
print(f"val:   {val.height}")
print(f"test:  {test.height:}")
assert train.height + val.height + test.height == df.height # sanity check

train.write_parquet(f"{DATA_DIR}/train.parquet", compression="zstd")
val.write_parquet(f"{DATA_DIR}/val.parquet", compression="zstd")
test.write_parquet(f"{DATA_DIR}/test.parquet", compression="zstd")
print("Done.")
