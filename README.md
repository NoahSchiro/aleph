# Aleph

### Download data

I put these files in a `data` dir:

```bash
# Book data
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_books.json.gz
# User / book interactions
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_interactiosn.csv
# Book id to book_id mapping
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/book_id_map.csv
# User id mapping (might not be needed)
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/user_id_map.csv
# Author map
wget https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_book_authors.json.gz
```

Unpack the gz files with: `gunzip *.gz`

### Method 1: Bayesian Personalized Ranking with Matrix Factorization

[Bayesian Personalized Ranking](https://arxiv.org/pdf/1205.2618) (BPR) is a method for learning
recommendations from implicit feedback (only item interaction, not ratings). Instead of trying to
predict an absolute score for each user-item pair, BPR optimizes for the relative ordering: for each
user, it assumes items they've interacted with should be ranked higher than items they haven't. The
model is updated via stochastic gradient *ascent* to increase the predicted score gap between the
positive and negative item.

[Matrix Factorization](https://developers.google.com/machine-learning/recommendation/collaborative/matrix)
within the context of machine learning is the simpliest embedding model, where we learn a simple set
of weight to be applied to a user and items the users are selecting from (in this project, books).

To run the training for BPR, run this:
```
uv run bpr.py
```

The default settings yield the following results for `k=10` (i.e. do the top 10 predictions get a
positive result, a book someone would want to read?):
| Metric | Score |
|--------|-------|
| Recall | 0.411 |
| NDCG   | 0.255 |

### Method 2: Two tower approach

For the two tower approach, we have a module (a tower) which casts a user / book into an embedded
space. One tower for users, one for books, hence the name. Once we have these vector representations
of a user and a book, we multipy these together elementwise and sum. In the end we get a score for
each user / book combination.

Otherwise, loading data, prepping, etc. is largely the same. We are still using implicit feedback here,
so even 1-star reviews count as a positive interaction. Accounting for review data will be next!

To run the training for two tower, try this:
```
uv run two_tower.py
```
| Metric | Score |
|--------|-------|
| Recall | 0.840 |
| NDCG   | 0.659 |

#### Giving the reviewer more data about book interactions

So far, we have just been treating any user interaction as positive. However, our dataset breaks this
up into shelved books, read but unrated books, and rated books. These are different scales of interaction
and we want to weight them differently. I have found this to work best:

| Signal | Weight |
|--------|--------|
|shelved|0.3|
|read, unrated|0.6|
|1 star|used as negative example|
|2 star|used as negative example|
|3 stars|0.7|
|4 stars|1.0|
|5 stars|1.3|

To run the training with weights:
```
uv run two_tower.py --weighted
```
| Metric | Score |
|--------|-------|
| Recall | 0.831 |
| NDCG   | 0.700 |

So essentially not any better! My running theory here is that most of the value is already captured
through user interaction. Most of the time when people read a book, they finish it, and most of the
time they rate it 3+ stars.

### Deploy to AWS

I thought it would be really cool to deploy this to a webpage so that people can actually try the
model out.

Strategy:
- Use github.io for the frontend site. I don't want to pay for the frontend stuff. The frontend will
be handled in another repo which I will link [here]().
- Use Lambda as the backend. Costs money, but should be pretty minimal because I don't expect too many
users.

Two routes:
- `/search`: Allows users to search for books within the database that we have. Adds it to a list of books they have read
- `/recommend`: Given the list of books they have read, return `k` recommendations

We can precompute some inference artifacts that are a) expensive and b) needed for every recommend / search call.

This can be done with:
```
uv run inference_artifacts.py --ckpt ./path/to/trained/model.pt
```

The following needs to be uploaded to S3:
- `book_lookup.parquet`
- `item_embeddings.npy`

Note that we don't need the trained model. The useful information from the model has been encoded in
`item_embeddings.npy`.

Next we need an API which loads these artifacts on start and serves the two endpoints. To run locally:
```
uv run uvicorn inference_api:app
```

To test the search (if you have `jq`, otherwise just omit the pipe):
```
curl "http://localhost:8000/search?q=harry%20potter | jq"
```

And then get recommendations (the book id passed in is for Harry Potter and the Sorcerer's Stone):
```
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "book_ids": [663092],
    "top_k": 10
  }' | jq
```
