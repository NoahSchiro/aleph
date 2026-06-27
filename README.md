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
| Metric | Score |
|--------|-------|
| Recall | 0.839 |
| NDCG   | 0.701 |

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
|||
