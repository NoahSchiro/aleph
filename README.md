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

[Bayesian Personalized Ranking](https://arxiv.org/pdf/1205.2618) (BPR) is a method... TODO

[Matrix Factorization](https://developers.google.com/machine-learning/recommendation/collaborative/matrix) within the context of machine learning is the simpliest embedding model, where we learn a simple set of weight to be applied to a user and items the users are selecting from (in this project, books).
