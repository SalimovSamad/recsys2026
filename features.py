import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from common import DAYS_PER_YEAR

EPOCH_YEAR = 1970.0
N_ACTIVITY_BUCKETS = 16
N_MEAN_RATING_BUCKETS = 11
AGE_EDGES = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0])
N_AGE_BUCKETS = len(AGE_EDGES) + 2
MAX_AUTHORS_PER_BOOK = 3
MIN_BOOKS_FOR_AUTHOR_TOKEN = 3
MIN_PUBLICATION_YEAR = 1990
N_PAGE_BINS = 8
N_DESCRIPTION_LENGTH_BINS = 4
TOP_CATEGORY_VALUES = {"format": 12, "language_code": 6, "publisher": 40}
TFIDF_PARAMS = dict(max_features=4000, min_df=5, stop_words="english", sublinear_tf=True)
KEYWORDS_PER_BOOK = 6


def age_token_ids(pub_year, t_days, base):
    age = torch.floor(EPOCH_YEAR + t_days / DAYS_PER_YEAR) - pub_year

    bucket = torch.bucketize(age.nan_to_num(0.0), AGE_EDGES, right=True)
    return base + torch.where(torch.isnan(pub_year), N_AGE_BUCKETS - 1, bucket)


def pad(rows):
    out = np.zeros((len(rows), max(map(len, rows))), dtype=np.int64)
    for i, row in enumerate(rows):
        out[i, :len(row)] = row
    return out


def bin_tokens(prefix, x, n_bins):
    edges = np.unique(np.nanquantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return [[f"{prefix}:na" if np.isnan(v) else f"{prefix}:{b}"] for v, b in zip(x, np.searchsorted(edges, x))]


def item_columns(books):
    authors = books.authors
    author_books = pd.Series([a for L in authors for a in L]).value_counts()
    authors_col = [
        [f"a:{a}" if author_books[a] >= MIN_BOOKS_FOR_AUTHOR_TOKEN else "a:rare" for a in list(L)[:MAX_AUTHORS_PER_BOOK]] or ["a:na"]
        for L in authors
    ]

    year_col = [["y:na" if np.isnan(y) else f"y:{int(max(y, MIN_PUBLICATION_YEAR))}"] for y in books.publication_year.values]
    pages_col = bin_tokens("p", books.num_pages.astype(float).values, N_PAGE_BINS)

    categorical_cols = []
    for col, top in TOP_CATEGORY_VALUES.items():
        values = books[col].fillna("").replace("", "na")
        keep = set(values[values != "na"].value_counts().index[:top])
        categorical_cols.append([[f"{col}:{v if v in keep or v == 'na' else 'other'}"] for v in values])

    ebook = books.is_ebook.astype(str).str.lower().values
    flags_col = [[f"ebook:{e}", f"series:{int(s > 0)}"] for e, s in zip(ebook, books.series.values)]

    description = books.description.fillna("").astype(str)
    length_col = bin_tokens("dl", description.str.len().replace(0, np.nan).astype(float).values, N_DESCRIPTION_LENGTH_BINS)

    tfidf = TfidfVectorizer(**TFIDF_PARAMS)
    tfidf_matrix = tfidf.fit_transform(description).tocsr()
    words = np.array(tfidf.get_feature_names_out())
    keywords_col = []
    for i in range(len(books)):
        start, end = tfidf_matrix.indptr[i], tfidf_matrix.indptr[i + 1]
        best = tfidf_matrix.indices[start:end][np.argsort(-tfidf_matrix.data[start:end])[:KEYWORDS_PER_BOOK]]
        keywords_col.append([f"w:{w}" for w in words[best]])

    return [authors_col, year_col, pages_col, *categorical_cols, flags_col, length_col, keywords_col]


class FMFeatures:
    def __init__(self, split, use_meta):
        self.use_meta = use_meta
        self.vocab = {}
        self.user_id_tok = np.array([self._tok(f"u:{u}") for u in range(split.n_users)])
        self.item_id_tok = np.array([self._tok(f"i:{i}") for i in range(split.n_items)])

        columns = []
        if use_meta:
            for a in range(N_ACTIVITY_BUCKETS):
                self._tok(f"ua:{a}")
            for m in range(N_MEAN_RATING_BUCKETS):
                self._tok(f"um:{m}")

            self.age_base = len(self.vocab) + 1
            for b in range(N_AGE_BUCKETS):
                self._tok(f"age:{b}")

            self.pub_year = torch.tensor(split.books.publication_year.values, dtype=torch.float32)
            columns = item_columns(split.books)

        for column in columns:
            for tokens in column:
                for t in tokens:
                    self._tok(t)

        self.n_feat = len(self.vocab) + 1
        self.item_fields = pad([
            [self.item_id_tok[i]] + [self.vocab[t] for column in columns for t in column[i]]
            for i in range(split.n_items)
        ])

    def _tok(self, s):
        return self.vocab.setdefault(s, len(self.vocab) + 1)

    def user_fields(self, task):
        if not self.use_meta:
            return self.user_id_tok[:, None]

        df = task.fit_df
        count = np.bincount(df.u.values, minlength=task.n_users)
        mean = np.bincount(df.u.values, weights=df.rating.values, minlength=task.n_users) / np.maximum(count, 1)

        activity = [self.vocab[f"ua:{min(int(np.log2(max(c, 1))), N_ACTIVITY_BUCKETS - 1)}"] for c in count]
        mean_rating = [self.vocab[f"um:{min(int(round(2 * m)), N_MEAN_RATING_BUCKETS - 1)}"] for m in mean]
        return np.stack([self.user_id_tok, activity, mean_rating], axis=1)
