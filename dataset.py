import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre"
FILES = {
    "interactions": "goodreads_interactions_comics_graphic.json.gz",
    "books": "goodreads_books_comics_graphic.json.gz",
}
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SAMPLING_SEED = 42
POOL_MIN_INTERACTIONS = 10
N_SAMPLED_USERS = 16_000
MIN_USER_INTERACTIONS = 10
MIN_ITEM_INTERACTIONS = 20

DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
INTERACTION_COLS = ["user_id", "book_id", "rating", "date_updated"]
BOOK_COLS = [
    "book_id", "title", "title_without_series", "description", "authors", "publisher",
    "num_pages", "publication_year", "format", "language_code", "is_ebook", "series",
]


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name in FILES.values():
        if not (RAW_DIR / name).exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", RAW_DIR / name)


def load_interactions(path):
    rows = {c: [] for c in INTERACTION_COLS}
    with gzip.open(path, "rt") as f:
        for line in f:
            record = json.loads(line)
            for c in INTERACTION_COLS:
                rows[c].append(record[c])

    df = pd.DataFrame(rows)
    df["rating"] = df["rating"].astype("int8")
    df["ts"] = pd.to_datetime(df.pop("date_updated"), format=DATE_FORMAT, utc=True)
    return df


def load_books(path):
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            record = json.loads(line)
            rows.append({c: record.get(c) for c in BOOK_COLS})

    books = pd.DataFrame(rows)
    books["authors"] = books["authors"].map(lambda a: [x["author_id"] for x in a] if a else [])
    books["series"] = books["series"].map(lambda s: len(s) if s else 0)
    for c in ("num_pages", "publication_year"):
        books[c] = pd.to_numeric(books[c], errors="coerce")
    return books


def kcore(df, min_user, min_item):
    while True:
        n = len(df)
        df = df[df.groupby("user_id").user_id.transform("size").ge(min_user)]
        df = df[df.groupby("book_id").book_id.transform("size").ge(min_item)]
        if len(df) == n:
            return df


def build_subsample(df):
    rated = df[df.rating > 0]
    pool = kcore(rated, POOL_MIN_INTERACTIONS, POOL_MIN_INTERACTIONS)

    users = np.sort(pool.user_id.unique())
    rng = np.random.default_rng(SAMPLING_SEED)
    sampled_users = rng.choice(users, size=min(N_SAMPLED_USERS, len(users)), replace=False)

    sub = kcore(pool[pool.user_id.isin(sampled_users)], MIN_USER_INTERACTIONS, MIN_ITEM_INTERACTIONS)
    return sub.sort_values("ts", kind="stable").reset_index(drop=True)


def prepare():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    inter = build_subsample(load_interactions(RAW_DIR / FILES["interactions"]))
    books = load_books(RAW_DIR / FILES["books"])
    books = books[books.book_id.isin(inter.book_id.unique())].reset_index(drop=True)

    inter.to_parquet(PROCESSED_DIR / "interactions.parquet")
    books.to_parquet(PROCESSED_DIR / "books.parquet")


def load_processed():
    return pd.read_parquet(PROCESSED_DIR / "interactions.parquet"), pd.read_parquet(PROCESSED_DIR / "books.parquet")
