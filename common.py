import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

SEED = 42
K = 10
REL_THRESHOLD = 4
VAL_QUANTILE = 0.8
TEST_QUANTILE = 0.9
DAYS_PER_YEAR = 365.2425
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
USERS_PER_BATCH = 1024
BOOTSTRAP_ROUNDS = 2000
CI_ALPHA = 0.05


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    n_users: int
    n_items: int
    t_val: float
    t_test: float
    books: pd.DataFrame


def make_split(inter, books):
    user_ids = np.sort(inter.user_id.unique())
    item_ids = np.sort(inter.book_id.unique())

    df = pd.DataFrame({
        "u": np.searchsorted(user_ids, inter.user_id.values).astype(np.int32),
        "i": np.searchsorted(item_ids, inter.book_id.values).astype(np.int32),
        "rating": inter.rating.values,
        "t": ((inter.ts - EPOCH) / pd.Timedelta(days=1)).values,
    }).sort_values("t", kind="stable").reset_index(drop=True)

    t_val, t_test = df.t.quantile(VAL_QUANTILE), df.t.quantile(TEST_QUANTILE)
    return Split(
        train=df[df.t < t_val].reset_index(drop=True),
        val=df[(df.t >= t_val) & (df.t < t_test)].reset_index(drop=True),
        test=df[df.t >= t_test].reset_index(drop=True),
        n_users=len(user_ids),
        n_items=len(item_ids),
        t_val=t_val,
        t_test=t_test,
        books=books.set_index("book_id").loc[item_ids].reset_index(),
    )


def to_csr(df, n_users, n_items, threshold=1):
    d = df[df.rating >= threshold]
    return sp.csr_matrix((np.ones(len(d), dtype=np.float32), (d.u.values, d.i.values)), shape=(n_users, n_items))


@dataclass(repr=False)
class Task:
    split: Split
    fit_df: pd.DataFrame
    target_df: pd.DataFrame
    T: float
    seen: sp.csr_matrix
    candidate_mask: np.ndarray
    eval_users: np.ndarray
    rel: np.ndarray
    stats: dict

    @property
    def n_users(self):
        return self.split.n_users

    @property
    def n_items(self):
        return self.split.n_items

    def matrix(self, threshold=1):
        return to_csr(self.fit_df, self.n_users, self.n_items, threshold)


def make_task(split, stage):
    if stage == "val":
        fit_df, target_df, T = split.train, split.val, split.t_val
    else:
        fit_df, target_df, T = pd.concat([split.train, split.val], ignore_index=True), split.test, split.t_test

    seen = to_csr(fit_df, split.n_users, split.n_items)
    candidate_mask = seen.getnnz(0) > 0
    has_history = seen.getnnz(1) > 0

    # cold items and cold users cannot be served, so their relevant interactions are dropped
    relevant = target_df[target_df.rating >= REL_THRESHOLD]
    kept = relevant[candidate_mask[relevant.i.values] & has_history[relevant.u.values]]
    eval_users = np.sort(kept.u.unique())
    rel = np.zeros((len(eval_users), split.n_items), dtype=bool)
    rel[np.searchsorted(eval_users, kept.u.values), kept.i.values] = True

    stats = dict(
        n_fit=len(fit_df),
        n_target=len(target_df),
        n_rel_total=len(relevant),
        n_rel_kept=len(kept),
        n_eval_users=len(eval_users),
        n_candidates=int(candidate_mask.sum()),
        users_target=int(target_df.u.nunique()),
    )
    return Task(split, fit_df, target_df, T, seen, candidate_mask, eval_users, rel, stats)


def recommend(model, task):
    recs = []
    for s in range(0, len(task.eval_users), USERS_PER_BATCH):
        users = task.eval_users[s:s + USERS_PER_BATCH]
        scores = np.array(model.score(users), dtype=np.float32)

        scores[:, ~task.candidate_mask] = -np.inf
        rows, cols = task.seen[users].nonzero()
        scores[rows, cols] = -np.inf

        top = np.argpartition(-scores, K, axis=1)[:, :K]
        order = np.argsort(-np.take_along_axis(scores, top, 1), axis=1)
        recs.append(np.take_along_axis(top, order, 1))

    return np.vstack(recs)


def evaluate(recs, task):
    hits = task.rel[np.arange(len(recs))[:, None], recs].astype(np.float32)
    n_rel = task.rel.sum(1).astype(np.float32)

    discount = 1.0 / np.log2(np.arange(2, K + 2))
    ideal_dcg = np.cumsum(discount)[np.minimum(n_rel, K).astype(int) - 1]
    precision = np.cumsum(hits, 1) / np.arange(1, K + 1)

    per_user = {
        f"NDCG@{K}": (hits * discount).sum(1) / ideal_dcg,
        f"Recall@{K}": hits.sum(1) / n_rel,
        f"MAP@{K}": (precision * hits).sum(1) / np.minimum(n_rel, K),
        f"HitRate@{K}": (hits.sum(1) > 0).astype(np.float32),
    }
    mean = {name: float(values.mean()) for name, values in per_user.items()}
    mean[f"Coverage@{K}"] = len(np.unique(recs)) / int(task.candidate_mask.sum())
    return mean, per_user


def run(model, task):
    recs = recommend(model.fit(task), task)
    mean, per_user = evaluate(recs, task)
    return mean, per_user, recs


def bootstrap_ci(x, seed=SEED):
    rng = np.random.default_rng(seed)

    means = x[rng.integers(0, len(x), size=(BOOTSTRAP_ROUNDS, len(x)))].mean(1)
    return float(np.quantile(means, CI_ALPHA / 2)), float(np.quantile(means, 1 - CI_ALPHA / 2))
