import numpy as np
import scipy.sparse as sp
import torch

from common import DAYS_PER_YEAR, SEED

BATCH = 16384
INIT_STD = 0.05
MIN_RATING = 1
MAX_RATING = 5
EPS = 1e-9


class TimeSVDpp:
    def __init__(self, factors=32, lr=0.01, reg=0.05, epochs=30, beta=0.4, n_bins=10,
                 use_time=True, pop_blend=0.0, seed=SEED):
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.beta = beta
        self.n_bins = n_bins
        self.use_time = use_time
        self.pop_blend = pop_blend
        self.seed = seed

    def _dev(self, u, t):
        """User drift: sign(t - t_u) * |t - t_u| ** beta, t_u is the mean time of the user's ratings (in years)."""
        d = (t - self.mean_time[u]) / DAYS_PER_YEAR
        return torch.sign(d) * d.abs().pow(self.beta)

    def _implicit(self, u):
        """Implicit part of the user factor: sum of y_j over the user's items / sqrt(n_items)."""
        return torch.sparse.mm(self.Z.index_select(0, u), self.Y)

    def _predict(self, u, i, t):
        user_factor = self.P[u] + self._implicit(u)
        rating = self.mu + self.bu[u] + self.bi[i]

        if self.use_time:
            dev = self._dev(u, t)
            time_bin = torch.floor((t - self.t_start) / self.span * self.n_bins).long().clamp(0, self.n_bins - 1)
            user_factor = user_factor + self.AP[u] * dev[:, None]
            rating = rating + self.au[u] * dev + self.bi_bin[i, time_bin]

        return rating + (self.Q[i] * user_factor).sum(1)

    def fit(self, task, on_epoch=None):
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        n_users, n_items = task.n_users, task.n_items
        df = task.fit_df

        users = torch.tensor(df.u.values, dtype=torch.long)
        items = torch.tensor(df.i.values, dtype=torch.long)
        times = torch.tensor(df.t.values, dtype=torch.float32)
        ratings = torch.tensor(df.rating.values, dtype=torch.float32)

        count = np.maximum(np.bincount(df.u.values, minlength=n_users), 1)
        mean_time = np.bincount(df.u.values, weights=df.t.values, minlength=n_users) / count
        self.mean_time = torch.tensor(mean_time, dtype=torch.float32)
        self.t_start = float(df.t.min())
        self.span = float(task.T - df.t.min())
        self.mu = float(ratings.mean())
        self.T = task.T

        Z = (sp.diags((count ** -0.5).astype(np.float32)) @ task.seen).tocoo()
        self.Z = torch.sparse_coo_tensor(np.vstack([Z.row, Z.col]), Z.data, (n_users, n_items)).coalesce()

        def zeros(*shape):
            return torch.zeros(*shape, requires_grad=True)

        def randn(*shape):
            return (INIT_STD * torch.randn(*shape, generator=generator)).requires_grad_()

        self.bu = zeros(n_users)
        self.au = zeros(n_users)
        self.bi = zeros(n_items)
        self.bi_bin = zeros(n_items, self.n_bins)
        self.P = randn(n_users, self.factors)
        self.AP = zeros(n_users, self.factors)
        self.Q = randn(n_items, self.factors)
        self.Y = randn(n_items, self.factors)
        params = [self.bu, self.au, self.bi, self.bi_bin, self.P, self.AP, self.Q, self.Y]
        optimizer = torch.optim.Adam(params, lr=self.lr)

        log_pop = np.log1p(np.bincount(df.i.values, minlength=n_items)).astype(np.float32)
        self.log_pop = torch.tensor((log_pop - log_pop.mean()) / (log_pop.std() + EPS), dtype=torch.float32)

        for epoch in range(1, self.epochs + 1):
            for batch in torch.randperm(len(users), generator=generator).split(BATCH):
                u, i, t, r = users[batch], items[batch], times[batch], ratings[batch]

                prediction = self._predict(u, i, t)
                loss = ((prediction - r) ** 2).mean()

                penalty = (self.bu[u] ** 2 + self.bi[i] ** 2 + (self.P[u] ** 2).sum(1) + (self.Q[i] ** 2).sum(1)).mean()
                if self.use_time:
                    penalty = penalty + (self.au[u] ** 2 + (self.AP[u] ** 2).sum(1) + (self.bi_bin[i] ** 2).sum(1)).mean()
                penalty = penalty + (self.Y ** 2).sum() / n_items

                optimizer.zero_grad()
                (loss + self.reg * penalty).backward()
                optimizer.step()

            if on_epoch is not None and on_epoch(self, epoch):
                break

        return self

    @torch.no_grad()
    def rmse(self, df):
        """RMSE of the predicted ratings of df."""
        users = torch.tensor(df.u.values, dtype=torch.long)
        items = torch.tensor(df.i.values, dtype=torch.long)
        times = torch.tensor(df.t.values, dtype=torch.float32)
        ratings = torch.tensor(df.rating.values, dtype=torch.float32)

        prediction = self._predict(users, items, times).clamp(MIN_RATING, MAX_RATING)
        return float(((prediction - ratings) ** 2).mean().sqrt())

    @torch.no_grad()
    def score(self, users):
        """Ranking score at the recommendation time T (user-only terms do not change the order of items)."""
        u = torch.as_tensor(users, dtype=torch.long)
        user_factor = self.P[u] + self._implicit(u)
        item_bias = self.bi[None, :]

        if self.use_time:
            dev = self._dev(u, torch.full((len(u),), float(self.T)))
            user_factor = user_factor + self.AP[u] * dev[:, None]
            item_bias = item_bias + self.bi_bin[:, -1][None, :]

        scores = user_factor @ self.Q.T + item_bias
        return (scores + self.pop_blend * self.log_pop[None, :]).numpy()
