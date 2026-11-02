import numpy as np
import torch
import torch.nn.functional as F

from common import SEED
from features import FMFeatures, age_token_ids

BATCH = 4096
INIT_STD = 0.05


class FM:
    def __init__(self, dim=32, lr=5e-3, l2=1e-4, epochs=20, n_neg=2, threshold=4,
                 use_meta=True, meta_reg=1.0, seed=SEED):
        self.dim = dim
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.n_neg = n_neg
        self.threshold = threshold
        self.use_meta = use_meta
        self.meta_reg = meta_reg
        self.seed = seed

    def _forward(self, fields):
        """FM score of every row of token ids and its L2 penalty (metadata tokens are scaled by meta_reg)"""
        e = self.E(fields)
        linear = self.w(fields).squeeze(-1)
        pairwise = 0.5 * ((e.sum(1) ** 2).sum(1) - (e ** 2).sum((1, 2)))

        weight = self.token_reg[fields]
        penalty = (weight[..., None] * e ** 2).sum() + (weight * linear ** 2).sum()
        return linear.sum(1) + pairwise, penalty

    def _rows(self, u, i, t):
        fields = [self.user_fields[u], self.item_fields[i]]
        if self.use_meta:
            fields.append(age_token_ids(self.pub_year[i], t, self.feat.age_base)[:, None])
        return torch.cat(fields, 1)

    def fit(self, task, on_epoch=None):
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        self.T = task.T

        self.feat = FMFeatures(task.split, self.use_meta)
        self.user_fields = torch.tensor(self.feat.user_fields(task))
        self.item_fields = torch.tensor(self.feat.item_fields)
        self.pub_year = self.feat.pub_year if self.use_meta else None

        self.E = torch.nn.Embedding(self.feat.n_feat, self.dim, padding_idx=0)
        self.w = torch.nn.Embedding(self.feat.n_feat, 1, padding_idx=0)
        torch.nn.init.normal_(self.E.weight, std=INIT_STD)
        torch.nn.init.zeros_(self.w.weight)
        with torch.no_grad():
            self.E.weight[0].zero_()

        id_tokens = np.concatenate([self.feat.user_id_tok, self.feat.item_id_tok])
        self.token_reg = torch.full((self.feat.n_feat,), float(self.meta_reg))
        self.token_reg[torch.tensor(id_tokens)] = 1.0
        self.token_reg[0] = 0.0
        optimizer = torch.optim.Adam(list(self.E.parameters()) + list(self.w.parameters()), lr=self.lr)

        df = task.fit_df[task.fit_df.rating >= self.threshold]
        users = torch.tensor(df.u.values, dtype=torch.long)
        items = torch.tensor(df.i.values, dtype=torch.long)
        times = torch.tensor(df.t.values, dtype=torch.float32)
        candidates = torch.tensor(np.where(task.candidate_mask)[0], dtype=torch.long)

        for epoch in range(1, self.epochs + 1):
            for batch in torch.randperm(len(users), generator=generator).split(BATCH):
                u, i, t = users[batch], items[batch], times[batch]
                u_neg, t_neg = u.repeat(self.n_neg), t.repeat(self.n_neg)
                j = candidates[torch.randint(len(candidates), (len(u_neg),), generator=generator)]

                pos, pos_reg = self._forward(self._rows(u, i, t))
                neg, neg_reg = self._forward(self._rows(u_neg, j, t_neg))
                bpr = F.softplus(-(pos.repeat(self.n_neg) - neg)).mean()
                loss = bpr + self.l2 * (pos_reg / len(u) + neg_reg / len(u_neg))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if on_epoch is not None and on_epoch(self, epoch):
                break

        return self

    @torch.no_grad()
    def score(self, users):
        fields = self.item_fields
        if self.use_meta:
            age = age_token_ids(self.pub_year, torch.full((len(fields),), float(self.T)), self.feat.age_base)
            fields = torch.cat([fields, age[:, None]], 1)

        e = self.E(fields)
        e_item = e.sum(1)
        item_bias = self.w(fields).squeeze(-1).sum(1) + 0.5 * ((e_item ** 2).sum(1) - (e ** 2).sum((1, 2)))

        e_user = self.E(self.user_fields[torch.as_tensor(users, dtype=torch.long)]).sum(1)
        return (e_user @ e_item.T + item_bias[None, :]).numpy()
