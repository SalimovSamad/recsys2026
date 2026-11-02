import numpy as np
import scipy.linalg as la

from common import SEED

INIT_STD = 0.01


def solve_rows(C, Y, reg, alpha):
    gram = (Y.T @ Y).astype(np.float64) + reg * np.eye(Y.shape[1])
    out = np.zeros((C.shape[0], Y.shape[1]), dtype=np.float32)

    for r in range(C.shape[0]):
        start, end = C.indptr[r], C.indptr[r + 1]
        if start == end:
            continue

        Yr = Y[C.indices[start:end]]
        c = alpha * C.data[start:end]
        out[r] = la.solve(gram + (Yr.T * c) @ Yr, (1.0 + c) @ Yr, assume_a="pos", check_finite=False)

    return out


class IALS:
    def __init__(self, factors=64, reg=0.1, alpha=20.0, epochs=10, threshold=1, seed=SEED):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.epochs = epochs
        self.threshold = threshold
        self.seed = seed

    def fit(self, task, on_epoch=None):
        X = task.matrix(self.threshold)
        XT = X.T.tocsr()

        rng = np.random.default_rng(self.seed)
        self.U = (INIT_STD * rng.standard_normal((X.shape[0], self.factors))).astype(np.float32)
        self.V = (INIT_STD * rng.standard_normal((X.shape[1], self.factors))).astype(np.float32)

        for epoch in range(1, self.epochs + 1):
            self.U = solve_rows(X, self.V, self.reg, self.alpha)
            self.V = solve_rows(XT, self.U, self.reg, self.alpha)

            if on_epoch is not None and on_epoch(self, epoch):
                break

        return self

    def score(self, users):
        return self.U[users] @ self.V.T
