import numpy as np
import scipy.linalg as la


class EASE:
    def __init__(self, lam=500.0, threshold=1):
        self.lam = lam
        self.threshold = threshold

    def fit(self, task):
        X = task.matrix(self.threshold)

        gram = (X.T @ X).toarray().astype(np.float64)
        gram[np.diag_indices_from(gram)] += self.lam
        P = la.inv(gram, overwrite_a=True, check_finite=False)

        B = P / -np.diag(P)
        B[np.diag_indices_from(B)] = 0.0

        self.B = B.astype(np.float32)
        self.X = X
        return self

    def score(self, users):
        return self.X[users] @ self.B
