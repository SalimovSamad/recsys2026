import numpy as np
import scipy.sparse as sp

EPS = 1e-9


class ItemKNN:
    def __init__(self, k=100, threshold=1):
        self.k = k
        self.threshold = threshold

    def fit(self, task):
        X = task.matrix(self.threshold)
        norm = np.sqrt(X.sum(0).A1) + EPS

        S = (X.T @ X).toarray()
        S /= norm[:, None]
        S /= norm[None, :]
        np.fill_diagonal(S, 0.0)

        top = np.argpartition(-S, self.k, axis=1)[:, :self.k]
        rows = np.repeat(np.arange(len(S)), self.k)
        S_k = sp.csr_matrix((np.take_along_axis(S, top, 1).ravel(), (rows, top.ravel())), shape=S.shape)

        self.similarity = S_k.T.tocsr().astype(np.float32)
        self.X = X
        return self

    def score(self, users):
        return (self.X[users] @ self.similarity).toarray()
