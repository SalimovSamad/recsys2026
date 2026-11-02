import numpy as np

from common import REL_THRESHOLD


class Popularity:
    def fit(self, task):
        self.popularity = task.matrix(REL_THRESHOLD).sum(0).A1
        return self

    def score(self, users):
        return np.tile(self.popularity, (len(users), 1))
