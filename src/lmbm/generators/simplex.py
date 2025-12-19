from typing import Callable

import numpy as np
from lmbm.functions import Softmax, sample_simplex_matrix


class SimplexCombiner:

    def __init__(self, S: np.ndarray = None, n: int = 2,
                 context: int = 1000, beta: float = 1 / 2,
                 prob_distribution_normalizer: Callable[[np.ndarray], np.ndarray] = None):
        if S is None:
            S = sample_simplex_matrix(n, ddf=.9)
        self.S = S
        self.context = context
        self.beta = beta
        if prob_distribution_normalizer is None:
            prob_distribution_normalizer = Softmax()
        self.to_prob_distribution = prob_distribution_normalizer

    def encode(self, seq: np.ndarray) -> np.ndarray:
        seq = seq[-self.context:]
        a = self.beta ** (np.arange(len(seq))[::-1])  # Larger beta for more recent tokens
        an = self.to_prob_distribution(a)
        return self.S[:, seq].dot(an)
