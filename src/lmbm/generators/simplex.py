from typing import Callable

import numpy as np
from lmbm.functions import Softmax, sample_simplex_matrix, Lexicon


class SimplexCombiner:

    def __init__(self, S: np.ndarray = None, lexicon: Lexicon = None,
                 context: int = 1000, beta: float = 1 / 2,
                 prob_distribution_normalizer: Callable[[np.ndarray], np.ndarray] = None):
        n = lexicon.size
        if S is None:
            S = sample_simplex_matrix(n, ddf=.9)
        self.S = S
        self.lexicon = lexicon
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

    def generate(self, n: int, input: np.ndarray = None) -> list:
        if input is None:
            input = np.random.choice(self.lexicon.lexicon, 1)
        for i in range(n):
            p = self.encode(input)
            token = np.random.choice(self.lexicon.lexicon, 1, p=p)
            input = np.hstack([input, token])
        return input
