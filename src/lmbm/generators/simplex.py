from typing import Callable

import numpy as np
from lmbm.functions import Softmax, sample_simplex_matrix, Lexicon
from lmbm.encoding import SequenceToGamma
import time
from scipy.stats import entropy


class SimplexCombiner:

    def __init__(self, S: np.ndarray = None, lexicon: Lexicon = None,
                 context: int = 1000, beta: float = 1 / 2,
                 prob_distribution_normalizer: Callable[[np.ndarray], np.ndarray] = None,
                 seq2real=None):
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

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.S)
        self.eigenvectors_inv = np.linalg.inv(self.eigenvectors)

        self.entropy_estimate = EntropyEstimate()

        if seq2real is None:
            enc = SequenceToGamma(alpha=1, lambda_=1)
            self.seq2real = lambda x: 2 + enc(x)
        else:
            self.seq2real = seq2real

    def encode(self, seq: np.ndarray) -> np.ndarray:

        seq = seq[-self.context:]

        a = self.beta ** (np.arange(len(seq))[::-1])  # Larger beta for more recent tokens

        an = self.to_prob_distribution(a)

        # start = time.time()
        # exponent = np.round(self.seq2real(seq))
        exponent = self.seq2real(seq)
        # print('exp: ', time.time() - start)

        # start = time.time()
        diagonal_items = np.power(self.eigenvalues, exponent, dtype=complex)
        # print('diagonal_items: ', time.time() - start)
        # start = time.time()
        W = (self.eigenvectors * diagonal_items).dot(
            self.eigenvectors_inv[:, seq])
        # print('W: ', time.time() - start)
        # print()
        return W.dot(an).real  # TODO Safe? Always real?

    def generate(self, n: int, items: list = None) -> list:
        assert n > 0, "N must be at least 1."
        if items is None:
            items = np.zeros(n, dtype=np.int32)
            items[0] = int(np.random.choice(self.lexicon.lexicon, 1)[0])
        for i in range(1, n):
            p = self.encode(items[:i])
            p = np.round(p, 12)  # I observe numerical zeroes below zero
            self.entropy_estimate.add_sample(p)
            token = int(np.random.choice(self.lexicon.lexicon, 1, p=p)[0])
            items[i] = token
        return items


class EntropyEstimate:
    def __init__(self):
        self.running_estimate = 0
        self.n_samples = 0

    def add_sample(self, p: np.ndarray[float]):
        # entropy = -p.dot(np.log(p))
        self.running_estimate = \
            ((self.n_samples * self.running_estimate + entropy(p, nan_policy='omit')) / (
                        self.n_samples + 1))
        self.n_samples += 1
