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

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.S)
        self.eigenvectors_inv = np.linalg.inv(self.eigenvectors)

    def encode(self, seq: np.ndarray) -> np.ndarray:
        seq = seq[-self.context:]
        a = self.beta ** (np.arange(len(seq))[::-1])  # Larger beta for more recent tokens

        # TODO Add eigenvalue decomposition to consider long range dependencies.
        #  Note we need a deterministic mapping to a real for the exponent, adequately distributed (e.g. gamma + 2)
        #  We want to see if 1.01 is still easy to learn
        #  Heck, is 1 even easy to learn? Try an MLP or LSTM
        # Core idea:
        # v,V = np.linalg.eig(S) # This can go in constructor
        # W = V.dot(np.diag(v**2.4)).dot(np.linalg.inv(V))
        # return W[:, seq].dot(an)

        an = self.to_prob_distribution(a)

        exponent = self.map_to_exponent(seq)
        W = self.eigenvectors.dot(np.diag(self.eigenvalues**exponent)).dot(self.eigenvectors_inv[:, seq])
        return W.dot(an)
        # return self.S[:, seq].dot(an)

    def map_to_exponent(self, seq: np.ndarray) -> float:
        """
        Map sequence to real >= 2.
        :param seq:
        :return:
        """
        # TODO
        return 2.0

    def generate(self, n: int, input: np.ndarray = None) -> list:
        if input is None:
            input = np.random.choice(self.lexicon.lexicon, 1)
        for i in range(n):
            p = self.encode(input)
            token = np.random.choice(self.lexicon.lexicon, 1, p=p)
            input = np.hstack([input, token])
        return input
