import numpy as np
from scipy.stats import entropy

from lmbm.encoding import SequenceToGamma, SequenceToSimplex
from lmbm.functions import sample_simplex_matrix


class SimplexCombiner:

    def __init__(self, S: np.ndarray = None, n: int = 1, context: int = 1000,
                 seq2real=None, seq2simplex=None):
        """

        :param S: Matrix of word embeddings as columns.
        :param n: Vocabulary size.
        :param context: Context length, i.e. max number of elements in the input sequence used to compute the next token distribution.
        :param seq2real:
        """
        if S is None:
            assert n is not None, "S and n cannot both be None."
            S = sample_simplex_matrix(n, ddf=.9)
        self.S = S
        n = S.shape[0]
        self.lexicon = np.arange(n)
        self.context = context

        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.S)
        self.eigenvectors_inv = np.linalg.inv(self.eigenvectors)

        self.entropy_estimate = EntropyEstimate()

        if seq2real is None:
            enc = SequenceToGamma(alpha=1, lambda_=1)
            self.seq2real = lambda x: 2 + enc(x)
        else:
            self.seq2real = seq2real

        if seq2simplex is None:
            self.seq2simplex = SequenceToSimplex(n_components=context, theta=.9)
        else:
            self.seq2simplex = seq2simplex

    def encode(self, seq: np.ndarray) -> np.ndarray:

        seq = seq[-self.context:]
        an = self.seq2simplex(seq)
        an = an[-len(seq):]
        if len(seq) != self.context:
            an = an/np.sum(an)
        exponent = self.seq2real(seq)
        diagonal_items = np.power(self.eigenvalues, exponent, dtype=complex)
        W = (self.eigenvectors * diagonal_items).dot(
            self.eigenvectors_inv[:, seq])
        assert an.shape[0] == W.shape[1], \
            f"Dimension mismatch. Coefficients vector: {an.shape}. W: {W.shape}"
        return W.dot(an).real  # TODO Safe? Always real?

    def generate(self, n: int, items: list = None) -> list:
        assert n > 0, "N must be at least 1."
        if items is None:
            items = np.zeros(n, dtype=np.int32)
            items[0] = int(np.random.choice(self.lexicon, 1)[0])
        for i in range(1, n):
            p = self.encode(items[:i])
            p = np.round(p, 12)  # I observe numerical zeroes below zero
            self.entropy_estimate.add_sample(p)
            token = int(np.random.choice(self.lexicon, 1, p=p)[0])
            items[i] = token
        return items


class EntropyEstimate:
    def __init__(self):
        self.running_estimate = 0
        self.n_samples = 0

    def add_sample(self, p: np.ndarray[float]):
        self.running_estimate = \
            ((self.n_samples * self.running_estimate + entropy(p, nan_policy='omit')) / (
                    self.n_samples + 1))
        self.n_samples += 1
