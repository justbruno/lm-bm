import numpy as np
from scipy.stats import entropy

from lmbm.encoding import SequenceToGamma, SequenceToSimplex
from lmbm.functions import sample_simplex_matrix


class SimplexCombiner:

    def __init__(self, S: np.ndarray = None, n: int = 1, context: int = 1000,
                 seq2real=None, seq2simplex=None, seed: int = None):
        """

        :param S: Matrix of word embeddings as columns.
        :param n: Vocabulary size.
        :param context: Context length, i.e. max number of elements in the input sequence used to compute the next token distribution.
        :param seq2real:
        :param seed: Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)
        if S is None:
            assert n is not None, "S and n cannot both be None."
            S = sample_simplex_matrix(n, ddf=.9, rng=self.rng)
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
            self.seq2simplex = SequenceToSimplex(n_components=context, theta=.01)
        else:
            self.seq2simplex = seq2simplex

    def encode(self, seq: np.ndarray) -> np.ndarray:

        seq = seq[-self.context:]
        an = self.seq2simplex(seq)
        an = an[-len(seq):]
        if len(seq) != self.context:
            an = an/np.sum(an)
        exponent = self.seq2real(seq)
        diagonal_items = np.power(np.abs(self.eigenvalues), exponent)
        W = (self.eigenvectors * diagonal_items).dot(
            self.eigenvectors_inv[:, seq])
        assert an.shape[0] == W.shape[1], \
            f"Dimension mismatch. Coefficients vector: {an.shape}. W: {W.shape}"
        return W.dot(an).real

    def generate(self, n: int, seq: list = None) -> list:
        assert n > 0, "N must be at least 1."
        correction = 0
        if seq is None:
            seq = self.rng.choice(self.lexicon, 1)
            correction = 1
        items = np.hstack([seq, np.zeros(n - correction, dtype=np.int32)])
        start = len(seq)
        end = start + n - correction
        for i in range(start, end):
            p = self.encode(items[:i])
            p = np.clip(p, 0, None)
            p = p / np.sum(p)
            self.entropy_estimate.add_sample(p)
            token = int(self.rng.choice(self.lexicon, 1, p=p)[0])
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
