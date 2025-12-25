import numpy as np


class Softmax:
    def __init__(self, t: float = 1) -> np.ndarray:
        """
        :param t: Temperature.
        """
        self.t = t

    def __call__(self, x: np.ndarray) -> np.ndarray:
        m = np.max(x)  # To prevent overflow
        v = np.exp((x - m) / self.t)
        return v / np.sum(v)


def sample_simplex_matrix(n: int = 2, ddf: float = 0) -> np.ndarray:
    """
    Generate a square random matrix whose columns are points in the simplex.
    :param n: Number and dimension of the points.
    :param ddf: Diagonal deflation factor.
    :return: An ndarray containing the matrix.
    """
    S = np.random.random((n, n))
    S -= ddf * np.diag(np.diag(S))
    return S / np.sum(S, axis=0)


def sample_simplex_matrix_concentrated(n: int = 2, ddf: int = 0, noise: float = 0) -> np.ndarray:
    """
    Generate a square random matrix whose columns are points in the simplex.
    :param noise: Uniform noise multiplier.
    :param n: Number and dimension of the points.
    :param ddf: Diagonal deflation factor.
    :return: An ndarray containing the matrix.
    """
    S = np.eye(n) + np.random.random((n, n)) * noise
    S = np.roll(S, shift=1, axis=1)
    S -= ddf * np.diag(np.diag(S))
    return S / np.sum(S, axis=0)