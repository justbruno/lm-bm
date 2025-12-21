import hashlib
from scipy.special import gammaincinv
from typing import Union, Sequence


class SequenceToGamma:
    """
    Maps sequences of elements to positive real values following a Gamma distribution.

    Uses SHA-256 for efficient hashing and inverse transform sampling to achieve
    the target Gamma distribution.

    Attributes:
        alpha (float): Shape parameter of the Gamma distribution (α > 0)
        lambda_ (float): Rate parameter of the Gamma distribution (λ > 0)
                        Alternatively, scale parameter theta = 1/lambda_
        scale (float): Equivalent to 1/lambda_ (scale parameterization)
    """

    def __init__(self, alpha: float, lambda_: float = 1.0):
        """
        Initialize the Gamma distribution parameters.

        Args:
            alpha (float): Shape parameter (α > 0). Controls distribution skewness.
            lambda_ (float): Rate parameter (λ > 0). Default is 1.0.
                           Use scale = 1/lambda_ for scale parameterization.

        Raises:
            ValueError: If alpha <= 0 or lambda_ <= 0
        """
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if lambda_ <= 0:
            raise ValueError(f"lambda_ must be positive, got {lambda_}")

        self.alpha = alpha
        self.lambda_ = lambda_
        self.scale = 1.0 / lambda_  # For convenience

    def _hash_to_uniform(self, sequence: Union[Sequence, str, bytes]) -> float:
        """
        Convert a sequence to a uniform random variable in [0, 1).

        Uses SHA-256 for fast, deterministic hashing.

        Args:
            sequence: The input sequence (list, tuple, string, or bytes)

        Returns:
            float: A value in [0, 1)
        """
        # Convert sequence to bytes
        if isinstance(sequence, bytes):
            data = sequence
        elif isinstance(sequence, str):
            data = sequence.encode('utf-8')
        else:
            # For lists/tuples, convert to string representation then encode
            data = str(sequence).encode('utf-8')

        # Hash using SHA-256
        hash_digest = hashlib.sha256(data).digest()

        # Convert first 8 bytes to integer and normalize to [0, 1)
        hash_int = int.from_bytes(hash_digest[:8], byteorder='big')
        uniform = (hash_int % (2 ** 63)) / (2 ** 63)

        return uniform

    def __call__(self, sequence: Union[Sequence, str, bytes]) -> float:
        """
        Map a sequence to a Gamma-distributed value.

        Args:
            sequence: The input sequence (list, tuple, string, or bytes)

        Returns:
            float: A positive real value following Gamma(α, λ)
        """
        # Step 1: Hash sequence to uniform [0, 1)
        uniform = self._hash_to_uniform(sequence)

        # Step 2: Inverse transform sampling
        # Using regularized incomplete gamma function inverse (gammaincinv)
        # gammaincinv(a, y) computes the inverse of the regularized lower incomplete gamma
        # For Gamma(alpha, lambda), X = gammaincinv(alpha, uniform) / lambda_
        gamma_value = gammaincinv(self.alpha, uniform) / self.lambda_

        return gamma_value

    def map(self, sequence: Union[Sequence, str, bytes]) -> float:
        """
        Alias for __call__. Maps a sequence to a Gamma-distributed value.

        Args:
            sequence: The input sequence

        Returns:
            float: A positive real value following Gamma(α, λ)
        """
        return self(sequence)

    def __repr__(self) -> str:
        return f"SequenceToGamma(alpha={self.alpha}, lambda_={self.lambda_})"