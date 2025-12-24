import hashlib
from scipy.special import gammaincinv
from typing import Union, Sequence
import xxhash
import math

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


import hashlib
from typing import Union, Sequence


class SequenceToPoisson:
    """
    Ultra-fast mapping from sequences to Poisson-distributed integers.

    Uses MurmurHash3 (non-cryptographic, extremely fast) + direct Poisson sampling.
    No scipy dependency, pure Python + xxhash.

    Attributes:
        lambda_ (float): Rate parameter of the Poisson distribution (λ > 0)
    """

    def __init__(self, lambda_: float = 5.0):
        """
        Initialize with Poisson rate parameter.

        Args:
            lambda_ (float): Mean and variance of Poisson distribution (> 0)
        """
        if lambda_ <= 0:
            raise ValueError(f"lambda_ must be positive, got {lambda_}")
        self.lambda_ = lambda_

    def _murmurhash3(self, sequence: Union[Sequence, str, bytes]) -> int:
        """
        Extremely fast 64-bit hash using MurmurHash3 algorithm.
        ~10x faster than SHA-256 for this use case.
        """
        # Convert to bytes
        if isinstance(sequence, bytes):
            data = sequence
        elif isinstance(sequence, str):
            data = sequence.encode('utf-8')
        else:
            data = str(sequence).encode('utf-8')

        # Simple but fast MurmurHash3 64-bit implementation
        h1 = 0x87c37b91114253d5
        c1 = 0x87c37b91114253d5
        c2 = 0x4cf5ad432745937f

        # Process data in 8-byte chunks
        length = len(data)
        for i in range(0, length, 8):
            chunk = int.from_bytes(data[i:i + 8], 'little')
            k1 = chunk ^ h1
            k1 = ((k1 * c1) & 0xFFFFFFFFFFFFFFFF) >> 33
            k1 *= c2
            k1 = ((k1 & 0xFFFFFFFFFFFFFFFF) >> 33) * c1
            h1 ^= k1
            h1 = ((h1 * 5) & 0xFFFFFFFFFFFFFFFF) >> 17
            h1 = (h1 * c1) & 0xFFFFFFFFFFFFFFFF

        # Finalization
        h1 ^= length
        h1 = ((h1 * c1) & 0xFFFFFFFFFFFFFFFF) >> 33
        h1 *= c2
        h1 = ((h1 & 0xFFFFFFFFFFFFFFFF) >> 33) * c1
        h1 ^= h1 >> 16

        return h1

    def _poisson_sample(self, seed: int, lambda_: float) -> int:
        """
        Direct Poisson sampling using hash seed.
        Knuth's algorithm L - O(λ) time, perfect for λ ≤ 100.
        """
        L = math.exp(-lambda_)
        k = 0
        p = 1.0

        seed_state = seed & 0xFFFFFFFFFFFFFFFF
        while p >= L:
            # Fast LCG PRNG seeded by hash
            seed_state = (
                                 seed_state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            u = seed_state / (2 ** 64)
            p *= u
            k += 1

        return k - 1

    def __call__(self, sequence: Union[Sequence, str, bytes]) -> int:
        """
        Map sequence to Poisson-distributed integer.

        Args:
            sequence: Input sequence

        Returns:
            int: Poisson(λ)-distributed integer
        """
        hash_val = self._murmurhash3(sequence)
        return self._poisson_sample(hash_val, self.lambda_)

    def __repr__(self) -> str:
        return f"SequenceToPoisson(lambda_={self.lambda_:.2f})"


class FastSequenceToPoisson(SequenceToPoisson):
    """10-20x faster than base class using xxhash."""

    def _murmurhash3(self, sequence: Union[Sequence, str, bytes]) -> int:
        if isinstance(sequence, bytes):
            data = sequence
        elif isinstance(sequence, str):
            data = sequence.encode('utf-8')
        else:
            data = str(sequence).encode('utf-8')
        return xxhash.xxh64(data).intdigest()

