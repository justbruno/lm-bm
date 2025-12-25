"""
Advanced pytest fixtures and utilities for lmbm testing.

This module provides reusable fixtures, parametrization helpers,
and custom assertions for comprehensive testing.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from dataclasses import dataclass

from lmbm.dataset import LongSequenceDataset, MultiSequenceDataset
from lmbm.models.lstm import NextIntLSTM
from lmbm.models.mlp import NextIntMLP
from lmbm.models.transformer import NextIntTransformer
from lmbm.encoding import SequenceToGamma, SequenceToPoisson, SequenceToSimplex
from lmbm.functions import Softmax, sample_simplex_matrix


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def small_vocab():
    """Small vocabulary size for quick tests."""
    return 50


@pytest.fixture
def medium_vocab():
    """Medium vocabulary size for standard tests."""
    return 100


@pytest.fixture
def small_sequence_length():
    """Short sequence for fast iteration."""
    return 5


@pytest.fixture
def medium_sequence_length():
    """Standard sequence length."""
    return 10


@pytest.fixture
def small_batch_size():
    """Small batch size for memory efficiency."""
    return 4


@pytest.fixture
def medium_batch_size():
    """Standard batch size."""
    return 16


@pytest.fixture
def random_token_sequence(small_vocab, medium_sequence_length):
    """Random token sequence for testing."""
    return torch.randint(0, small_vocab, (100,))


@pytest.fixture
def random_batch(small_vocab, small_sequence_length, small_batch_size):
    """Random batch of sequences."""
    return torch.randint(0, small_vocab, (small_batch_size, small_sequence_length))


@pytest.fixture
def deterministic_seed():
    """Set deterministic seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield
    # Cleanup (optional)


# ============================================================================
# MODEL FIXTURES
# ============================================================================

@pytest.fixture
def lstm_model(small_vocab, small_sequence_length):
    """LSTM model for testing."""
    return NextIntLSTM(
        vocab_size=small_vocab,
        embed_dim=16,
        hidden_dim=32,
        num_layers=2,
        dropout=0.1
    )


@pytest.fixture
def mlp_model(small_vocab, small_sequence_length):
    """MLP model for testing."""
    return NextIntMLP(
        vocab_size=small_vocab,
        embed_dim=16,
        seq_len=small_sequence_length,
        hidden_dim=32,
        num_hidden_layers=2
    )


@pytest.fixture
def transformer_model(small_vocab, small_sequence_length):
    """Transformer model for testing."""
    return NextIntTransformer(
        vocab_size=small_vocab,
        seq_len=small_sequence_length,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.1
    )


@pytest.fixture(params=['lstm', 'mlp', 'transformer'])
def all_models(request, small_vocab, small_sequence_length):
    """Parametrized fixture for all model types."""
    if request.param == 'lstm':
        return NextIntLSTM(
            vocab_size=small_vocab,
            embed_dim=16,
            hidden_dim=32,
            num_layers=2
        )
    elif request.param == 'mlp':
        return NextIntMLP(
            vocab_size=small_vocab,
            embed_dim=16,
            seq_len=small_sequence_length,
            hidden_dim=32,
            num_hidden_layers=2
        )
    else:  # transformer
        return NextIntTransformer(
            vocab_size=small_vocab,
            seq_len=small_sequence_length,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64
        )


# ============================================================================
# DATASET FIXTURES
# ============================================================================

@pytest.fixture
def simple_dataset(random_token_sequence, small_sequence_length):
    """Simple single-sequence dataset."""
    return LongSequenceDataset(random_token_sequence, seq_len=small_sequence_length)


@pytest.fixture
def multi_dataset(small_sequence_length):
    """Multi-sequence dataset."""
    sequences = [
        torch.randint(0, 50, (100,)),
        torch.randint(0, 50, (80,)),
        torch.randint(0, 50, (120,))
    ]
    return MultiSequenceDataset(sequences, seq_len=small_sequence_length)


@pytest.fixture
def dataloader(simple_dataset, small_batch_size):
    """DataLoader for dataset."""
    return torch.utils.data.DataLoader(
        simple_dataset,
        batch_size=small_batch_size,
        shuffle=True
    )


# ============================================================================
# ENCODING FIXTURES
# ============================================================================

@pytest.fixture
def gamma_encoder():
    """SequenceToGamma encoder."""
    return SequenceToGamma(alpha=2.0, lambda_=1.0)


@pytest.fixture
def poisson_encoder():
    """SequenceToPoisson encoder."""
    return SequenceToPoisson(lambda_=5.0)


@pytest.fixture
def simplex_mapper():
    """SequenceToSimplex mapper."""
    return SequenceToSimplex(n_components=8, theta=0.5)


@pytest.fixture
def softmax():
    """Softmax utility."""
    return Softmax(t=1.0)


# ============================================================================
# PARAMETRIZATION HELPERS
# ============================================================================

# Common parameter sets for tests
VOCAB_SIZES = [10, 50, 100]
SEQUENCE_LENGTHS = [5, 10, 20]
BATCH_SIZES = [1, 4, 8, 16]
EMBEDDING_DIMS = [8, 16, 32]
HIDDEN_DIMS = [16, 32, 64]

PARAMETRIZE_VOCAB_SIZES = pytest.mark.parametrize("vocab_size", VOCAB_SIZES)
PARAMETRIZE_SEQUENCE_LENGTHS = pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS)
PARAMETRIZE_BATCH_SIZES = pytest.mark.parametrize("batch_size", BATCH_SIZES)
PARAMETRIZE_EMBEDDING_DIMS = pytest.mark.parametrize("embed_dim", EMBEDDING_DIMS)


# ============================================================================
# CUSTOM ASSERTIONS
# ============================================================================

class TensorAssertions:
    """Custom assertions for PyTorch tensors."""
    
    @staticmethod
    def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {tensor.shape}"
    
    @staticmethod
    def assert_tensor_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype):
        """Assert tensor has expected dtype."""
        assert tensor.dtype == expected_dtype, \
            f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    @staticmethod
    def assert_tensor_device(tensor: torch.Tensor, expected_device: str):
        """Assert tensor is on expected device."""
        assert str(tensor.device) == expected_device, \
            f"Expected device {expected_device}, got {tensor.device}"
    
    @staticmethod
    def assert_tensors_close(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                            rtol: float = 1e-5, atol: float = 1e-8):
        """Assert two tensors are numerically close."""
        assert torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol), \
            f"Tensors differ: max diff = {(tensor1 - tensor2).abs().max().item()}"
    
    @staticmethod
    def assert_no_nan_inf(tensor: torch.Tensor):
        """Assert tensor contains no NaN or Inf values."""
        assert not torch.isnan(tensor).any(), "Tensor contains NaN"
        assert not torch.isinf(tensor).any(), "Tensor contains Inf"


class NumpyAssertions:
    """Custom assertions for NumPy arrays."""
    
    @staticmethod
    def assert_array_shape(array: np.ndarray, expected_shape: Tuple):
        """Assert array has expected shape."""
        assert array.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {array.shape}"
    
    @staticmethod
    def assert_probability_vector(array: np.ndarray, name: str = "vector"):
        """Assert array is a valid probability vector (sums to 1, values in [0,1])."""
        assert np.allclose(np.sum(array), 1.0), \
            f"{name} does not sum to 1.0: sum={np.sum(array)}"
        assert np.all(array >= 0), f"{name} contains negative values"
        assert np.all(array <= 1), f"{name} contains values > 1"
    
    @staticmethod
    def assert_simplex_matrix(matrix: np.ndarray, name: str = "matrix"):
        """Assert matrix is valid simplex (columns are probability vectors)."""
        col_sums = np.sum(matrix, axis=0)
        assert np.allclose(col_sums, 1.0), \
            f"{name} columns do not sum to 1.0: sums={col_sums}"
        assert np.all(matrix >= 0), f"{name} contains negative values"
    
    @staticmethod
    def assert_no_nan_inf(array: np.ndarray):
        """Assert array contains no NaN or Inf values."""
        assert not np.isnan(array).any(), "Array contains NaN"
        assert not np.isinf(array).any(), "Array contains Inf"


@pytest.fixture
def tensor_assert():
    """Fixture providing tensor assertions."""
    return TensorAssertions


@pytest.fixture
def array_assert():
    """Fixture providing numpy array assertions."""
    return NumpyAssertions


# ============================================================================
# MODEL TESTING UTILITIES
# ============================================================================

@dataclass
class ModelTestConfig:
    """Configuration for model testing."""
    vocab_size: int = 50
    seq_len: int = 10
    batch_size: int = 4
    device: str = 'cpu'
    
    def create_batch(self):
        """Create a test batch."""
        return torch.randint(0, self.vocab_size, 
                           (self.batch_size, self.seq_len))


@pytest.fixture
def model_test_config():
    """Model testing configuration."""
    return ModelTestConfig()


class ModelTestUtility:
    """Utilities for testing neural network models."""
    
    @staticmethod
    def test_forward_pass(model: nn.Module, input_tensor: torch.Tensor,
                         expected_output_shape: Tuple):
        """Test model forward pass."""
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == expected_output_shape
        TensorAssertions.assert_no_nan_inf(output)
        return output
    
    @staticmethod
    def test_gradients(model: nn.Module, input_tensor: torch.Tensor,
                       target_tensor: torch.Tensor):
        """Test that gradients flow through model."""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        model.train()
        output = model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        
        # Check that gradients were computed
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.all(param.grad == 0), \
                    f"No gradient computed for {name}"
        
        optimizer.step()
    
    @staticmethod
    def test_model_state_save_load(model: nn.Module, tmpdir):
        """Test model state save/load."""
        import tempfile
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
            torch.save(model.state_dict(), path)
        
        # Load
        model2 = model.__class__(**model.__dict__)
        model2.load_state_dict(torch.load(path))
        
        # Verify same state
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)


@pytest.fixture
def model_utility():
    """Model testing utility."""
    return ModelTestUtility


# ============================================================================
# STATISTICAL TESTING UTILITIES
# ============================================================================

class DistributionTester:
    """Utilities for testing distributions."""
    
    @staticmethod
    def kolmogorov_smirnov_test(samples: np.ndarray, cdf_func, 
                                alpha: float = 0.05) -> bool:
        """Simple KS test for distribution."""
        from scipy.stats import kstest
        stat, pvalue = kstest(samples, cdf_func)
        return pvalue > alpha
    
    @staticmethod
    def anderson_darling_test(samples: np.ndarray, distribution: str = 'norm',
                             alpha: float = 0.05) -> bool:
        """Anderson-Darling test for distribution."""
        from scipy.stats import anderson
        result = anderson(samples, dist=distribution)
        # Compare with critical values
        return result.statistic < result.critical_values[-1]
    
    @staticmethod
    def empirical_mean_variance(samples: np.ndarray, 
                               expected_mean: float, 
                               expected_var: float,
                               tolerance: float = 0.15):
        """Test empirical mean and variance."""
        emp_mean = np.mean(samples)
        emp_var = np.var(samples)
        
        mean_error = abs(emp_mean - expected_mean) / expected_mean
        var_error = abs(emp_var - expected_var) / expected_var
        
        assert mean_error < tolerance, \
            f"Mean error {mean_error:.2%} > tolerance {tolerance:.2%}"
        assert var_error < tolerance, \
            f"Variance error {var_error:.2%} > tolerance {tolerance:.2%}"


@pytest.fixture
def distribution_tester():
    """Distribution testing utility."""
    return DistributionTester


# ============================================================================
# PERFORMANCE TESTING UTILITIES
# ============================================================================

import time
from contextlib import contextmanager


@contextmanager
def timer(name: str = "Operation"):
    """Context manager for timing code blocks."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"\n{name} took {elapsed:.4f}s")


class PerformanceTester:
    """Utilities for performance testing."""
    
    @staticmethod
    def time_function(func, *args, n_runs: int = 100, **kwargs) -> Tuple[float, float]:
        """Time function execution, return mean and std."""
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            times.append(time.perf_counter() - start)
        
        return np.mean(times), np.std(times)
    
    @staticmethod
    def benchmark_models(models: dict, input_batch: torch.Tensor,
                        n_runs: int = 100) -> dict:
        """Benchmark multiple models."""
        results = {}
        
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                times = []
                for _ in range(n_runs):
                    start = time.perf_counter()
                    _ = model(input_batch)
                    times.append(time.perf_counter() - start)
            
            results[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            }
        
        return results


@pytest.fixture
def performance_tester():
    """Performance testing utility."""
    return PerformanceTester


# ============================================================================
# EXAMPLE USAGE IN TESTS
# ============================================================================

def test_example_with_fixtures(lstm_model, random_batch, tensor_assert):
    """Example test using fixtures and custom assertions."""
    lstm_model.eval()
    output = lstm_model(random_batch)
    
    # Use custom assertions
    tensor_assert.assert_tensor_shape(output, (4, 50))
    tensor_assert.assert_no_nan_inf(output)


@PARAMETRIZE_VOCAB_SIZES
@PARAMETRIZE_BATCH_SIZES
def test_example_parametrized(vocab_size, batch_size):
    """Example parametrized test."""
    model = NextIntMLP(
        vocab_size=vocab_size,
        embed_dim=16,
        seq_len=10,
        hidden_dim=32
    )
    
    batch = torch.randint(0, vocab_size, (batch_size, 10))
    output = model(batch)
    
    assert output.shape == (batch_size, vocab_size)


def test_example_performance(performance_tester, lstm_model, mlp_model, random_batch):
    """Example performance test."""
    models = {'lstm': lstm_model, 'mlp': mlp_model}
    results = performance_tester.benchmark_models(models, random_batch, n_runs=100)
    
    print("\nBenchmark Results:")
    for name, metrics in results.items():
        print(f"  {name}: {metrics['mean']*1000:.2f}ms ± {metrics['std']*1000:.2f}ms")
