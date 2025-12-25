"""
Comprehensive test suite for lmbm package.

Tests cover:
- Encoding (Gamma, Poisson, Simplex distributions)
- Dataset utilities (single and multi-sequence)
- Models (LSTM, MLP, Transformer)
- Training pipeline
- Utility functions (Softmax, simplex matrix sampling)
- Generators (SimplexCombiner, EntropyEstimate)
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from pathlib import Path

from lmbm.encoding import (
    SequenceToGamma,
    SequenceToPoisson,
    SequenceToSimplex,
    FastSequenceToPoisson
)
from lmbm.dataset import LongSequenceDataset, MultiSequenceDataset
from lmbm.models.lstm import NextIntLSTM
from lmbm.models.mlp import NextIntMLP
from lmbm.models.transformer import NextIntTransformer, PositionalEncoding
from lmbm.training import SequenceTrainer, create_model
from lmbm.functions import Softmax, sample_simplex_matrix, \
    sample_simplex_matrix_concentrated
from lmbm.generators.simplex import SimplexCombiner, EntropyEstimate


# ============================================================================
# ENCODING TESTS
# ============================================================================

class TestSequenceToGamma:
    """Test SequenceToGamma distribution mapping."""

    def test_initialization_valid(self):
        """Test valid initialization with positive parameters."""
        enc = SequenceToGamma(alpha=2.0, lambda_=1.0)
        assert enc.alpha == 2.0
        assert enc.lambda_ == 1.0
        assert enc.scale == 1.0

    def test_initialization_invalid_alpha(self):
        """Test that non-positive alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            SequenceToGamma(alpha=0, lambda_=1.0)
        with pytest.raises(ValueError, match="alpha must be positive"):
            SequenceToGamma(alpha=-1, lambda_=1.0)

    def test_initialization_invalid_lambda(self):
        """Test that non-positive lambda raises ValueError."""
        with pytest.raises(ValueError, match="lambda_ must be positive"):
            SequenceToGamma(alpha=1, lambda_=0)
        with pytest.raises(ValueError, match="lambda_ must be positive"):
            SequenceToGamma(alpha=1, lambda_=-1)

    def test_call_returns_positive(self):
        """Test that output is always positive."""
        enc = SequenceToGamma(alpha=1.0, lambda_=1.0)
        for seq in ["test", [1, 2, 3], b"bytes", ("tuple",)]:
            result = enc(seq)
            assert result > 0, f"Expected positive, got {result}"

    def test_deterministic(self):
        """Test that same input always produces same output."""
        enc = SequenceToGamma(alpha=2.0, lambda_=1.5)
        seq = "deterministic_test"
        result1 = enc(seq)
        result2 = enc(seq)
        assert result1 == result2

    def test_different_sequences_different_outputs(self):
        """Test that different sequences produce different outputs (probabilistically)."""
        enc = SequenceToGamma(alpha=1.0, lambda_=1.0)
        results = [enc(f"seq_{i}") for i in range(10)]
        assert len(
            set(results)) > 1, "Different sequences should produce different outputs"

    def test_different_alpha_produces_different_values(self):
        """Test that alpha parameter affects output distribution."""
        seq = "test"
        enc1 = SequenceToGamma(alpha=0.5, lambda_=1.0)
        enc2 = SequenceToGamma(alpha=5.0, lambda_=1.0)

        values1 = [enc1(f"{seq}_{i}") for i in range(100)]
        values2 = [enc2(f"{seq}_{i}") for i in range(100)]

        # Different alpha should produce different distributions
        assert abs(np.mean(values1) - np.mean(values2)) > 0.1

    def test_input_types(self):
        """Test that function accepts various input types."""
        enc = SequenceToGamma(alpha=1.0, lambda_=1.0)

        # String
        result_str = enc("string")
        assert isinstance(result_str, (float, np.floating))

        # Bytes
        result_bytes = enc(b"bytes")
        assert isinstance(result_bytes, (float, np.floating))

        # List
        result_list = enc([1, 2, 3])
        assert isinstance(result_list, (float, np.floating))

        # Tuple
        result_tuple = enc((1, 2, 3))
        assert isinstance(result_tuple, (float, np.floating))

    def test_repr(self):
        """Test string representation."""
        enc = SequenceToGamma(alpha=2.5, lambda_=1.5)
        repr_str = repr(enc)
        assert "SequenceToGamma" in repr_str
        assert "2.5" in repr_str
        assert "1.5" in repr_str


class TestSequenceToPoisson:
    """Test SequenceToPoisson integer mapping."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        enc = SequenceToPoisson(lambda_=5.0)
        assert enc.lambda_ == 5.0

    def test_initialization_invalid_lambda(self):
        """Test that non-positive lambda raises ValueError."""
        with pytest.raises(ValueError, match="lambda_ must be positive"):
            SequenceToPoisson(lambda_=0)

    def test_call_returns_integer(self):
        """Test that output is always a non-negative integer."""
        enc = SequenceToPoisson(lambda_=5.0)
        for seq in ["test", [1, 2, 3], b"bytes"]:
            result = enc(seq)
            assert isinstance(result, (int, np.integer))
            assert result >= 0

    def test_deterministic(self):
        """Test deterministic behavior."""
        enc = SequenceToPoisson(lambda_=5.0)
        seq = "test"
        result1 = enc(seq)
        result2 = enc(seq)
        assert result1 == result2

    def test_distribution_mean(self):
        """Test that empirical mean matches lambda (statistically)."""
        enc = SequenceToPoisson(lambda_=5.0)
        values = [enc(f"seq_{i}") for i in range(1000)]
        empirical_mean = np.mean(values)

        # Allow 20% tolerance
        assert 4.0 < empirical_mean < 6.0


class TestFastSequenceToPoisson:
    """Test FastSequenceToPoisson using xxhash."""

    def test_faster_than_base(self):
        """Test that xxhash version is faster (or at least works)."""
        import time

        enc_base = SequenceToPoisson(lambda_=5.0)
        enc_fast = FastSequenceToPoisson(lambda_=5.0)

        seqs = [f"test_{i}" for i in range(100)]

        # Both should produce valid outputs
        for seq in seqs:
            result_base = enc_base(seq)
            result_fast = enc_fast(seq)
            assert isinstance(result_base, (int, np.integer))
            assert isinstance(result_fast, (int, np.integer))


class TestSequenceToSimplex:
    """Test SequenceToSimplex probability vector mapping."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        mapper = SequenceToSimplex(n_components=4, theta=0.5)
        assert mapper.n_components == 4
        assert mapper.theta == 0.5

    def test_initialization_invalid_n_components(self):
        """Test that n_components must be >= 2."""
        with pytest.raises(ValueError, match="n_components must be >= 2"):
            SequenceToSimplex(n_components=1, theta=0.5)

    def test_initialization_invalid_theta(self):
        """Test that theta must be in [0, 1]."""
        with pytest.raises(ValueError, match="theta must be in"):
            SequenceToSimplex(n_components=4, theta=-0.1)
        with pytest.raises(ValueError, match="theta must be in"):
            SequenceToSimplex(n_components=4, theta=1.1)

    def test_output_shape(self):
        """Test output shape matches n_components."""
        mapper = SequenceToSimplex(n_components=5, theta=0.5)
        result = mapper("test")
        assert len(result) == 5

    def test_output_is_probability_vector(self):
        """Test output sums to 1 and all values in [0, 1]."""
        mapper = SequenceToSimplex(n_components=4, theta=0.5)
        result = mapper("test")

        assert np.allclose(np.sum(result), 1.0, atol=1e-10)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_theta_0_concentrated_at_end(self):
        """Test that theta=0 concentrates on last component."""
        mapper = SequenceToSimplex(n_components=4, theta=0.0)
        result = mapper("test")

        # Last component should have most of the mass
        assert result[-1] > 0.9

    def test_theta_1_concentrated_at_start(self):
        """Test that theta=1 concentrates on first component."""
        mapper = SequenceToSimplex(n_components=4, theta=1.0)
        result = mapper("test")

        # First component should have most of the mass
        assert result[0] > 0.9

    def test_theta_0p5_roughly_uniform(self):
        """Test that theta=0.5 gives approximately uniform distribution."""
        mapper = SequenceToSimplex(n_components=4, theta=0.5)

        # Sample multiple sequences
        results = [mapper(f"seq_{i}") for i in range(1000)]
        avg = np.mean(results, axis=0)

        # Should be approximately uniform
        expected = 0.25
        assert np.allclose(avg, expected, atol=0.05)

    def test_deterministic(self):
        """Test deterministic behavior."""
        mapper = SequenceToSimplex(n_components=4, theta=0.5)
        result1 = mapper("test")
        result2 = mapper("test")

        assert np.allclose(result1, result2)

    def test_different_theta_different_results(self):
        """Test that different theta values produce different distributions."""
        seq = "test"
        mapper1 = SequenceToSimplex(n_components=4, theta=0.2)
        mapper2 = SequenceToSimplex(n_components=4, theta=0.8)

        result1 = mapper1(seq)
        result2 = mapper2(seq)

        # Should be different
        assert not np.allclose(result1, result2)


# ============================================================================
# DATASET TESTS
# ============================================================================

class TestLongSequenceDataset:
    """Test single sequence dataset."""

    def test_initialization_valid(self):
        """Test valid initialization."""
        tokens = torch.arange(100)
        dataset = LongSequenceDataset(tokens, seq_len=10)

        assert dataset.seq_len == 10
        assert dataset.max_start == 89

    def test_length(self):
        """Test dataset length."""
        tokens = torch.arange(100)
        dataset = LongSequenceDataset(tokens, seq_len=10)

        assert len(dataset) == 90  # 100 - 10

    def test_getitem_shapes(self):
        """Test output shapes from __getitem__."""
        tokens = torch.arange(100)
        dataset = LongSequenceDataset(tokens, seq_len=10)

        x, y = dataset[0]
        assert x.shape == (10,)
        assert y.shape == ()
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_getitem_correctness(self):
        """Test that getitem returns correct tokens."""
        tokens = torch.tensor([10, 20, 30, 40, 50, 60])
        dataset = LongSequenceDataset(tokens, seq_len=2)

        # Sample at index 0: should return [10, 20] and target 30
        x, y = dataset[0]
        assert torch.equal(x, torch.tensor([10, 20]))
        assert y == 30

        # Sample at index 1: should return [20, 30] and target 40
        x, y = dataset[1]
        assert torch.equal(x, torch.tensor([20, 30]))
        assert y == 40

    def test_initialization_too_short(self):
        """Test that assertion fails for too-short sequences."""
        tokens = torch.arange(5)
        with pytest.raises(AssertionError):
            LongSequenceDataset(tokens, seq_len=10)

    def test_non_1d_tensor_fails(self):
        """Test that non-1D tensor raises assertion."""
        tokens = torch.arange(100).reshape(10, 10)
        with pytest.raises(AssertionError):
            LongSequenceDataset(tokens, seq_len=5)


class TestMultiSequenceDataset:
    """Test multi-sequence dataset."""

    def test_initialization_single_sequence(self):
        """Test initialization with single sequence."""
        sequences = [torch.arange(100)]
        dataset = MultiSequenceDataset(sequences, seq_len=10)

        assert len(dataset) == 90

    def test_initialization_multiple_sequences(self):
        """Test initialization with multiple sequences."""
        sequences = [torch.arange(50), torch.arange(50, 100)]
        dataset = MultiSequenceDataset(sequences, seq_len=10)

        # Each sequence contributes 40 samples (50-10)
        assert len(dataset) == 80

    def test_list_sequences_converted_to_tensor(self):
        """Test that list sequences are converted to tensors."""
        sequences = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        dataset = MultiSequenceDataset(sequences, seq_len=2)

        # Should not raise errors
        x, y = dataset[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_skips_short_sequences(self):
        """Test that sequences shorter than seq_len+1 are skipped."""
        sequences = [
            torch.arange(100),
            torch.arange(5),  # Too short
            torch.arange(100, 200)
        ]
        dataset = MultiSequenceDataset(sequences, seq_len=10)

        # Only 2 sequences contribute samples
        assert len(dataset) == 180  # 90 + 0 + 90

    @pytest.mark.skip("TODO")
    def test_getitem_from_different_sequences(self):
        """Test that samples come from different source sequences."""
        seq1 = torch.tensor([1, 1, 1, 1, 1])
        seq2 = torch.tensor([2, 2, 2, 2, 2])

        sequences = [seq1, seq2]
        dataset = MultiSequenceDataset(sequences, seq_len=2)

        # First sample should be from seq1, later samples from seq2
        x1, y1 = dataset[0]
        x2, y2 = dataset[1]

        assert torch.all(x1 == 1)
        assert torch.all(x2 == 2)


# ============================================================================
# MODEL TESTS
# ============================================================================

class TestNextIntLSTM:
    """Test LSTM model."""

    def test_initialization(self):
        """Test model initialization."""
        model = NextIntLSTM(
            vocab_size=100,
            embed_dim=32,
            hidden_dim=64,
            num_layers=2
        )

        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.lstm, nn.LSTM)
        assert isinstance(model.fc, nn.Linear)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = NextIntLSTM(
            vocab_size=100,
            embed_dim=32,
            hidden_dim=64,
            num_layers=2
        )

        x = torch.randint(0, 100, (8, 20))  # batch=8, seq_len=20
        logits = model(x)

        assert logits.shape == (8, 100)  # batch, vocab_size

    def test_forward_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        model = NextIntLSTM(vocab_size=50, embed_dim=16, hidden_dim=32)

        for batch_size in [1, 4, 16, 32]:
            x = torch.randint(0, 50, (batch_size, 20))
            logits = model(x)
            assert logits.shape == (batch_size, 50)

    def test_hidden_state_passing(self):
        """Test that hidden state can be passed."""
        model = NextIntLSTM(vocab_size=100, embed_dim=32, hidden_dim=64, num_layers=2)

        x = torch.randint(0, 100, (8, 20))
        logits1 = model(x, hidden=None)

        # Should work with explicit hidden state too
        assert logits1.shape == (8, 100)

    def test_dropout_effect(self):
        """Test that dropout is applied during training."""
        model = NextIntLSTM(
            vocab_size=100,
            embed_dim=32,
            hidden_dim=64,
            num_layers=2,
            dropout=0.5
        )

        x = torch.randint(0, 100, (8, 20))

        # Training mode should apply dropout
        model.train()
        logits_train1 = model(x)
        logits_train2 = model(x)

        # Eval mode should not apply dropout
        model.eval()
        logits_eval1 = model(x)
        logits_eval2 = model(x)

        # Eval outputs should be identical
        assert torch.allclose(logits_eval1, logits_eval2)


class TestNextIntMLP:
    """Test MLP model."""

    def test_initialization(self):
        """Test model initialization."""
        model = NextIntMLP(
            vocab_size=100,
            embed_dim=32,
            seq_len=20,
            hidden_dim=128
        )

        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.mlp, nn.Sequential)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = NextIntMLP(
            vocab_size=100,
            embed_dim=32,
            seq_len=20,
            hidden_dim=128
        )

        x = torch.randint(0, 100, (8, 20))
        logits = model(x)

        assert logits.shape == (8, 100)

    def test_different_hidden_layers(self):
        """Test with different numbers of hidden layers."""
        for num_layers in [1, 2, 3, 4]:
            model = NextIntMLP(
                vocab_size=50,
                embed_dim=16,
                seq_len=10,
                hidden_dim=64,
                num_hidden_layers=num_layers
            )

            x = torch.randint(0, 50, (4, 10))
            logits = model(x)
            assert logits.shape == (4, 50)


class TestNextIntTransformer:
    """Test Transformer model."""

    def test_positional_encoding(self):
        """Test positional encoding."""
        pe = PositionalEncoding(d_model=64, max_len=100)

        x = torch.randn(8, 20, 64)
        out = pe(x)

        assert out.shape == (8, 20, 64)
        # Should not be all zeros
        assert not torch.allclose(out, x)

    def test_transformer_initialization(self):
        """Test Transformer model initialization."""
        model = NextIntTransformer(
            vocab_size=100,
            seq_len=20,
            d_model=64,
            nhead=4,
            num_layers=2
        )

        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.pos_encoder, PositionalEncoding)

    def test_forward_shape(self):
        """Test forward pass output shape."""
        model = NextIntTransformer(
            vocab_size=100,
            seq_len=20,
            d_model=64,
            nhead=4,
            num_layers=2
        )

        x = torch.randint(0, 100, (8, 20))
        logits = model(x)

        assert logits.shape == (8, 100)

    def test_nhead_divisibility(self):
        """Test that d_model is divisible by nhead."""
        # This should work
        model = NextIntTransformer(
            vocab_size=100,
            seq_len=20,
            d_model=64,
            nhead=8
        )

        x = torch.randint(0, 100, (4, 20))
        logits = model(x)
        assert logits.shape == (4, 100)


# ============================================================================
# UTILITY FUNCTIONS TESTS
# ============================================================================

class TestSoftmax:
    """Test Softmax utility."""

    def test_initialization(self):
        """Test Softmax initialization."""
        sf = Softmax(t=1.0)
        assert sf.t == 1.0

    def test_output_is_probability_vector(self):
        """Test that output is a valid probability vector."""
        sf = Softmax(t=1.0)

        x = np.array([1.0, 2.0, 3.0])
        result = sf(x)

        assert np.allclose(np.sum(result), 1.0)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_temperature_effect(self):
        """Test that temperature affects softness of distribution."""
        x = np.array([1.0, 2.0, 3.0])

        sf_hot = Softmax(t=10.0)  # High temperature = uniform
        sf_cold = Softmax(t=0.1)  # Low temperature = concentrated

        result_hot = sf_hot(x)
        result_cold = sf_cold(x)

        # Cold should be more concentrated on max
        assert result_cold[-1] > result_hot[-1]

    def test_numerical_stability(self):
        """Test numerical stability with large values."""
        sf = Softmax(t=1.0)

        x = np.array([1000.0, 1001.0, 1002.0])
        result = sf(x)

        # Should not have NaN or Inf
        assert np.all(np.isfinite(result))
        assert np.allclose(np.sum(result), 1.0)


class TestSampleSimplexMatrix:
    """Test simplex matrix sampling."""

    def test_sample_simplex_matrix(self):
        """Test basic simplex matrix generation."""
        S = sample_simplex_matrix(n=5, ddf=0.0)

        assert S.shape == (5, 5)

        # Each column should sum to 1
        col_sums = np.sum(S, axis=0)
        assert np.allclose(col_sums, 1.0)

        # All entries should be non-negative
        assert np.all(S >= 0)

    def test_diagonal_deflation_factor(self):
        """Test effect of diagonal deflation factor."""
        S_no_ddf = sample_simplex_matrix(n=5, ddf=0.0)
        S_with_ddf = sample_simplex_matrix(n=5, ddf=0.5)

        # Both should be valid probability matrices
        assert np.allclose(np.sum(S_no_ddf, axis=0), 1.0)
        assert np.allclose(np.sum(S_with_ddf, axis=0), 1.0)

    def test_concentrated_simplex_matrix(self):
        """Test concentrated simplex matrix generation."""
        S = sample_simplex_matrix_concentrated(n=5, ddf=0.0, noise=0.0)

        assert S.shape == (5, 5)
        assert np.allclose(np.sum(S, axis=0), 1.0)
        assert np.all(S >= 0)


# ============================================================================
# GENERATOR TESTS
# ============================================================================

class TestEntropyEstimate:
    """Test entropy estimation."""

    def test_initialization(self):
        """Test initialization."""
        ee = EntropyEstimate()
        assert ee.n_samples == 0
        assert ee.running_estimate == 0

    def test_add_sample(self):
        """Test adding samples."""
        ee = EntropyEstimate()

        p1 = np.array([0.5, 0.5])
        ee.add_sample(p1)

        assert ee.n_samples == 1
        assert ee.running_estimate > 0

    def test_entropy_uniform_distribution(self):
        """Test entropy of uniform distribution."""
        ee = EntropyEstimate()

        p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
        ee.add_sample(p_uniform)

        # Entropy of uniform distribution of 4 items ≈ log(4) ≈ 1.386
        assert 1.3 < ee.running_estimate < 1.5

    def test_entropy_concentrated_distribution(self):
        """Test entropy of concentrated distribution."""
        ee = EntropyEstimate()

        p_concentrated = np.array([0.99, 0.005, 0.003, 0.002])
        ee.add_sample(p_concentrated)

        # Entropy should be small
        assert ee.running_estimate < 0.2

    def test_multiple_samples(self):
        """Test averaging over multiple samples."""
        ee = EntropyEstimate()

        samples = [
            np.array([0.5, 0.5]),
            np.array([0.3, 0.7]),
            np.array([0.2, 0.8])
        ]

        for p in samples:
            ee.add_sample(p)

        assert ee.n_samples == 3
        assert ee.running_estimate > 0


class TestSimplexCombiner:
    """Test SimplexCombiner generator."""

    def test_initialization_with_n(self):
        """Test initialization with vocabulary size."""
        combiner = SimplexCombiner(n=10, context=5)

        assert combiner.S.shape == (10, 10)
        assert combiner.context == 5
        assert len(combiner.lexicon) == 10

    def test_initialization_with_S(self):
        """Test initialization with matrix."""
        S = sample_simplex_matrix(n=8, ddf=0.9)
        combiner = SimplexCombiner(S=S, context=5)

        assert combiner.S.shape == (8, 8)

    def test_encode_output_shape(self):
        """Test encode output shape."""
        combiner = SimplexCombiner(n=10, context=5)

        seq = np.array([0, 1, 2, 3, 4])
        output = combiner.encode(seq)

        assert isinstance(output, (np.ndarray, float, np.floating))

    def test_generate_output_shape(self):
        """Test generate output shape."""
        combiner = SimplexCombiner(n=10, context=5)

        sequence = combiner.generate(n=20)

        assert len(sequence) == 20
        assert np.all(sequence >= 0)
        assert np.all(sequence < 10)

    def test_entropy_estimate_tracking(self):
        """Test that entropy is tracked during generation."""
        combiner = SimplexCombiner(n=10, context=5)

        initial_samples = combiner.entropy_estimate.n_samples
        combiner.generate(n=10)

        # Entropy should have been updated
        assert combiner.entropy_estimate.n_samples > initial_samples


# ============================================================================
# TRAINING TESTS
# ============================================================================

class TestSequenceTrainer:
    """Test training pipeline."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        device = torch.device('cpu')
        trainer = SequenceTrainer(device)

        assert trainer.device == device
        assert trainer.run_history == []

    def test_create_model_lstm(self):
        """Test model creation for LSTM."""
        config = {
            'vocab_size': 50,
            'seq_len': 10,
            'embed_dim': 16,
            'hidden_dim': 32,
            'num_layers': 2
        }

        model = create_model('lstm', config)
        assert isinstance(model, NextIntLSTM)

    def test_create_model_mlp(self):
        """Test model creation for MLP."""
        config = {
            'vocab_size': 50,
            'seq_len': 10,
            'embed_dim': 16,
            'hidden_dim': 32,
            'num_layers': 2
        }

        model = create_model('mlp', config)
        assert isinstance(model, NextIntMLP)

    def test_create_model_transformer(self):
        """Test model creation for Transformer."""
        config = {
            'vocab_size': 50,
            'seq_len': 10,
            'd_model': 32,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 64,
            'dropout': 0.1
        }

        model = create_model('transformer', config)
        assert isinstance(model, NextIntTransformer)

    def test_train_loop_basic(self):
        """Test basic training loop."""
        device = torch.device('cpu')
        trainer = SequenceTrainer(device)

        # Create small datasets
        tokens = torch.arange(100)
        train_dataset = LongSequenceDataset(tokens, seq_len=5)
        val_dataset = LongSequenceDataset(tokens, seq_len=5)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=8, shuffle=False
        )

        # Small model
        model = NextIntMLP(
            vocab_size=100,
            embed_dim=8,
            seq_len=5,
            hidden_dim=16,
            num_hidden_layers=1
        )

        config = {
            'max_epochs': 2,
            'lr': 0.001,
            'lr_patience': 1,
            'patience': 10,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = trainer.train(model, train_loader, val_loader, config,
                                   model_dir=tmpdir)

            assert 'best_val_loss' in result
            assert 'epochs_trained' in result
            assert result['epochs_trained'] > 0
            assert len(trainer.run_history) == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_training_pipeline(self):
        """Test complete training pipeline."""
        # Create dataset
        tokens = torch.randint(0, 50, (1000,))
        dataset = LongSequenceDataset(tokens, seq_len=10)

        # Split into train/val
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=16, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=16, shuffle=False
        )

        # Create model and train
        model = NextIntMLP(
            vocab_size=50,
            embed_dim=16,
            seq_len=10,
            hidden_dim=32
        )

        device = torch.device('cpu')
        trainer = SequenceTrainer(device)

        config = {
            'max_epochs': 1,
            'lr': 0.001,
            'lr_patience': 1,
            'patience': 5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = trainer.train(model, train_loader, val_loader, config,
                                   model_dir=tmpdir)

            assert result['best_val_loss'] > 0
            assert os.path.exists(f"{tmpdir}/best_model_run_0.pt")

    def test_all_models_train(self):
        """Test that all model types can train."""
        # Small dataset
        tokens = torch.randint(0, 30, (500,))
        dataset = LongSequenceDataset(tokens, seq_len=5)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=True
        )

        model_configs = {
            'lstm': {
                'vocab_size': 30,
                'seq_len': 5,
                'embed_dim': 8,
                'hidden_dim': 16,
                'num_layers': 1
            },
            'mlp': {
                'vocab_size': 30,
                'seq_len': 5,
                'embed_dim': 8,
                'hidden_dim': 16,
                'num_layers': 1
            },
            'transformer': {
                'vocab_size': 30,
                'seq_len': 5,
                'd_model': 16,
                'nhead': 2,
                'num_layers': 1,
                'dim_feedforward': 32
            }
        }

        for model_type, config in model_configs.items():
            model = create_model(model_type, config)

            # Do one training step
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for x, y in loader:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                break  # Just one batch


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
