"""Tests for random initial condition generator."""

import numpy as np
import pytest

try:
    from ch_timedisc.intital_conditions.random import Random
except ImportError:
    pytest.skip("dolfinx not available", allow_module_level=True)


@pytest.fixture
def random_gen_default():
    """Create Random generator with default parameters."""
    return Random()


@pytest.fixture
def random_gen_seeded():
    """Create Random generator with fixed seed for reproducibility."""
    return Random(seed=42)


@pytest.fixture
def sample_points():
    """Create sample points for testing."""
    # 2D domain with 100 points
    return np.random.rand(2, 100)


class TestRandomInitialization:
    """Test Random class initialization."""

    def test_random_creates_instance(self):
        """Test that Random creates a valid instance."""
        gen = Random()
        assert gen is not None
        assert isinstance(gen, Random)

    def test_random_default_parameters(self):
        """Test that Random has correct default parameters."""
        gen = Random()
        assert gen.mean == 0.5
        assert gen.std == 0.1
        assert gen.seed is None

    def test_random_custom_parameters(self):
        """Test that Random accepts custom parameters."""
        gen = Random(mean=0.3, std=0.2, seed=123)
        assert gen.mean == 0.3
        assert gen.std == 0.2
        assert gen.seed == 123


class TestRandomCallable:
    """Test Random as a callable."""

    def test_random_is_callable(self, random_gen_default):
        """Test that Random instance is callable."""
        assert callable(random_gen_default)

    def test_random_returns_array(self, random_gen_default, sample_points):
        """Test that calling Random returns a numpy array."""
        result = random_gen_default(sample_points)
        assert isinstance(result, np.ndarray)

    def test_random_correct_output_shape(self, random_gen_default, sample_points):
        """Test that output has correct shape."""
        result = random_gen_default(sample_points)
        expected_shape = (sample_points.shape[1],)
        assert result.shape == expected_shape

    def test_random_output_bounded(self, random_gen_default, sample_points):
        """Test that output is bounded to [0, 1]."""
        result = random_gen_default(sample_points)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_random_no_nan_values(self, random_gen_default, sample_points):
        """Test that output contains no NaN values."""
        result = random_gen_default(sample_points)
        assert not np.any(np.isnan(result))


class TestRandomReproducibility:
    """Test random reproducibility with seed."""

    def test_random_reproducible_with_seed(self):
        """Test that same seed produces same results."""
        gen1 = Random(seed=42)
        gen2 = Random(seed=42)

        points = np.random.rand(2, 50)

        result1 = gen1(points)
        result2 = gen2(points)

        np.testing.assert_array_equal(result1, result2)

    def test_random_different_without_seed(self):
        """Test that different calls without seed produce different results."""
        gen = Random()
        points = np.random.rand(2, 50)

        result1 = gen(points)
        result2 = gen(points)

        # Very unlikely to be exactly equal
        assert not np.allclose(result1, result2)


class TestRandomStatistics:
    """Test statistical properties of random generator."""

    def test_random_mean_approximately_correct(self):
        """Test that generated values have approximately correct mean."""
        gen = Random(mean=0.5, std=0.05, seed=42)

        # Generate many samples
        points = np.random.rand(2, 10000)
        result = gen(points)

        # Mean should be close to specified mean (but clipping affects this)
        assert 0.45 < np.mean(result) < 0.55

    def test_random_different_mean(self):
        """Test that different mean values produce different outputs."""
        gen_low = Random(mean=0.2, seed=42)
        gen_high = Random(mean=0.8, seed=42)

        # Reset seed for reproducible sequence
        np.random.seed(42)
        points = np.random.rand(2, 1000)

        result_low = gen_low(points)
        result_high = gen_high(points)

        # Low mean should produce lower values on average
        assert np.mean(result_low) < np.mean(result_high)

    def test_random_different_std(self):
        """Test that different std values produce different distributions."""
        gen_low_std = Random(mean=0.5, std=0.01, seed=42)
        gen_high_std = Random(mean=0.5, std=0.3, seed=42)

        points = np.random.rand(2, 1000)

        result_low_std = gen_low_std(points)
        result_high_std = gen_high_std(points)

        # Low std should produce less spread values
        assert np.std(result_low_std) < np.std(result_high_std)


class TestRandomEdgeCases:
    """Test edge cases and special scenarios."""

    def test_random_single_point(self):
        """Test Random with a single point."""
        gen = Random(seed=42)
        points = np.array([[0.5], [0.5]])
        result = gen(points)

        assert result.shape == (1,)
        assert 0.0 <= result[0] <= 1.0

    def test_random_many_points(self):
        """Test Random with many points."""
        gen = Random(seed=42)
        points = np.random.rand(2, 10000)
        result = gen(points)

        assert result.shape == (10000,)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_random_3d_points(self):
        """Test Random with 3D points."""
        gen = Random(seed=42)
        points = np.random.rand(3, 100)
        result = gen(points)

        assert result.shape == (100,)
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_random_extreme_mean_zero(self):
        """Test Random with mean close to 0."""
        gen = Random(mean=0.0, std=0.1, seed=42)
        points = np.random.rand(2, 1000)
        result = gen(points)

        # Should be clipped to [0, 1]
        assert np.all((result >= 0.0) & (result <= 1.0))
        assert np.mean(result) > 0.0  # Should have some positive values

    def test_random_extreme_mean_one(self):
        """Test Random with mean close to 1."""
        gen = Random(mean=1.0, std=0.1, seed=42)
        points = np.random.rand(2, 1000)
        result = gen(points)

        # Should be clipped to [0, 1]
        assert np.all((result >= 0.0) & (result <= 1.0))
        assert np.mean(result) < 1.0  # Should have some values below 1
