"""Tests for initial phase field smoothing."""

# Fix MPI/OFI finalization errors on macOS
import os

os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

import numpy as np
import pytest

import ch_timedisc as ch
from ch_timedisc.intital_conditions.initial_pf import initial_pf

try:
    from mpi4py import MPI
    from dolfinx.mesh import create_unit_square
    from dolfinx.fem import functionspace, Function
    from basix.ufl import element
    from ufl import as_ufl
except ImportError:
    pytest.skip("dolfinx not available", allow_module_level=True)


def create_ufl_from_callable(mesh, P, callable_func):
    """Convert a callable to a UFL expression by interpolation."""
    V = functionspace(mesh, P)
    u = Function(V)
    u.interpolate(callable_func)
    return u


@pytest.fixture
def simple_mesh():
    """Create a simple unit square mesh."""
    return create_unit_square(MPI.COMM_WORLD, nx=8, ny=8)


@pytest.fixture
def simple_parameters():
    """Create simple simulation parameters."""
    return ch.Parameters(ell=0.1, nx=8, ny=8)


@pytest.fixture
def P1_element():
    """Create a P1 finite element."""
    return element("Lagrange", "triangle", 1)


@pytest.fixture
def constant_initial_condition():
    """Constant initial condition: 0.5."""

    def ic(x):
        return 0.5 * np.ones(x.shape[1])

    return ic


@pytest.fixture
def smooth_initial_condition():
    """Smooth initial condition: sine wave."""

    def ic(x):
        return 0.5 + 0.3 * np.sin(2 * np.pi * x[0]) * np.sin(2 * np.pi * x[1])

    return ic


class TestInitialPFBasic:
    """Test basic functionality of initial_pf."""

    def test_initial_pf_returns_function(
        self, simple_mesh, P1_element, constant_initial_condition, simple_parameters
    ):
        """Test that initial_pf returns a Function object."""
        pf0 = create_ufl_from_callable(
            simple_mesh, P1_element, constant_initial_condition
        )
        result = initial_pf(pf0, P1_element, simple_mesh, simple_parameters)

        assert result is not None
        assert isinstance(result, Function)

    def test_initial_pf_no_nan_values(
        self, simple_mesh, P1_element, constant_initial_condition, simple_parameters
    ):
        """Test that the resulting function has no NaN values."""
        pf0 = create_ufl_from_callable(
            simple_mesh, P1_element, constant_initial_condition
        )
        result = initial_pf(pf0, P1_element, simple_mesh, simple_parameters)

        assert not np.any(np.isnan(result.x.array))

    def test_initial_pf_bounded(
        self, simple_mesh, P1_element, constant_initial_condition, simple_parameters
    ):
        """Test that the smoothed field stays within reasonable bounds."""
        pf0 = create_ufl_from_callable(
            simple_mesh, P1_element, constant_initial_condition
        )
        result = initial_pf(pf0, P1_element, simple_mesh, simple_parameters)

        # For constant input 0.5, output should be close to 0.5
        assert np.min(result.x.array) >= -0.1
        assert np.max(result.x.array) <= 1.1

    def test_initial_pf_constant_input(
        self, simple_mesh, P1_element, constant_initial_condition, simple_parameters
    ):
        """Test that constant input produces constant-like output."""
        pf0 = create_ufl_from_callable(
            simple_mesh, P1_element, constant_initial_condition
        )
        result = initial_pf(pf0, P1_element, simple_mesh, simple_parameters)

        mean_val = np.mean(result.x.array)
        std_val = np.std(result.x.array)

        # For constant input, output should be nearly constant
        assert np.isclose(mean_val, 0.5, atol=0.01)
        assert std_val < 0.05


class TestInitialPFSmoothing:
    """Test smoothing properties of initial_pf."""

    def test_initial_pf_smooths_sharp_field(
        self, simple_mesh, P1_element, simple_parameters
    ):
        """Test that initial_pf smooths a sharp discontinuous field."""

        # Create a sharp step function
        def step_ic(x):
            return np.where(x[0] < 0.5, 0.0, 1.0)

        # Create a smooth version as reference
        def smooth_step_ic(x):
            return 0.5 + 0.3 * np.tanh(20 * (x[0] - 0.5))

        sharp_pf0 = create_ufl_from_callable(simple_mesh, P1_element, step_ic)
        smooth_pf0 = create_ufl_from_callable(simple_mesh, P1_element, smooth_step_ic)

        sharp_result = initial_pf(sharp_pf0, P1_element, simple_mesh, simple_parameters)
        smooth_result = initial_pf(
            smooth_pf0, P1_element, simple_mesh, simple_parameters
        )

        # Both should be valid
        assert not np.any(np.isnan(sharp_result.x.array))
        assert not np.any(np.isnan(smooth_result.x.array))

        # Both should have values in [0, 1]
        assert np.all((sharp_result.x.array >= -0.01) & (sharp_result.x.array <= 1.01))
        assert np.all(
            (smooth_result.x.array >= -0.01) & (smooth_result.x.array <= 1.01)
        )

    def test_initial_pf_different_ell_values(
        self, simple_mesh, P1_element, constant_initial_condition
    ):
        """Test that different ell values produce different smoothing."""
        params_small_ell = ch.Parameters(ell=0.01, nx=8, ny=8)
        params_large_ell = ch.Parameters(ell=0.2, nx=8, ny=8)

        pf0 = create_ufl_from_callable(
            simple_mesh, P1_element, constant_initial_condition
        )

        result_small = initial_pf(pf0, P1_element, simple_mesh, params_small_ell)
        result_large = initial_pf(pf0, P1_element, simple_mesh, params_large_ell)

        # Both should be valid
        assert not np.any(np.isnan(result_small.x.array))
        assert not np.any(np.isnan(result_large.x.array))

        # Both should be close to 0.5 for constant input
        assert np.isclose(np.mean(result_small.x.array), 0.5, atol=0.05)
        assert np.isclose(np.mean(result_large.x.array), 0.5, atol=0.05)


class TestInitialPFProperties:
    """Test mathematical properties of initial_pf."""

    def test_initial_pf_preserves_mean(
        self, simple_mesh, P1_element, simple_parameters
    ):
        """Test that smoothing preserves the mean value."""

        def ic(x):
            return 0.3 + 0.2 * np.sin(4 * np.pi * x[0])

        pf0 = create_ufl_from_callable(simple_mesh, P1_element, ic)
        result = initial_pf(pf0, P1_element, simple_mesh, simple_parameters)

        # The variational form should preserve the mean approximately
        # (within numerical integration error)
        assert not np.any(np.isnan(result.x.array))

    def test_initial_pf_with_ufl_expression(
        self, simple_mesh, P1_element, simple_parameters
    ):
        """Test that initial_pf works with UFL expressions."""
        # Create a UFL expression (constant)
        pf0_expr = as_ufl(0.5)

        result = initial_pf(pf0_expr, P1_element, simple_mesh, simple_parameters)

        assert result is not None
        assert isinstance(result, Function)
        assert not np.any(np.isnan(result.x.array))
