"""Tests for FEMHandler finite element setup."""

# Fix MPI/OFI finalization errors on macOS
import os

os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

import numpy as np
import pytest

from ch_timedisc import DoubleWell, Parameters
from ch_timedisc.fem import FEMHandler

try:
    from mpi4py import MPI
    from dolfinx.mesh import create_unit_square
except ImportError:
    pytest.skip("dolfinx not available", allow_module_level=True)


@pytest.fixture
def simple_mesh():
    """Create a simple unit square mesh."""
    return create_unit_square(MPI.COMM_WORLD, nx=4, ny=4)


@pytest.fixture
def simple_parameters():
    """Create simple simulation parameters."""
    return Parameters(ell=0.1, nx=4, ny=4)


@pytest.fixture
def simple_doublewell():
    """Create a simple double well potential."""
    return DoubleWell(scaling=1.0)


@pytest.fixture
def simple_initial_condition():
    """Simple initial condition: constant 0.5."""

    def ic(x):
        return 0.5 * np.ones(x.shape[1])

    return ic


class TestFEMHandlerInitialization:
    """Test FEMHandler initialization and setup."""

    def test_fem_handler_creates_function_space(
        self,
        simple_mesh,
        simple_initial_condition,
        simple_parameters,
        simple_doublewell,
    ):
        """Test that FEMHandler creates a valid function space."""
        handler = FEMHandler(
            simple_mesh,
            simple_initial_condition,
            simple_parameters,
            simple_doublewell,
        )

        assert handler.V is not None
        assert handler.xi is not None
        assert handler.xi_old is not None

    def test_fem_handler_initializes_solutions(
        self,
        simple_mesh,
        simple_initial_condition,
        simple_parameters,
        simple_doublewell,
    ):
        """Test that solution functions are initialized without NaN values."""
        handler = FEMHandler(
            simple_mesh,
            simple_initial_condition,
            simple_parameters,
            simple_doublewell,
        )

        # Check xi is initialized
        assert not np.any(np.isnan(handler.xi.x.array))
        # Check xi_old is initialized
        assert not np.any(np.isnan(handler.xi_old.x.array))

    def test_fem_handler_xi_xi_old_consistent(
        self,
        simple_mesh,
        simple_initial_condition,
        simple_parameters,
        simple_doublewell,
    ):
        """Test that current and old solutions start with same values."""
        handler = FEMHandler(
            simple_mesh,
            simple_initial_condition,
            simple_parameters,
            simple_doublewell,
        )

        # xi and xi_old should be initialized to same values
        assert np.allclose(handler.xi.x.array, handler.xi_old.x.array)

    def test_fem_handler_split_components(
        self,
        simple_mesh,
        simple_initial_condition,
        simple_parameters,
        simple_doublewell,
    ):
        """Test that split components are properly created."""
        handler = FEMHandler(
            simple_mesh,
            simple_initial_condition,
            simple_parameters,
            simple_doublewell,
        )

        # Test functions should be split
        assert handler.eta_pf is not None
        assert handler.eta_mu is not None

        # Solution components should be split
        assert handler.pf is not None
        assert handler.mu is not None

        # Old solution components should be split
        assert handler.pf_old is not None
        assert handler.mu_old is not None

    def test_fem_handler_test_functions(
        self,
        simple_mesh,
        simple_initial_condition,
        simple_parameters,
        simple_doublewell,
    ):
        """Test that test function eta is properly initialized."""
        handler = FEMHandler(
            simple_mesh,
            simple_initial_condition,
            simple_parameters,
            simple_doublewell,
        )

        assert handler.eta is not None

    def test_fem_handler_different_initial_conditions(
        self, simple_mesh, simple_parameters, simple_doublewell
    ):
        """Test FEMHandler with different initial conditions."""

        # Linear initial condition
        def linear_ic(x):
            return x[0]

        handler1 = FEMHandler(
            simple_mesh, linear_ic, simple_parameters, simple_doublewell
        )
        assert handler1.xi is not None

        # Step function-like initial condition
        def step_ic(x):
            return np.where(x[0] < 0.5, 0.0, 1.0)

        handler2 = FEMHandler(
            simple_mesh, step_ic, simple_parameters, simple_doublewell
        )
        assert handler2.xi is not None

    def test_fem_handler_array_dimensions(
        self,
        simple_mesh,
        simple_initial_condition,
        simple_parameters,
        simple_doublewell,
    ):
        """Test that arrays have expected dimensions."""
        handler = FEMHandler(
            simple_mesh,
            simple_initial_condition,
            simple_parameters,
            simple_doublewell,
        )

        # Mixed function space should have 2x the DOFs of scalar space
        n_dof = handler.V.dofmap.index_map.size_local
        assert n_dof > 0
        # Should have 2 components (pf, mu) in mixed space
        assert n_dof % 2 == 0
