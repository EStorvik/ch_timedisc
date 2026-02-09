import numpy as np
from typing import Dict, Any, Optional

from dolfinx import default_real_type
from petsc4py import PETSc


class Parameters:
    """
    Configuration parameters for Cahn-Hilliard time discretization simulations.

    This class encapsulates all parameters needed for the finite element simulation,
    including model parameters, time discretization settings, spatial discretization,
    nonlinear solver settings, and PETSc options.
    """

    def __init__(
        self,
        dt: float = 1e-5,
        T: Optional[float] = None,
        num_time_steps: Optional[int] = None,
        gamma: float = 1,
        ell: float = 0.025,
        mobility: float = 1.0,
        nx: int = 64,
        ny: int = 64,
        nz: int = 64,
        tol: float = 1.0e-6,
        max_iter: int = 200,
        t0: float = 0,
    ) -> None:
        """
        Initialize simulation parameters with sensible defaults.

        Args:
            dt: Time step size. Default: 1e-5
            T: Total simulation time. Provide either T or num_time_steps, not both.
            num_time_steps: Number of time steps to simulate. Provide either T or num_time_steps, not both.
            gamma: Surface tension parameter. Default: 1
            ell: Interface thickness/length scale. Default: 0.025
            mobility: Mobility parameter controlling dynamics. Default: 1.0
            nx: Number of mesh points in x-direction. Default: 64
            ny: Number of mesh points in y-direction. Default: 64
            nz: Number of mesh points in z-direction. Default: 64
            tol: Nonlinear solver tolerance. Default: 1.0e-6
            max_iter: Maximum nonlinear solver iterations. Default: 200
            t0: Initial time. Default: 0

        Raises:
            ValueError: If both T and num_time_steps are provided.
        """

        # Model
        self.gamma: float = gamma
        self.ell: float = ell
        self.mobility: float = mobility

        # Time Discretization
        self.dt: float = dt
        self.t0: float = t0

        # Ensure exactly one of T or num_time_steps is provided
        if T is not None and num_time_steps is not None:
            raise ValueError(
                "Cannot specify both 'T' and 'num_time_steps'. "
                "Provide only one and the other will be calculated."
            )
        elif T is None and num_time_steps is None:
            # Set T and use to calculate num_time_steps
            self.T: float = 1.0
            self.num_time_steps: int = int((self.T - t0) / dt)

        elif num_time_steps is None:
            assert T is not None
            # Use T to calculate num_time_steps
            self.num_time_steps = int((T - t0) / dt)
            self.T = T
        else:
            # Use num_time_steps to calculate T
            self.num_time_steps = num_time_steps
            self.T = self.dt * self.num_time_steps + t0

        # Spatial Discretization
        self.nx: int = nx
        self.ny: int = ny
        self.nz: int = nz

        # Nonlinear iteration parameters
        self.tol: float = tol
        self.max_iter: int = max_iter

        # Determine linear solver based on available PETSc packages
        sys = PETSc.Sys()  # type: ignore
        if sys.hasExternalPackage("superlu_dist"):
            linear_solver = "superlu_dist"
        elif sys.hasExternalPackage("mumps"):
            linear_solver = "mumps"
        else:
            linear_solver = "petsc"
        self.petsc_options: Dict[str, Any] = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_stol": self.tol,
            "snes_atol": 0,
            "snes_rtol": 0,
            "snes_max_it": self.max_iter,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": linear_solver,
            # "snes_monitor": None,
            # "snes_view": None,
        }
