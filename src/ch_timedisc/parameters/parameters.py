import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

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
        adaptive_threshold_increase: float = -0.001,
        adaptive_threshold_decrease: float = -0.01,
        adaptive_factor: float = 1.5,
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

        self.adaptive_threshold_increase: float = adaptive_threshold_increase
        self.adaptive_threshold_decrease: float = adaptive_threshold_decrease
        self.adaptive_factor: float = adaptive_factor

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

        print(self)

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> "Parameters":
        """Load parameters from a JSON file.

        Args:
            json_path: Path to the JSON file containing parameter values.

        Returns:
            Parameters object initialized with values from the JSON file.

        Example:
            >>> params = Parameters.from_json("parameters.json")
        """
        json_path = Path(json_path)
        with open(json_path, "r") as f:
            param_dict = json.load(f)

        return cls(**param_dict)

    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save parameters to a JSON file.

        Args:
            json_path: Path where the JSON file will be written.

        Example:
            >>> params = Parameters(dt=1e-5, T=1.0)
            >>> params.to_json("parameters.json")
        """
        json_path = Path(json_path)
        param_dict = {
            "dt": self.dt,
            "T": self.T,
            "num_time_steps": self.num_time_steps,
            "adaptive_threshold_increase": self.adaptive_threshold_increase,
            "adaptive_threshold_decrease": self.adaptive_threshold_decrease,
            "gamma": self.gamma,
            "ell": self.ell,
            "mobility": self.mobility,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "tol": self.tol,
            "max_iter": self.max_iter,
            "t0": self.t0,
        }

        with open(json_path, "w") as f:
            json.dump(param_dict, indent=2, fp=f)

    def __str__(self) -> str:
        """Return a nicely formatted string representation of the parameters.

        Returns:
            Formatted string displaying all parameter values.
        """
        return f"""
╔═══════════════════════════════════════════════════════╗
║          Cahn-Hilliard Simulation Parameters          ║
╠═══════════════════════════════════════════════════════╣
║ Model Parameters                                      ║
╠═══════════════════════════════════════════════════════╣
║  γ (gamma):                            {self.gamma:>13.6e}  ║
║  ℓ (ell):                              {self.ell:>13.6e}  ║
║  m (mobility):                         {self.mobility:>13.6e}  ║
╠═══════════════════════════════════════════════════════╣
║ Time Discretization                                   ║
╠═══════════════════════════════════════════════════════╣
║  dt:                                   {self.dt:>13.6e}  ║
║  t0:                                   {self.t0:>13.6e}  ║
║  T (final time):                       {self.T:>13.6e}  ║
║  num_time_steps:                          {self.num_time_steps:>10d}  ║
║  adaptive_threshold_increase:          {self.adaptive_threshold_increase:>13.6e}  ║
║  adaptive_threshold_decrease:          {self.adaptive_threshold_decrease:>13.6e}  ║
║  adaptive_factor:                      {self.adaptive_factor:>13.6e}  ║
╠═══════════════════════════════════════════════════════╣
║ Spatial Discretization                                ║
╠═══════════════════════════════════════════════════════╣
║  nx:                                      {self.nx:>10d}  ║
║  ny:                                      {self.ny:>10d}  ║
║  nz:                                      {self.nz:>10d}  ║
╠═══════════════════════════════════════════════════════╣
║ Nonlinear Solver                                      ║
╠═══════════════════════════════════════════════════════╣
║  tolerance:                            {self.tol:>13.6e}  ║
║  max_iter:                                {self.max_iter:>10d}  ║
║  linear solver:                      {self.petsc_options['pc_factor_mat_solver_type']:>15s}  ║
╚═══════════════════════════════════════════════════════╝
"""

    def __repr__(self) -> str:
        """Return a concise representation suitable for debugging.

        Returns:
            String representation showing key parameters.
        """
        return (
            f"Parameters(dt={self.dt:.2e}, T={self.T:.2e}, "
            f"num_time_steps={self.num_time_steps}, "
            f"gamma={self.gamma}, ell={self.ell}, mobility={self.mobility}, "
            f"nx={self.nx}, ny={self.ny}, nz={self.nz})"
        )

    def print_parameters(self) -> None:
        """Print the parameters in a nicely formatted way.

        Example:
            >>> params = Parameters(dt=1e-5, T=1.0)
            >>> params.print_parameters()
        """
        print(str(self))
