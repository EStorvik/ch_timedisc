import numpy as np

from dolfinx import default_real_type
from petsc4py import PETSc


class Parameters:

    def __init__(self):

        # Model
        self.gamma = 1
        self.ell = 0.025
        self.mobility = 1.0

        # Time Discretization
        self.dt = 1.0e-6
        self.t0 = 0
        self.num_time_steps = 10
        self.T = self.dt * self.num_time_steps

        # Spatial Discretization
        self.nx = self.ny = self.nz = 64

        # Nonlinear iteration parameters
        self.tol = 1.0e-6
        self.max_iter = 200

        use_superlu = PETSc.IntType == np.int64  # or PETSc.ScalarType == np.complex64
        sys = PETSc.Sys()  # type: ignore
        if sys.hasExternalPackage("mumps") and not use_superlu:
            linear_solver = "mumps"
        elif sys.hasExternalPackage("superlu_dist"):
            linear_solver = "superlu_dist"
        else:
            linear_solver = "petsc"
        self.petsc_options = {
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
