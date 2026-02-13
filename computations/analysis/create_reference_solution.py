# Fix MPI/OFI finalization errors on macOS
import os

os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

# Suppress duplicate rpath linker warnings during JIT compilation
os.environ["LDFLAGS"] = os.environ.get("LDFLAGS", "") + " -Wl,-w"

import sys

from dolfinx import mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import Mesh
from typing import Optional

from dolfinx.io import XDMFFile
import numpy as np
from mpi4py import MPI

import matplotlib.pyplot as plt

from pathlib import Path


import ch_timedisc as ch


# Define material parameters
parameters: ch.Parameters = ch.Parameters.from_json(
    "reference_solution/parameters.json"
)


# Double well
doublewell: ch.DoubleWell = ch.DoubleWell()

# Mesh
msh: Mesh = mesh.create_unit_square(
    MPI.COMM_WORLD, parameters.nx, parameters.ny, cell_type=mesh.CellType.triangle
)

# Initial condition
# initialcondition = ch.Cross2D(width=0.3)
initialcondition: ch.Random = ch.Random(seed=42)

# Set up femhandler
femhandler: ch.FEMHandler = ch.FEMHandler(
    msh, initialcondition=initialcondition, parameters=parameters, doublewell=doublewell
)


energy: ch.Energy = ch.Energy(
    femhandler=femhandler, parameters=parameters, doublewell=doublewell
)

# Linear variational forms
imp_euler: ch.VariationalImplicitEuler = ch.VariationalImplicitEuler(
    femhandler=femhandler, parameters=parameters, doublewell=doublewell
)


# Set up nonlinear problem
problem: NonlinearProblem = NonlinearProblem(
    imp_euler.F,
    femhandler.xi,
    petsc_options_prefix="ch_implicit_",
    petsc_options=parameters.petsc_options,
)


# Pyvista plot
viz: ch.PyvistaVizualization = ch.PyvistaVizualization(
    femhandler.V.sub(0), femhandler.xi.sub(0), 0.0
)

# Output file (absolute path under computations/output)
output_dir = Path(__file__).resolve().parent.parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "ch_reference_solution_random.xdmf"
output_file_pf = XDMFFile(MPI.COMM_WORLD, str(output_path), "w")
output_file_pf.write_mesh(msh)


# Time stepping
adaptive_time_step: ch.AdaptiveTimeStep = ch.AdaptiveTimeStepEnergyDiff(
    energy=energy,
    parameters=parameters,
    femhandler=femhandler,
    variational_form=imp_euler,
    verbose=False,
)


# Set up time marching
numpy_output_path = Path(__file__).resolve().parent.parent.parent / "reference_solution"
time_marching: ch.TimeMarching = ch.TimeMarching(
    femhandler=femhandler,
    parameters=parameters,
    energy=energy,
    problem=problem,
    viz=viz,
    adaptive_time_step=adaptive_time_step,
    verbose=True,
    output_file=output_file_pf,
    numpy_output_dir=numpy_output_path,
)

# Perform time marching
time_marching()

# Close output file
output_file_pf.close()

viz.final_plot(femhandler.xi.sub(0))
