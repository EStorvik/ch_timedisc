# Fix MPI/OFI finalization errors on macOS
import os

os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

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
parameters: ch.Parameters = ch.Parameters()

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
output_path = output_dir / "ch_implicit_random_e_m5.xdmf"
output_file_pf = XDMFFile(MPI.COMM_WORLD, str(output_path), "w")
output_file_pf.write_mesh(msh)


# Time stepping
adaptive_time_step: ch.AdaptiveTimeStep = ch.AdaptiveTimeStepEnergyDiff(
    energy=energy,
    parameters=parameters,
    femhandler=femhandler,
    variational_form=imp_euler,
    factor=1.5,
    threshold_high=-0.01,
    threshold_low=-0.005,
    verbose=True,
)


# Set up time marching
time_marching: ch.TimeMarching = ch.TimeMarching(
    femhandler=femhandler,
    parameters=parameters,
    energy=energy,
    problem=problem,
    adaptive_time_step=adaptive_time_step,
    viz=viz,
    # output_file = output_file_pf,
)

# Perform time marching
time_marching()


viz.final_plot(femhandler.xi.sub(0))


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"  # or 'sans-serif'
plt.rcParams["font.size"] = 16

plt.figure("Energy evolution")
plt.plot(time_marching.time_vec, energy.energy_vec)
plt.show()


# plt.figure("dt Energy")
# plt.plot(
#     time_marching.time_vec[2:],
#     energy.energy_dt_vec()[1:],
#     label=r"$\partial_t\mathcal{E}$",
# )
# plt.plot(
#     time_marching.time_vec[2:],
#     energy.gradmu_squared_vec[2:],
#     label=r"$-m\|\nabla\mu\|^2$",
# )

# plt.legend()

# plt.figure("dte - mnmu^2")
# plt.plot(
#     time_marching.time_vec[2:],
#     np.array(energy.energy_dt_vec()[1:]) - np.array(energy.gradmu_squared_vec[2:]),
# )
# plt.show()
# # output_file_pf.close()
