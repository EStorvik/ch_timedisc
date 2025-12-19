# Fix MPI/OFI finalization errors on macOS
import os

os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

import dolfinx
from basix.ufl import element, mixed_element
from dolfinx import mesh, default_real_type
from dolfinx.fem import Function, functionspace, assemble_scalar, form
from dolfinx.fem.petsc import NonlinearProblem
from petsc4py import PETSc

from dolfinx.io import XDMFFile
import numpy as np
from mpi4py import MPI
from ufl import TestFunction, dx, grad, inner, split, Measure

import matplotlib.pyplot as plt


import ch_timedisc as ch


# Define material parameters
parameters = ch.Parameters()

# Double well
doublewell = ch.DoubleWell()

# Mesh
msh = mesh.create_unit_square(
    MPI.COMM_WORLD, parameters.nx, parameters.ny, cell_type=mesh.CellType.triangle
)


# Initial conditions
initialcondition = ch.Cross2D(width=0.3)
# initialcondition = ch.Random()

# FEM Handler
femhandler = ch.FEMHandler(
    msh=msh,
    initialcondition=initialcondition,
    parameters=parameters,
    doublewell=doublewell,
)

energy = ch.Energy(femhandler=femhandler, parameters=parameters, doublewell=doublewell)

# Linear variational forms
crank_nicholson = ch.VariationalCrankNicholson(
    femhandler=femhandler, parameters=parameters, doublewell=doublewell
)


# Set up nonlinear problem
problem = NonlinearProblem(
    crank_nicholson.F,
    femhandler.xi,
    petsc_options_prefix="ch_cn_",
    petsc_options=parameters.petsc_options,
)


# Pyvista plot
viz = ch.visualization.PyvistaVizualization(
    femhandler.V.sub(0), femhandler.xi.sub(0), 0.0
)

# Output file
# output_file_pf = XDMFFile(MPI.COMM_WORLD, "../output/ch_implicit.xdmf", "w")
# output_file_pf.write_mesh(msh)

# Time stepping
time_marching = ch.TimeMarching(
    femhandler=femhandler,
    parameters=parameters,
    energy=energy,
    problem=problem,
    viz=viz,
)
time_marching()

viz.final_plot(femhandler.xi.sub(0))


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"  # or 'sans-serif'
plt.rcParams["font.size"] = 16

# plt.figure("Energy evolution")
# plt.plot(time_vec, energy.energy_vec)
# plt.show()


plt.figure("dt Energy")
plt.plot(
    time_marching.time_vec[2:],
    energy.energy_dt_vec()[1:],
    label=r"$\partial_t\mathcal{E}$",
)
plt.plot(
    time_marching.time_vec[2:],
    energy.gradmu_squared_vec[2:],
    label=r"$-m\|\nabla\mu\|^2$",
)

plt.legend()

plt.figure("dte - mnmu^2")
plt.plot(
    time_marching.time_vec[2:],
    np.array(energy.energy_dt_vec()[1:]) - np.array(energy.gradmu_squared_vec[2:]),
)
plt.show()
# output_file_pf.close()
