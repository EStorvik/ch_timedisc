# Fix MPI/OFI finalization errors on macOS
import os

os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

from dolfinx import mesh
from dolfinx.fem.petsc import NonlinearProblem

from dolfinx.io import XDMFFile
import numpy as np
from mpi4py import MPI

import matplotlib.pyplot as plt


import ch_timedisc as ch


dts = [1e-5, 1e-4, 1e-3, 1e-2]
T = 5e-2


def run_imp_euler(dt, T):

    num_time_steps = int(T / dt)
    print(num_time_steps)

    # Define material parameters
    parameters = ch.Parameters(dt=dt, num_time_steps=num_time_steps)

    # Double well
    doublewell = ch.DoubleWell()

    # Mesh
    msh = mesh.create_unit_square(
        MPI.COMM_WORLD, parameters.nx, parameters.ny, cell_type=mesh.CellType.triangle
    )

    # Initial condition
    initialcondition = ch.Cross2D(width=0.3)

    # Set up femhandler
    femhandler = ch.FEMHandler(
        msh,
        initialcondition=initialcondition,
        parameters=parameters,
        doublewell=doublewell,
    )

    # Energy computations
    energy = ch.Energy(femhandler.pf, femhandler.mu, parameters, doublewell)

    # Linear variational form
    implicit_euler = ch.VariationalImplicitEuler(
        femhandler=femhandler, parameters=parameters, doublewell=doublewell
    )

    # Set up nonlinear problem
    problem = NonlinearProblem(
        implicit_euler.F,
        femhandler.xi,
        petsc_options_prefix="ch_implicit_euler_" + f"dt_{dt}_",
        petsc_options=parameters.petsc_options,
    )

    # Set up time marching
    time_marching = ch.TimeMarching(
        femhandler=femhandler, parameters=parameters, energy=energy, problem=problem
    )

    # Perform time marching
    time_marching()

    return energy, time_marching


output_folder = "log/implicit_euler_test/"
plt.figure("Energy")

for i, dt in enumerate(dts):
    energy, time_marching = run_imp_euler(dt=dt, T=T)
    results = {
        "dt": np.array([dt]),
        "Time": np.array(time_marching.time_vec),
        "Energy": np.array(energy.energy_vec),
    }

    np.savez(output_folder + f"dt_{i}.npz", **results)
    plt.plot(time_marching.time_vec, energy.energy_vec, label=f"dt = {dt}")

plt.legend()

plt.show()
