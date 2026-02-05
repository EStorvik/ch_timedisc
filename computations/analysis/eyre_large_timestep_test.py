# Fix MPI/OFI finalization errors on macOS
import os

os.environ["FI_PROVIDER"] = "tcp"
os.environ["MPICH_OFI_STARTUP_CONNECT"] = "0"

from dolfinx import mesh
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import Mesh
from typing import Tuple, List

from dolfinx.io import XDMFFile
import numpy as np
from mpi4py import MPI

import matplotlib.pyplot as plt


import ch_timedisc as ch


dts: List[float] = [1e-5, 1e-4, 1e-3, 1e-2]
T: float = 5e-2


def run_eyre(dt: float, T: float) -> Tuple[ch.Energy, ch.TimeMarching]:

    num_time_steps: int = int(T / dt)
    print(num_time_steps)

    # Define material parameters
    parameters: ch.Parameters = ch.Parameters(dt=dt, num_time_steps=num_time_steps)

    # Double well
    doublewell: ch.DoubleWell = ch.DoubleWell()

    # Mesh
    msh: Mesh = mesh.create_unit_square(
        MPI.COMM_WORLD, parameters.nx, parameters.ny, cell_type=mesh.CellType.triangle
    )

    # Initial condition
    initialcondition: ch.Cross2D = ch.Cross2D(width=0.3)

    # Set up femhandler
    femhandler: ch.FEMHandler = ch.FEMHandler(
        msh,
        initialcondition=initialcondition,
        parameters=parameters,
        doublewell=doublewell,
    )

    # Energy computations
    energy: ch.Energy = ch.Energy(femhandler, parameters, doublewell)

    # Linear variational form
    eyre: ch.VariationalEyre = ch.VariationalEyre(
        femhandler=femhandler, parameters=parameters, doublewell=doublewell
    )

    # Set up nonlinear problem
    problem: NonlinearProblem = NonlinearProblem(
        eyre.F,
        femhandler.xi,
        petsc_options_prefix="ch_eyre_" + f"dt_{dt}_",
        petsc_options=parameters.petsc_options,
    )

    # Set up time marching
    time_marching: ch.TimeMarching = ch.TimeMarching(
        femhandler=femhandler, parameters=parameters, energy=energy, problem=problem
    )

    # Perform time marching
    time_marching()

    return energy, time_marching


output_folder: str = "log/eyre_test/"
plt.figure("Energy")

for i, dt in enumerate(dts):
    energy, time_marching = run_eyre(dt=dt, T=T)
    results = {
        "dt": np.array([dt]),
        "Time": np.array(time_marching.time_vec),
        "Energy": np.array(energy.energy_vec),
    }

    np.savez(output_folder + f"dt_{i}.npz", **results)  # type: ignore[arg-type]
    plt.plot(time_marching.time_vec, energy.energy_vec, label=f"dt = {dt}")

plt.legend()

plt.show()
