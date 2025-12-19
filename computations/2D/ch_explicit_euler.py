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
gamma = parameters.gamma
ell = parameters.ell
mobility = parameters.mobility
dt = parameters.dt

# Double well
doublewell = ch.DoubleWell()

# Mesh
msh = mesh.create_unit_square(
    MPI.COMM_WORLD, parameters.nx, parameters.ny, cell_type=mesh.CellType.triangle
)

# Finite elements
P1 = element("Lagrange", msh.basix_cell(), 1)
ME = mixed_element([P1, P1])

# Function spaces
V = functionspace(msh, ME)

# Test function
eta = TestFunction(V)
eta_pf, eta_mu = split(eta)


# Solution function
xi = Function(V)
pf, mu = split(xi)

xi_old = Function(V)
pf_old, mu_old = split(xi_old)


# Initial conditions
initialcondition = ch.Cross2D(width=0.3)
# initialcondition = ch.Random()
xi.sub(0).interpolate(initialcondition)

# Interpolate initial condition to mu
mu0 = ch.intitial_mu(pf, P1, msh, parameters=parameters, doublewell=doublewell)
xi.sub(1).interpolate(mu0)
xi.x.scatter_forward()

# Copy to old
xi_old.x.array[:] = xi.x.array
xi_old.x.scatter_forward()

energy = ch.Energy(pf, mu, parameters, doublewell)

# Linear variational forms
F_pf = (
    inner(pf - pf_old, eta_pf) * dx + dt * mobility * inner(grad(mu), grad(eta_pf)) * dx
)
F_mu = (
    inner(mu, eta_mu) * dx
    - gamma * ell * inner(grad(pf_old), grad(eta_mu)) * dx
    - gamma / ell * doublewell.prime(pf_old) * eta_mu * dx
)
F = F_pf + F_mu


# Set up nonlinear problem
problem = NonlinearProblem(
    F, xi, petsc_options_prefix="ch_implicit_", petsc_options=parameters.petsc_options
)


# Pyvista plot
viz = ch.visualization.PyvistaVizualization(V.sub(0), xi.sub(0), 0.0)

# Output file
# output_file_pf = XDMFFile(MPI.COMM_WORLD, "../output/ch_implicit.xdmf", "w")
# output_file_pf.write_mesh(msh)

# Time stepping
t = parameters.t0


time_vec = [t]

for i in range(parameters.num_time_steps):
    # Set old time-step functions
    xi_old.x.array[:] = xi.x.array
    xi_old.x.scatter_forward()

    # Update current time
    t += dt

    # Solve
    n, converged = problem.solve()

    if not converged:
        print(f"WARNING: Newton solver did not converge at time step {i}")

    # print(f"Used {n} newton iterations to converge at time step {i}.")

    # energy_int = assemble_scalar(form(energy_i(pf, dx=Measure("dx", domain=msh))))
    time_vec.append(t)
    # energy_vec.append(energy_int)

    viz.update(xi.sub(0), t)
    print(energy(pf, mu))
    # print(f"The energy at time step {i} is {energy_int}.")

    # # Output
    # pf_out, _ = xi_n.split()
    # output_file_pf.write_function(pf_out, t)


viz.final_plot(xi.sub(0))


plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"  # or 'sans-serif'
plt.rcParams["font.size"] = 16

# plt.figure("Energy evolution")
# plt.plot(time_vec, energy.energy_vec)
# plt.show()


plt.figure("dt Energy")
plt.plot(time_vec[2:], energy.energy_dt_vec()[1:], label=r"$\partial_t\mathcal{E}$")
plt.plot(time_vec[2:], energy.gradmu_squared_vec[2:], label=r"$-m\|\nabla\mu\|^2$")

plt.legend()

plt.figure("dte - mnmu^2")
plt.plot(
    time_vec[2:],
    np.array(energy.energy_dt_vec()[1:]) - np.array(energy.gradmu_squared_vec[2:]),
)
plt.show()
# output_file_pf.close()
