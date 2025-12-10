"""Chemical potential initialization from phase field."""

import ch_timedisc as ch
from ufl import grad, inner
from dolfinx.fem import Function, functionspace
from ufl import TestFunction, TrialFunction, dx, split
from dolfinx.fem.petsc import LinearProblem


def intitial_mu(pf0, P, msh, parameters: ch.Parameters, doublewell: ch.DoubleWell):

    V = functionspace(msh, P)
    u = TrialFunction(V)
    v = TestFunction(V)

    ell = parameters.ell

    # Variational form: ∫ mu * v dx = ell ∫ ∇pf·∇v dx + (1/ell) ∫ f'(pf) v dx
    a = inner(u, v) * dx
    L = (
        ell * inner(grad(pf0), grad(v)) * dx
        + (1 / ell) * doublewell.prime(pf0) * v * dx
    )

    problem = LinearProblem(
        a,
        L,
        bcs=[],
        petsc_options_prefix="initial_mu_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "ksp_error_if_not_converged": True,
        },
    )
    mu0 = problem.solve()

    return mu0
