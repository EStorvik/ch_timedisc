"""Chemical potential initialization from phase field."""

import ch_timedisc as ch
from ufl import grad, inner
from dolfinx.fem import Function, functionspace
from ufl import TestFunction, TrialFunction, dx, split
from dolfinx.fem.petsc import LinearProblem


def initial_pf(pf0, P, msh, parameters: ch.Parameters):

    V = functionspace(msh, P)
    u = TrialFunction(V)
    v = TestFunction(V)

    ell = parameters.ell

    a = 1 / ell * inner(u, v) * dx + ell * (inner(grad(u), grad(v))) * dx
    L = 1 / ell * inner(pf0, v) * dx

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
    pf = problem.solve()

    return pf
