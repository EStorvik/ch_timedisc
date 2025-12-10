"""Chemical potential initialization from phase field."""

import ch_timedisc as ch
from ufl import grad, inner
from dolfinx.fem import Function
from dolfinx.fem.petsc import LinearProblem


def interpolate_mu(
    pf, mu, eta_mu, V, parameters: ch.Parameters, doublewell: ch.DoubleWell
):

    from ufl import TestFunction, TrialFunction, dx, split

    x = TrialFunction(V)
    _, u = split(x)

    ell = parameters.ell

    # Variational form: ∫ mu * v dx = ell ∫ ∇pf·∇v dx + (1/ell) ∫ f'(pf) v dx
    a = inner(u, eta_mu) * dx
    L = (
        ell * inner(grad(pf), grad(eta_mu)) * dx
        + (1 / ell) * doublewell.prime(pf) * eta_mu * dx
    )

    problem = LinearProblem(a, L, bcs=[])
    mu = problem.solve()

    return mu
