"""Chemical potential initialization from phase field. This is actually not needed."""

from typing import TYPE_CHECKING, Any, cast

import ch_timedisc as ch
from ufl import grad, inner
from dolfinx.fem import Function, functionspace
from ufl import TestFunction, TrialFunction, dx
from dolfinx.fem.petsc import LinearProblem
from ufl.core.expr import Expr as UFLExpr

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh


def initial_mu(
    pf0: UFLExpr,
    P: Any,
    msh: "Mesh",
    parameters: ch.Parameters,
    doublewell: ch.DoubleWell,
) -> Function:
    """Initialize chemical potential from initial phase field.

    Solves the variational problem:
        ∫ mu * v dx = ell ∫ ∇pf·∇v dx + (1/ell) ∫ f'(pf) v dx

    where mu is the chemical potential, pf is the phase field, and f' is the
    derivative of the double-well potential. This recovers the chemical
    potential field consistent with the phase field at initialization.

    Args:
        pf0: Initial phase field expression or function.
        P: Finite element basis for the scalar field.
        msh: Computational mesh.
        parameters: Simulation parameters containing length scale (ell).
        doublewell: Double well potential object for computing derivatives.

    Returns:
        Initial chemical potential function in the scalar function space.
    """

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
    mu0 = cast(Function, problem.solve())

    return mu0
