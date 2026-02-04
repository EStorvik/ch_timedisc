"""Phase field smoothing from initial condition."""

from typing import TYPE_CHECKING, Any

import ch_timedisc as ch
from ufl import grad, inner
from dolfinx.fem import Function, functionspace
from ufl import TestFunction, TrialFunction, dx, split
from dolfinx.fem.petsc import LinearProblem
from ufl.core.expr import Expr as UFLExpr

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh


def initial_pf(
    pf0: UFLExpr,
    P: Any,
    msh: "Mesh",
    parameters: ch.Parameters,
) -> Function:
    """Smooth initial phase field using regularization.

    Solves the regularized problem to smooth the initial phase field:
        ∫ (1/ell) * pf * v dx + ∫ ell * ∇pf·∇v dx = ∫ (1/ell) * pf0 * v dx

    where ell is a regularization length scale. This creates a smoothed
    phase field representation of the initial condition.

    Args:
        pf0: Initial phase field condition (callable or expression).
        P: Finite element basis for the scalar field.
        msh: Computational mesh.
        parameters: Simulation parameters containing length scale (ell).

    Returns:
        Smoothed phase field function in the scalar function space.
    """

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
