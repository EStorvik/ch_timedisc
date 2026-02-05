"""Finite element method setup and management for Cahn-Hilliard simulations."""

from typing import TYPE_CHECKING, Callable

from basix.ufl import element, mixed_element
from dolfinx.fem import Function, functionspace
from dolfinx.fem.function import FunctionSpace
from ufl import Argument, split, TestFunction
from ufl.core.expr import Expr as UFLExpr
import ch_timedisc as ch

if TYPE_CHECKING:
    from dolfinx.mesh import Mesh
    from ch_timedisc.parameters import Parameters
    from ch_timedisc.doublewell import DoubleWell


class FEMHandler:
    """Manages finite element spaces and solution variables for Cahn-Hilliard problems.

    This class sets up a mixed finite element space for the phase field (pf) and
    chemical potential (mu), initializes solution functions, and prepares test
    functions for variational formulations.

    Attributes:
        V (FunctionSpace): Mixed function space for (pf, mu).
        eta (TestFunction): Test function on the mixed space.
        eta_pf (ufl.Coefficient): Test function component for phase field.
        eta_mu (ufl.Coefficient): Test function component for chemical potential.
        xi (Function): Current solution (pf, mu) at time t.
        pf (ufl.Coefficient): Current phase field component.
        mu (ufl.Coefficient): Current chemical potential component.
        xi_old (Function): Previous solution at time t-dt.
        pf_old (ufl.Coefficient): Previous phase field component.
        mu_old (ufl.Coefficient): Previous chemical potential comp''''onent.
        initialcondition: Initial condition function for phase field.
    """

    def __init__(
        self,
        msh: "Mesh",
        initialcondition: Callable,
        parameters: "Parameters",
        doublewell: "DoubleWell",
    ) -> None:
        """Initialize FEM setup for Cahn-Hilliard equation.

        Args:
            msh (Mesh): The computational domain mesh.
            initialcondition: Initial condition function for phase field
                (callable that returns values at mesh points).
            parameters (Parameters): Simulation parameters object.
            doublewell (DoubleWell): Double well potential object.
        """
        # Finite elements: P1 Lagrange for both components
        P1 = element("Lagrange", msh.basix_cell(), 1)
        ME = mixed_element([P1, P1])

        # Function spaces
        self.V: FunctionSpace = functionspace(msh, ME)

        # Test function on mixed space
        self.eta: Argument = TestFunction(self.V)
        self.eta_pf: UFLExpr
        self.eta_mu: UFLExpr
        self.eta_pf, self.eta_mu = split(self.eta)

        # Solution functions
        self.xi: Function = Function(self.V)
        self.pf: UFLExpr
        self.mu: UFLExpr
        self.pf, self.mu = split(self.xi)

        self.xi_old: Function = Function(self.V)
        self.pf_old: UFLExpr
        self.mu_old: UFLExpr
        self.pf_old, self.mu_old = split(self.xi_old)

        # Initialize phase field
        self.initialcondition = initialcondition
        self.xi.sub(0).interpolate(initialcondition)
        pf0: UFLExpr = ch.initial_pf(self.pf, P1, msh, parameters=parameters)
        self.xi.sub(0).interpolate(pf0)
        self.xi.x.scatter_forward()

        # Initialize chemical potential from phase field
        mu0: UFLExpr = ch.initial_mu(
            self.pf, P1, msh, parameters=parameters, doublewell=doublewell
        )
        self.xi.sub(1).interpolate(mu0)
        self.xi.x.scatter_forward()

        # Copy to old for time stepping
        self.xi_old.x.array[:] = self.xi.x.array
        self.xi_old.x.scatter_forward()
