"""Adaptive time stepping for the Cahn-Hilliard equation.

This module provides the abstract base class for adaptive time step control
based on different criteria. Concrete implementations define specific adaptation
strategies (e.g., based on energy dissipation, gradient of chemical potential,
error estimates, etc.).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional
from dolfinx.fem.petsc import NonlinearProblem


if TYPE_CHECKING:
    from ch_timedisc import VariationalForm
    from ch_timedisc import Parameters
    from ch_timedisc import FEMHandler


class AdaptiveTimeStep(ABC):
    """Abstract base class for adaptive time step controllers.

    This class defines the interface for adaptive time stepping strategies
    that adjust the time step size based on problem-specific criteria.
    Different concrete implementations can implement different adaptation
    strategies (e.g., energy-based, gradient-based, error-based).

    Subclasses must implement the `criterion()` method that returns whether
    to "increase", "decrease", or "keep" the time step based on the current
    solution state.

    Parameters
    ----------
    factor : float
        Factor to multiply/divide time step by.
        - Increase: dt = dt * factor
        - Decrease: dt = dt / factor
    variational_form : VariationalForm
        Variational form that needs updating when dt changes.
    parameters : Parameters
        Simulation parameters containing dt and other settings.
    femhandler : FEMHandler
        Finite element handler with solution variables.
    verbose : bool
        If True, prints time step updates.

    Attributes
    ----------
    factor : float
        Multiplicative factor for time step adaptation.
    variational_form : VariationalForm
        Reference to the variational form object.
    parameters : Parameters
        Reference to the parameters object.
    femhandler : FEMHandler
        Reference to the FEM handler.
    verbose : bool
        Verbosity flag for printing updates.
    """

    def __init__(
        self,
        factor: float,
        variational_form: "VariationalForm",
        parameters: "Parameters",
        femhandler: "FEMHandler",
        verbose: bool,
    ) -> None:
        """Initialize the adaptive time step controller.

        Args:
            factor: Factor to multiply/divide dt by during adaptation.
            variational_form: Variational form to update when dt changes.
            parameters: Simulation parameters containing dt.
            femhandler: Finite element handler with solution variables.
            verbose: If True, prints update information.
        """

        self.factor = factor
        self.verbose = verbose
        self.variational_form = variational_form
        self.parameters = parameters
        self.femhandler = femhandler

    @abstractmethod
    def criterion(self) -> Literal["decrease", "increase", "keep"]:
        """Evaluate the adaptation criterion based on the current solution state.

        This method must be implemented by concrete subclasses to determine
        whether to increase, decrease, or keep the time step size.

        Returns:
            "increase": Increase the time step
            "decrease": Decrease the time step
            "keep": Keep the time step unchanged
        """
        pass

    def update_dt(
        self, state: str, time_step: Optional[int] = None
    ) -> NonlinearProblem:
        """Update the time step size and recreate the nonlinear problem.

        This method modifies parameters.dt based on the state, updates the
        variational form with the new dt, and creates a new NonlinearProblem
        with the updated form.

        Args:
            state: Either "increase" or "decrease" to modify dt.
            time_step: Current time step number for verbose output.

        Returns:
            Updated NonlinearProblem with the new time step size.
        """

        if state == "decrease":
            self.parameters.dt /= self.factor

        elif state == "increase":
            self.parameters.dt *= self.factor

        if self.verbose and time_step is not None:
            print(
                f"Updated time step size at time step {time_step} is dt = {self.parameters.dt}"
            )

        self.variational_form.update()
        problem: NonlinearProblem = NonlinearProblem(
            self.variational_form.F,
            self.femhandler.xi,
            petsc_options_prefix="ch_",
            petsc_options=self.parameters.petsc_options,
        )
        return problem
