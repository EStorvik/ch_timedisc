"""Adaptive time stepping for the Cahn-Hilliard equation.

This module provides adaptive time step control based on different criteria.
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
    strategies.

    Parameters
    ----------
    factor : float
       factor to increase or decrease time step by
        decrease: dt = dt / factor
        increase: dt = dt * factor

    Attributes
    ----------
    factor : float
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
            factor: Energy object that computes energy-related quantities.
            verbose: prints update information.
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
        """Update time step size"""

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
