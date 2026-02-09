"""Adaptive time stepping for the Cahn-Hilliard equation.

This module provides adaptive time step control based on different criteria.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal


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

    def __init__(self, dt0: float, factor: float, verbose: bool) -> None:
        """Initialize the adaptive time step controller.

        Args:
            factor: Energy object that computes energy-related quantities.
            verbose: prints update information.
        """

        self.factor = factor
        self.verbose = verbose
        self.dt = dt0

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

    def update_dt(self, dt: float, state: str, time_step: int) -> float:
        """Update time step size"""

        if state == "decrease":
            n_dt = dt / self.factor

        elif state == "increase":
            n_dt = dt * self.factor

        if self.verbose:
            print(f"Updated time step size at time step {time_step} is dt = {n_dt}")

        self.dt = n_dt

        return n_dt
