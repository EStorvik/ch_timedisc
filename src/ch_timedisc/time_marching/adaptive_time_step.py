"""Adaptive time stepping for the Cahn-Hilliard equation.

This module provides adaptive time step control based on the gradient of the
chemical potential, ensuring numerical stability and computational efficiency.
"""

from typing import TYPE_CHECKING, Any

import ch_timedisc as ch
from dolfinx.fem.petsc import NonlinearProblem

if TYPE_CHECKING:
    from ch_timedisc.visualization import Energy
    from ch_timedisc.fem import FEMHandler
    from ch_timedisc.parameters import Parameters


class AdaptiveTimeStep:
    """Adaptive time step controller for Cahn-Hilliard simulations.

    This class implements an adaptive time stepping strategy that adjusts the
    time step size based on the squared gradient of the chemical potential.
    The time step is increased when the solution is smooth (small gradients)
    and decreased when sharp features develop (large gradients).

    The adaptation strategy:
    - Doubles dt if dt * ||∇μ||² < 10 (smooth solution, can take larger steps)
    - Halves dt if dt * ||∇μ||² > 100 (sharp features, need smaller steps)
    - Keeps dt unchanged if 10 ≤ dt * ||∇μ||² ≤ 100

    Parameters
    ----------
    energy : ch.Energy
        Energy object that computes energy-related quantities including
        dt_gradmusquared() which returns dt * ||∇μ||².
    parameters : ch.Parameters
        Parameters object containing simulation parameters including the
        time step size (dt) which will be modified by this controller.
    verbose : bool, optional
        If True, prints the updated time step size after each adaptation.
        Default is False.

    Attributes
    ----------
    energy : ch.Energy
        Reference to the energy object.
    parameters : ch.Parameters
        Reference to the parameters object.
    verbose : bool
        Verbosity flag for printing time step updates.

    Examples
    --------
    >>> import ch_timedisc as ch
    >>> # Setup simulation components
    >>> parameters = ch.Parameters(dt=0.01)
    >>> energy = ch.Energy(...)  # Initialize with appropriate arguments
    >>> adaptive_dt = ch.AdaptiveTimeStep(energy, parameters, verbose=True)
    >>>
    >>> # In time stepping loop
    >>> for step in range(num_steps):
    >>>     # Solve the system
    >>>     solver.solve()
    >>>     # Adapt time step based on solution characteristics
    >>>     adaptive_dt()

    Notes
    -----
    The threshold values (10 and 100) are heuristic and may need adjustment
    based on the specific problem characteristics and desired accuracy.
    """

    def __init__(
        self,
        energy: "Energy",
        femhandler: "FEMHandler",
        parameters: "Parameters",
        variational_form: Any,
        verbose: bool = False,
    ) -> None:
        """Initialize the adaptive time step controller.

        Args:
            energy: Energy object that computes energy-related quantities.
            femhandler: Finite element handler with solution variables.
            parameters: Parameters object containing simulation parameters.
            variational_form: Variational form object for updating the problem.
            verbose: Enable printing of time step updates. Defaults to False.
        """
        self.verbose: bool = verbose
        self.energy: "Energy" = energy
        self.parameters: "Parameters" = parameters
        self.femhandler: "FEMHandler" = femhandler
        self.variational_form: Any = variational_form

    def __call__(self) -> NonlinearProblem:
        """Adapt the time step based on the current solution state.

        Evaluates dt * ||∇μ||² and adjusts the time step size accordingly:
        - If < 10: doubles the time step (solution is smooth)
        - If > 100: halves the time step (sharp features present)
        - Otherwise: keeps the time step unchanged

        The time step size in self.parameters.dt is modified in-place.

        Returns:
            Updated nonlinear problem for the next time step.

        Notes
        -----
        This method should be called after each successful time step to
        evaluate whether the time step should be adjusted for the next step.
        """
        dt_grad_mu_sq = self.energy.dt_gradmusquared()

        if dt_grad_mu_sq < 10:
            self.parameters.dt *= 2
            problem = self.update_problem()
        # elif dt_grad_mu_sq > 100:
        #     self.parameters.dt *= 0.5
        #     problem = self.update_problem()
        else:
            problem = self.update_problem()
        if self.verbose:
            print(f"Time step size is: {self.parameters.dt}")

        return problem

    def update_problem(self) -> NonlinearProblem:
        """Update the nonlinear problem with the current parameters.

        Returns:
            Updated nonlinear problem with the current time step.
        """
        self.variational_form.update(self.parameters)

        # Set up nonlinear problem
        problem = NonlinearProblem(
            self.variational_form.F,
            self.femhandler.xi,
            petsc_options_prefix="ch_implicit_",
            petsc_options=self.parameters.petsc_options,
        )

        return problem
