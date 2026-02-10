"""Adaptive time stepping based on gradient of chemical potential.

This module implements an adaptive time stepping strategy that adjusts the
time step size based on dt * ||∇μ||², where μ is the chemical potential.
This quantity represents the time-scaled energy dissipation rate in the
Cahn-Hilliard equation.
"""

from typing import TYPE_CHECKING, Literal

from .adaptive_time_step import AdaptiveTimeStep

if TYPE_CHECKING:
    from ch_timedisc.energy import Energy
    from ch_timedisc import VariationalForm
    from ch_timedisc import Parameters
    from ch_timedisc import FEMHandler


class AdaptiveTimeStepGradMu(AdaptiveTimeStep):
    """Adaptive time step controller based on gradient of chemical potential.

    This class implements an adaptive time stepping strategy that adjusts the
    time step size based on the quantity dt * ||∇μ||², where μ is the chemical
    potential. This represents the time-scaled energy dissipation rate.

    The adaptation strategy:
    - Decreases dt if dt*||∇μ||² < threshold_decrease (large dissipation, needs smaller steps)
    - Increases dt if dt*||∇μ||² > threshold_increase (small dissipation, can take larger steps)
    - Keeps dt unchanged if threshold_decrease ≤ dt*||∇μ||² ≤ threshold_increase

    Note: Since the energy dissipation -m||∇μ||² is negative, thresholds are negative
    values where more negative means larger dissipation.

    Parameters
    ----------
    energy : ch.Energy
        Energy object that computes energy-related quantities including
        gradmu_squared() which returns -m||∇μ||².
    variational_form : ch.VariationalForm
        Variational form to update when dt changes.
    parameters : ch.Parameters
        Simulation parameters containing dt and other settings.
    femhandler : ch.FEMHandler
        Finite element handler with solution variables.
    factor : float, optional
        Factor to multiply/divide time step by. Default: 1.1
    threshold_increase : float, optional
        Upper threshold for -dt*m||∇μ||² (less negative). If exceeded, dt increases.
        Default: -0.001
    threshold_decrease : float, optional
        Lower threshold for -dt*m||∇μ||² (more negative). If below, dt decreases.
        Default: -0.01
    verbose : bool, optional
        If True, prints the updated time step size after each adaptation.
        Default: False

    Attributes
    ----------
    energy : Energy
        Reference to the energy object.
    threshold_increase : float
        Upper threshold (less negative) for dt increases.
    threshold_decrease : float
        Lower threshold (more negative) for dt decreases.

    Notes
    -----
    The threshold values are heuristic and may need adjustment
    based on the specific problem characteristics and desired accuracy.
    """

    def __init__(
        self,
        energy: "Energy",
        variational_form: "VariationalForm",
        parameters: "Parameters",
        femhandler: "FEMHandler",
        factor: float = 1.1,
        threshold_increase: float = -0.001,
        threshold_decrease: float = -0.01,
        verbose: bool = False,
    ) -> None:
        """Initialize the gradient-based adaptive time step controller.

        Args:
            energy: Energy object that computes energy-related quantities.
            variational_form: Variational form to update when dt changes.
            parameters: Simulation parameters containing dt.
            femhandler: Finite element handler with solution variables.
            factor: Factor to multiply/divide dt by. Default: 1.1
            threshold_increase: Upper threshold (less negative) for dt increases. Default: -0.001
            threshold_decrease: Lower threshold (more negative) for dt decreases. Default: -0.01
            verbose: If True, prints time step updates. Default: False
        """
        super().__init__(
            factor=factor,
            femhandler=femhandler,
            parameters=parameters,
            variational_form=variational_form,
            threshold_increase=threshold_increase,
            threshold_decrease=threshold_decrease,
            verbose=verbose,
        )
        self.energy = energy

    def criterion(self) -> Literal["decrease", "increase", "keep"]:
        """Evaluate the gradient-based adaptation criterion.

        Computes dt*(-m||∇μ||²) and determines whether to increase, decrease,
        or keep the time step based on the threshold values. Also checks if
        energy is increasing, which triggers a dt decrease.

        Returns:
            "decrease": If dt*(-m||∇μ||²) < threshold_decrease (more negative, large dissipation)
                       OR if energy is increasing (non-physical behavior)
            "increase": If dt*(-m||∇μ||²) > threshold_increase (less negative, small dissipation)
            "keep": If threshold_decrease ≤ dt*(-m||∇μ||²) ≤ threshold_increase
        """
        dtgradmu = -self.parameters.dt * self.energy.gradmu_squared()

        if (
            dtgradmu < self.threshold_decrease
            or self.energy.energy() - self.energy.energy_vec[-1] > 0
        ):
            return "decrease"
        elif dtgradmu > self.threshold_increase:
            return "increase"
        else:
            return "keep"
