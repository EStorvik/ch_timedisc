"""Adaptive time stepping based on energy dissipation (rate, not time scaled).

This module implements an adaptive time stepping strategy that adjusts the
time step size based on the squared gradient of the chemical potential,
which represents the energy dissipation rate.
"""

from typing import TYPE_CHECKING, Literal

from .adaptive_time_step import AdaptiveTimeStep

if TYPE_CHECKING:
    from ch_timedisc.energy import Energy
    from ch_timedisc import VariationalForm
    from ch_timedisc import Parameters
    from ch_timedisc import FEMHandler


class AdaptiveTimeStepEnergyDiff(AdaptiveTimeStep):
    """Adaptive time step controller based on energy dissipation.

    This class implements an adaptive time stepping strategy that adjusts the
    time step size based on the squared gradient of the chemical potential
    (dt * ||∇μ||²), which represents the energy dissipation rate.

    The adaptation strategy:
    - Increases dt if dt * ||∇μ||² < threshold_increase (smooth solution, can take larger steps)
    - Decreases dt if dt * ||∇μ||² > threshold_decrease (sharp features, need smaller steps)
    - Keeps dt unchanged if threshold_increase ≤ dt * ||∇μ||² ≤ threshold_decrease

    Parameters
    ----------
    energy : ch.Energy
        Energy object that computes energy-related quantities including
        dt_gradmusquared() which returns dt * ||∇μ||².
    factor : float, optional
        Factor to increase or decrease time step by. Default: 2.0
    threshold_increase : float, optional
        Lower threshold for - dt * m * ||∇μ||². Default: -0.001
    threshold_decrease : float, optional
        Upper threshold for - dt * m * ||∇μ||². Default: -0.01
    verbose : bool, optional
        If True, prints the updated time step size after each adaptation.
        Default is False.

    Attributes
    ----------
    threshold_increase : float
        Lower threshold for - dt * m * ||∇μ||².
    threshold_decrease : float
        Upper threshold for - dt * m * ||∇μ||².

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
        """Initialize the energy-based adaptive time step controller.

        Args:
            energy: Energy object that computes energy-related quantities.
            factor: Factor to increase or decrease time step by. Default: 2.0
            threshold_increase: Lower threshold for dt * ||∇μ||². Default: -0.001
            threshold_decrease: Upper threshold for dt * ||∇μ||². Default: -0.01
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
        """Evaluate the energy dissipation-based adaptation criterion.

        Computes dt * ||∇μ||² and determines whether to increase, decrease,
        or keep the time step based on the threshold values.

        Returns:
            "increase": If dt * ||∇μ||² < threshold_increase (smooth solution)
            "decrease": If dt * ||∇μ||² > threshold_decrease (sharp features)
            "keep": If threshold_increase ≤ dt * ||∇μ||² ≤ threshold_decrease
        """
        energy_diff = self.energy.energy() - self.energy.energy_vec[-1]

        if (
            energy_diff < self.threshold_decrease
            or self.energy.energy() - self.energy.energy_vec[-1] > 0
        ):
            return "decrease"
        elif energy_diff > self.threshold_increase:
            return "increase"
        else:
            return "keep"
