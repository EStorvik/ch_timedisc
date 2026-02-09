"""Adaptive time stepping based on energy dissipation (rate, not time scaled).

This module implements an adaptive time stepping strategy that adjusts the
time step size based on the squared gradient of the chemical potential,
which represents the energy dissipation rate.
"""

from typing import TYPE_CHECKING, Literal

from .adaptive_time_step import AdaptiveTimeStep

if TYPE_CHECKING:
    from ch_timedisc.energy import Energy


class AdaptiveTimeStepEnergyDiff(AdaptiveTimeStep):
    """Adaptive time step controller based on energy dissipation.

    This class implements an adaptive time stepping strategy that adjusts the
    time step size based on the squared gradient of the chemical potential
    (dt * ||∇μ||²), which represents the energy dissipation rate.

    The adaptation strategy:
    - Increases dt if dt * ||∇μ||² < threshold_low (smooth solution, can take larger steps)
    - Decreases dt if dt * ||∇μ||² > threshold_high (sharp features, need smaller steps)
    - Keeps dt unchanged if threshold_low ≤ dt * ||∇μ||² ≤ threshold_high

    Parameters
    ----------
    energy : ch.Energy
        Energy object that computes energy-related quantities including
        dt_gradmusquared() which returns dt * ||∇μ||².
    factor : float, optional
        Factor to increase or decrease time step by. Default: 2.0
    threshold_low : float, optional
        Lower threshold for - dt * m * ||∇μ||². Default: -0.001
    threshold_high : float, optional
        Upper threshold for - dt * m * ||∇μ||². Default: -0.01
    verbose : bool, optional
        If True, prints the updated time step size after each adaptation.
        Default is False.

    Attributes
    ----------
    threshold_low : float
        Lower threshold for - dt * m * ||∇μ||².
    threshold_high : float
        Upper threshold for - dt * m * ||∇μ||².

    Notes
    -----
    The threshold values are heuristic and may need adjustment
    based on the specific problem characteristics and desired accuracy.
    """

    def __init__(
        self,
        energy: "Energy",
        dt0: float,
        factor: float = 1.1,
        threshold_low: float = -0.001,
        threshold_high: float = -0.01,
        verbose: bool = False,
    ) -> None:
        """Initialize the energy-based adaptive time step controller.

        Args:
            energy: Energy object that computes energy-related quantities.
            factor: Factor to increase or decrease time step by. Default: 2.0
            threshold_low: Lower threshold for dt * ||∇μ||². Default: -0.001
            threshold_high: Upper threshold for dt * ||∇μ||². Default: -0.01
        """
        super().__init__(dt0=dt0, factor=factor, verbose=verbose)
        self.energy = energy
        self.threshold_low: float = threshold_low
        self.threshold_high: float = threshold_high

    def criterion(self) -> Literal["decrease", "increase", "keep"]:
        """Evaluate the energy dissipation-based adaptation criterion.

        Computes dt * ||∇μ||² and determines whether to increase, decrease,
        or keep the time step based on the threshold values.

        Returns:
            "increase": If dt * ||∇μ||² < threshold_low (smooth solution)
            "decrease": If dt * ||∇μ||² > threshold_high (sharp features)
            "keep": If threshold_low ≤ dt * ||∇μ||² ≤ threshold_high
        """
        dt_grad_mu_sq = -self.dt * self.energy.gradmu_squared()

        if (
            dt_grad_mu_sq < self.threshold_low
            or self.energy.energy() - self.energy.energy_vec[-1] > 0
        ):
            return "decrease"
        elif dt_grad_mu_sq > self.threshold_high:
            return "increase"
        else:
            return "keep"
