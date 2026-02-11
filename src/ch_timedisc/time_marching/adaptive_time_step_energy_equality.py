"""Adaptive time stepping based on energy equality constraint.

This module implements an adaptive time stepping strategy that adjusts the
time step size based on the energy equality: dE/dt = -||∇μ||².
The method monitors the deviation from this theoretical equality.
"""

from typing import TYPE_CHECKING, Literal

from .adaptive_time_step import AdaptiveTimeStep

if TYPE_CHECKING:
    from ch_timedisc.energy import Energy
    from ch_timedisc import VariationalForm
    from ch_timedisc import Parameters
    from ch_timedisc import FEMHandler


class AdaptiveTimeStepEnergyEquality(AdaptiveTimeStep):
    """Adaptive time step controller based on energy equality.

    This class implements an adaptive time stepping strategy that adjusts the
    time step size based on the energy equality: dE/dt = -||∇μ||².
    The controller monitors the deviation from this theoretical equality:
    energy_eq = (E(t) - E(t-dt))/dt - (-||∇μ||²)

    The adaptation strategy:
    - Increases dt if energy_eq < threshold_increase (equality well satisfied)
    - Decreases dt if energy_eq > threshold_decrease or E(t) - E(t-dt) > 0 (energy increasing)
    - Keeps dt unchanged if threshold_increase ≤ energy_eq ≤ threshold_decrease

    Parameters
    ----------
    energy : Energy
        Energy object that computes energy-related quantities.
    variational_form : VariationalForm
        Variational form for the time discretization.
    parameters : Parameters
        Parameters object containing simulation parameters including dt.
    femhandler : FEMHandler
        Finite element method handler.
    factor : float, optional
        Factor to increase or decrease time step by. Default: 1.1
    threshold_increase : float, optional
        Lower threshold for energy equality deviation. Default: -0.001
    threshold_decrease : float, optional
        Upper threshold for energy equality deviation. Default: -0.01
    verbose : bool, optional
        If True, prints the updated time step size after each adaptation.
        Default: False

    Attributes
    ----------
    energy : Energy
        Energy object for computing energy quantities.
    threshold_increase : float
        Lower threshold for energy equality deviation.
    threshold_decrease : float
        Upper threshold for energy equality deviation.

    Notes
    -----
    The threshold values are heuristic and may need adjustment
    based on the specific problem characteristics and desired accuracy.
    The energy should be non-increasing in time for physical consistency.
    """

    def __init__(
        self,
        energy: "Energy",
        variational_form: "VariationalForm",
        parameters: "Parameters",
        femhandler: "FEMHandler",
        verbose: bool = False,
    ) -> None:
        """Initialize the energy equality-based adaptive time step controller.

        Args:
            energy: Energy object that computes energy-related quantities.
            variational_form: Variational form for the time discretization.
            parameters: Parameters object containing simulation parameters.
            femhandler: Finite element method handler.
            factor: Factor to increase or decrease time step by. Default: 1.1
            threshold_increase: Lower threshold for energy equality deviation. Default: -0.001
            threshold_decrease: Upper threshold for energy equality deviation. Default: -0.01
            verbose: If True, prints the updated time step size. Default: False
        """
        super().__init__(
            femhandler=femhandler,
            parameters=parameters,
            variational_form=variational_form,
            verbose=verbose,
        )
        self.energy = energy

    def criterion(self) -> Literal["decrease", "increase", "keep"]:
        """Evaluate the energy equality-based adaptation criterion.

        Computes the deviation from energy equality:
        energy_eq = (E(t) - E(t-dt))/dt - (-||∇μ||²)

        Returns:
            "increase": If energy_eq < threshold_increase (equality well satisfied)
            "decrease": If energy_eq > threshold_decrease or energy increases
            "keep": If threshold_increase ≤ energy_eq ≤ threshold_decrease
        """
        energy_eq = (
            self.energy.energy() - self.energy.energy_vec[-1]
        ) / self.parameters.dt - self.energy.gradmu_squared()

        if (
            energy_eq > self.threshold_decrease
            or self.energy.energy() - self.energy.energy_vec[-1] > 0
        ):
            return "decrease"
        elif energy_eq < self.threshold_increase:
            return "increase"
        else:
            return "keep"
