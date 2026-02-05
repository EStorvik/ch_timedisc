"""Energy tracking and analysis for Cahn-Hilliard simulations."""

from typing import TYPE_CHECKING, List, Union

import ch_timedisc as ch
from ufl import Form, dx, grad, inner, Measure
from dolfinx.fem import assemble_scalar, form, Function

if TYPE_CHECKING:
    from ch_timedisc.fem import FEMHandler
    from ch_timedisc.parameters import Parameters
    from ch_timedisc.doublewell import DoubleWell
    from ufl.core.expr import Expr as UFLExpr
    import numpy as np

PfType = Union[float, "np.ndarray", "Function", "UFLExpr"]


class Energy:
    """Track and analyze energy evolution in Cahn-Hilliard phase field simulations.

    This class computes the Cahn-Hilliard free energy and tracks its evolution
    over time, including the energy dissipation rate through the gradient of
    the chemical potential.

    Attributes:
        ell (float): Interface thickness parameter.
        t_vec (list): Time values at each recorded step.
        dt (float): Time step size.
        mobility (float): Mobility parameter.
        doublewell (DoubleWell): Double well potential function.
        energy_vec (list): Free energy values at each time step.
        gradmu_squared_vec (list): Squared gradient of chemical potential at each step.
    """

    def __init__(
        self,
        femhandler: "FEMHandler",
        parameters: "Parameters",
        doublewell: "DoubleWell",
    ) -> None:
        """Initialize the energy tracker.

        Args:
            femhandler: Finite element handler with solution variables.
            parameters: Simulation parameters containing ell, dt, mobility.
            doublewell: Double well potential object.
        """
        self.ell: float = parameters.ell
        self.parameters: "Parameters" = parameters
        self.mobility: float = parameters.mobility
        self.doublewell: "DoubleWell" = doublewell
        self.energy_vec: List[float] = []
        self.dt_energy_vec: List[float] = []
        self.ddt_energy_vec: List[float] = []
        self.gradmu_squared_vec: List[float] = []
        self.femhandler = femhandler
        self()

    def __call__(self) -> float:
        """Compute and record the current energy and dissipation.

        Returns:
            The computed energy value at the current solution (tracked by femhandler).
        """
        pf = self.femhandler.pf
        mu = self.femhandler.mu

        # Compute energy and add to list
        energy: float = assemble_scalar(form(self.energy(pf))).real
        self.energy_vec.append(energy)

        # Compute gradmu squared and add to list
        self.gradmu_squared_vec.append(self.gradmu_squared(mu))

        # Compute dt_E
        if len(self.energy_vec) > 1:
            self.dt_energy_vec.append(
                (self.energy_vec[-1] - self.energy_vec[-2]) / self.parameters.dt
            )

        # Compute ddt_E
        if len(self.dt_energy_vec) > 1:
            self.ddt_energy_vec.append(
                (self.dt_energy_vec[-1] - self.dt_energy_vec[-2]) / self.parameters.dt
            )

        return energy

    def energy(self, pf: PfType) -> Form:
        """Compute the Cahn-Hilliard free energy functional.

        The energy consists of the bulk free energy (double well potential)
        and the gradient energy (interface energy).

        Args:
            pf: Phase field function.

        Returns:
            The energy form to be assembled.
        """
        return (
            1 / self.ell * self.doublewell(pf)
            + self.ell / 2 * inner(grad(pf), grad(pf))
        ) * dx

    def gradmu_squared(self, mu: PfType) -> float:
        """Compute the squared gradient of chemical potential (dissipation term).

        This represents the energy dissipation rate in the Cahn-Hilliard equation.

        Args:
            mu: Chemical potential.

        Returns:
            Mobility-weighted squared gradient integral.
        """
        return (
            -self.mobility * assemble_scalar(form(inner(grad(mu), grad(mu)) * dx)).real
        )

    def dt_gradmusquared(self) -> float:
        """Compute the time derivative of squared gradient of chemical potential.

        Returns:
            Time rate of change of squared gradient.
        """
        return (
            self.gradmu_squared_vec[-1] - self.gradmu_squared_vec[-2]
        ) / self.parameters.dt

    def energy_dt_vec(self) -> List[float]:
        """Compute the discrete time derivative of energy.

        Returns:
            Time derivative of energy at each step (dE/dt).
        """
        energy_dt_vec: List[float] = []
        for i in range(len(self.energy_vec) - 1):
            energy_dt_vec.append(
                (self.energy_vec[i + 1] - self.energy_vec[i]) / self.parameters.dt
            )
        return energy_dt_vec
