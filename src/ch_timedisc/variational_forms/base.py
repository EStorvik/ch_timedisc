"""Base class for variational forms in Cahn-Hilliard equation discretizations."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ufl import Form, dx, inner, grad

if TYPE_CHECKING:
    from ch_timedisc.fem import FEMHandler
    from ch_timedisc.parameters import Parameters
    from ch_timedisc.doublewell import DoubleWell


class VariationalForm(ABC):
    """Abstract base class for variational form discretizations.

    All Cahn-Hilliard variational forms share a common structure:
    - Phase field equation (F_pf) is identical across all schemes
    - Chemical potential equation (F_mu) varies by discretization scheme
    - Combined form (F) is always F_pf + F_mu

    Subclasses only need to implement the `_build_F_mu` method.

    Attributes:
        F_pf: Variational form for phase field equation.
        F_mu: Variational form for chemical potential equation.
        F: Combined variational form (F_pf + F_mu).
    """

    def __init__(
        self,
        femhandler: "FEMHandler",
        parameters: "Parameters",
        doublewell: "DoubleWell",
    ) -> None:
        """Initialize variational forms.

        Args:
            femhandler: Finite element handler with spaces and functions.
            parameters: Simulation parameters (dt, gamma, ell, mobility).
            doublewell: Double well potential with derivative methods.
        """
        self.femhandler = femhandler
        self.parameters = parameters
        self.doublewell = doublewell

        # Build forms on initialization
        self._build_forms()

    def _build_forms(self) -> None:
        """Build all variational forms."""
        self.F_pf: Form = self._build_F_pf()
        self.F_mu: Form = self._build_F_mu()
        self.F: Form = self.F_pf + self.F_mu

    def _build_F_pf(self) -> Form:
        """Build phase field variational form (identical for all schemes)."""
        f = self.femhandler
        p = self.parameters

        return (
            inner(f.pf - f.pf_old, f.eta_pf) * dx
            + p.dt * p.mobility * inner(grad(f.mu), grad(f.eta_pf)) * dx
        )

    @abstractmethod
    def _build_F_mu(self) -> Form:
        """Build chemical potential variational form (scheme-specific).

        This method must be implemented by subclasses to define the specific
        discretization scheme for the chemical potential equation.

        Returns:
            UFL form for the chemical potential equation.
        """
        pass

    def update(self) -> None:
        """Update variational forms with new parameters.

        This is useful for adaptive time stepping where parameters change.

        Args:
            parameters: Updated simulation parameters.
        """
        self._build_forms()
