"""Accurate dissipation time discretization variational forms for Cahn-Hilliard equation."""

from ufl import Form, dx, inner, grad

from .base import VariationalForm


class VariationalAccurateDissipation(VariationalForm):
    """Accurate dissipation time discretization scheme for the Cahn-Hilliard equation.

    The accurate dissipation scheme is an energy stable
    discretization that uses Taylor expansions of the potential derivative
    to achieve improved accuracy while maintaining energy dissipation. It
    decouples the phase field and chemical potential equations through
    a convex-concave splitting approach.

    The scheme computes:
        F_pf: Time evolution of phase field
        F_mu: Chemical potential equation with Taylor-expanded potential derivative
        F: Combined variational form

    Attributes:
        F_pf (ufl.Form): Variational form for phase field equation.
        F_mu (ufl.Form): Variational form for chemical potential equation.
        F (ufl.Form): Complete variational form (F_pf + F_mu).
    """

    def _build_F_mu(self) -> Form:
        """Build chemical potential form using Taylor expansion.

        The scheme uses Taylor expansions of the potential derivative centered at
        the current phase field value, with terms involving higher-order derivatives
        (doubleprime, tripleprime, quadprime) and powers of the phase field change.
        This provides improved temporal accuracy while preserving energy stability.

        Returns:
            UFL form for the chemical potential equation.
        """
        f = self.femhandler
        p = self.parameters

        # Chemical potential equation with accurate dissipation Taylor expansion
        # Uses derivatives up to quadprime for improved temporal accuracy
        # mu = -gamma*ell*Δpf_avg + (gamma/ell)*f'_Taylor(pf, pf_old)
        return (
            inner(f.mu, f.eta_mu) * dx
            - 0.5
            * p.gamma
            * p.ell
            * inner((grad(f.pf) + grad(f.pf_old)), grad(f.eta_mu))
            * dx
            - p.gamma
            / p.ell
            * (
                self.doublewell.prime(f.pf)
                - 0.5 * self.doublewell.doubleprime(f.pf) * (f.pf - f.pf_old)
                + 1 / 6 * self.doublewell.tripleprime(f.pf) * (f.pf - f.pf_old) ** 2
                - 1 / 24 * self.doublewell.quadprime(f.pf) * (f.pf - f.pf_old) ** 3
            )
            * f.eta_mu
            * dx
        )
