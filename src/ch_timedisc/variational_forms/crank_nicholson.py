"""Crank-Nicholson time discretization variational forms for Cahn-Hilliard equation."""

from ufl import Form, dx, inner, grad

from .base import VariationalForm


class VariationalCrankNicholson(VariationalForm):
    """Crank-Nicholson time discretization scheme for the Cahn-Hilliard equation.

    The Crank-Nicholson scheme is a second-order, energy stable time discretization
    that uses midpoint evaluation of the nonlinear terms to achieve temporal accuracy
    while maintaining energy dissipation. It decouples the phase field and chemical
    potential equations through symmetric averaging.

    The scheme computes:
        F_pf: Time evolution of phase field
        F_mu: Chemical potential equation with midpoint-averaged potential derivative
        F: Combined variational form

    Attributes:
        F_pf (ufl.Form): Variational form for phase field equation.
        F_mu (ufl.Form): Variational form for chemical potential equation.
        F (ufl.Form): Complete variational form (F_pf + F_mu).
    """

    def _build_F_mu(self) -> Form:
        """Build chemical potential form using symmetric averaging (midpoint).

        The scheme uses symmetric averages (midpoint evaluations) of the Laplacian
        and potential derivative between current and previous time steps. This
        approach is second-order accurate in time and maintains energy stability
        properties.

        Returns:
            UFL form for the chemical potential equation.
        """
        f = self.femhandler
        p = self.parameters

        # Chemical potential equation with Crank-Nicholson symmetric averaging
        # Uses midpoint evaluation: 0.5*(value_n + value_n-1)
        # mu = -gamma*ell*0.5*(Δpf_n + Δpf_n-1) + (gamma/ell)*0.5*(f'(pf_n) + f'(pf_n-1))
        return (
            inner(f.mu, f.eta_mu) * dx
            - 0.5
            * p.gamma
            * p.ell
            * inner((grad(f.pf) + grad(f.pf_old)), grad(f.eta_mu))
            * dx
            - 0.5
            * p.gamma
            / p.ell
            * (self.doublewell.prime(f.pf) + self.doublewell.prime(f.pf_old))
            * f.eta_mu
            * dx
        )
