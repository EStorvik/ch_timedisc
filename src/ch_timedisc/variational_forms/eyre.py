"""Eyre time discretization variational forms for Cahn-Hilliard equation."""

from ufl import dx, inner, grad

from .base import VariationalForm


class VariationalEyre(VariationalForm):
    """Eyre time discretization scheme for the Cahn-Hilliard equation.

    The Eyre scheme is an energy stable, first-order time discretization that
    decouples the phase field and chemical potential equations. It uses a
    convex-concave splitting of the potential to ensure discrete energy
    dissipation.

    The scheme computes:
        F_pf: Time evolution of phase field
        F_mu: Chemical potential equation with split potential derivative
        F: Combined variational form

    Attributes:
        F_pf (ufl.Form): Variational form for phase field equation.
        F_mu (ufl.Form): Variational form for chemical potential equation.
        F (ufl.Form): Complete variational form (F_pf + F_mu).
    """

    def _build_F_mu(self):
        """Build chemical potential form using convex-concave splitting.

        The Eyre scheme decouples the system by using the convex part of the
        potential derivative (cprime) at the current time and the concave part
        (eprime) at the previous time, ensuring energy stability.

        Returns:
            UFL form for the chemical potential equation.
        """
        f = self.femhandler
        p = self.parameters

        # Chemical potential equation with convex-concave split
        # mu = -gamma*ell*Δpf + (gamma/ell)*(cprime(pf_n) - eprime(pf_n-1))
        return (
            inner(f.mu, f.eta_mu) * dx
            - p.gamma * p.ell * inner(grad(f.pf), grad(f.eta_mu)) * dx
            - (p.gamma / p.ell)
            * (self.doublewell.cprime(f.pf) - self.doublewell.eprime(f.pf_old))
            * f.eta_mu
            * dx
        )
