"""Implicit Euler time discretization variational forms for Cahn-Hilliard equation."""

from ufl import Form, dx, inner, grad

from .base import VariationalForm


class VariationalImplicitEuler(VariationalForm):
    """Implicit Euler time discretization scheme for the Cahn-Hilliard equation.

    The Implicit Euler scheme is an energy stable, first-order time discretization that
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

    def _build_F_mu(self) -> Form:
        """Build chemical potential form for Implicit Euler scheme.

        The Implicit Euler scheme decouples the system by using the full
        potential derivative at the current time.

        Returns:
            UFL form for the chemical potential equation.
        """
        f = self.femhandler
        p = self.parameters

        # Chemical potential equation with convex-concave split
        # mu = -gamma*ell*Δpf + (gamma/ell)*f'(pf_n)
        return (
            inner(f.mu, f.eta_mu) * dx
            - p.gamma * p.ell * inner(grad(f.pf), grad(f.eta_mu)) * dx
            - (p.gamma / p.ell) * (self.doublewell.prime(f.pf)) * f.eta_mu * dx
        )
