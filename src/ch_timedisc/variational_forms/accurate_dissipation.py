"""Accurate dissipation time discretization variational forms for Cahn-Hilliard equation."""

from ufl import dx, inner, grad


class VariationalAccurateDissipation:
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

    def __init__(self, femhandler, parameters, doublewell):
        """Initialize accurate dissipation variational forms.

        The scheme uses Taylor expansions of the potential derivative centered at
        the current phase field value, with terms involving higher-order derivatives
        (doubleprime, tripleprime, quadprime) and powers of the phase field change.
        This provides improved temporal accuracy while preserving energy stability.

        Args:
            femhandler (FEMHandler): Finite element handler with spaces and functions.
            parameters (Parameters): Simulation parameters (dt, gamma, ell, mobility).
            doublewell (DoubleWell): Double well potential with derivative methods.
        """
        f = femhandler
        p = parameters

        # Phase field equation: time discretization with decoupled mobility
        self.F_pf = (
            inner(f.pf - f.pf_old, f.eta_pf) * dx
            + p.dt * p.mobility * inner(grad(f.mu), grad(f.eta_pf)) * dx
        )
        # Chemical potential equation with accurate dissipation Taylor expansion
        # Uses derivatives up to quadprime for improved temporal accuracy
        # mu = -gamma*ell*Δpf_avg + (gamma/ell)*f'_Taylor(pf, pf_old)
        self.F_mu = (
            inner(f.mu, f.eta_mu) * dx
            - 0.5
            * p.gamma
            * p.ell
            * inner((grad(f.pf) + grad(f.pf_old)), grad(f.eta_mu))
            * dx
            - p.gamma
            / p.ell
            * (
                doublewell.prime(f.pf)
                - 0.5 * doublewell.doubleprime(f.pf) * (f.pf - f.pf_old)
                + 1 / 6 * doublewell.tripleprime(f.pf) * (f.pf - f.pf_old) ** 2
                - 1 / 24 * doublewell.quadprime(f.pf) * (f.pf - f.pf_old) ** 3
            )
            * f.eta_mu
            * dx
        )

        # Combined variational form for nonlinear solver
        self.F = self.F_pf + self.F_mu
