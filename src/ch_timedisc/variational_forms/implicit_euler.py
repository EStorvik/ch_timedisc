"""Implicit Euler time discretization variational forms for Cahn-Hilliard equation."""

from ufl import dx, inner, grad


class VariationalImplicitEuler:
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

    def __init__(self, femhandler, parameters, doublewell):
        """Initialize Implicit Euler variational forms.

        The Implicit Euler scheme decouples the system by using the convex part of the
        potential derivative (cprime) at the current time and the concave part
        (eprime) at the previous time, ensuring energy stability.

        Args:
            femhandler (FEMHandler): Finite element handler with spaces and functions.
            parameters (Parameters): Simulation parameters (dt, gamma, ell, mobility).
            doublewell (DoubleWell): Double well potential with cprime and eprime.
        """
        f = femhandler
        p = parameters

        # Phase field equation: time discretization with decoupled mobility
        self.F_pf = (
            inner(f.pf - f.pf_old, f.eta_pf) * dx
            + p.dt * p.mobility * inner(grad(f.mu), grad(f.eta_pf)) * dx
        )
        # Chemical potential equation with convex-concave split
        # mu = -gamma*ell*Δpf + (gamma/ell)*(cprime(pf_n) - eprime(pf_n-1))
        self.F_mu = (
            inner(f.mu, f.eta_mu) * dx
            - p.gamma * p.ell * inner(grad(f.pf), grad(f.eta_mu)) * dx
            - (p.gamma / p.ell) * (doublewell.prime(f.pf)) * f.eta_mu * dx
        )

        # Combined variational form for nonlinear solver
        self.F = self.F_pf + self.F_mu
