"""Energy tracking and analysis for Cahn-Hilliard simulations."""

import ch_timedisc as ch
from ufl import dx, grad, inner, Measure
from dolfinx.fem import assemble_scalar, form


class Energy():
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
    
    def __init__(self, pf0, mu0, parameters: ch.Parameters, doublewell: ch.DoubleWell):
        """Initialize the energy tracker.
        
        Args:
            pf0 (Function): Initial phase field.
            mu0 (Function): Initial chemical potential.
            parameters (Parameters): Simulation parameters containing ell, dt, mobility.
            doublewell (DoubleWell): Double well potential object.
        """
        self.ell = parameters.ell
        self.t_vec = [parameters.t0]
        self.dt = parameters.dt
        self.mobility = parameters.mobility
        self.doublewell = doublewell
        self.energy_vec = []
        self.gradmu_squared_vec = []
        self(pf0, mu0)

    def energy(self, pf):
        """Compute the Cahn-Hilliard free energy functional.
        
        The energy consists of the bulk free energy (double well potential)
        and the gradient energy (interface energy).
        
        Args:
            pf (Function): Phase field function.
            
        Returns:
            ufl.Form: The energy form to be assembled.
        """
        return (1 / self.ell * self.doublewell(pf) + self.ell / 2 * inner(grad(pf), grad(pf))) * dx

    def __call__(self, pf, mu):
        """Compute and record the current energy and dissipation.
        
        Args:
            pf (Function): Current phase field.
            mu (Function): Current chemical potential.
            
        Returns:
            float: The computed energy value.
        """
        energy = assemble_scalar(form(self.energy(pf)))
        self.energy_vec.append(energy)
        self.gradmu_squared_vec.append(self.gradmu_squared(mu))
        return energy

    def energy_dt_vec(self):
        """Compute the discrete time derivative of energy.
        
        Returns:
            list: Time derivative of energy at each step (dE/dt).
        """
        energy_dt_vec = []
        for i in range(len(self.energy_vec) - 1):
            energy_dt_vec.append((self.energy_vec[i+1] - self.energy_vec[i]) / self.dt)
        return energy_dt_vec

    def gradmu_squared(self, mu):
        """Compute the squared gradient of chemical potential (dissipation term).
        
        This represents the energy dissipation rate in the Cahn-Hilliard equation.
        
        Args:
            mu (Function): Chemical potential.
            
        Returns:
            float: Mobility-weighted squared gradient integral.
        """
        return -self.mobility * assemble_scalar(form(inner(grad(mu), grad(mu)) * dx))
