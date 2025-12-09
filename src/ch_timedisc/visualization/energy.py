import ch_timedisc as ch
from ufl import dx, grad, inner, Measure
from dolfinx.fem import assemble_scalar, form



class Energy():
    def __init__(self, pf0, parameters: ch.Parameters, doublewell: ch.DoubleWell, msh):
        self.ell = parameters.ell
        self.t_vec = [parameters.t0]
        self.dt = parameters.dt
        self.doublewell = doublewell
        self.msh = msh
        self.energy_vec = [self(pf0)]

    def energy(self, pf):
        return (1 / self.ell * self.doublewell(pf) + self.ell / 2 * inner(grad(pf), grad(pf))) * dx

    def __call__(self, pf):
        energy = assemble_scalar(form(self.energy(pf), metadata={"quadrature_degree": 4}))
        self.energy_vec.append(energy)
        return energy

    def energy_dt(self):
        energy_dt_vec = []
        for i in len(self.energy_vec-1):
            energy_dt_vec.append((self.energy_vec[i+1]-self.energy_vec[i+1])/self.dt)

    
