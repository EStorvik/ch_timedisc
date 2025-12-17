"""CH Timedisc package convenience imports."""

from .parameters.parameters import Parameters
from .doublewell.doublewell import DoubleWell
from .intital_conditions.cross import Cross2D, Cross3D
from .intital_conditions.random import Random
from .intital_conditions.initial_mu import initial_mu
from .intital_conditions.initial_pf import initial_pf
from .visualization.pyvista_visualization import PyvistaVizualization
from .visualization.energy import Energy
from .fem.fem_handler import FEMHandler
from .variational_forms.eyre import VariationalEyre
from .time_marching.time_marching import TimeMarching

__all__ = [
    "Parameters",
    "DoubleWell",
    "Cross2D",
    "Cross3D",
    "Random",
    "PyvistaVizualization",
    "Energy",
    "FEMHandler",
    "VariationalEyre",
    "TimeMarching",
    "initial_mu",
    "initial_pf",
]
__version__ = "0.1.0"


