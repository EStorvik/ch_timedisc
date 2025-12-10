"""CH Timedisc package convenience imports."""

from .parameters.parameters import Parameters
from .doublewell.doublewell import DoubleWell
from .intital_conditions.cross import Cross2D, Cross3D
from .intital_conditions.random import Random
from .intital_conditions.interpolate_mu import interpolate_mu
from .visualization.pyvista_visualization import PyvistaVizualization
from .visualization.energy import Energy

__all__ = ["Parameters", "DoubleWell", "Cross2D", "Cross3D", "Random", "PyvistaVizualization", "Energy", "interpolate_mu"]
__version__ = "0.1.0"
