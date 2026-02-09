"""CH Timedisc package convenience imports."""

from .parameters.parameters import Parameters
from .fem.fem_handler import FEMHandler
from .doublewell.doublewell import DoubleWell
from .intital_conditions.cross import Cross2D, Cross3D
from .intital_conditions.random import Random
from .intital_conditions.initial_mu import initial_mu
from .intital_conditions.initial_pf import initial_pf
from .visualization.pyvista_visualization import PyvistaVizualization
from .energy.energy import Energy
from .variational_forms.base import VariationalForm
from .variational_forms.eyre import VariationalEyre
from .variational_forms.implicit_euler import VariationalImplicitEuler
from .variational_forms.accurate_dissipation import VariationalAccurateDissipation
from .variational_forms.crank_nicholson import VariationalCrankNicholson
from .time_marching.time_marching import TimeMarching
from .time_marching.adaptive_time_step import AdaptiveTimeStep
from .time_marching.adaptive_time_step_energy_diff import AdaptiveTimeStepEnergyDiff
from .time_marching.adaptive_time_step_gradmu import AdaptiveTimeStepGradMu

__all__ = [
    "Parameters",
    "DoubleWell",
    "Cross2D",
    "Cross3D",
    "Random",
    "PyvistaVizualization",
    "Energy",
    "FEMHandler",
    "VariationalForm",
    "VariationalEyre",
    "VariationalImplicitEuler",
    "VariationalAccurateDissipation",
    "VariationalCrankNicholson",
    "TimeMarching",
    "AdaptiveTimeStep",
    "AdaptiveTimeStepEnergyDiff",
    "AdaptiveTimeStepGradMu",
    "initial_mu",
    "initial_pf",
]
__version__ = "0.1.0"
