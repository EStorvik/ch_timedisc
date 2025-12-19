"""Variational forms for time discretization schemes."""

from .eyre import VariationalEyre
from .implicit_euler import VariationalImplicitEuler
from .accurate_dissipation import VariationalAccurateDissipation
from .crank_nicholson import VariationalCrankNicholson

__all__ = [
    "VariationalEyre",
    "VariationalImplicitEuler",
    "VariationalAccurateDissipation",
    "VariationalCrankNicholson",
]
