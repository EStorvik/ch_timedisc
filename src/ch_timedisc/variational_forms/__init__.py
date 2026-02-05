"""Variational forms for time discretization schemes."""

from .base import VariationalForm
from .eyre import VariationalEyre
from .implicit_euler import VariationalImplicitEuler
from .accurate_dissipation import VariationalAccurateDissipation
from .crank_nicholson import VariationalCrankNicholson

__all__ = [
    "VariationalForm",
    "VariationalEyre",
    "VariationalImplicitEuler",
    "VariationalAccurateDissipation",
    "VariationalCrankNicholson",
]
