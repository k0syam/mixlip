"""Shared type aliases used across the mixlip package."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

# Physical quantity arrays
EnergyScalar: TypeAlias = float  # eV (total energy)
ForceArray: TypeAlias = NDArray[np.float64]  # (N, 3)  eV/Å
StressArray: TypeAlias = NDArray[np.float64]  # (6,)   eV/Å³, Voigt order: xx yy zz yz xz xy
MagmomArray: TypeAlias = NDArray[np.float64]  # (N,)   μB
