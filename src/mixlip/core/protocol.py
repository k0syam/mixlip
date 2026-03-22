"""Core protocol and result types for MLIP calculators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from mixlip.core.types import EnergyScalar, ForceArray, MagmomArray, StressArray

if TYPE_CHECKING:
    import ase
    from pymatgen.core import Structure


@dataclass
class PredictionResult:
    """Single-structure prediction output.

    All units follow ASE convention:
    - energy: eV (total)
    - forces: eV/Å, shape (N, 3)
    - stress: eV/Å³, Voigt 6-component (xx yy zz yz xz xy), or None
    - magmoms: μB, shape (N,), or None
    """

    energy: EnergyScalar
    forces: ForceArray
    stress: StressArray | None = None
    magmoms: MagmomArray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.forces = np.asarray(self.forces, dtype=np.float64)
        if self.stress is not None:
            self.stress = np.asarray(self.stress, dtype=np.float64)
            if self.stress.shape == (3, 3):
                self.stress = _full_3x3_to_voigt_6(self.stress)
        if self.magmoms is not None:
            self.magmoms = np.asarray(self.magmoms, dtype=np.float64)


def _full_3x3_to_voigt_6(stress_3x3: np.ndarray) -> np.ndarray:
    """Convert 3x3 stress tensor to 6-component Voigt vector."""
    return np.array(
        [
            stress_3x3[0, 0],
            stress_3x3[1, 1],
            stress_3x3[2, 2],
            stress_3x3[1, 2],
            stress_3x3[0, 2],
            stress_3x3[0, 1],
        ],
        dtype=np.float64,
    )


@runtime_checkable
class MLIPCalculatorProtocol(Protocol):
    """Structural protocol every calculator adapter must satisfy.

    Also exposes an .ase_calculator property for drop-in ASE compatibility.
    """

    model_name: str
    supported_properties: frozenset[str]

    def predict(self, structure: Structure) -> PredictionResult: ...

    def predict_batch(self, structures: list[Structure]) -> list[PredictionResult]: ...

    @property
    def ase_calculator(self) -> ase.calculators.calculator.Calculator: ...
