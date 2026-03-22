"""Abstract base class for all MixLIP calculator adapters."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from mixlip.core.protocol import PredictionResult

if TYPE_CHECKING:
    import ase
    from pymatgen.core import Structure

    from mixlip.core.config import ModelConfig


class MixLIPCalculator(Calculator):
    """ASE Calculator base that also satisfies the MLIPCalculatorProtocol.

    Subclasses implement:
    - _load_backend(config) → the upstream calculator/model object
    - _run_backend(atoms) → PredictionResult

    Both calculate() [ASE contract] and predict() [Protocol contract]
    delegate to _run_backend internally.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    model_name: str = "base"
    supported_properties: frozenset[str] = frozenset({"energy", "forces", "stress"})

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self._backend = self._load_backend(config)

    @abstractmethod
    def _load_backend(self, config: ModelConfig):
        """Load and return the upstream model/calculator object."""

    @abstractmethod
    def _run_backend(self, atoms: ase.Atoms) -> PredictionResult:
        """Run inference with the upstream backend and return a PredictionResult."""

    # ------------------------------------------------------------------
    # ASE Calculator interface
    # ------------------------------------------------------------------

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        result = self._run_backend(self.atoms)
        self.results["energy"] = result.energy
        self.results["free_energy"] = result.energy
        self.results["forces"] = result.forces
        if result.stress is not None:
            self.results["stress"] = result.stress
        if result.magmoms is not None:
            self.results["magmoms"] = result.magmoms

    # ------------------------------------------------------------------
    # Protocol interface
    # ------------------------------------------------------------------

    def predict(self, structure: Structure) -> PredictionResult:
        """Run inference on a pymatgen Structure."""
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(structure)
        return self._run_backend(atoms)

    def predict_batch(self, structures: list[Structure]) -> list[PredictionResult]:
        """Run inference on a list of pymatgen Structures (sequential default)."""
        return [self.predict(s) for s in structures]

    @property
    def ase_calculator(self) -> Calculator:
        """Return self as an ASE Calculator (for Protocol compatibility)."""
        return self
