"""Ensemble calculator: mean + std across multiple MLIP models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mixlip.core.protocol import PredictionResult

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from mixlip.calculators.base import MixLIPCalculator


class EnsembleCalculator:
    """Run multiple MixLIPCalculators and return mean + uncertainty (std).

    The ensemble std over energies/forces is a lightweight uncertainty proxy
    without requiring MC-Dropout or ensembled weights.

    Parameters
    ----------
    calculators:
        List of MixLIPCalculator instances (can mix backends).
    """

    def __init__(self, calculators: list[MixLIPCalculator]):
        if len(calculators) < 2:
            raise ValueError("EnsembleCalculator requires at least 2 calculators.")
        self.calculators = calculators
        self.model_name = "ensemble[" + ",".join(c.model_name for c in calculators) + "]"

    def predict(self, structure: Structure) -> PredictionResult:
        """Return mean prediction and attach std to metadata."""
        results = [calc.predict(structure) for calc in self.calculators]

        energies = np.array([r.energy for r in results])
        forces = np.stack([r.forces for r in results], axis=0)  # (M, N, 3)

        mean_energy = float(energies.mean())
        std_energy = float(energies.std())
        mean_forces = forces.mean(axis=0)
        std_forces = forces.std(axis=0)

        stress = None
        std_stress = None
        stresses = [r.stress for r in results if r.stress is not None]
        if len(stresses) == len(results):
            stress_arr = np.stack(stresses, axis=0)
            stress = stress_arr.mean(axis=0)
            std_stress = stress_arr.std(axis=0)

        return PredictionResult(
            energy=mean_energy,
            forces=mean_forces,
            stress=stress,
            metadata={
                "model": self.model_name,
                "energy_std": std_energy,
                "forces_std": std_forces.tolist(),
                "stress_std": std_stress.tolist() if std_stress is not None else None,
                "n_models": len(results),
            },
        )

    def predict_batch(self, structures: list[Structure]) -> list[PredictionResult]:
        return [self.predict(s) for s in structures]
