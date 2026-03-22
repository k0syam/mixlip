"""EquiformerV2 / FAIRChem calculator adapter.

Supports fairchem-core (formerly OCP / fair-chem).
Install: pip install mixlip[fairchem]
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from mixlip.calculators.base import MixLIPCalculator
from mixlip.core.protocol import PredictionResult
from mixlip.core.registry import register_calculator

if TYPE_CHECKING:
    import ase
    from mixlip.core.config import ModelConfig


@register_calculator("equiformerv2")
class EquiformerV2Adapter(MixLIPCalculator):
    model_name = "equiformerv2"
    supported_properties = frozenset({"energy", "forces", "stress"})

    def _load_backend(self, config: ModelConfig):
        try:
            from fairchem.core import OCPCalculator
        except ImportError as e:
            raise ImportError(
                "fairchem-core is required: pip install mixlip[fairchem]"
            ) from e

        checkpoint = config.checkpoint
        if checkpoint in ("pretrained", "equiformerv2"):
            checkpoint = "EquiformerV2-153M-S2EF-OC20-All+MD"
        return OCPCalculator(checkpoint=checkpoint, cpu=config.device == "cpu")

    def _run_backend(self, atoms: ase.Atoms) -> PredictionResult:
        t0 = time.perf_counter()
        self._backend.calculate(atoms, properties=["energy", "forces"])
        r = self._backend.results
        stress = r.get("stress")
        return PredictionResult(
            energy=float(r["energy"]),
            forces=r["forces"].copy(),
            stress=stress,
            metadata={
                "model": self.model_name,
                "checkpoint": self.config.checkpoint,
                "elapsed_s": time.perf_counter() - t0,
            },
        )
