"""ALIGNN calculator adapter.

Install: pip install mixlip[alignn]
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


@register_calculator("alignn")
class ALIGNNAdapter(MixLIPCalculator):
    model_name = "alignn"
    supported_properties = frozenset({"energy", "forces", "stress"})

    def _load_backend(self, config: ModelConfig):
        try:
            from alignn.ff.ff import AlignnAtomwiseCalculator, default_path
        except ImportError as e:
            raise ImportError("alignn is required: pip install mixlip[alignn]") from e

        checkpoint = config.checkpoint
        if checkpoint in ("pretrained", "alignn"):
            model_path = default_path()
        else:
            model_path = checkpoint
        return AlignnAtomwiseCalculator(path=model_path, device=config.device)

    def _run_backend(self, atoms: ase.Atoms) -> PredictionResult:
        t0 = time.perf_counter()
        self._backend.calculate(atoms, properties=["energy", "forces", "stress"])
        r = self._backend.results
        return PredictionResult(
            energy=float(r["energy"]),
            forces=r["forces"].copy(),
            stress=r.get("stress"),
            metadata={
                "model": self.model_name,
                "elapsed_s": time.perf_counter() - t0,
            },
        )
