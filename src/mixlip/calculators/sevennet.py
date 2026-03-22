"""SevenNet calculator adapter.

Install: pip install mixlip[sevennet]
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


@register_calculator("sevennet")
class SevenNetAdapter(MixLIPCalculator):
    model_name = "sevennet"
    supported_properties = frozenset({"energy", "forces", "stress"})

    def _load_backend(self, config: ModelConfig):
        try:
            from sevenn.sevennet_calculator import SevenNetCalculator
        except ImportError as e:
            raise ImportError("sevenn is required: pip install mixlip[sevennet]") from e

        checkpoint = config.checkpoint
        if checkpoint in ("pretrained", "sevennet"):
            checkpoint = "7net-0"  # default pretrained model
        return SevenNetCalculator(checkpoint, device=config.device)

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
