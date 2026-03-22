"""ORB calculator adapter.

Install: pip install mixlip[orb]
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


@register_calculator("orb")
class ORBAdapter(MixLIPCalculator):
    model_name = "orb"
    supported_properties = frozenset({"energy", "forces", "stress"})

    def _load_backend(self, config: ModelConfig):
        try:
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
        except ImportError as e:
            raise ImportError("orb-models is required: pip install mixlip[orb]") from e

        checkpoint = config.checkpoint
        if checkpoint in ("pretrained", "orb"):
            model = pretrained.orb_v2(device=config.device)
        else:
            model = pretrained.orb_v2(weights_path=checkpoint, device=config.device)
        return ORBCalculator(model, device=config.device)

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
