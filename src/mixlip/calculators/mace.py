"""MACE calculator adapter.

Supports:
- mace-mp pretrained models (mace_mp())
- Custom MACE checkpoints (MACECalculator)

Install: pip install mixlip[mace]
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


@register_calculator("mace")
class MACEAdapter(MixLIPCalculator):
    model_name = "mace"
    supported_properties = frozenset({"energy", "forces", "stress"})

    def _load_backend(self, config: ModelConfig):
        try:
            from mace.calculators import MACECalculator, mace_mp
        except ImportError as e:
            raise ImportError("mace-torch is required: pip install mixlip[mace]") from e

        if config.checkpoint in ("pretrained", "mace-mp", "mace_mp"):
            return mace_mp(
                model=config.model_size,
                device=config.device,
                default_dtype=config.dtype,
            )
        return MACECalculator(
            model_paths=config.checkpoint,
            device=config.device,
            default_dtype=config.dtype,
        )

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
                "checkpoint": self.config.checkpoint,
                "elapsed_s": time.perf_counter() - t0,
            },
        )
