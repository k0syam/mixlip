"""CHGNet calculator adapter.

Install: pip install mixlip[chgnet]
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


@register_calculator("chgnet")
class CHGNetAdapter(MixLIPCalculator):
    model_name = "chgnet"
    supported_properties = frozenset({"energy", "forces", "stress", "magmoms"})

    def _load_backend(self, config: ModelConfig):
        try:
            from chgnet.model import CHGNet
            from chgnet.model.dynamics import CHGNetCalculator
        except ImportError as e:
            raise ImportError("chgnet is required: pip install mixlip[chgnet]") from e

        if config.checkpoint in ("pretrained", "chgnet"):
            model = CHGNet.load()
        else:
            model = CHGNet.from_file(config.checkpoint)
        return CHGNetCalculator(model=model, use_device=config.device)

    def _run_backend(self, atoms: ase.Atoms) -> PredictionResult:
        t0 = time.perf_counter()
        self._backend.calculate(atoms, properties=["energy", "forces", "stress"])
        r = self._backend.results
        return PredictionResult(
            energy=float(r["energy"]),
            forces=r["forces"].copy(),
            stress=r.get("stress"),
            magmoms=r.get("magmoms"),
            metadata={
                "model": self.model_name,
                "elapsed_s": time.perf_counter() - t0,
            },
        )
