"""M3GNet / matgl calculator adapter.

Install: pip install mixlip[matgl]
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


@register_calculator("m3gnet")
class M3GNetAdapter(MixLIPCalculator):
    model_name = "m3gnet"
    supported_properties = frozenset({"energy", "forces", "stress"})

    def _load_backend(self, config: ModelConfig):
        try:
            import matgl
            from matgl.ext.ase import PESCalculator
        except ImportError as e:
            raise ImportError("matgl is required: pip install mixlip[matgl]") from e

        checkpoint = config.checkpoint
        if checkpoint in ("pretrained", "m3gnet"):
            model = matgl.load_model("M3GNet-MP-2021.2.8-PES")
        else:
            model = matgl.load_model(checkpoint)
        try:
            # matgl>=2 returns stress in GPa by default; mixlip uses ASE units (eV/Å³)
            return PESCalculator(potential=model, stress_unit="eV/A3")
        except TypeError:  # older matgl without the stress_unit option (already eV/Å³)
            return PESCalculator(potential=model)

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
