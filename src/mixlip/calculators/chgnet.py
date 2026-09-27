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
    import torch

    from mixlip.core.config import ModelConfig
    from mixlip.data.schema import AtomicSample

# CHGNet's raw model output ("s") is a per-graph 3x3 stress tensor in GPa.
# mixlip's convention (see PredictionResult) is Voigt-6 in eV/Å³.
_GPA_TO_EV_PER_A3 = 1.0 / 160.21766208


def _voigt6_from_3x3(t: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 stress tensor to Voigt-6 (xx yy zz yz xz xy), keeping autograd."""
    import torch

    return torch.stack([t[0, 0], t[1, 1], t[2, 2], t[1, 2], t[0, 2], t[0, 1]])


@register_calculator("chgnet")
class CHGNetAdapter(MixLIPCalculator):
    model_name = "chgnet"
    supported_properties = frozenset({"energy", "forces", "stress", "magmoms"})
    supports_training = True

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

    # ------------------------------------------------------------------
    # Training / distillation interface
    # ------------------------------------------------------------------

    @property
    def trainable_module(self):
        return self._backend.model

    def training_forward(self, samples: list[AtomicSample]) -> dict[str, torch.Tensor]:
        import torch

        model = self._backend.model
        device = self._backend.device

        task = "ef"
        if self.config.compute_stress:
            task += "s"
        if any(s.magmoms is not None for s in samples):
            task += "m"

        # CHGNet computes forces/stress as autograd.grad(energy, positions/strain)
        # *inside* its own forward pass. Lightning runs validation/test steps
        # under torch.no_grad(), which would make that internal grad() call fail
        # even though we don't need to backprop *through* this call ourselves
        # there — so force grad on regardless of the ambient context.
        with torch.enable_grad():
            graphs = [model.graph_converter(s.structure).to(device) for s in samples]
            raw = model(graphs, task=task)

            energy = raw["e"].float()
            if model.is_intensive:
                energy = energy * raw["atoms_per_graph"].to(energy.dtype)

            out: dict[str, torch.Tensor] = {
                "energy": energy,
                "forces": torch.cat(raw["f"], dim=0).float(),
            }
            if "s" in raw:
                out["stress"] = torch.stack(
                    [_voigt6_from_3x3(s * _GPA_TO_EV_PER_A3) for s in raw["s"]], dim=0
                ).float()
            if "m" in raw:
                out["magmoms"] = torch.cat(raw["m"], dim=0).float()
        return out
