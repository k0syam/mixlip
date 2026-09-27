"""PyTorch Lightning module for MLIP training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import torch

from mixlip.data.schema import collate_labels
from mixlip.training.loss import WeightedEFSLoss

if TYPE_CHECKING:
    from mixlip.calculators.base import MixLIPCalculator
    from mixlip.core.config import TrainingConfig
    from mixlip.data.schema import AtomicSample


class CalculatorTrainingWrapper(torch.nn.Module):
    """Adapts a MixLIPCalculator to the nn.Module interface Lightning expects.

    forward(samples: list[AtomicSample]) delegates to
    `calculator.training_forward`, so each backend controls exactly how its
    own native graph representation is built and how outputs map to mixlip's
    {energy, forces, stress, magmoms} convention (see MixLIPCalculator).

    `calculator.trainable_module` is registered as a submodule so that
    `self.parameters()` — and therefore the optimizer — sees the real
    upstream weights, not a disconnected copy.
    """

    def __init__(self, calculator: MixLIPCalculator):
        super().__init__()
        if not calculator.supports_training:
            # Also raised (with the same message) by calculator.trainable_module
            # below, but failing here gives a clearer stack for CLI callers.
            calculator.trainable_module  # noqa: B018 - raises NotImplementedError
        self.calculator = calculator
        self.torch_model = calculator.trainable_module

    def forward(self, samples: list[AtomicSample]) -> dict[str, torch.Tensor]:
        return self.calculator.training_forward(samples)


class MLIPLightningModule(L.LightningModule):
    """Wraps a trainable MLIP model behind the Lightning training interface.

    `model` is expected to accept a list[AtomicSample] batch (as produced by
    MLIPDataModule with as_graph=False) and return a dict with keys "energy"
    (per-structure), "forces" (per-atom, optional), "stress" (per-structure
    Voigt, optional) — see CalculatorTrainingWrapper for the standard adapter
    from a MixLIPCalculator.
    """

    def __init__(self, model: torch.nn.Module, config: TrainingConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = WeightedEFSLoss(config.loss)
        self.save_hyperparameters(ignore=["model"])

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # Lightning's default transfer recurses into dataclass fields looking
        # for anything with a `.to(device)` method — and AtomicSample.structure
        # (a pymatgen Structure) happens to define an unrelated `.to(fmt=...)`
        # for file export, which that blind duck-typing then calls incorrectly.
        # AtomicSample batches carry no tensors to move: each backend's
        # training_forward places its own graph tensors on `device` itself.
        return batch

    def forward(self, batch: list[AtomicSample]) -> dict[str, torch.Tensor]:
        return self.model(batch)

    def training_step(self, batch: list[AtomicSample], batch_idx: int) -> torch.Tensor:
        pred = self(batch)
        targets = collate_labels(batch, device=pred["energy"].device)
        losses = self.loss_fn(pred, targets)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
            batch_size=len(batch),
        )
        return losses["total"]

    def validation_step(self, batch: list[AtomicSample], batch_idx: int) -> None:
        pred = self(batch)
        targets = collate_labels(batch, device=pred["energy"].device)
        losses = self.loss_fn(pred, targets)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            on_epoch=True,
            batch_size=len(batch),
        )

    def test_step(self, batch: list[AtomicSample], batch_idx: int) -> None:
        pred = self(batch)
        targets = collate_labels(batch, device=pred["energy"].device)
        losses = self.loss_fn(pred, targets)
        self.log_dict(
            {f"test/{k}": v for k, v in losses.items()},
            on_epoch=True,
            batch_size=len(batch),
        )

    def configure_optimizers(self):
        cfg = self.config
        optim_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[
            cfg.optimizer
        ]
        optimizer = optim_cls(self.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        scheduler_cfg: dict = {}
        if cfg.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.max_epochs
            )
            scheduler_cfg = {"scheduler": scheduler, "interval": "epoch"}
        elif cfg.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            scheduler_cfg = {"scheduler": scheduler, "interval": "epoch"}
        elif cfg.scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=cfg.patience // 5
            )
            scheduler_cfg = {
                "scheduler": scheduler,
                "monitor": "val/total",
                "interval": "epoch",
            }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_cfg}
