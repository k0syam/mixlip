"""PyTorch Lightning module for MLIP training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import lightning as L
import torch

from mixlip.training.loss import WeightedEFSLoss

if TYPE_CHECKING:
    from mixlip.core.config import TrainingConfig


class MLIPLightningModule(L.LightningModule):
    """Wraps any torch-based MLIP model behind the Lightning training interface.

    The model must accept a torch_geometric.Data batch and return a dict
    with keys: "energy" (per-structure), "forces" (per-atom, optional),
    "stress" (per-structure Voigt, optional).
    """

    def __init__(self, model: torch.nn.Module, config: TrainingConfig):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = WeightedEFSLoss(config.loss)
        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch: Any) -> dict[str, torch.Tensor]:
        return self.model(batch)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        pred = self(batch)
        losses = self.loss_fn(pred, batch)
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=True,
            batch_size=batch.num_graphs,
        )
        return losses["total"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        pred = self(batch)
        losses = self.loss_fn(pred, batch)
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            on_epoch=True,
            batch_size=batch.num_graphs,
        )

    def test_step(self, batch: Any, batch_idx: int) -> None:
        pred = self(batch)
        losses = self.loss_fn(pred, batch)
        self.log_dict(
            {f"test/{k}": v for k, v in losses.items()},
            on_epoch=True,
            batch_size=batch.num_graphs,
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
