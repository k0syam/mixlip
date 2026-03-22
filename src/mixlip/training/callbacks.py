"""Training callbacks."""

from __future__ import annotations

import lightning as L
import torch


class EMACallback(L.Callback):
    """Exponential Moving Average of model weights.

    Maintains an EMA copy of parameters and swaps them in for
    validation/test, then restores originals for training.
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                self._shadow[name] = param.data.clone()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self._shadow:
                self._shadow[name] = (
                    self.decay * self._shadow[name] + (1 - self.decay) * param.data
                )

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._backup = {
            name: param.data.clone()
            for name, param in pl_module.named_parameters()
            if param.requires_grad
        }
        for name, param in pl_module.named_parameters():
            if name in self._shadow:
                param.data.copy_(self._shadow[name])

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        for name, param in pl_module.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup.clear()


class MetricLoggerCallback(L.Callback):
    """Log per-epoch summary to stdout using rich."""

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f"Epoch {trainer.current_epoch}")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            for key, val in trainer.callback_metrics.items():
                if "val" in key:
                    table.add_row(key, f"{float(val):.4f}")
            console.print(table)
        except ImportError:
            pass
