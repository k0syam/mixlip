"""mixlip train — fine-tune an MLIP on a dataset."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Fine-tune an MLIP on a dataset.")
console = Console()


@app.callback(invoke_without_command=True)
def train(
    config: Path = typer.Option(..., "--config", "-c", help="Path to training config YAML/TOML"),
    devices: int = typer.Option(1, "--devices", help="Number of GPUs"),
    fast_dev_run: bool = typer.Option(False, "--fast-dev-run", help="Run 1 batch for debugging"),
):
    """Fine-tune an MLIP from a YAML/TOML config file."""
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from mixlip.core.config import load_config
    from mixlip.data.datamodule import MLIPDataModule
    from mixlip.training.callbacks import EMACallback, MetricLoggerCallback

    cfg = load_config(config)
    console.print(f"[bold]Training config:[/bold] {config}")
    console.print(f"  Backend: [cyan]{cfg.model.backend}[/cyan]")
    console.print(f"  Dataset: [cyan]{cfg.data.train_path}[/cyan]")

    # Load model
    from mixlip.calculators import load as load_calc

    calc = load_calc(cfg.model.backend, cfg.model)
    # The underlying torch model is at calc._backend or calc._backend.model
    # This is backend-specific; use a generic wrapper
    model = _extract_torch_model(calc)

    from mixlip.training.module import MLIPLightningModule

    module = MLIPLightningModule(model, cfg)
    datamodule = MLIPDataModule(cfg.data)

    callbacks = [
        ModelCheckpoint(monitor="val/total", save_top_k=3, mode="min"),
        EarlyStopping(monitor="val/total", patience=cfg.patience, mode="min"),
        MetricLoggerCallback(),
    ]
    if cfg.use_ema:
        callbacks.append(EMACallback(cfg.ema_decay))

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        devices=devices,
        gradient_clip_val=cfg.grad_clip_norm,
        callbacks=callbacks,
        log_every_n_steps=cfg.log_every_n_steps,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(module, datamodule=datamodule)
    console.print("[bold green]Training complete.[/bold green]")


def _extract_torch_model(calc):
    """Best-effort extraction of torch.nn.Module from a MixLIPCalculator."""
    import torch.nn as nn

    backend = calc._backend
    # Try common attribute names used by upstream libraries
    for attr in ("model", "net", "_model", "calculator"):
        if hasattr(backend, attr) and isinstance(getattr(backend, attr), nn.Module):
            return getattr(backend, attr)
    if isinstance(backend, nn.Module):
        return backend
    raise RuntimeError(
        f"Could not extract a torch.nn.Module from {type(backend).__name__}. "
        "Use the Python API directly and pass the model explicitly."
    )
