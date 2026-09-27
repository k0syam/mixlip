"""mixlip distill — knowledge distillation between MLIPs."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Knowledge distillation between MLIP models.")
console = Console()


@app.command("run")
def distill_run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to distillation config"),
    devices: int = typer.Option(1, "--devices"),
):
    """Run distillation from a config file."""
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from mixlip.calculators import load as load_calc
    from mixlip.core.config import load_config
    from mixlip.data.datamodule import MLIPDataModule
    from mixlip.training.distill import DistillationModule

    cfg = load_config(config)
    assert cfg.distill is not None, "--config must have a [distill] section"

    console.print("[bold]Distillation:[/bold]")
    console.print(f"  Teacher: [yellow]{cfg.distill.teacher.backend}[/yellow]")
    console.print(f"  Student: [cyan]{cfg.distill.student.backend}[/cyan]")

    # Teacher is inference-only (.predict()), so any backend works here
    # regardless of training support.
    teacher_calc = load_calc(cfg.distill.teacher.backend, cfg.distill.teacher)
    student_calc = load_calc(cfg.distill.student.backend, cfg.distill.student)
    if not student_calc.supports_training:
        console.print(
            f"[red]Student backend '{cfg.distill.student.backend}' does not support "
            f"training yet.[/red] Currently supported: chgnet."
        )
        raise typer.Exit(1)

    from mixlip.training.module import CalculatorTrainingWrapper

    module = DistillationModule(
        CalculatorTrainingWrapper(student_calc), cfg, teacher=teacher_calc
    )
    datamodule = MLIPDataModule(cfg.data)

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        devices=devices,
        gradient_clip_val=cfg.grad_clip_norm,
        callbacks=[
            ModelCheckpoint(monitor="val/total", save_top_k=3, mode="min"),
            EarlyStopping(monitor="val/total", patience=cfg.patience, mode="min"),
        ],
    )
    trainer.fit(module, datamodule=datamodule)
    console.print("[bold green]Distillation complete.[/bold green]")


@app.command("generate-labels")
def generate_labels(
    teacher: str = typer.Option(..., "--teacher", "-t", help="Teacher backend (e.g. mace)"),
    input_path: Path = typer.Option(..., "--input", "-i", help="Input dataset (extxyz/hdf5)"),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output HDF5 path"),
    device: str = typer.Option("cuda", "--device"),
    checkpoint: str = typer.Option("pretrained", "--checkpoint"),
):
    """Pre-compute teacher labels for offline distillation."""
    from mixlip.calculators import load as load_calc
    from mixlip.core.config import ModelConfig
    from mixlip.data.dataset import MLIPDataset
    from mixlip.training.distill import generate_teacher_labels

    teacher_cfg = ModelConfig(backend=teacher, checkpoint=checkpoint, device=device)  # type: ignore[arg-type]
    teacher_calc = load_calc(teacher, teacher_cfg)

    dataset = MLIPDataset.from_file(input_path, as_graph=False)
    samples = [dataset[i] for i in range(len(dataset))]

    generate_teacher_labels(samples, teacher_calc, output_path)
    console.print(f"[bold green]Teacher labels saved to {output_path}[/bold green]")
