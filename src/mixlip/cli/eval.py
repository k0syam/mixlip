"""mixlip eval — evaluate an MLIP on a labeled dataset."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Evaluate MLIP accuracy on a labeled dataset.")
console = Console()


@app.callback(invoke_without_command=True)
def evaluate(
    backend: str = typer.Option(..., "--backend", "-b"),
    dataset: Path = typer.Option(..., "--dataset", "-d"),
    device: str = typer.Option("cuda", "--device"),
    checkpoint: str = typer.Option("pretrained", "--checkpoint"),
    n_samples: int = typer.Option(0, "--n", help="Max samples (0 = all)"),
):
    """Evaluate energy/forces/stress MAE on a labeled dataset."""
    from mixlip.calculators import load as load_calc
    from mixlip.core.config import ModelConfig
    from mixlip.data.dataset import MLIPDataset
    from mixlip.training.metrics import compute_all_metrics
    from rich.progress import track

    cfg = ModelConfig(backend=backend, checkpoint=checkpoint, device=device)  # type: ignore[arg-type]
    calc = load_calc(backend, cfg)

    ds = MLIPDataset.from_file(dataset, as_graph=False)
    indices = range(min(n_samples, len(ds)) if n_samples > 0 else len(ds))

    results = []
    for i in track(indices, description=f"Evaluating {backend}..."):
        sample = ds[i]
        pred = calc.predict(sample.structure)
        results.append({"pred": pred, "true": sample})

    metrics = compute_all_metrics(results)

    table = Table(title=f"{backend} — {dataset.name}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    for key, val in metrics.items():
        table.add_row(key, f"{val:.4f}")
    console.print(table)
