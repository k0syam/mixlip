"""mixlip generate — dataset generation from various sources."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Generate or label datasets.")
console = Console()


@app.command("from-mp")
def from_mp(
    formula: Optional[str] = typer.Option(None, "--formula", "-f"),
    ids: Optional[str] = typer.Option(None, "--ids", help="Comma-separated MP IDs"),
    output: Path = typer.Option(Path("dataset.extxyz"), "--output", "-o"),
    max_results: int = typer.Option(1000, "--max"),
    api_key: Optional[str] = typer.Option(None, "--api-key"),
):
    """Fetch structures from the Materials Project."""
    from mixlip.data.loaders.mp_loader import load_from_mp
    from mixlip.data.writers.extxyz_writer import write_extxyz

    query = formula or (ids.split(",") if ids else None)
    if query is None:
        console.print("[red]Provide --formula or --ids[/red]")
        raise typer.Exit(1)

    console.print(f"Fetching from Materials Project: {query}")
    samples = load_from_mp(query, api_key=api_key, max_results=max_results)
    write_extxyz(samples, output)
    console.print(f"[green]Saved {len(samples)} structures to {output}[/green]")


@app.command("label")
def label(
    input_path: Path = typer.Option(..., "--input", "-i"),
    output_path: Path = typer.Option(..., "--output", "-o"),
    backend: str = typer.Option("mace", "--backend", "-b"),
    device: str = typer.Option("cuda", "--device"),
    checkpoint: str = typer.Option("pretrained", "--checkpoint"),
    fmt: str = typer.Option("extxyz", "--format"),
):
    """Label unlabeled structures with an MLIP (energy/forces/stress)."""
    from mixlip.calculators import load as load_calc
    from mixlip.core.config import ModelConfig
    from mixlip.data.dataset import MLIPDataset
    from mixlip.data.schema import AtomicSample
    from mixlip.data.writers.extxyz_writer import write_extxyz
    from rich.progress import track

    calc_cfg = ModelConfig(backend=backend, checkpoint=checkpoint, device=device)  # type: ignore[arg-type]
    calc = load_calc(backend, calc_cfg)

    dataset = MLIPDataset.from_file(input_path, as_graph=False)
    labeled: list[AtomicSample] = []

    for i in track(range(len(dataset)), description="Labeling..."):
        sample: AtomicSample = dataset[i]
        result = calc.predict(sample.structure)
        labeled.append(
            AtomicSample(
                structure=sample.structure,
                energy=result.energy,
                forces=result.forces,
                stress=result.stress,
                magmoms=result.magmoms,
                source=f"{backend}:{checkpoint}",
            )
        )

    write_extxyz(labeled, output_path)
    console.print(f"[green]Labeled {len(labeled)} structures → {output_path}[/green]")
