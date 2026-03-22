"""mixlip CLI entry point."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="mixlip",
    help="MixLIP: unified ML interatomic potential toolkit",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


# Import sub-command modules (they register themselves via app.command())
from mixlip.cli import train as _train  # noqa: E402, F401
from mixlip.cli import distill as _distill  # noqa: E402, F401
from mixlip.cli import generate as _generate  # noqa: E402, F401
from mixlip.cli import eval as _eval  # noqa: E402, F401

app.add_typer(_train.app, name="train")
app.add_typer(_distill.app, name="distill")
app.add_typer(_generate.app, name="generate")
app.add_typer(_eval.app, name="eval")


@app.command()
def list_backends():
    """List all available calculator backends."""
    from mixlip.core.registry import list_backends as _list

    # Trigger registration by importing all adapter modules
    import importlib

    for mod in [
        "mixlip.calculators.mace",
        "mixlip.calculators.chgnet",
        "mixlip.calculators.sevennet",
        "mixlip.calculators.equiformerv2",
        "mixlip.calculators.m3gnet",
        "mixlip.calculators.alignn",
        "mixlip.calculators.orb",
    ]:
        try:
            importlib.import_module(mod)
        except ImportError:
            pass

    backends = _list()
    console.print("[bold]Available backends:[/bold]")
    for b in backends:
        console.print(f"  [green]•[/green] {b}")


if __name__ == "__main__":
    app()
