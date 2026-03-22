"""Molecular dynamics setup helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ase
    from mixlip.calculators.base import MixLIPCalculator


def run_nvt(
    atoms: ase.Atoms,
    calculator: MixLIPCalculator,
    temperature_K: float = 300.0,
    timestep_fs: float = 1.0,
    n_steps: int = 1000,
    friction: float = 0.01,
    trajectory_path: str | None = None,
    logfile: str | None = None,
    log_interval: int = 10,
) -> ase.Atoms:
    """Run NVT (Langevin) MD and return the final Atoms object."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = atoms.copy()
    atoms.calc = calculator.ase_calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        friction=friction,
        trajectory=trajectory_path,
        logfile=logfile,
        loginterval=log_interval,
    )
    dyn.run(n_steps)
    return atoms


def run_npt(
    atoms: ase.Atoms,
    calculator: MixLIPCalculator,
    temperature_K: float = 300.0,
    pressure_gpa: float = 0.0,
    timestep_fs: float = 1.0,
    n_steps: int = 1000,
    trajectory_path: str | None = None,
    logfile: str | None = None,
    log_interval: int = 10,
) -> ase.Atoms:
    """Run NPT (Nose-Hoover / Berendsen) MD and return the final Atoms object."""
    from ase import units
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    atoms = atoms.copy()
    atoms.calc = calculator.ase_calculator
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)

    dyn = NPTBerendsen(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_K,
        pressure_au=pressure_gpa * units.Pascal * 1e9,
        trajectory=trajectory_path,
        logfile=logfile,
        loginterval=log_interval,
    )
    dyn.run(n_steps)
    return atoms
