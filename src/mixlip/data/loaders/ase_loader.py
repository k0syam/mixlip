"""Loaders for ASE-compatible formats: extxyz, ASE DB, trajectories."""

from __future__ import annotations

from pathlib import Path

from mixlip.data.schema import AtomicSample


def load_extxyz(path: Path) -> list[AtomicSample]:
    """Load an extended XYZ file into a list of AtomicSamples."""
    import ase.io

    frames = ase.io.read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    return [AtomicSample.from_ase_atoms(atoms) for atoms in frames]


def load_ase_db(path: Path) -> list[AtomicSample]:
    """Load an ASE SQLite database into a list of AtomicSamples."""
    from ase.db import connect

    db = connect(str(path))
    samples = []
    for row in db.select():
        atoms = row.toatoms(attach_calculator=False)
        # ASE DB rows store energy/forces as arrays on the row
        kwargs: dict = {}
        if hasattr(row, "energy"):
            kwargs["energy"] = row.energy
        if hasattr(row, "forces"):
            kwargs["forces"] = row.forces
        if hasattr(row, "stress"):
            kwargs["stress"] = row.stress
        samples.append(AtomicSample.from_ase_atoms(atoms, **kwargs))
    return samples


def load_trajectory(path: Path) -> list[AtomicSample]:
    """Load an ASE trajectory file (e.g. .traj, VASP XDATCAR)."""
    import ase.io

    frames = ase.io.read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    return [AtomicSample.from_ase_atoms(atoms) for atoms in frames]
