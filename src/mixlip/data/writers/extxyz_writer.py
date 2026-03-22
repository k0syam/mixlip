"""Write AtomicSamples to extended XYZ format."""

from __future__ import annotations

from pathlib import Path

from mixlip.data.schema import AtomicSample


def write_extxyz(samples: list[AtomicSample], path: Path, append: bool = False) -> None:
    """Write AtomicSamples to an extxyz file.

    Parameters
    ----------
    samples:
        List of AtomicSample objects to write.
    path:
        Output file path.
    append:
        If True, append to an existing file instead of overwriting.
    """
    import ase.io
    from pymatgen.io.ase import AseAtomsAdaptor

    mode = "a" if append else "w"
    atoms_list = []
    for sample in samples:
        atoms = AseAtomsAdaptor.get_atoms(sample.structure)
        if sample.energy is not None:
            from ase.calculators.singlepoint import SinglePointCalculator

            calc_kwargs: dict = {"energy": sample.energy}
            if sample.forces is not None:
                calc_kwargs["forces"] = sample.forces
            if sample.stress is not None:
                calc_kwargs["stress"] = sample.stress
            atoms.calc = SinglePointCalculator(atoms, **calc_kwargs)
        atoms.info["source"] = sample.source
        atoms.info["weight"] = sample.weight
        atoms_list.append(atoms)

    ase.io.write(str(path), atoms_list, format="extxyz", append=append)
