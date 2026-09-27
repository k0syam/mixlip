"""Write AtomicSamples to HDF5 format for fast random access.

Schema
------
/n_atoms          (N,)       int32   number of atoms per sample
/energy           (N,)       float64 or masked
/weight           (N,)       float32
/source           (N,)       variable-length UTF-8 string
/atomic_numbers   (total,)   int32   concatenated
/cart_coords      (total,3)  float64 concatenated
/lattice          (N,3,3)    float64
/forces           (total,3)  float64 or masked
/stress           (N,6)      float64 or masked
/magmoms          (total,)   float64 or masked
/metadata_json    (N,)       variable-length UTF-8 string, JSON-encoded per sample

Each per-sample field is accessed by slicing with the cumulative n_atoms offset.
`/metadata_json` round-trips `AtomicSample.metadata` (e.g. the `teacher_energy`/
`teacher_forces`/`teacher_stress` fields written by `generate_teacher_labels`
for offline distillation). Files written before this field existed simply
lack the dataset; `load_hdf5` falls back to `metadata={}` for those.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mixlip.data.schema import AtomicSample


def write_hdf5(samples: list[AtomicSample], path: Path) -> None:
    """Write a list of AtomicSamples to an HDF5 file."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is required: pip install h5py") from e

    n = len(samples)
    n_atoms_arr = np.array([s.n_atoms for s in samples], dtype=np.int32)
    total = int(n_atoms_arr.sum())

    has_energy = any(s.energy is not None for s in samples)
    has_forces = any(s.forces is not None for s in samples)
    has_stress = any(s.stress is not None for s in samples)
    has_magmoms = any(s.magmoms is not None for s in samples)

    with h5py.File(str(path), "w") as f:
        f.create_dataset("n_atoms", data=n_atoms_arr)
        f.create_dataset("weight", data=np.array([s.weight for s in samples], dtype=np.float32))
        f.create_dataset(
            "source",
            data=np.array([s.source for s in samples], dtype=h5py.string_dtype()),
        )

        atomic_numbers = np.zeros(total, dtype=np.int32)
        cart_coords = np.zeros((total, 3), dtype=np.float64)
        lattice = np.zeros((n, 3, 3), dtype=np.float64)
        energy = np.full(n, np.nan, dtype=np.float64) if has_energy else None
        forces = np.full((total, 3), np.nan, dtype=np.float64) if has_forces else None
        stress = np.full((n, 6), np.nan, dtype=np.float64) if has_stress else None
        magmoms = np.full(total, np.nan, dtype=np.float64) if has_magmoms else None

        offset = 0
        for i, sample in enumerate(samples):
            na = sample.n_atoms
            atomic_numbers[offset : offset + na] = [
                sp.Z for sp in sample.structure.species
            ]
            cart_coords[offset : offset + na] = sample.structure.cart_coords
            lattice[i] = sample.structure.lattice.matrix
            if has_energy and sample.energy is not None:
                energy[i] = sample.energy
            if has_forces and sample.forces is not None:
                forces[offset : offset + na] = sample.forces
            if has_stress and sample.stress is not None:
                stress[i] = sample.stress
            if has_magmoms and sample.magmoms is not None:
                magmoms[offset : offset + na] = sample.magmoms
            offset += na

        f.create_dataset("atomic_numbers", data=atomic_numbers)
        f.create_dataset("cart_coords", data=cart_coords)
        f.create_dataset("lattice", data=lattice)
        if has_energy:
            f.create_dataset("energy", data=energy)
        if has_forces:
            f.create_dataset("forces", data=forces)
        if has_stress:
            f.create_dataset("stress", data=stress)
        if has_magmoms:
            f.create_dataset("magmoms", data=magmoms)

        metadata_json = np.array(
            [json.dumps(s.metadata, default=str) for s in samples],
            dtype=h5py.string_dtype(),
        )
        f.create_dataset("metadata_json", data=metadata_json)


def load_hdf5(path: Path) -> list[AtomicSample]:
    """Load AtomicSamples from an HDF5 file written by write_hdf5."""
    try:
        import h5py
    except ImportError as e:
        raise ImportError("h5py is required: pip install h5py") from e

    from pymatgen.core import Element, Lattice, Structure

    samples = []
    with h5py.File(str(path), "r") as f:
        n_atoms_arr = f["n_atoms"][:]
        weights = f["weight"][:]
        sources = [s.decode() if isinstance(s, bytes) else s for s in f["source"][:]]
        atomic_numbers = f["atomic_numbers"][:]
        cart_coords = f["cart_coords"][:]
        lattice = f["lattice"][:]

        energy = f["energy"][:] if "energy" in f else None
        forces = f["forces"][:] if "forces" in f else None
        stress = f["stress"][:] if "stress" in f else None
        magmoms = f["magmoms"][:] if "magmoms" in f else None
        if "metadata_json" in f:
            metadata_list = [
                json.loads(m.decode() if isinstance(m, bytes) else m)
                for m in f["metadata_json"][:]
            ]
        else:
            # Files written before metadata_json existed.
            metadata_list = [{} for _ in range(len(n_atoms_arr))]

        offset = 0
        for i, na in enumerate(n_atoms_arr):
            species = [Element.from_Z(int(z)) for z in atomic_numbers[offset : offset + na]]
            struct = Structure(
                lattice=Lattice(lattice[i]),
                species=species,
                coords=cart_coords[offset : offset + na],
                coords_are_cartesian=True,
            )
            e = energy[i] if energy is not None and not np.isnan(energy[i]) else None
            f_arr = forces[offset : offset + na] if forces is not None else None
            if f_arr is not None and np.all(np.isnan(f_arr)):
                f_arr = None
            s_arr = stress[i] if stress is not None else None
            if s_arr is not None and np.all(np.isnan(s_arr)):
                s_arr = None
            m_arr = magmoms[offset : offset + na] if magmoms is not None else None
            if m_arr is not None and np.all(np.isnan(m_arr)):
                m_arr = None

            samples.append(
                AtomicSample(
                    structure=struct,
                    energy=e,
                    forces=f_arr,
                    stress=s_arr,
                    magmoms=m_arr,
                    weight=float(weights[i]),
                    source=sources[i],
                    metadata=metadata_list[i],
                )
            )
            offset += na
    return samples
