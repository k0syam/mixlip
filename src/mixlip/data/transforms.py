"""Data augmentation transforms for AtomicSample objects."""

from __future__ import annotations

import numpy as np

from mixlip.data.schema import AtomicSample


def rattle(sample: AtomicSample, stdev: float = 0.01, seed: int | None = None) -> AtomicSample:
    """Add random Gaussian displacement to atomic positions.

    Forces and energy are invalidated (set to None) because they no longer
    correspond to the rattled structure.
    """
    from pymatgen.core import Structure

    rng = np.random.default_rng(seed)
    displacement = rng.normal(0, stdev, size=(sample.n_atoms, 3))
    new_cart = np.array(sample.structure.cart_coords) + displacement
    new_struct = Structure(
        lattice=sample.structure.lattice,
        species=sample.structure.species,
        coords=new_cart,
        coords_are_cartesian=True,
    )
    return AtomicSample(
        structure=new_struct,
        energy=None,
        forces=None,
        stress=None,
        magmoms=sample.magmoms,
        weight=sample.weight,
        source=sample.source,
        metadata={**sample.metadata, "augmentation": "rattle", "rattle_stdev": stdev},
    )


def strain(
    sample: AtomicSample, eps: float = 0.01, seed: int | None = None
) -> AtomicSample:
    """Apply a random symmetric strain to the cell (Voigt components ε_ij)."""
    from pymatgen.core import Lattice, Structure

    rng = np.random.default_rng(seed)
    strain_voigt = rng.uniform(-eps, eps, size=6)
    # Convert Voigt to 3x3 strain tensor
    e = np.array(
        [
            [strain_voigt[0], strain_voigt[5] / 2, strain_voigt[4] / 2],
            [strain_voigt[5] / 2, strain_voigt[1], strain_voigt[3] / 2],
            [strain_voigt[4] / 2, strain_voigt[3] / 2, strain_voigt[2]],
        ]
    )
    deform = np.eye(3) + e
    new_matrix = sample.structure.lattice.matrix @ deform
    new_struct = Structure(
        lattice=Lattice(new_matrix),
        species=sample.structure.species,
        coords=sample.structure.frac_coords,
    )
    return AtomicSample(
        structure=new_struct,
        energy=None,
        forces=None,
        stress=None,
        magmoms=sample.magmoms,
        weight=sample.weight,
        source=sample.source,
        metadata={**sample.metadata, "augmentation": "strain", "strain_eps": eps},
    )


def make_supercell(sample: AtomicSample, scaling_matrix) -> AtomicSample:
    """Create a supercell from a sample."""
    new_struct = sample.structure.make_supercell(scaling_matrix, in_place=False)
    n_rep = int(round(len(new_struct) / sample.n_atoms))
    forces = np.tile(sample.forces, (n_rep, 1)) if sample.forces is not None else None
    magmoms = np.tile(sample.magmoms, n_rep) if sample.magmoms is not None else None
    energy = sample.energy * n_rep if sample.energy is not None else None
    return AtomicSample(
        structure=new_struct,
        energy=energy,
        forces=forces,
        stress=sample.stress,  # intensive quantity, unchanged
        magmoms=magmoms,
        weight=sample.weight,
        source=sample.source,
        metadata={**sample.metadata, "supercell": True},
    )
