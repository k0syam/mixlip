"""AtomicSample — the canonical labeled-structure container."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pymatgen.core import Structure


@dataclass
class AtomicSample:
    """Single labeled structure.

    All labels are optional to support partial-label datasets
    (e.g., energies only, no stress).

    Units:
    - energy: eV (total)
    - forces: eV/Å, shape (N, 3)
    - stress: eV/Å³, Voigt 6-component, shape (6,)
    - magmoms: μB, shape (N,)
    """

    structure: Structure
    energy: float | None = None
    forces: NDArray[np.float64] | None = None
    stress: NDArray[np.float64] | None = None
    magmoms: NDArray[np.float64] | None = None
    weight: float = 1.0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.forces is not None:
            self.forces = np.asarray(self.forces, dtype=np.float64)
        if self.stress is not None:
            self.stress = np.asarray(self.stress, dtype=np.float64)
        if self.magmoms is not None:
            self.magmoms = np.asarray(self.magmoms, dtype=np.float64)

    @classmethod
    def from_ase_atoms(cls, atoms, **label_kwargs) -> AtomicSample:
        """Construct from an ASE Atoms object.

        Energy/forces/stress can be taken from the attached calculator
        or passed explicitly via label_kwargs.
        """
        from pymatgen.io.ase import AseAtomsAdaptor

        structure = AseAtomsAdaptor.get_structure(atoms)
        # Extract labels from the calculator if not explicitly provided
        if "energy" not in label_kwargs:
            try:
                label_kwargs["energy"] = float(atoms.get_potential_energy())
            except Exception:
                pass
        if "forces" not in label_kwargs:
            try:
                label_kwargs["forces"] = atoms.get_forces()
            except Exception:
                pass
        if "stress" not in label_kwargs:
            try:
                # ASE stress is (6,) Voigt in eV/Å³ when atoms has pbc
                label_kwargs["stress"] = atoms.get_stress(voigt=True)
            except Exception:
                pass
        return cls(structure=structure, **label_kwargs)

    @property
    def n_atoms(self) -> int:
        return len(self.structure)


def collate_labels(samples: list[AtomicSample], device=None) -> SimpleNamespace:
    """Collate a batch of AtomicSample labels into an object WeightedEFSLoss can read.

    Returns a SimpleNamespace exposing `.energy`, `.forces`, `.stress`, `.magmoms`
    (each concatenated/stacked across the batch, set only if EVERY sample in the
    batch has that label — otherwise left unset, so `getattr(batch, name, None)`
    correctly yields None), plus `.num_atoms`, `.weight`, and `.num_graphs`.

    Shapes/units mirror what `mixlip.data.graph.structure_to_graph` puts on its
    torch_geometric.Data output, so the same WeightedEFSLoss works whether
    predictions come from a generic PyG-graph model or from a backend-native
    `MixLIPCalculator.training_forward` (see `mixlip.training.module`).

    Parameters
    ----------
    device:
        Optional torch device to place the label tensors on. Pass the same
        device as the model's predictions (e.g. `pred["energy"].device`) so
        WeightedEFSLoss doesn't hit a CPU/GPU mismatch — labels default to
        CPU otherwise (fine for CPU training or standalone use/tests).
    """
    import torch

    batch = SimpleNamespace()
    batch.num_graphs = len(samples)
    batch.num_atoms = torch.tensor([s.n_atoms for s in samples], dtype=torch.long, device=device)
    batch.weight = torch.tensor(
        [s.weight for s in samples], dtype=torch.float32, device=device
    )

    if all(s.energy is not None for s in samples):
        batch.energy = torch.tensor(
            [s.energy for s in samples], dtype=torch.float64, device=device
        )
    if all(s.forces is not None for s in samples):
        batch.forces = torch.tensor(
            np.concatenate([s.forces for s in samples], axis=0),
            dtype=torch.float32,
            device=device,
        )
    if all(s.stress is not None for s in samples):
        batch.stress = torch.tensor(
            np.stack([s.stress for s in samples], axis=0), dtype=torch.float32, device=device
        )
    if all(s.magmoms is not None for s in samples):
        batch.magmoms = torch.tensor(
            np.concatenate([s.magmoms for s in samples], axis=0),
            dtype=torch.float32,
            device=device,
        )
    return batch
