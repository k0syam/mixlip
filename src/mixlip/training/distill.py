"""Knowledge distillation training module.

Architecture-agnostic distillation: student learns from both DFT hard
labels and teacher soft labels (energy/forces/stress predictions).

Teacher need not share architecture with student — this enables e.g.
MACE-MP (teacher) → M3GNet (student) cross-architecture distillation.

Two distillation modes are supported:

1. Online distillation
   Teacher runs inference on each batch during training. Requires teacher
   to be loaded in memory alongside student. Slower but no pre-computation.

2. Pre-computed (offline) distillation
   Teacher labels are stored in the dataset (e.g. as `teacher_energy`,
   `teacher_forces`, `teacher_stress` batch attributes). Teacher model
   is not instantiated. Fast and memory-efficient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from mixlip.training.loss import WeightedEFSLoss
from mixlip.training.module import MLIPLightningModule

if TYPE_CHECKING:
    from mixlip.calculators.base import MixLIPCalculator
    from mixlip.core.config import TrainingConfig


class DistillationModule(MLIPLightningModule):
    """Knowledge distillation: student ← teacher + DFT.

    Loss = alpha_task * L(student, DFT) + alpha_distill * L(student, teacher)

    Parameters
    ----------
    student_model:
        A torch.nn.Module that accepts a torch_geometric.Data batch and
        returns {energy, forces, stress} dict.
    teacher:
        Optional MixLIPCalculator instance (online distillation).
        If None, teacher labels must be pre-computed in the dataset.
    config:
        TrainingConfig with a populated .distill field.
    """

    def __init__(
        self,
        student_model: torch.nn.Module,
        config: TrainingConfig,
        teacher: MixLIPCalculator | None = None,
    ):
        super().__init__(student_model, config)
        self.teacher = teacher
        if teacher is not None:
            # Freeze teacher parameters (if it exposes any)
            if hasattr(teacher, "parameters"):
                for p in teacher.parameters():
                    p.requires_grad_(False)

        dcfg = config.distill
        assert dcfg is not None, "TrainingConfig.distill must be set for DistillationModule"
        self.alpha_task = dcfg.alpha_task
        self.alpha_distill = dcfg.alpha_distill
        self.distill_loss_fn = WeightedEFSLoss(dcfg.loss)

    def _get_teacher_pred(self, batch) -> dict[str, torch.Tensor]:
        """Return teacher predictions as a dict of tensors.

        If pre-computed labels are present on the batch (teacher_energy, etc.),
        use those. Otherwise run live teacher inference.
        """
        # -- pre-computed path (fastest) --
        if hasattr(batch, "teacher_energy"):
            pred: dict[str, torch.Tensor] = {"energy": batch.teacher_energy}
            if hasattr(batch, "teacher_forces"):
                pred["forces"] = batch.teacher_forces
            if hasattr(batch, "teacher_stress"):
                pred["stress"] = batch.teacher_stress
            return pred

        # -- live inference path --
        if self.teacher is None:
            raise RuntimeError(
                "Teacher model not provided and no pre-computed teacher labels found in batch. "
                "Either pass teacher= to DistillationModule or pre-compute labels with "
                "`mixlip distill generate-labels`."
            )

        device = next(self.model.parameters()).device
        teacher_pred: dict[str, torch.Tensor] = {}

        # The teacher is a MixLIPCalculator (ASE-based). We iterate per structure
        # because ASE calculators don't natively batch torch_geometric graphs.
        energies, forces_list, stresses = [], [], []
        from pymatgen.core import Lattice, Structure, Element

        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), batch.num_atoms.cumsum(0)]
        )
        for i in range(batch.num_graphs):
            s = int(offsets[i])
            e = int(offsets[i + 1])
            z = batch.atomic_numbers[s:e].cpu().numpy()
            pos = batch.pos[s:e].cpu().numpy()
            cell = batch.cell[i].cpu().numpy() if batch.cell.dim() == 3 else batch.cell.cpu().numpy()
            species = [Element.from_Z(int(zi)) for zi in z]
            struct = Structure(Lattice(cell), species, pos, coords_are_cartesian=True)
            with torch.no_grad():
                result = self.teacher.predict(struct)
            energies.append(result.energy)
            forces_list.append(result.forces)
            if result.stress is not None:
                stresses.append(result.stress)

        teacher_pred["energy"] = torch.tensor(
            energies, dtype=torch.float32, device=device
        )
        teacher_pred["forces"] = torch.tensor(
            __import__("numpy").concatenate(forces_list, axis=0),
            dtype=torch.float32,
            device=device,
        )
        if stresses:
            teacher_pred["stress"] = torch.tensor(
                __import__("numpy").stack(stresses, axis=0),
                dtype=torch.float32,
                device=device,
            )
        return teacher_pred

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        student_pred = self(batch)

        # Task loss (vs DFT hard labels)
        task_losses = self.loss_fn(student_pred, batch)

        # Distillation loss (vs teacher soft labels)
        with torch.no_grad():
            teacher_pred = self._get_teacher_pred(batch)
        # Treat teacher predictions as "true" labels for distill loss
        class _TeacherBatch:
            pass

        tb = _TeacherBatch()
        tb.energy = teacher_pred.get("energy")
        tb.forces = teacher_pred.get("forces")
        tb.stress = teacher_pred.get("stress")
        tb.num_atoms = batch.num_atoms
        tb.weight = getattr(batch, "weight", torch.ones(batch.num_graphs, device=batch.num_atoms.device))

        distill_losses = self.distill_loss_fn(student_pred, tb)

        total = self.alpha_task * task_losses["total"] + self.alpha_distill * distill_losses["total"]

        self.log("train/task_loss", task_losses["total"], batch_size=batch.num_graphs)
        self.log("train/distill_loss", distill_losses["total"], batch_size=batch.num_graphs)
        self.log("train/total", total, batch_size=batch.num_graphs)
        return total


def generate_teacher_labels(
    samples,
    teacher: MixLIPCalculator,
    output_path,
    batch_size: int = 32,
) -> None:
    """Pre-compute teacher labels for a list of AtomicSamples and save to HDF5.

    This is useful for offline distillation: run this once to cache teacher
    predictions, then train student with the cached labels (no teacher in memory).

    Parameters
    ----------
    samples:
        List of AtomicSample objects (structures only needed; labels optional).
    teacher:
        Instantiated MixLIPCalculator to use as teacher.
    output_path:
        Path to output HDF5 file.
    batch_size:
        Number of structures to process at once.
    """
    import numpy as np
    from pathlib import Path
    from mixlip.data.schema import AtomicSample
    from mixlip.data.writers.hdf5_writer import write_hdf5
    from rich.progress import track

    labeled: list[AtomicSample] = []
    for sample in track(samples, description="Generating teacher labels..."):
        result = teacher.predict(sample.structure)
        labeled.append(
            AtomicSample(
                structure=sample.structure,
                energy=sample.energy,       # keep original DFT energy if present
                forces=sample.forces,
                stress=sample.stress,
                magmoms=sample.magmoms,
                weight=sample.weight,
                source=sample.source,
                metadata={
                    **sample.metadata,
                    "teacher_energy": result.energy,
                    "teacher_forces": result.forces.tolist(),
                    "teacher_stress": result.stress.tolist() if result.stress is not None else None,
                    "teacher_model": teacher.model_name,
                },
            )
        )
    write_hdf5(labeled, Path(output_path))
