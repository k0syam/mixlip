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
   Teacher labels are stored per-sample in `AtomicSample.metadata` (as
   `teacher_energy`, `teacher_forces`, `teacher_stress` — see
   `generate_teacher_labels` below). Teacher model is not instantiated.
   Fast and memory-efficient.

The teacher only ever needs `.predict()` (the ASE/pymatgen inference path
every MixLIPCalculator implements), so any of the 7 backends can be a
teacher regardless of whether it supports training — only the *student*
needs `supports_training = True`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mixlip.data.schema import collate_labels
from mixlip.training.loss import WeightedEFSLoss
from mixlip.training.module import MLIPLightningModule

if TYPE_CHECKING:
    from mixlip.calculators.base import MixLIPCalculator
    from mixlip.core.config import TrainingConfig
    from mixlip.data.schema import AtomicSample


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
            # Freeze teacher parameters, if it happens to be a real nn.Module.
            # A MixLIPCalculator (the normal case — teacher is used purely via
            # .predict()) is an ASE Calculator, not an nn.Module: it has its own
            # unrelated `.parameters` dict attribute (ASE's calculator kwargs),
            # so `hasattr(teacher, "parameters")` would be True but not callable.
            maybe_parameters = getattr(teacher, "parameters", None)
            if callable(maybe_parameters):
                for p in maybe_parameters():
                    p.requires_grad_(False)

        dcfg = config.distill
        assert dcfg is not None, "TrainingConfig.distill must be set for DistillationModule"
        self.alpha_task = dcfg.alpha_task
        self.alpha_distill = dcfg.alpha_distill
        self.distill_loss_fn = WeightedEFSLoss(dcfg.loss)

    def _get_teacher_pred(self, batch: list[AtomicSample]) -> dict[str, torch.Tensor]:
        """Return teacher predictions as a dict of tensors.

        If every sample in the batch carries pre-computed labels (written by
        `generate_teacher_labels` into `sample.metadata`), use those directly —
        no teacher model needed. Otherwise run live teacher inference via
        `self.teacher.predict(sample.structure)`.
        """
        device = next(self.model.parameters()).device

        # -- pre-computed path (fastest, no teacher model required) --
        if all("teacher_energy" in s.metadata for s in batch):
            pred: dict[str, torch.Tensor] = {
                "energy": torch.tensor(
                    [s.metadata["teacher_energy"] for s in batch],
                    dtype=torch.float32,
                    device=device,
                )
            }
            if all("teacher_forces" in s.metadata for s in batch):
                pred["forces"] = torch.tensor(
                    np.concatenate(
                        [np.asarray(s.metadata["teacher_forces"]) for s in batch], axis=0
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            if all(s.metadata.get("teacher_stress") is not None for s in batch):
                pred["stress"] = torch.tensor(
                    np.stack(
                        [np.asarray(s.metadata["teacher_stress"]) for s in batch], axis=0
                    ),
                    dtype=torch.float32,
                    device=device,
                )
            return pred

        # -- live inference path --
        if self.teacher is None:
            raise RuntimeError(
                "Teacher model not provided and no pre-computed teacher labels found in "
                "sample.metadata. Either pass teacher= to DistillationModule or "
                "pre-compute labels with `mixlip distill generate-labels`."
            )

        energies, forces_list, stresses = [], [], []
        for sample in batch:
            result = self.teacher.predict(sample.structure)
            energies.append(result.energy)
            forces_list.append(result.forces)
            if result.stress is not None:
                stresses.append(result.stress)

        teacher_pred: dict[str, torch.Tensor] = {
            "energy": torch.tensor(energies, dtype=torch.float32, device=device),
            "forces": torch.tensor(
                np.concatenate(forces_list, axis=0), dtype=torch.float32, device=device
            ),
        }
        if len(stresses) == len(batch):
            teacher_pred["stress"] = torch.tensor(
                np.stack(stresses, axis=0), dtype=torch.float32, device=device
            )
        return teacher_pred

    def training_step(self, batch: list[AtomicSample], batch_idx: int) -> torch.Tensor:
        student_pred = self(batch)
        targets = collate_labels(batch, device=student_pred["energy"].device)

        # Task loss (vs DFT hard labels)
        task_losses = self.loss_fn(student_pred, targets)

        # Distillation loss (vs teacher soft labels)
        with torch.no_grad():
            teacher_pred = self._get_teacher_pred(batch)
        # Treat teacher predictions as "true" labels for distill loss
        from types import SimpleNamespace

        tb = SimpleNamespace(
            energy=teacher_pred.get("energy"),
            forces=teacher_pred.get("forces"),
            stress=teacher_pred.get("stress"),
            num_atoms=targets.num_atoms,
            weight=targets.weight,
        )

        distill_losses = self.distill_loss_fn(student_pred, tb)

        total = self.alpha_task * task_losses["total"] + self.alpha_distill * distill_losses["total"]

        self.log("train/task_loss", task_losses["total"], batch_size=len(batch))
        self.log("train/distill_loss", distill_losses["total"], batch_size=len(batch))
        self.log("train/total", total, batch_size=len(batch))
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
    from pathlib import Path

    from rich.progress import track

    from mixlip.data.schema import AtomicSample
    from mixlip.data.writers.hdf5_writer import write_hdf5

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
