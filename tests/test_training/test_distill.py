"""Unit tests for DistillationModule._get_teacher_pred (no real models needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mixlip.data.schema import AtomicSample


class _FakeStudent(torch.nn.Module):
    """Minimal trainable stand-in for CalculatorTrainingWrapper in these tests."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, samples):
        n = sum(s.n_atoms for s in samples)
        return {
            "energy": self.w * torch.ones(len(samples)),
            "forces": torch.zeros(n, 3),
        }


def _make_config():
    from mixlip.core.config import DistillConfig, ModelConfig, TrainingConfig, DataConfig

    model_cfg = ModelConfig(backend="chgnet", device="cpu")
    return TrainingConfig(
        model=model_cfg,
        data=DataConfig(train_path="unused.extxyz"),
        distill=DistillConfig(teacher=model_cfg, student=model_cfg),
    )


def _labeled_sample(structure, teacher_energy, teacher_forces) -> AtomicSample:
    return AtomicSample(
        structure=structure,
        energy=-10.0,
        forces=np.zeros((len(structure), 3)),
        metadata={
            "teacher_energy": teacher_energy,
            "teacher_forces": teacher_forces,
        },
    )


def test_get_teacher_pred_uses_precomputed_metadata_without_teacher(silicon_structure):
    from mixlip.training.distill import DistillationModule

    n = len(silicon_structure)
    batch = [
        _labeled_sample(silicon_structure, -21.0, np.zeros((n, 3))),
        _labeled_sample(silicon_structure, -22.0, np.ones((n, 3))),
    ]

    module = DistillationModule(_FakeStudent(), _make_config(), teacher=None)
    pred = module._get_teacher_pred(batch)

    assert torch.allclose(pred["energy"], torch.tensor([-21.0, -22.0]))
    assert pred["forces"].shape == (2 * n, 3)
    assert "stress" not in pred  # no teacher_stress in metadata


def test_get_teacher_pred_raises_without_teacher_or_metadata(silicon_structure):
    from mixlip.training.distill import DistillationModule

    batch = [AtomicSample(structure=silicon_structure)]
    module = DistillationModule(_FakeStudent(), _make_config(), teacher=None)

    with pytest.raises(RuntimeError, match="Teacher model not provided"):
        module._get_teacher_pred(batch)


def test_init_with_ase_style_teacher_does_not_crash(silicon_structure):
    """A MixLIPCalculator teacher is an ASE Calculator, which has its own
    unrelated `.parameters` dict attribute (calculator kwargs) — not a
    callable `.parameters()` like torch.nn.Module. Construction must not
    mistake one for the other (regression test for a `hasattr` vs
    `callable(getattr(...))` bug)."""
    from mixlip.training.distill import DistillationModule

    class _AseStyleTeacher:
        parameters = {"some": "ase-calculator-kwarg"}  # not callable, like ASE's

        def predict(self, structure):
            from mixlip.core.protocol import PredictionResult

            return PredictionResult(energy=-1.0, forces=np.zeros((len(structure), 3)))

    # Must not raise TypeError: 'dict' object is not callable.
    DistillationModule(_FakeStudent(), _make_config(), teacher=_AseStyleTeacher())


def test_get_teacher_pred_falls_back_to_live_teacher(silicon_structure):
    from mixlip.training.distill import DistillationModule
    from mixlip.core.protocol import PredictionResult

    class _FakeTeacher:
        def predict(self, structure):
            n = len(structure)
            return PredictionResult(energy=-5.0, forces=np.zeros((n, 3)))

    batch = [AtomicSample(structure=silicon_structure)]
    module = DistillationModule(_FakeStudent(), _make_config(), teacher=_FakeTeacher())
    pred = module._get_teacher_pred(batch)

    assert pred["energy"].item() == pytest.approx(-5.0)
    assert pred["forces"].shape == (len(silicon_structure), 3)
