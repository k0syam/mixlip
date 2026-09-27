"""Integration test for the CHGNet training/distillation adapter.

Requires chgnet (and its pretrained weights) to be available — marked
`integration` and skipped by default (see pyproject.toml / README).
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_chgnet_trainable_module_is_real_model(silicon_structure):
    chgnet = pytest.importorskip("chgnet")
    from mixlip.calculators import load

    calc = load("chgnet", device="cpu")
    assert calc.supports_training is True
    assert calc.trainable_module is calc._backend.model


@pytest.mark.integration
def test_chgnet_training_forward_shapes_and_grad(silicon_sample, small_dataset):
    pytest.importorskip("chgnet")
    import torch

    from mixlip.calculators import load

    calc = load("chgnet", device="cpu")
    samples = small_dataset[:2]

    pred = calc.training_forward(samples)

    n_total_atoms = sum(s.n_atoms for s in samples)
    assert pred["energy"].shape == (2,)
    assert pred["forces"].shape == (n_total_atoms, 3)
    assert pred["stress"].shape == (2, 6)
    assert pred["energy"].requires_grad
    assert pred["forces"].requires_grad


@pytest.mark.integration
def test_chgnet_training_forward_backward_populates_grad(small_dataset):
    pytest.importorskip("chgnet")
    import torch

    from mixlip.calculators import load
    from mixlip.data.schema import collate_labels
    from mixlip.training.loss import WeightedEFSLoss
    from mixlip.core.config import LossConfig

    calc = load("chgnet", device="cpu")
    samples = small_dataset[:2]

    for p in calc.trainable_module.parameters():
        assert p.grad is None

    pred = calc.training_forward(samples)
    targets = collate_labels(samples)
    loss_fn = WeightedEFSLoss(LossConfig())
    losses = loss_fn(pred, targets)
    losses["total"].backward()

    grads = [p.grad for p in calc.trainable_module.parameters() if p.requires_grad]
    assert len(grads) > 0
    assert any(g is not None and torch.any(g != 0) for g in grads)


@pytest.mark.integration
def test_chgnet_training_forward_energy_matches_predict_order(silicon_structure):
    """Energy from training_forward and from .predict() should have the same sign
    and be within a broad tolerance of each other (graphs differ slightly:
    ASE-calculator inference vs. the raw model.graph_converter used here)."""
    pytest.importorskip("chgnet")
    from mixlip.calculators import load
    from mixlip.data.schema import AtomicSample

    calc = load("chgnet", device="cpu")
    sample = AtomicSample(structure=silicon_structure)

    pred = calc.training_forward([sample])
    energy_train = pred["energy"].item()

    result = calc.predict(silicon_structure)
    assert energy_train == pytest.approx(result.energy, rel=1e-3)
