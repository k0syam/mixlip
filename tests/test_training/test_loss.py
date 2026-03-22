"""Tests for WeightedEFSLoss."""

import pytest
import torch

from mixlip.core.config import LossConfig
from mixlip.training.loss import WeightedEFSLoss


def _make_batch(n_atoms=4, n_graphs=2):
    """Create a minimal fake batch for loss testing."""

    class FakeBatch:
        pass

    b = FakeBatch()
    b.num_atoms = torch.tensor([n_atoms, n_atoms])
    b.num_graphs = n_graphs
    b.energy = torch.tensor([-21.0, -22.0], dtype=torch.float64)
    b.forces = torch.randn(n_atoms * n_graphs, 3)
    b.stress = torch.randn(n_graphs, 6)
    b.weight = torch.ones(n_graphs)
    return b


def test_loss_returns_total():
    cfg = LossConfig()
    loss_fn = WeightedEFSLoss(cfg)
    batch = _make_batch()
    pred = {
        "energy": torch.tensor([-21.5, -22.5], dtype=torch.float32),
        "forces": torch.randn(8, 3),
        "stress": torch.randn(2, 6),
    }
    losses = loss_fn(pred, batch)
    assert "total" in losses
    assert losses["total"].item() > 0


def test_loss_partial_labels():
    """Loss should work even with no stress in batch."""
    cfg = LossConfig()
    loss_fn = WeightedEFSLoss(cfg)

    class MinimalBatch:
        num_atoms = torch.tensor([2, 2])
        num_graphs = 2
        energy = torch.tensor([-4.0, -4.0], dtype=torch.float64)
        forces = torch.zeros(4, 3)
        stress = None
        weight = torch.ones(2)

    pred = {
        "energy": torch.tensor([-4.1, -3.9], dtype=torch.float32),
        "forces": torch.zeros(4, 3),
    }
    losses = loss_fn(pred, MinimalBatch())
    assert "stress" not in losses
    assert losses["total"].item() >= 0


@pytest.mark.parametrize("criterion", ["mse", "mae", "huber"])
def test_loss_criteria(criterion):
    cfg = LossConfig(criterion=criterion)
    loss_fn = WeightedEFSLoss(cfg)
    batch = _make_batch()
    pred = {
        "energy": torch.tensor([-21.0, -22.0], dtype=torch.float32),
        "forces": batch.forces.clone(),
        "stress": batch.stress.clone(),
    }
    losses = loss_fn(pred, batch)
    # Perfect prediction → near-zero loss
    assert losses["energy"].item() < 1e-4
    assert losses["forces"].item() < 1e-4
