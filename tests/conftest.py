"""Shared fixtures for mixlip tests."""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from mixlip.data.schema import AtomicSample


@pytest.fixture
def silicon_structure() -> Structure:
    """Diamond cubic silicon unit cell."""
    a = 5.431
    return Structure(
        lattice=Lattice.cubic(a),
        species=["Si", "Si"],
        coords=[[0, 0, 0], [0.25, 0.25, 0.25]],
    )


@pytest.fixture
def silicon_sample(silicon_structure) -> AtomicSample:
    """AtomicSample for silicon with fake DFT labels."""
    n = len(silicon_structure)
    return AtomicSample(
        structure=silicon_structure,
        energy=-10.845 * n,
        forces=np.random.default_rng(0).normal(0, 0.1, (n, 3)),
        stress=np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0]),
        source="test",
    )


@pytest.fixture
def small_dataset(silicon_sample) -> list[AtomicSample]:
    """A small list of AtomicSamples for dataset tests."""
    rng = np.random.default_rng(42)
    samples = []
    for i in range(10):
        s = AtomicSample(
            structure=silicon_sample.structure,
            energy=silicon_sample.energy + rng.normal(0, 0.01),
            forces=silicon_sample.forces + rng.normal(0, 0.01, silicon_sample.forces.shape),
            stress=silicon_sample.stress + rng.normal(0, 0.001, 6),
            source="test",
        )
        samples.append(s)
    return samples
