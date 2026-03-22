"""Tests for AtomicSample."""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from mixlip.data.schema import AtomicSample


def test_basic_construction(silicon_structure):
    sample = AtomicSample(structure=silicon_structure, energy=-21.69)
    assert sample.energy == pytest.approx(-21.69)
    assert sample.forces is None
    assert sample.n_atoms == 2


def test_force_array_coercion(silicon_structure):
    forces_list = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
    sample = AtomicSample(structure=silicon_structure, forces=forces_list)
    assert sample.forces.shape == (2, 3)
    assert sample.forces.dtype == np.float64


def test_stress_voigt_coercion(silicon_structure):
    stress_3x3 = np.diag([0.01, 0.02, 0.03])
    sample = AtomicSample(structure=silicon_structure, stress=stress_3x3)
    # AtomicSample does NOT auto-convert 3x3 to Voigt (that is done by PredictionResult)
    # But it does coerce to numpy array
    assert isinstance(sample.stress, np.ndarray)


def test_from_ase_atoms(silicon_structure):
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(silicon_structure)
    sample = AtomicSample.from_ase_atoms(atoms, energy=-21.69)
    assert sample.energy == pytest.approx(-21.69)
    assert sample.n_atoms == 2
