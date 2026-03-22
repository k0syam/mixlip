"""Integration test for MACE adapter (requires mace-torch installed)."""

import pytest


@pytest.mark.integration
def test_mace_predict(silicon_structure):
    mace = pytest.importorskip("mace")
    from mixlip.calculators import load

    calc = load("mace", device="cpu", checkpoint="mace-mp", model_size="small")
    result = calc.predict(silicon_structure)

    assert result.energy < 0
    assert result.forces.shape == (2, 3)
    assert result.stress is not None and result.stress.shape == (6,)


@pytest.mark.integration
def test_mace_ase_calculator(silicon_structure):
    pytest.importorskip("mace")
    from pymatgen.io.ase import AseAtomsAdaptor
    from mixlip.calculators import load

    calc = load("mace", device="cpu", checkpoint="mace-mp", model_size="small")
    atoms = AseAtomsAdaptor.get_atoms(silicon_structure)
    atoms.calc = calc.ase_calculator
    energy = atoms.get_potential_energy()
    assert energy < 0
