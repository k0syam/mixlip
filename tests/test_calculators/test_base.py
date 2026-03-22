"""Tests for PredictionResult and protocol."""

import numpy as np
import pytest

from mixlip.core.protocol import PredictionResult, _full_3x3_to_voigt_6


def test_prediction_result_basic():
    forces = np.zeros((2, 3))
    r = PredictionResult(energy=-10.5, forces=forces)
    assert r.energy == pytest.approx(-10.5)
    assert r.forces.shape == (2, 3)
    assert r.stress is None


def test_prediction_result_3x3_to_voigt():
    """3x3 stress tensor should be auto-converted to Voigt 6-component."""
    stress_3x3 = np.diag([1.0, 2.0, 3.0])
    r = PredictionResult(energy=0.0, forces=np.zeros((1, 3)), stress=stress_3x3)
    assert r.stress.shape == (6,)
    assert r.stress[0] == pytest.approx(1.0)
    assert r.stress[1] == pytest.approx(2.0)
    assert r.stress[2] == pytest.approx(3.0)
    assert r.stress[3] == pytest.approx(0.0)


def test_voigt_conversion():
    tensor = np.array([[1, 6, 5], [6, 2, 4], [5, 4, 3]], dtype=float)
    voigt = _full_3x3_to_voigt_6(tensor)
    assert voigt.tolist() == pytest.approx([1, 2, 3, 4, 5, 6])


def test_registry():
    from mixlip.core.registry import _CALCULATOR_REGISTRY, list_backends

    # Registry starts empty before any adapter is imported
    # (in a fresh interpreter). We just check the function works.
    backends = list_backends()
    assert isinstance(backends, list)
