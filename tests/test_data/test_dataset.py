"""Tests for MLIPDataset and writers."""

import tempfile
from pathlib import Path

import pytest

from mixlip.data.dataset import MLIPDataset
from mixlip.data.writers.extxyz_writer import write_extxyz


def test_dataset_len(small_dataset):
    ds = MLIPDataset(small_dataset, as_graph=False)
    assert len(ds) == 10


def test_dataset_getitem_no_graph(small_dataset, silicon_sample):
    ds = MLIPDataset(small_dataset, as_graph=False)
    item = ds[0]
    assert item.n_atoms == silicon_sample.n_atoms


def test_extxyz_roundtrip(small_dataset):
    with tempfile.NamedTemporaryFile(suffix=".extxyz", delete=False) as f:
        path = Path(f.name)
    write_extxyz(small_dataset, path)
    loaded = MLIPDataset.from_extxyz(path, as_graph=False)
    assert len(loaded) == len(small_dataset)
    item = loaded[0]
    assert hasattr(item, "structure")
    path.unlink()


def test_hdf5_roundtrip(small_dataset):
    pytest.importorskip("h5py")
    from mixlip.data.writers.hdf5_writer import load_hdf5, write_hdf5

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = Path(f.name)
    write_hdf5(small_dataset, path)
    loaded = load_hdf5(path)
    assert len(loaded) == len(small_dataset)
    assert loaded[0].energy == pytest.approx(small_dataset[0].energy, rel=1e-5)
    path.unlink()


@pytest.mark.parametrize("suffix", [".h5", ".hdf5"])
def test_dataset_from_hdf5_and_from_file(small_dataset, tmp_path, suffix):
    pytest.importorskip("h5py")
    from mixlip.data.writers.hdf5_writer import write_hdf5

    path = tmp_path / f"data{suffix}"
    write_hdf5(small_dataset, path)

    for ds in (
        MLIPDataset.from_hdf5(path, as_graph=False),
        MLIPDataset.from_file(path, as_graph=False),
    ):
        assert len(ds) == len(small_dataset)
        assert ds[0].energy == pytest.approx(small_dataset[0].energy, rel=1e-5)


def test_hdf5_metadata_roundtrip(silicon_structure, tmp_path):
    """metadata (e.g. teacher_energy/teacher_forces for offline distillation)
    must survive a write_hdf5 -> load_hdf5 round-trip."""
    pytest.importorskip("h5py")
    from mixlip.data.schema import AtomicSample
    from mixlip.data.writers.hdf5_writer import load_hdf5, write_hdf5

    samples = [
        AtomicSample(
            structure=silicon_structure,
            energy=-10.0,
            metadata={"teacher_energy": -10.5, "teacher_forces": [[0.0, 0.0, 0.0]] * 2},
        ),
        AtomicSample(structure=silicon_structure, energy=-11.0),  # no metadata
    ]

    path = tmp_path / "labeled.h5"
    write_hdf5(samples, path)
    loaded = load_hdf5(path)

    assert loaded[0].metadata["teacher_energy"] == pytest.approx(-10.5)
    assert loaded[0].metadata["teacher_forces"] == [[0.0, 0.0, 0.0]] * 2
    assert loaded[1].metadata == {}


def test_hdf5_load_without_metadata_dataset_defaults_to_empty(small_dataset, tmp_path):
    """Files written before metadata_json existed should still load fine."""
    pytest.importorskip("h5py")
    import h5py

    from mixlip.data.writers.hdf5_writer import load_hdf5, write_hdf5

    path = tmp_path / "old_format.h5"
    write_hdf5(small_dataset, path)
    with h5py.File(path, "a") as f:
        del f["metadata_json"]

    loaded = load_hdf5(path)
    assert all(s.metadata == {} for s in loaded)
