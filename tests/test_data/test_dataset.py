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
    from mixlip.data.writers.hdf5_writer import write_hdf5, load_hdf5

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
