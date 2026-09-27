"""MLIPDataset — PyTorch Dataset wrapping a list of AtomicSamples."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from mixlip.data.schema import AtomicSample


class MLIPDataset(Dataset):
    """Dataset of AtomicSample objects, optionally converted to graphs.

    Parameters
    ----------
    samples:
        List of AtomicSample objects.
    cutoff:
        Neighbor cutoff radius in Å (used when as_graph=True).
    max_neighbors:
        Max neighbors per atom.
    as_graph:
        If True, convert samples to torch_geometric.Data on access.
    transform:
        Optional callable applied to each sample before graph conversion.
    """

    def __init__(
        self,
        samples: list[AtomicSample],
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        as_graph: bool = True,
        transform: Callable | None = None,
    ):
        self.samples = samples
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.as_graph = as_graph
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.as_graph:
            from mixlip.data.graph import structure_to_graph

            return structure_to_graph(sample, self.cutoff, self.max_neighbors)
        return sample

    @classmethod
    def from_extxyz(cls, path: Path | str, **kwargs) -> MLIPDataset:
        from mixlip.data.loaders.ase_loader import load_extxyz

        return cls(load_extxyz(Path(path)), **kwargs)

    @classmethod
    def from_ase_db(cls, path: Path | str, **kwargs) -> MLIPDataset:
        from mixlip.data.loaders.ase_loader import load_ase_db

        return cls(load_ase_db(Path(path)), **kwargs)

    @classmethod
    def from_hdf5(cls, path: Path | str, **kwargs) -> MLIPDataset:
        from mixlip.data.writers.hdf5_writer import load_hdf5

        return cls(load_hdf5(Path(path)), **kwargs)

    @classmethod
    def from_file(cls, path: Path | str, fmt: str | None = None, **kwargs) -> MLIPDataset:
        """Auto-detect format from extension if fmt is None."""
        path = Path(path)
        fmt = fmt or path.suffix.lstrip(".")
        dispatch = {
            "xyz": cls.from_extxyz,
            "extxyz": cls.from_extxyz,
            "db": cls.from_ase_db,
            "h5": cls.from_hdf5,
            "hdf5": cls.from_hdf5,
        }
        if fmt not in dispatch:
            raise ValueError(f"Unsupported format '{fmt}'. Use one of: {list(dispatch)}")
        return dispatch[fmt](path, **kwargs)
