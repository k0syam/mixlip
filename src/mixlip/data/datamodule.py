"""PyTorch Lightning DataModule for MLIP training."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import random_split

from mixlip.core.config import DataConfig
from mixlip.data.dataset import MLIPDataset


class MLIPDataModule(L.LightningDataModule):
    """LightningDataModule that wraps MLIPDataset for train/val/test splits.

    Parameters
    ----------
    config:
        DataConfig with paths, format, cutoff, batch_size, etc.
    """

    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self._train: MLIPDataset | None = None
        self._val: MLIPDataset | None = None
        self._test: MLIPDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        cfg = self.config

        if cfg.val_path is not None:
            self._train = MLIPDataset.from_file(
                cfg.train_path, cutoff=cfg.cutoff_radius, max_neighbors=cfg.max_neighbors
            )
            self._val = MLIPDataset.from_file(
                cfg.val_path, cutoff=cfg.cutoff_radius, max_neighbors=cfg.max_neighbors
            )
        else:
            full = MLIPDataset.from_file(
                cfg.train_path, cutoff=cfg.cutoff_radius, max_neighbors=cfg.max_neighbors
            )
            n_val = max(1, int(len(full) * cfg.val_fraction))
            n_train = len(full) - n_val
            self._train, self._val = random_split(full, [n_train, n_val])

        if cfg.test_path is not None:
            self._test = MLIPDataset.from_file(
                cfg.test_path, cutoff=cfg.cutoff_radius, max_neighbors=cfg.max_neighbors
            )

    def train_dataloader(self):
        from torch_geometric.loader import DataLoader

        return DataLoader(
            self._train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        from torch_geometric.loader import DataLoader

        return DataLoader(
            self._val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

    def test_dataloader(self):
        if self._test is None:
            raise RuntimeError("No test_path was configured.")
        from torch_geometric.loader import DataLoader

        return DataLoader(
            self._test,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
