"""PyTorch Lightning DataModule for MLIP training."""

from __future__ import annotations

import lightning as L
from torch.utils.data import random_split

from mixlip.core.config import DataConfig
from mixlip.data.dataset import MLIPDataset


def _identity_collate(batch: list) -> list:
    """Pass a batch of AtomicSample through unchanged.

    A plain top-level function (not a lambda) so it stays picklable for
    multiprocessing DataLoader workers (num_workers > 0), including on
    Windows where the default start method requires pickling collate_fn.
    """
    return batch


class MLIPDataModule(L.LightningDataModule):
    """LightningDataModule that wraps MLIPDataset for train/val/test splits.

    Parameters
    ----------
    config:
        DataConfig with paths, format, batch_size, etc. `cutoff_radius` and
        `max_neighbors` only apply when `as_graph=True` (see below) — a
        backend's own `training_forward` builds its native graph with its
        own (usually pretraining-fixed) cutoffs instead.
    as_graph:
        If False (the default, and what `mixlip train`/`mixlip distill run`
        use), batches are plain `list[AtomicSample]`, meant to be consumed by
        a MixLIPCalculator's `training_forward` via CalculatorTrainingWrapper
        (see `mixlip.training.module`). If True, batches are
        torch_geometric.data.Batch objects built by `structure_to_graph`,
        for a custom from-scratch model operating on that generic schema.
    """

    def __init__(self, config: DataConfig, as_graph: bool = False):
        super().__init__()
        self.config = config
        self.as_graph = as_graph
        self._train: MLIPDataset | None = None
        self._val: MLIPDataset | None = None
        self._test: MLIPDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        cfg = self.config

        def _load(path):
            return MLIPDataset.from_file(
                path,
                cutoff=cfg.cutoff_radius,
                max_neighbors=cfg.max_neighbors,
                as_graph=self.as_graph,
            )

        if cfg.val_path is not None:
            self._train = _load(cfg.train_path)
            self._val = _load(cfg.val_path)
        else:
            full = _load(cfg.train_path)
            n_val = max(1, int(len(full) * cfg.val_fraction))
            n_train = len(full) - n_val
            self._train, self._val = random_split(full, [n_train, n_val])

        if cfg.test_path is not None:
            self._test = _load(cfg.test_path)

    def _dataloader_cls(self):
        if self.as_graph:
            from torch_geometric.loader import DataLoader

            return DataLoader, {}
        from torch.utils.data import DataLoader

        # AtomicSample batches are plain Python objects, not tensors — pass
        # them through unchanged rather than letting the default collate_fn
        # (which expects tensors) try and fail to stack them.
        return DataLoader, {"collate_fn": _identity_collate}

    def train_dataloader(self):
        loader_cls, extra_kwargs = self._dataloader_cls()
        return loader_cls(
            self._train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.as_graph,
            **extra_kwargs,
        )

    def val_dataloader(self):
        loader_cls, extra_kwargs = self._dataloader_cls()
        return loader_cls(
            self._val,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            **extra_kwargs,
        )

    def test_dataloader(self):
        if self._test is None:
            raise RuntimeError("No test_path was configured.")
        loader_cls, extra_kwargs = self._dataloader_cls()
        return loader_cls(
            self._test,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            **extra_kwargs,
        )
