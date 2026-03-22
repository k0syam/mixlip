"""Pydantic v2 configuration models for mixlip."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    backend: Literal["mace", "chgnet", "sevennet", "equiformerv2", "m3gnet", "alignn", "orb"]
    checkpoint: str = "pretrained"  # "pretrained" | local path | HF model id
    model_size: str = "medium"      # backend-specific size token
    device: str = "cuda"
    dtype: str = "float32"
    compute_stress: bool = True
    extra: dict = Field(default_factory=dict)


class DataConfig(BaseModel):
    train_path: Path
    val_path: Path | None = None
    test_path: Path | None = None
    format: Literal["extxyz", "hdf5", "ase_db", "vasp"] = "extxyz"
    cutoff_radius: float = 6.0
    max_neighbors: int = 50
    val_fraction: float = 0.1
    batch_size: int = 32
    num_workers: int = 4


class LossConfig(BaseModel):
    energy_weight: float = 1.0
    forces_weight: float = 100.0
    stress_weight: float = 10.0
    magmom_weight: float = 1.0
    criterion: Literal["mse", "mae", "huber"] = "huber"
    huber_delta: float = 0.01


class DistillConfig(BaseModel):
    teacher: ModelConfig
    student: ModelConfig
    temperature: float = 1.0
    alpha_task: float = 0.5     # weight on hard labels (DFT data)
    alpha_distill: float = 0.5  # weight on teacher soft labels
    loss: LossConfig = Field(default_factory=LossConfig)


class TrainingConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 500
    patience: int = 50
    grad_clip_norm: float = 10.0
    use_ema: bool = True
    ema_decay: float = 0.99
    scheduler: Literal["cosine", "exponential", "plateau"] = "cosine"
    log_every_n_steps: int = 10
    wandb: bool = False
    distill: DistillConfig | None = None  # None = normal training


def load_config(path: Path) -> TrainingConfig:
    """Load and validate a TrainingConfig from a .yaml or .toml file."""
    import tomllib

    import yaml

    text = path.read_text()
    if path.suffix == ".toml":
        raw = tomllib.loads(text)
    else:
        raw = yaml.safe_load(text)
    return TrainingConfig.model_validate(raw)
