"""MixLIP: unified ML interatomic potential toolkit."""

__version__ = "0.1.0"

from mixlip.calculators import load as load_calculator
from mixlip.core import (
    ModelConfig,
    PredictionResult,
    get_calculator,
    list_backends,
)
from mixlip.data import AtomicSample, MLIPDataModule, MLIPDataset
from mixlip.inference import EnsembleCalculator, relax_structure, run_nvt, run_npt
from mixlip.training import DistillationModule, MLIPLightningModule, WeightedEFSLoss

__all__ = [
    "__version__",
    # Quick access
    "load_calculator",
    # Core
    "ModelConfig",
    "PredictionResult",
    "get_calculator",
    "list_backends",
    # Data
    "AtomicSample",
    "MLIPDataset",
    "MLIPDataModule",
    # Training
    "WeightedEFSLoss",
    "MLIPLightningModule",
    "DistillationModule",
    # Inference
    "EnsembleCalculator",
    "relax_structure",
    "run_nvt",
    "run_npt",
]
