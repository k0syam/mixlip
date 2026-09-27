"""Abstract base class for all MixLIP calculator adapters."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from ase.calculators.calculator import Calculator, all_changes

from mixlip.core.protocol import PredictionResult

if TYPE_CHECKING:
    import ase
    import torch
    from pymatgen.core import Structure

    from mixlip.core.config import ModelConfig
    from mixlip.data.schema import AtomicSample


#: Backends whose `training_forward`/`trainable_module` are implemented.
#: Kept as a plain constant (rather than only the per-class `supports_training`
#: flag) so error messages can list every supported backend in one place.
_TRAINING_SUPPORTED_BACKENDS = ("chgnet",)


class MixLIPCalculator(Calculator):
    """ASE Calculator base that also satisfies the MLIPCalculatorProtocol.

    Subclasses implement:
    - _load_backend(config) → the upstream calculator/model object
    - _run_backend(atoms) → PredictionResult

    Both calculate() [ASE contract] and predict() [Protocol contract]
    delegate to _run_backend internally.

    Subclasses that also support fine-tuning/distillation (see
    `_TRAINING_SUPPORTED_BACKENDS`) additionally implement:
    - trainable_module → the upstream torch.nn.Module holding trainable weights
    - training_forward(samples) → differentiable {energy, forces, stress, magmoms}
    and set `supports_training = True`. A calculator only used as a distillation
    *teacher* never needs these — `.predict()` is enough for that role.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]
    model_name: str = "base"
    supported_properties: frozenset[str] = frozenset({"energy", "forces", "stress"})
    supports_training: bool = False

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self._backend = self._load_backend(config)

    @abstractmethod
    def _load_backend(self, config: ModelConfig):
        """Load and return the upstream model/calculator object."""

    @abstractmethod
    def _run_backend(self, atoms: ase.Atoms) -> PredictionResult:
        """Run inference with the upstream backend and return a PredictionResult."""

    # ------------------------------------------------------------------
    # ASE Calculator interface
    # ------------------------------------------------------------------

    def calculate(
        self,
        atoms: ase.Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        result = self._run_backend(self.atoms)
        self.results["energy"] = result.energy
        self.results["free_energy"] = result.energy
        self.results["forces"] = result.forces
        if result.stress is not None:
            self.results["stress"] = result.stress
        if result.magmoms is not None:
            self.results["magmoms"] = result.magmoms

    # ------------------------------------------------------------------
    # Protocol interface
    # ------------------------------------------------------------------

    def predict(self, structure: Structure) -> PredictionResult:
        """Run inference on a pymatgen Structure."""
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(structure)
        return self._run_backend(atoms)

    def predict_batch(self, structures: list[Structure]) -> list[PredictionResult]:
        """Run inference on a list of pymatgen Structures (sequential default)."""
        return [self.predict(s) for s in structures]

    @property
    def ase_calculator(self) -> Calculator:
        """Return self as an ASE Calculator (for Protocol compatibility)."""
        return self

    # ------------------------------------------------------------------
    # Training / distillation interface (optional — see supports_training)
    # ------------------------------------------------------------------

    @property
    def trainable_module(self) -> torch.nn.Module:
        """The upstream torch.nn.Module holding this backend's trainable weights.

        Used by `mixlip train`/`mixlip distill run` to register parameters with
        the optimizer (via `CalculatorTrainingWrapper` in `mixlip.training.module`).
        Only meaningful when `supports_training` is True.
        """
        raise NotImplementedError(
            f"Backend '{self.model_name}' does not support training/fine-tuning yet. "
            f"Currently supported: {', '.join(_TRAINING_SUPPORTED_BACKENDS)}."
        )

    def training_forward(self, samples: list[AtomicSample]) -> dict[str, torch.Tensor]:
        """Differentiable forward pass for training/distillation.

        Parameters
        ----------
        samples:
            A batch of AtomicSample (only `.structure` is read here; use
            `mixlip.data.schema.collate_labels` to get the corresponding
            target tensors for the loss function).

        Returns
        -------
        dict[str, Tensor] with keys:
            "energy":  (B,)              total eV
            "forces":  (sum(n_atoms), 3) eV/Å, concatenated in sample order
            "stress":  (B, 6)            eV/Å³, Voigt (xx yy zz yz xz xy) — optional
            "magmoms": (sum(n_atoms),)   μB, concatenated in sample order — optional
        Every returned tensor must be differentiable w.r.t. `self.trainable_module`'s
        parameters (i.e. computed without `torch.no_grad()`).
        """
        raise NotImplementedError(
            f"Training is not yet supported for backend '{self.model_name}'. "
            f"Currently supported: {', '.join(_TRAINING_SUPPORTED_BACKENDS)}. "
            f"Note this backend can still be used as a distillation *teacher* "
            f"(inference only, via .predict()) — only the student needs training support."
        )
