"""Decorator-based registry for MixLIPCalculator backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from mixlip.calculators.base import MixLIPCalculator
    from mixlip.core.config import ModelConfig

_CALCULATOR_REGISTRY: dict[str, Type[MixLIPCalculator]] = {}


def register_calculator(name: str):
    """Class decorator that registers a MixLIPCalculator subclass by name."""

    def decorator(cls: Type[MixLIPCalculator]) -> Type[MixLIPCalculator]:
        _CALCULATOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_calculator(name: str, config: ModelConfig) -> MixLIPCalculator:
    """Instantiate a registered calculator by backend name."""
    if name not in _CALCULATOR_REGISTRY:
        available = list(_CALCULATOR_REGISTRY)
        raise KeyError(f"Unknown backend '{name}'. Available: {available}")
    return _CALCULATOR_REGISTRY[name](config)


def list_backends() -> list[str]:
    """Return names of all registered calculator backends."""
    return list(_CALCULATOR_REGISTRY)
