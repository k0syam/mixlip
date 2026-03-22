"""Calculator adapters for various MLIP backends.

Backends are registered lazily — importing this module only imports the
registry dispatch layer. Each concrete adapter is imported on demand
to avoid hard dependencies on uninstalled backends.
"""

from __future__ import annotations

from mixlip.core.config import ModelConfig
from mixlip.core.registry import get_calculator, list_backends

_ADAPTER_MODULES = {
    "mace": "mixlip.calculators.mace",
    "chgnet": "mixlip.calculators.chgnet",
    "sevennet": "mixlip.calculators.sevennet",
    "equiformerv2": "mixlip.calculators.equiformerv2",
    "m3gnet": "mixlip.calculators.m3gnet",
    "alignn": "mixlip.calculators.alignn",
    "orb": "mixlip.calculators.orb",
}


def load(backend: str, config: ModelConfig | None = None, **kwargs):
    """Load a calculator by backend name.

    Parameters
    ----------
    backend:
        One of "mace", "chgnet", "sevennet", "equiformerv2", "m3gnet",
        "alignn", "orb".
    config:
        ModelConfig instance. If None, a default config is constructed
        from ``backend`` and any extra ``kwargs``.
    **kwargs:
        Passed to ModelConfig when config=None (e.g. device="cpu").
    """
    import importlib

    if backend not in _ADAPTER_MODULES:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {list(_ADAPTER_MODULES)}"
        )
    # Trigger the @register_calculator decorator by importing the module
    importlib.import_module(_ADAPTER_MODULES[backend])

    if config is None:
        config = ModelConfig(backend=backend, **kwargs)  # type: ignore[arg-type]
    return get_calculator(backend, config)


__all__ = ["load", "list_backends"]
