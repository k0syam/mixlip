"""Materials Project API loader.

Requires: pip install mp-api
Set MP_API_KEY env var or pass api_key directly.
"""

from __future__ import annotations

import os
from typing import Sequence

from mixlip.data.schema import AtomicSample


def load_from_mp(
    formula_or_ids: str | Sequence[str],
    api_key: str | None = None,
    fields: Sequence[str] = ("structure", "energy_per_atom", "nsites"),
    max_results: int = 1000,
) -> list[AtomicSample]:
    """Fetch structures from the Materials Project.

    Parameters
    ----------
    formula_or_ids:
        Chemical formula (e.g. "Fe2O3") or list of MP IDs (e.g. ["mp-1234"]).
    api_key:
        MP API key. Falls back to MP_API_KEY env var.
    fields:
        Fields to request.
    max_results:
        Maximum number of results to return.

    Returns
    -------
    List of AtomicSample objects (energy in eV/atom * n_sites).
    """
    try:
        from mp_api.client import MPRester
    except ImportError as e:
        raise ImportError("mp-api is required: pip install mp-api") from e

    key = api_key or os.environ.get("MP_API_KEY")
    if key is None:
        raise ValueError(
            "Materials Project API key required. Pass api_key= or set MP_API_KEY env var."
        )

    samples: list[AtomicSample] = []
    with MPRester(key) as mpr:
        if isinstance(formula_or_ids, str) and not formula_or_ids.startswith("mp-"):
            docs = mpr.materials.summary.search(
                formula=formula_or_ids,
                fields=list(fields),
                num_chunks=1,
                chunk_size=max_results,
            )
        else:
            ids = (
                [formula_or_ids]
                if isinstance(formula_or_ids, str)
                else list(formula_or_ids)
            )
            docs = mpr.materials.summary.search(
                material_ids=ids,
                fields=list(fields),
            )

        for doc in docs:
            struct = doc.structure
            energy = None
            if hasattr(doc, "energy_per_atom") and doc.energy_per_atom is not None:
                energy = doc.energy_per_atom * len(struct)
            samples.append(
                AtomicSample(
                    structure=struct,
                    energy=energy,
                    source="mp",
                    metadata={"mp_id": getattr(doc, "material_id", None)},
                )
            )
    return samples
