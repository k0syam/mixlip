"""Convert AtomicSample → torch_geometric.Data graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mixlip.data.schema import AtomicSample


def structure_to_graph(
    sample: AtomicSample,
    cutoff: float = 6.0,
    max_neighbors: int = 50,
):
    """Convert an AtomicSample to a torch_geometric.Data object.

    Uses pymatgen's fast Cython neighbor search (same as matgl/CHGNet).
    Edge vectors are PBC-aware displacement vectors in Cartesian Å.

    Returns
    -------
    torch_geometric.data.Data with fields:
        atomic_numbers : (N,)
        pos            : (N, 3)  Cartesian coordinates
        edge_index     : (2, E)
        edge_vec       : (E, 3)  Cartesian displacement vectors (receiver - sender)
        edge_dist      : (E,)    distances in Å
        cell           : (1, 3, 3)
        energy         : scalar or None
        forces         : (N, 3) or None
        stress         : (6,) or None
        magmoms        : (N,) or None
        weight         : scalar
        num_atoms      : scalar
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as e:
        raise ImportError(
            "torch and torch_geometric are required for graph construction. "
            "Install PyTorch first, then: pip install mixlip[data]"
        ) from e

    from pymatgen.optimization.neighbors import find_points_in_spheres

    struct = sample.structure
    cart_coords = np.array(struct.cart_coords, dtype=np.float64)
    lattice = np.array(struct.lattice.matrix, dtype=np.float64)

    center_idx, neigh_idx, offset_vecs, distances = find_points_in_spheres(
        cart_coords,
        cart_coords,
        r=cutoff,
        pbc=np.array([1, 1, 1], dtype=np.int64),
        lattice=lattice,
    )

    # Remove self-loops
    mask = distances > 1e-8
    center_idx = center_idx[mask]
    neigh_idx = neigh_idx[mask]
    distances = distances[mask]

    # Displacement vectors: neighbour position - center position (with PBC offset)
    edge_vec = (
        cart_coords[neigh_idx]
        + offset_vecs[mask] @ lattice
        - cart_coords[center_idx]
    )

    # Optionally trim to max_neighbors (keep shortest bonds per center atom)
    if max_neighbors is not None:
        center_idx, neigh_idx, edge_vec, distances = _trim_neighbors(
            center_idx, neigh_idx, edge_vec, distances, max_neighbors
        )

    atomic_numbers = torch.tensor(
        [s.Z for s in struct.species], dtype=torch.long
    )

    data = Data(
        atomic_numbers=atomic_numbers,
        pos=torch.tensor(cart_coords, dtype=torch.float32),
        edge_index=torch.tensor(
            np.stack([center_idx, neigh_idx], axis=0), dtype=torch.long
        ),
        edge_vec=torch.tensor(edge_vec, dtype=torch.float32),
        edge_dist=torch.tensor(distances, dtype=torch.float32),
        cell=torch.tensor(lattice, dtype=torch.float32).unsqueeze(0),
        weight=torch.tensor(sample.weight, dtype=torch.float32),
        num_atoms=torch.tensor(len(struct), dtype=torch.long),
    )

    if sample.energy is not None:
        data.energy = torch.tensor(sample.energy, dtype=torch.float64)
    if sample.forces is not None:
        data.forces = torch.tensor(sample.forces, dtype=torch.float32)
    if sample.stress is not None:
        data.stress = torch.tensor(sample.stress, dtype=torch.float32)
    if sample.magmoms is not None:
        data.magmoms = torch.tensor(sample.magmoms, dtype=torch.float32)

    return data


def _trim_neighbors(center_idx, neigh_idx, edge_vec, distances, max_neighbors):
    """Keep only the max_neighbors shortest bonds per center atom."""
    import numpy as np

    n_atoms = int(center_idx.max()) + 1 if len(center_idx) > 0 else 0
    keep = []
    for i in range(n_atoms):
        mask_i = center_idx == i
        idx_i = np.where(mask_i)[0]
        if len(idx_i) > max_neighbors:
            order = np.argsort(distances[idx_i])[:max_neighbors]
            idx_i = idx_i[order]
        keep.append(idx_i)
    keep = np.concatenate(keep) if keep else np.array([], dtype=np.int64)
    return center_idx[keep], neigh_idx[keep], edge_vec[keep], distances[keep]
