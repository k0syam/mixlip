"""Evaluation metrics for MLIP predictions."""

from __future__ import annotations

import numpy as np


def energy_mae_per_atom(
    pred_energies: np.ndarray,
    true_energies: np.ndarray,
    n_atoms: np.ndarray,
) -> float:
    """Mean absolute error of energy per atom in meV/atom."""
    pred_per_atom = pred_energies / n_atoms
    true_per_atom = true_energies / n_atoms
    return float(np.mean(np.abs(pred_per_atom - true_per_atom)) * 1000)


def forces_mae(pred_forces: np.ndarray, true_forces: np.ndarray) -> float:
    """Mean absolute error of force components in meV/Å."""
    return float(np.mean(np.abs(pred_forces - true_forces)) * 1000)


def forces_rmse(pred_forces: np.ndarray, true_forces: np.ndarray) -> float:
    """Root mean square error of force magnitude in meV/Å."""
    diff = pred_forces - true_forces
    return float(np.sqrt(np.mean(diff**2)) * 1000)


def stress_mae(pred_stress: np.ndarray, true_stress: np.ndarray) -> float:
    """Mean absolute error of stress components in GPa."""
    # 1 eV/Å³ = 160.218 GPa
    return float(np.mean(np.abs(pred_stress - true_stress)) * 160.218)


def forces_cosine_similarity(pred_forces: np.ndarray, true_forces: np.ndarray) -> float:
    """Mean cosine similarity between predicted and true force vectors."""
    dot = np.einsum("ij,ij->i", pred_forces, true_forces)
    pred_norm = np.linalg.norm(pred_forces, axis=1)
    true_norm = np.linalg.norm(true_forces, axis=1)
    denom = pred_norm * true_norm
    mask = denom > 1e-8
    cos_sim = np.where(mask, dot / denom, 0.0)
    return float(np.mean(cos_sim))


def compute_all_metrics(
    results: list[dict],  # [{"pred": PredictionResult, "true": AtomicSample}]
) -> dict[str, float]:
    """Compute all metrics given a list of (pred, true) pairs."""
    from mixlip.core.protocol import PredictionResult
    from mixlip.data.schema import AtomicSample

    pred_e, true_e, n_atoms_list = [], [], []
    pred_f_list, true_f_list = [], []
    pred_s_list, true_s_list = [], []

    for item in results:
        pred: PredictionResult = item["pred"]
        true: AtomicSample = item["true"]
        n = true.n_atoms
        pred_e.append(pred.energy)
        true_e.append(true.energy)
        n_atoms_list.append(n)
        if pred.forces is not None and true.forces is not None:
            pred_f_list.append(pred.forces)
            true_f_list.append(true.forces)
        if pred.stress is not None and true.stress is not None:
            pred_s_list.append(pred.stress)
            true_s_list.append(true.stress)

    metrics: dict[str, float] = {}
    if pred_e and true_e and all(x is not None for x in true_e):
        metrics["energy_mae_meV_per_atom"] = energy_mae_per_atom(
            np.array(pred_e), np.array(true_e), np.array(n_atoms_list)
        )

    if pred_f_list:
        pf = np.concatenate(pred_f_list, axis=0)
        tf = np.concatenate(true_f_list, axis=0)
        metrics["forces_mae_meV_per_A"] = forces_mae(pf, tf)
        metrics["forces_rmse_meV_per_A"] = forces_rmse(pf, tf)
        metrics["forces_cos_sim"] = forces_cosine_similarity(pf, tf)

    if pred_s_list:
        ps = np.stack(pred_s_list, axis=0)
        ts = np.stack(true_s_list, axis=0)
        metrics["stress_mae_GPa"] = stress_mae(ps, ts)

    return metrics
