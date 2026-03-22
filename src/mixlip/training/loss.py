"""Weighted energy/forces/stress loss function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from mixlip.core.config import LossConfig


class WeightedEFSLoss(nn.Module):
    """Weighted sum of energy, forces, stress (and optionally magmom) losses.

    - Energy loss is per-atom normalized.
    - Forces loss is element-wise over (N_total, 3).
    - Stress loss is over 6 Voigt components per structure.
    - Missing labels (None / missing attribute) are automatically skipped.
    - Per-sample ``weight`` field scales the energy loss contribution.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self.w_e = config.energy_weight
        self.w_f = config.forces_weight
        self.w_s = config.stress_weight
        self.w_m = config.magmom_weight
        self._base = self._build_criterion(config)

    @staticmethod
    def _build_criterion(config: LossConfig) -> nn.Module:
        if config.criterion == "mse":
            return nn.MSELoss(reduction="none")
        if config.criterion == "mae":
            return nn.L1Loss(reduction="none")
        return nn.HuberLoss(reduction="none", delta=config.huber_delta)

    def forward(self, pred: dict, batch) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}

        # --- energy ---
        e_pred = pred.get("energy")
        e_true = getattr(batch, "energy", None)
        if e_pred is not None and e_true is not None:
            n = batch.num_atoms.float()
            e_per_atom_pred = e_pred / n
            e_per_atom_true = e_true.float() / n
            weight = getattr(batch, "weight", torch.ones_like(e_per_atom_pred))
            losses["energy"] = (
                self._base(e_per_atom_pred, e_per_atom_true) * weight
            ).mean() * self.w_e

        # --- forces ---
        f_pred = pred.get("forces")
        f_true = getattr(batch, "forces", None)
        if f_pred is not None and f_true is not None:
            losses["forces"] = self._base(f_pred, f_true).mean() * self.w_f

        # --- stress ---
        s_pred = pred.get("stress")
        s_true = getattr(batch, "stress", None)
        if s_pred is not None and s_true is not None:
            losses["stress"] = self._base(s_pred, s_true).mean() * self.w_s

        # --- magmoms ---
        m_pred = pred.get("magmoms")
        m_true = getattr(batch, "magmoms", None)
        if m_pred is not None and m_true is not None:
            losses["magmoms"] = self._base(m_pred, m_true).mean() * self.w_m

        losses["total"] = sum(losses.values()) if losses else torch.tensor(0.0)
        return losses
