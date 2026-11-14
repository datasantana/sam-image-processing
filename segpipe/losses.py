"""Pérdidas configurables. Todas ignoran ``ignore_index`` (píxeles sin anotar)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .classes import IGNORE_INDEX


def class_weights(pixel_counts: np.ndarray, kind: str) -> Optional[torch.Tensor]:
    """Pesos por clase desde los píxeles de train.

    inverse_freq (receta J0): w_c ∝ 1/n_c, normalizado a media 1 sobre las clases presentes.
    sqrt_inverse_freq: w_c ∝ 1/√n_c. Una clase ausente de train recibe peso 0.
    """
    if kind in (None, "none"):
        return None
    n = pixel_counts.astype(np.float64)
    present = n > 0
    w = np.zeros_like(n)
    if kind == "inverse_freq":
        w[present] = 1.0 / n[present]
    elif kind == "sqrt_inverse_freq":
        w[present] = 1.0 / np.sqrt(n[present])
    else:
        raise ValueError(f"loss.weights desconocido: {kind}")
    w[present] /= w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


class Combined(nn.Module):
    def __init__(self, parts):
        super().__init__()
        self.parts = nn.ModuleList(parts)

    def forward(self, logits, target):
        return sum(p(logits, target) for p in self.parts)


def build_loss(loss_cfg: dict, weights: Optional[torch.Tensor]) -> nn.Module:
    import segmentation_models_pytorch as smp
    name = loss_cfg["name"]
    ce = nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)
    dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=IGNORE_INDEX)
    focal = smp.losses.FocalLoss(mode="multiclass", gamma=float(loss_cfg.get("gamma", 2.0)), ignore_index=IGNORE_INDEX)
    if name == "dice_wce":
        return Combined([dice, ce])
    if name == "ce":
        return ce
    if name == "focal":
        return focal
    if name == "dice_focal":
        return Combined([dice, focal])
    raise ValueError(f"loss.name desconocido: {name}")
