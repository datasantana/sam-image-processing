"""Inferencia sobre la foto completa, devolviendo logits a resolución nativa.

- ``resize``: la foto se reescala a S×S (como en J0), y los logits se reescalan a H×W.
- ``sliding``: ventana deslizante de ``window`` px con solape ``overlap`` (promedio de logits),
  opcionalmente sobre la foto reescalada por ``scale`` (logits devueltos a H×W).
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from .data import to_tensor


def _positions(length: int, win: int, stride: int) -> list[int]:
    if length <= win:
        return [0]
    pos = list(range(0, length - win + 1, stride))
    if pos[-1] != length - win:
        pos.append(length - win)
    return pos


@torch.no_grad()
def predict_logits(model, image: np.ndarray, infer_cfg: dict, data_cfg: dict, device,
                   amp_dtype=None, multiple: int = 32, win_batch: int = 4) -> torch.Tensor:
    """Logits (C, H, W) float32 en ``device`` para una foto RGB uint8 (H, W, 3)."""
    import cv2
    model.eval()
    H, W = image.shape[:2]
    ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None
           else contextlib.nullcontext())
    mode = infer_cfg["mode"]
    if mode == "resize":
        s = int(data_cfg["resize"])
        x = to_tensor(cv2.resize(image, (s, s), interpolation=cv2.INTER_LINEAR), None,
                      data_cfg["mean"], data_cfg["std"])[None].to(device)
        with ctx:
            out = model(x).float()
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)[0]
    if mode != "sliding":
        raise ValueError(f"infer.mode desconocido: {mode}")

    scale = float(infer_cfg.get("scale", 1.0))
    img = image if scale == 1.0 else cv2.resize(image, (round(W * scale), round(H * scale)),
                                                 interpolation=cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    win = int(infer_cfg["window"])
    stride = max(1, int(round(win * (1 - float(infer_cfg["overlap"])))))
    # relleno: al menos una ventana y múltiplo de lo que pide la red
    ph = max(win, h); pw = max(win, w)
    x = to_tensor(img, None, data_cfg["mean"], data_cfg["std"]).to(device)
    x = F.pad(x, (0, pw - w, 0, ph - h))
    wh = min(win, ph); ww = min(win, pw)
    wh_p = -(-wh // multiple) * multiple; ww_p = -(-ww // multiple) * multiple
    acc = None
    cnt = torch.zeros((1, ph, pw), device=device)
    tiles = [(y, xx) for y in _positions(ph, wh, stride) for xx in _positions(pw, ww, stride)]
    for i in range(0, len(tiles), win_batch):
        chunk = tiles[i:i + win_batch]
        batch = torch.stack([x[:, y:y + wh, xx:xx + ww] for y, xx in chunk])
        if (wh_p, ww_p) != (wh, ww):
            batch = F.pad(batch, (0, ww_p - ww, 0, wh_p - wh))
        with ctx:
            out = model(batch).float()[:, :, :wh, :ww]
        if acc is None:
            acc = torch.zeros((out.shape[1], ph, pw), device=device)
        for (y, xx), o in zip(chunk, out):
            acc[:, y:y + wh, xx:xx + ww] += o
            cnt[:, y:y + wh, xx:xx + ww] += 1
    logits = (acc / cnt)[:, :h, :w]
    if scale != 1.0:
        logits = F.interpolate(logits[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return logits
