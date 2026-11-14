"""Configs de corrida en YAML con herencia (``base:``) para que cada paso de la ablación
declare solo lo que cambia respecto al anterior."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULTS: dict[str, Any] = {
    "name": None,
    "description": "",
    "dataset": "configs/datasets/pascual.yaml",
    "model": {"arch": "deeplabv3plus", "encoder": "resnet50", "weights": "imagenet",
              "hf_name": None},
    "data": {
        "mode": "resize",            # resize | crops
        "resize": 512,
        "crop": 1024,
        "samples_per_epoch": None,   # solo crops; None = nº de fotos de train
        "scale_range": None,         # crops: [min, max] reescalado aleatorio de la foto antes de recortar
        "sampler": "uniform",        # uniform | rare
        "rare_p": 0.5,
        "rare_classes": "auto",      # auto = clases con < rare_px_frac de los píxeles de train
        "rare_px_frac": 0.01,
        "augment": "legacy",         # legacy (aumentos del notebook original) | hflip
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "infer": {"mode": "resize", "window": 1024, "overlap": 0.5, "scale": 1.0},
    "loss": {"name": "dice_wce", "weights": "inverse_freq"},   # dice_wce | ce | focal ; weights: inverse_freq|sqrt_inverse_freq|none
    "optim": {"name": "adam", "lr": 1e-4, "weight_decay": 1e-4},
    "sched": {"name": "plateau", "factor": 0.5, "patience": 5, "power": 1.0},
    "train": {
        "batch": 8,            # batch efectivo; si no cabe en la GPU se usa acumulación de gradiente
        "epochs": 50,
        "patience": 10,        # early stopping (épocas sin mejorar la métrica de selección)
        "select": "miou",      # miou | maxmin
        "select_min_photos": 1,
        "amp": True,
        "grad_clip": None,
    },
    "eval": {"min_photos": 5, "n_boot": 2000},
}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_run_config(path: str | Path, overrides: Optional[dict] = None, _seen: Optional[set] = None) -> dict:
    """Carga un YAML de corrida resolviendo ``base:`` (relativo al archivo) y aplica ``overrides``."""
    path = Path(path)
    _seen = _seen or set()
    if path.resolve() in _seen:
        raise ValueError(f"Herencia circular en {path}")
    _seen.add(path.resolve())
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("base", None)
    cfg = load_run_config(path.parent / base, None, _seen) if base else copy.deepcopy(DEFAULTS)
    cfg["description"] = ""
    cfg = deep_merge(cfg, raw)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["name"] = raw.get("name") or path.stem   # el nombre nunca se hereda de la base
    cfg["base"] = base
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def diff(a: dict, b: dict, prefix: str = "") -> list[str]:
    """Lista legible de diferencias (para documentar el 'un solo cambio' de cada paso)."""
    out = []
    for k in sorted(set(a) | set(b)):
        if not prefix and k in ("name", "description", "base"):
            continue
        p = f"{prefix}{k}"
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            out += diff(va, vb, p + ".")
        elif va != vb:
            out.append(f"{p}: {va!r} -> {vb!r}")
    return out
