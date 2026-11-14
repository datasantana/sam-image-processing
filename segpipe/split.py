"""Split espacial por bloques con buffer, para imágenes UAV oblicuas.

Las fotos oblicuas vecinas ven el mismo terreno, así que un split aleatorio filtra información.
Aquí cada foto se representa por el **centro aproximado de su huella en tierra**, no por la
posición del dron:

    d = h / tan(|pitch|)        (h = RelativeAltitude, pitch = GimbalPitchDegree)
    centro = GPS + d · (sin(yaw), cos(yaw))   en metros (este, norte); yaw = GimbalYawDegree

Los centros se agrupan en una grilla de bloques de ``block_m`` metros. Los bloques completos se
reparten en ``k`` pliegues. En el pliegue ``f``: test = pliegue f, val = pliegue (f+1) mod k,
train = el resto **menos** las fotos a menos de ``buffer_m`` de cualquier foto de val o test.
El modo split fijo es el pliegue 0 de este esquema.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

EARTH_R = 6371008.8


def footprint_centers(rows: list[dict], geom: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (fx, fy, dx, dy): centro de huella y posición del dron en metros locales."""
    lat = np.array([float(r[geom["lat"]]) for r in rows])
    lon = np.array([float(r[geom["lon"]]) for r in rows])
    h = np.array([float(r[geom["height"]]) for r in rows])
    pitch = np.abs(np.array([float(r[geom["pitch"]]) for r in rows]))
    yaw = np.radians(np.array([float(r[geom["yaw"]]) for r in rows]))
    if np.any(pitch < 5):
        raise ValueError("Pitch del gimbal casi horizontal (<5°): la huella no está acotada")
    lat0, lon0 = lat.mean(), lon.mean()
    x = np.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    d = h / np.tan(np.radians(pitch))
    return x + d * np.sin(yaw), y + d * np.cos(yaw), x, y


def _pairwise_min(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    if len(b_xy) == 0:
        return np.full(len(a_xy), np.inf)
    d = np.hypot(a_xy[:, None, 0] - b_xy[None, :, 0], a_xy[:, None, 1] - b_xy[None, :, 1])
    return d.min(axis=1)


@dataclass
class Fold:
    fold: int
    train: list[str]
    val: list[str]
    test: list[str]
    excluded: list[str]          # quitadas de train por el buffer
    min_dist_train_heldout: float  # distancia mínima (m) entre centros train y val∪test


def assign_blocks(names: list[str], fx: np.ndarray, fy: np.ndarray, k: int, block_m: float,
                  presence: Optional[dict[str, set]] = None, seed: int = 0) -> dict[str, int]:
    """Asigna bloques completos a k pliegues (voraz, balanceando fotos y clases raras)."""
    bx = np.floor((fx - fx.min()) / block_m).astype(int)
    by = np.floor((fy - fy.min()) / block_m).astype(int)
    blocks: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(bx, by)):
        blocks.setdefault(key, []).append(i)
    if len(blocks) < k:
        raise ValueError(f"Solo hay {len(blocks)} bloques de {block_m} m para {k} pliegues; reduce block_m o k")
    rng = np.random.default_rng(seed)
    keys = list(blocks)
    rng.shuffle(keys)
    keys.sort(key=lambda kk: -len(blocks[kk]))  # estable: empates en orden aleatorio
    classes = sorted({c for s in (presence or {}).values() for c in s})
    tot = {c: max(1, sum(c in presence[n] for n in names)) for c in classes} if presence else {}
    n_fold = np.zeros(k)
    c_fold = {c: np.zeros(k) for c in classes}
    fold_of: dict[str, int] = {}
    for kk in keys:
        idx = blocks[kk]
        cost = n_fold / len(names)
        for c in classes:
            cnt = sum(c in presence[names[i]] for i in idx)
            if cnt:
                cost = cost + c_fold[c] / tot[c]
        f = int(np.argmin(cost + rng.random(k) * 1e-9))
        n_fold[f] += len(idx)
        for c in classes:
            c_fold[c][f] += sum(c in presence[names[i]] for i in idx)
        for i in idx:
            fold_of[names[i]] = f
    return fold_of


def make_folds(rows: list[dict], geom: dict, k: int = 5, block_m: float = 150.0, buffer_m: float = 50.0,
               presence: Optional[dict[str, set]] = None, seed: int = 0) -> list[Fold]:
    names = [r["name"] for r in rows]
    fx, fy, _, _ = footprint_centers(rows, geom)
    fold_of = assign_blocks(names, fx, fy, k, block_m, presence, seed)
    xy = np.stack([fx, fy], 1)
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for f in range(k):
        v = (f + 1) % k
        test = [n for n in names if fold_of[n] == f]
        val = [n for n in names if fold_of[n] == v]
        rest = [n for n in names if fold_of[n] not in (f, v)]
        held = xy[[idx[n] for n in test + val]]
        dmin = _pairwise_min(xy[[idx[n] for n in rest]], held) if rest else np.array([])
        train = [n for n, d in zip(rest, dmin) if d >= buffer_m]
        excl = [n for n, d in zip(rest, dmin) if d < buffer_m]
        md = float(_pairwise_min(xy[[idx[n] for n in train]], held).min()) if train else float("nan")
        out.append(Fold(f, train, val, test, excl, md))
    return out


def presence_from_class_stats(path: str | Path, class_names: list[str]) -> dict[str, set]:
    """Clases presentes (px > 0) por foto, desde class_stats.csv del pack."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["name"]: {c for c in class_names if int(r.get(c, 0) or 0) > 0} for r in csv.DictReader(f)}


def split_report(fold: Fold, presence: Optional[dict[str, set]], class_names: list[str]) -> dict:
    rep = {"fold": fold.fold, "n_train": len(fold.train), "n_val": len(fold.val), "n_test": len(fold.test),
           "n_excluded_buffer": len(fold.excluded), "min_dist_train_heldout_m": fold.min_dist_train_heldout}
    if presence:
        for part in ("train", "val", "test"):
            rep[f"photos_per_class_{part}"] = {c: sum(c in presence[n] for n in getattr(fold, part))
                                              for c in class_names}
    return rep


def save_folds(folds: list[Fold], out_dir: str | Path, params: dict,
               presence: Optional[dict[str, set]] = None, class_names: Optional[list[str]] = None) -> Path:
    """Guarda las listas de cada partición (JSON + un CSV por pliegue) y el reporte."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {"params": params, "folds": [asdict(f) for f in folds],
            "report": [split_report(f, presence, class_names or []) for f in folds]}
    (out / "splits.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    for f in folds:
        with open(out / f"fold{f.fold}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "partition"])
            for part in ("train", "val", "test", "excluded"):
                for n in getattr(f, part):
                    w.writerow([n, part])
    return out / "splits.json"


def load_folds(path: str | Path) -> list[Fold]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Fold(**f) for f in data["folds"]]


def format_report(reports: list[dict]) -> str:
    lines = []
    for r in reports:
        lines.append(f"Pliegue {r['fold']}: train {r['n_train']} | val {r['n_val']} | test {r['n_test']} | "
                     f"excluidas por buffer {r['n_excluded_buffer']} | dist. mín. train↔val/test "
                     f"{r['min_dist_train_heldout_m']:.1f} m")
        if "photos_per_class_val" in r:
            for part in ("val", "test"):
                d = r[f"photos_per_class_{part}"]
                lines.append(f"   fotos por clase en {part}: " + ", ".join(f"{c} {v}" for c, v in d.items()))
    return "\n".join(lines)
