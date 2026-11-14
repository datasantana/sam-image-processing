"""Split espacial por bloques con buffer, para imágenes UAV oblicuas ("fuga reducida").

Las fotos oblicuas vecinas ven el mismo terreno, así que un split aleatorio filtra información.
Cada foto se representa por su **huella en tierra**, no por la posición del dron:

- Con cámara calibrada (``geometry.camera``: focal y tamaño de píxel en mm, ancho y alto en px),
  se proyectan las 4 esquinas de la imagen sobre un plano a ``RelativeAltitude`` bajo el dron,
  orientando la cámara con GimbalYaw / GimbalPitch / GimbalRoll (punto principal en el centro de la
  imagen, sin distorsión, terreno plano). El centro de la foto es el **centroide del polígono**.
- Sin cámara: centro = GPS + h/tan(|pitch|) en la dirección del yaw (rayo del eje óptico).

Los centros se agrupan en una grilla de ``block_m`` metros (la fecha de vuelo no interviene). Los
bloques completos se reparten en ``k`` pliegues. En el pliegue ``f``: test = f, val = (f+1) mod k,
train = el resto **menos** las fotos a menos de ``buffer_m`` de cualquier foto de val o test.
Por pliegue se reporta la distancia mínima train↔test y el % de solape residual de huellas
(área de la unión de huellas de test cubierta por la unión de huellas de train).
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

EARTH_R = 6371008.8


# ----------------------------------------------------------------------------- geometría

def _local_xy(rows: list[dict], geom: dict) -> tuple[np.ndarray, np.ndarray]:
    lat = np.array([float(r[geom["lat"]]) for r in rows])
    lon = np.array([float(r[geom["lon"]]) for r in rows])
    lat0, lon0 = lat.mean(), lon.mean()
    x = np.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_R
    return x, y


def camera_fov(cam: dict) -> tuple[float, float]:
    """FOV horizontal y vertical (grados) de la cámara calibrada."""
    w = cam["width"] * cam["pixel_mm"]
    h = cam["height"] * cam["pixel_mm"]
    return (math.degrees(2 * math.atan(w / 2 / cam["focal_mm"])),
            math.degrees(2 * math.atan(h / 2 / cam["focal_mm"])))


def footprint_polygons(rows: list[dict], geom: dict) -> np.ndarray:
    """(N, 4, 2) esquinas de la huella en metros locales (este, norte): sup-izq, sup-der, inf-der, inf-izq."""
    cam = geom["camera"]
    x0, y0 = _local_xy(rows, geom)
    hw = cam["width"] * cam["pixel_mm"] / 2
    hh = cam["height"] * cam["pixel_mm"] / 2
    f = cam["focal_mm"]
    max_range = float(cam.get("max_range_m", 400.0))
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]  # +x derecha, +y abajo en la imagen
    out = np.zeros((len(rows), 4, 2))
    for i, r in enumerate(rows):
        h = float(r[geom["height"]])
        yaw = math.radians(float(r[geom["yaw"]]))
        pitch = math.radians(float(r[geom["pitch"]]))   # negativo = hacia abajo
        roll = math.radians(float(r.get(geom.get("roll", "GimbalRollDegree"), 0) or 0))
        d = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
        right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
        down = np.cross(d, right)
        if roll:
            right, down = (right * math.cos(roll) + down * math.sin(roll),
                           -right * math.sin(roll) + down * math.cos(roll))
        for j, (cx, cy) in enumerate(corners):
            ray = d * f + right * cx + down * cy
            if ray[2] < -1e-9:
                t = -h / ray[2]
                p = ray[:2] * t
                dist = float(np.hypot(*p))
                if dist > max_range:
                    p = p * (max_range / dist)
            else:  # el rayo no toca el suelo (sobre el horizonte): se recorta a max_range
                p = ray[:2] / (np.hypot(*ray[:2]) + 1e-12) * max_range
            out[i, j] = (x0[i] + p[0], y0[i] + p[1])
    return out


def polygon_centroid(poly: np.ndarray) -> tuple[float, float]:
    x, y = poly[:, 0], poly[:, 1]
    xs, ys = np.roll(x, -1), np.roll(y, -1)
    cross = x * ys - xs * y
    a = cross.sum() / 2
    if abs(a) < 1e-9:
        return float(x.mean()), float(y.mean())
    return float(((x + xs) * cross).sum() / (6 * a)), float(((y + ys) * cross).sum() / (6 * a))


def footprints(rows: list[dict], geom: dict) -> dict:
    """Centros de huella (cx, cy), posición del dron (dx, dy), polígonos (o None) y método usado."""
    pitch = np.abs(np.array([float(r[geom["pitch"]]) for r in rows]))
    if np.any(pitch < 5):
        raise ValueError("Pitch del gimbal casi horizontal (<5°): la huella no está acotada")
    x, y = _local_xy(rows, geom)
    if geom.get("camera"):
        polys = footprint_polygons(rows, geom)
        c = np.array([polygon_centroid(p) for p in polys])
        return {"cx": c[:, 0], "cy": c[:, 1], "dx": x, "dy": y, "polys": polys, "method": "camera_centroid"}
    h = np.array([float(r[geom["height"]]) for r in rows])
    yaw = np.radians(np.array([float(r[geom["yaw"]]) for r in rows]))
    d = h / np.tan(np.radians(pitch))
    return {"cx": x + d * np.sin(yaw), "cy": y + d * np.cos(yaw), "dx": x, "dy": y, "polys": None,
            "method": "principal_ray"}


def footprint_centers(rows: list[dict], geom: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(cx, cy, dx, dy): centro de huella y posición del dron en metros locales."""
    fp = footprints(rows, geom)
    return fp["cx"], fp["cy"], fp["dx"], fp["dy"]


# ----------------------------------------------------------------------------- solape (raster)

class FootprintRaster:
    """Rasteriza huellas convexas en una grilla de ``res_m`` metros para medir solapes de área."""

    def __init__(self, polys: np.ndarray, res_m: float = 2.0):
        lo = polys.reshape(-1, 2).min(0) - res_m
        hi = polys.reshape(-1, 2).max(0) + res_m
        xs = np.arange(lo[0], hi[0], res_m) + res_m / 2
        ys = np.arange(lo[1], hi[1], res_m) + res_m / 2
        gx, gy = np.meshgrid(xs, ys)
        self.res = res_m
        self.masks = np.stack([self._inside(p, gx, gy) for p in polys])

    @staticmethod
    def _inside(poly, gx, gy):
        p = poly if _signed_area(poly) > 0 else poly[::-1]
        m = np.ones(gx.shape, bool)
        for a, b in zip(p, np.roll(p, -1, axis=0)):
            m &= (b[0] - a[0]) * (gy - a[1]) - (b[1] - a[1]) * (gx - a[0]) >= 0
        return m

    def union(self, idx: list[int]) -> np.ndarray:
        return self.masks[idx].any(0) if idx else np.zeros(self.masks.shape[1:], bool)

    def covered_pct(self, target: list[int], by: list[int]) -> float:
        """% del área de la unión de ``target`` cubierta por la unión de ``by``."""
        t = self.union(target)
        if not t.any():
            return float("nan")
        return float(100 * (t & self.union(by)).sum() / t.sum())


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2)


# ----------------------------------------------------------------------------- pliegues

def _pairwise_min(a_xy: np.ndarray, b_xy: np.ndarray) -> np.ndarray:
    if len(a_xy) == 0:
        return np.array([])
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
    excluded: list[str]                 # quitadas de train por el buffer
    min_dist_train_heldout: float       # distancia mínima (m) entre centros train y val∪test
    min_dist_train_test: float = float("nan")
    overlap_train_test_pct: Optional[float] = None   # % del área de test cubierta por huellas de train
    overlap_train_val_pct: Optional[float] = None
    buffer_m: Optional[float] = None


def block_ids(cx: np.ndarray, cy: np.ndarray, block_m: float) -> list[tuple[int, int]]:
    bx = np.floor((cx - cx.min()) / block_m).astype(int)
    by = np.floor((cy - cy.min()) / block_m).astype(int)
    return list(zip(bx.tolist(), by.tolist()))


def assign_blocks(names: list[str], fx: np.ndarray, fy: np.ndarray, k: int, block_m: float,
                  presence: Optional[dict[str, set]] = None, seed: int = 0) -> dict[str, int]:
    """Asigna bloques completos a k pliegues (voraz, balanceando fotos y clases raras)."""
    blocks: dict[tuple, list[int]] = {}
    for i, key in enumerate(block_ids(fx, fy, block_m)):
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
            if sum(c in presence[names[i]] for i in idx):
                cost = cost + c_fold[c] / tot[c]
        f = int(np.argmin(cost + rng.random(k) * 1e-9))
        n_fold[f] += len(idx)
        for c in classes:
            c_fold[c][f] += sum(c in presence[names[i]] for i in idx)
        for i in idx:
            fold_of[names[i]] = f
    return fold_of


def make_folds(rows: list[dict], geom: dict, k: int = 5, block_m: float = 150.0, buffer_m: float = 50.0,
               presence: Optional[dict[str, set]] = None, seed: int = 0, overlap_res_m: float = 2.0) -> list[Fold]:
    names = [r["name"] for r in rows]
    fp = footprints(rows, geom)
    fold_of = assign_blocks(names, fp["cx"], fp["cy"], k, block_m, presence, seed)
    return folds_from_assignment(names, fp, fold_of, k, buffer_m, overlap_res_m)


def folds_from_assignment(names: list[str], fp: dict, fold_of: dict[str, int], k: int, buffer_m: float,
                          overlap_res_m: float = 2.0) -> list[Fold]:
    """Construye los pliegues (con buffer) a partir de una asignación foto → pliegue ya fijada."""
    xy = np.stack([fp["cx"], fp["cy"]], 1)
    idx = {n: i for i, n in enumerate(names)}
    raster = FootprintRaster(fp["polys"], overlap_res_m) if fp.get("polys") is not None else None
    out = []
    for f in range(k):
        v = (f + 1) % k
        test = [n for n in names if fold_of[n] == f]
        val = [n for n in names if fold_of[n] == v]
        rest = [n for n in names if fold_of[n] not in (f, v)]
        held = xy[[idx[n] for n in test + val]]
        dmin = _pairwise_min(xy[[idx[n] for n in rest]], held)
        train = [n for n, d in zip(rest, dmin) if d >= buffer_m]
        excl = [n for n, d in zip(rest, dmin) if d < buffer_m]
        tr_xy = xy[[idx[n] for n in train]]
        md = float(_pairwise_min(tr_xy, held).min()) if train else float("nan")
        mdt = float(_pairwise_min(tr_xy, xy[[idx[n] for n in test]]).min()) if train else float("nan")
        ov_t = ov_v = None
        if raster is not None:
            ti = [idx[n] for n in train]
            ov_t = raster.covered_pct([idx[n] for n in test], ti)
            ov_v = raster.covered_pct([idx[n] for n in val], ti)
        out.append(Fold(f, train, val, test, excl, md, mdt, ov_t, ov_v, buffer_m))
    return out


def fold_assignment(folds: list[Fold]) -> dict[str, int]:
    return {n: f.fold for f in folds for n in f.test}


# ----------------------------------------------------------------------------- diagnósticos

def flight_mix(names: list[str], cx: np.ndarray, cy: np.ndarray, block_m: float,
               group_regex: str = r"DJI_(\d{8})") -> dict:
    """Cuántos bloques mezclan fotos de vuelos (fechas) distintos. La fecha no se usa para agrupar."""
    groups = [(re.search(group_regex, n).group(1) if re.search(group_regex, n) else "?") for n in names]
    blocks: dict[tuple, set] = {}
    for b, g in zip(block_ids(cx, cy, block_m), groups):
        blocks.setdefault(b, set()).add(g)
    n_groups = sorted(set(groups))
    return {"groups": {g: groups.count(g) for g in n_groups}, "n_blocks": len(blocks),
            "blocks_with_several_groups": sum(len(s) > 1 for s in blocks.values()),
            "blocks_by_groups": {"+".join(sorted(s)): sum(1 for t in blocks.values() if t == s)
                                 for s in {frozenset(t) for t in blocks.values()}}}


def presence_from_class_stats(path: str | Path, class_names: list[str]) -> dict[str, set]:
    """Clases presentes (px > 0) por foto, desde class_stats.csv del pack."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["name"]: {c for c in class_names if int(r.get(c, 0) or 0) > 0} for r in csv.DictReader(f)}


def split_report(fold: Fold, presence: Optional[dict[str, set]], class_names: list[str]) -> dict:
    rep = {"fold": fold.fold, "buffer_m": fold.buffer_m, "n_train": len(fold.train), "n_val": len(fold.val),
           "n_test": len(fold.test), "n_excluded_buffer": len(fold.excluded),
           "min_dist_train_heldout_m": fold.min_dist_train_heldout,
           "min_dist_train_test_m": fold.min_dist_train_test,
           "overlap_train_test_pct": fold.overlap_train_test_pct,
           "overlap_train_val_pct": fold.overlap_train_val_pct}
    if presence:
        for part in ("train", "val", "test"):
            rep[f"photos_per_class_{part}"] = {c: sum(c in presence[n] for n in getattr(fold, part))
                                              for c in class_names}
    return rep


def save_folds(folds: list[Fold], out_dir: str | Path, params: dict,
               presence: Optional[dict[str, set]] = None, class_names: Optional[list[str]] = None,
               extra: Optional[dict] = None) -> Path:
    """Guarda las listas de cada partición (JSON + un CSV por pliegue) y el reporte."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {"params": params, "folds": [asdict(f) for f in folds],
            "report": [split_report(f, presence, class_names or []) for f in folds], **(extra or {})}
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


def _num(x, fmt="{:.1f}"):
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else fmt.format(x)


def format_report(reports: list[dict]) -> str:
    lines = []
    for r in reports:
        lines.append(f"Pliegue {r['fold']} (buffer {_num(r.get('buffer_m'), '{:.0f}')} m): train {r['n_train']} | "
                     f"val {r['n_val']} | test {r['n_test']} | excluidas por buffer {r['n_excluded_buffer']} | "
                     f"dist. mín. train↔test {_num(r.get('min_dist_train_test_m'))} m "
                     f"(train↔val∪test {_num(r['min_dist_train_heldout_m'])} m) | "
                     f"solape residual de huellas: test {_num(r.get('overlap_train_test_pct'))} %, "
                     f"val {_num(r.get('overlap_train_val_pct'))} %")
        if "photos_per_class_val" in r:
            for part in ("val", "test"):
                d = r[f"photos_per_class_{part}"]
                lines.append(f"   fotos por clase en {part}: " + ", ".join(f"{c} {v}" for c, v in d.items()))
    return "\n".join(lines)
