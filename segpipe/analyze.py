"""Análisis posterior de un estudio terminado (solo lectura de ``runs/``; sin torch).

Uso::

    python -m segpipe.analyze --study <carpeta del estudio> --manifest <manifest.csv> [--out analysis]
                              [--mapping configs/analysis/common_scheme.yaml] [--mauricio <xlsx>]

Lee lo que dejó la cola (``study.json``, ``runs/<config>/seed<s>/fold<k>/``) y escribe tablas,
figuras y ``REPORT.md`` en ``--out`` (por defecto ``<study>/analysis``). Nunca escribe dentro de
``runs/``.

Análisis:
  A1  Línea base de fuga: solape de huellas train→test del split aleatorio original
      (orden alfabético + ``train_test_split(train_size=0.8, random_state=42)``) frente a la CV
      por bloques con buffer 50 m y 75 m, con el mismo método de huellas de ``segpipe.split``.
  A2  Fuga residual: IoU medio por foto de test contra la fracción de su huella cubierta por las
      huellas de train de su pliegue; Spearman con IC bootstrap, parcial por pliegue, y la
      comparación pareada J0 vs J0-b75.
  A3  Esquema común con Mauricio (solo si el mapeo YAML tiene ``approved: true``).
  A4  Figuras (``segpipe.figures``).
  A5  REPORT.md (inventario, método y números; sin interpretación).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import metrics as M
from . import split as S

REPO = Path(__file__).resolve().parents[1]
LADDER_FIG = ["J0", "J0-pasos", "J1", "J2", "J3", "M-b3"]
CONFIG_ORDER = ["J0", "J0-pasos", "J1", "J2", "J3", "M-b3", "J0-b75"]
OUT_CLASS = "fuera_de_esquema"


# ============================================================================ localizar el estudio

def find_study(arg: Optional[str]) -> tuple[Path, str]:
    """Orden: --study, SEGPIPE_STUDY, Google Drive para escritorio. Devuelve (ruta, fuente)."""
    if arg:
        return Path(arg), "argumento --study"
    env = os.environ.get("SEGPIPE_STUDY")
    if env:
        return Path(env), "variable SEGPIPE_STUDY"
    for p in _drive_candidates():
        if (p / "study.json").exists():
            return p, "Google Drive para escritorio"
    raise SystemExit(
        "No encuentro el estudio: usa --study o SEGPIPE_STUDY. No hay Google Drive para escritorio "
        "montado; empaqueta el estudio sin *.pt en Colab (ver docs/analysis.md) y descomprímelo aquí.")


def _drive_candidates() -> list[Path]:
    out = []
    for letter in string.ascii_uppercase:
        for top in ("Mi unidad", "My Drive"):
            out.append(Path(f"{letter}:/") / top / "segpipe" / "pascual_v1")
    home = Path.home()
    for top in ("Google Drive/Mi unidad", "Google Drive/My Drive", "My Drive"):
        out.append(home / top / "segpipe" / "pascual_v1")
    return out


# ============================================================================ carga

@dataclass
class Run:
    config: str
    seed: int
    fold: int
    dir: Path
    done: dict
    split: dict
    cfg: dict
    log: list[dict]
    cms: np.ndarray              # (N, C, C) test
    names: list[str]
    env: dict = field(default_factory=dict)


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_runs(study: Path) -> tuple[list[Run], list[dict]]:
    """Lee todas las corridas terminadas. Devuelve (corridas, inventario por carpeta)."""
    runs, inv = [], []
    expected = ["DONE.json", "split.json", "config.json", "epoch_log.csv", "test/photo_confusion.npz",
                "test/metrics.json", "test/metrics_per_class.csv"]
    for fdir in sorted((study / "runs").glob("*/seed*/fold*")):
        config, seed, fold = fdir.parent.parent.name, int(fdir.parent.name[4:]), int(fdir.name[4:])
        present = {e: (fdir / e).exists() for e in expected}
        inv.append({"config": config, "seed": seed, "fold": fold,
                    **{e: present[e] for e in expected},
                    "best.pt": (fdir / "best.pt").exists(), "last.pt": (fdir / "last.pt").exists()})
        if not (present["DONE.json"] and present["test/photo_confusion.npz"] and present["split.json"]):
            continue
        z = np.load(fdir / "test" / "photo_confusion.npz", allow_pickle=False)
        rd = lambda n: json.loads((fdir / n).read_text(encoding="utf-8"))
        runs.append(Run(config, seed, fold, fdir, rd("DONE.json"), rd("split.json"),
                        rd("config.json") if present["config.json"] else {},
                        _read_csv(fdir / "epoch_log.csv") if present["epoch_log.csv"] else [],
                        z["cms"].astype(np.int64), [str(n) for n in z["names"]],
                        rd("env.json") if (fdir / "env.json").exists() else {}))
    return runs, inv


def configs_in(runs: list[Run]) -> list[str]:
    found = sorted({r.config for r in runs})
    return [c for c in CONFIG_ORDER if c in found] + [c for c in found if c not in CONFIG_ORDER]


@dataclass
class Geometry:
    names: list[str]
    idx: dict
    cx: np.ndarray
    cy: np.ndarray
    raster: S.FootprintRaster

    def coverage(self, photos: list[str], by: list[str]) -> np.ndarray:
        """% de la huella de cada foto de ``photos`` cubierta por la unión de huellas de ``by``."""
        u = self.raster.union([self.idx[n] for n in by])
        out = []
        for n in photos:
            m = self.raster.masks[self.idx[n]]
            out.append(100 * (m & u).sum() / m.sum() if m.any() else np.nan)
        return np.array(out, float)

    def union_overlap(self, test: list[str], train: list[str]) -> float:
        return self.raster.covered_pct([self.idx[n] for n in test], [self.idx[n] for n in train])

    def min_dist(self, a: list[str], b: list[str]) -> float:
        ia, ib = [self.idx[n] for n in a], [self.idx[n] for n in b]
        if not ia or not ib:
            return float("nan")
        d = np.hypot(self.cx[ia, None] - self.cx[None, ib], self.cy[ia, None] - self.cy[None, ib])
        return float(d.min())


def load_geometry(manifest: Path, geom: dict, res_m: float = 2.0) -> Geometry:
    rows = _read_csv(manifest)
    fp = S.footprints(rows, geom)
    if fp["polys"] is None:
        raise ValueError("La geometría no tiene cámara calibrada: no hay polígonos de huella")
    names = [r["name"] for r in rows]
    return Geometry(names, {n: i for i, n in enumerate(names)}, fp["cx"], fp["cy"],
                    S.FootprintRaster(fp["polys"], res_m))


# ============================================================================ estadística

def photo_iou(cms: np.ndarray) -> np.ndarray:
    """IoU medio por foto: media de IoU sobre las clases con unión > 0 en esa foto."""
    cms = np.asarray(cms, np.float64)
    tp = np.diagonal(cms, axis1=1, axis2=2)
    union = cms.sum(2) + cms.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nanmean(iou, axis=1)


def _rank(x):
    from scipy.stats import rankdata
    return rankdata(x)


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    rx, ry = _rank(x[ok]), _rank(y[ok])
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, groups) -> float:
    """Spearman controlando por grupo: rangos globales, se quita la media por grupo, Pearson."""
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], g[ok]
    if len(x) < 4:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    for v in np.unique(g):
        m = g == v
        rx[m] -= rx[m].mean()
        ry[m] -= ry[m].mean()
    if np.allclose(rx, 0) or np.allclose(ry, 0):
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_ci(stat, n: int, n_boot: int, seed: int, groups=None, alpha: float = 0.05) -> list[float]:
    """IC percentil remuestreando índices (estratificado por ``groups`` si se da)."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    if groups is not None:
        groups = np.asarray(groups)
        members = [np.flatnonzero(groups == v) for v in np.unique(groups)]
    for b in range(n_boot):
        if groups is None:
            ix = rng.integers(0, n, n)
        else:
            ix = np.concatenate([m[rng.integers(0, len(m), len(m))] for m in members])
        vals[b] = stat(ix)
    if not np.isfinite(vals).any():
        return [float("nan"), float("nan")]
    return [float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))]


def ols(y, X) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


# ============================================================================ A1

def random_split(names: list[str], train_size: float = 0.8, seed: int = 42) -> tuple[list[str], list[str]]:
    """Réplica exacta de ``sklearn.model_selection.train_test_split(sorted(names), train_size, random_state)``."""
    names = sorted(names)
    n = len(names)
    n_train = int(math.floor(train_size * n))
    n_test = n - n_train
    perm = np.random.RandomState(seed).permutation(n)
    return [names[i] for i in perm[n_test:n_test + n_train]], [names[i] for i in perm[:n_test]]


def cv_splits_by_buffer(runs: list[Run]) -> dict[float, dict[int, dict]]:
    """{buffer_m: {fold: split}} desde los split.json de las corridas (verifica que coincidan)."""
    out: dict[float, dict[int, dict]] = {}
    for r in runs:
        b = r.split.get("buffer_m")
        if b is None:
            continue
        prev = out.setdefault(float(b), {}).get(r.fold)
        if prev is None:
            out[float(b)][r.fold] = r.split
        elif sorted(prev["train"]) != sorted(r.split["train"]) or sorted(prev["test"]) != sorted(r.split["test"]):
            raise RuntimeError(f"Splits distintos con el mismo buffer {b} en el pliegue {r.fold} ({r.config})")
    return out


def analysis_a1(geo: Geometry, photos: list[str], runs: list[Run], seed: int = 42) -> dict:
    rows = []
    tr, te = random_split(photos, 0.8, seed)
    cov = geo.coverage(te, tr)
    rows.append({"esquema": "aleatorio_80_20", "buffer_m": None, "fold": "—", "n_train": len(tr),
                 "n_test": len(te), "solape_union_pct": geo.union_overlap(te, tr),
                 "cobertura_foto_mediana_pct": float(np.nanmedian(cov)),
                 "cobertura_foto_media_pct": float(np.nanmean(cov)),
                 "dist_min_train_test_m": geo.min_dist(tr, te), "solape_guardado_pct": None})
    for b, folds in sorted(cv_splits_by_buffer(runs).items()):
        for f, sp in sorted(folds.items()):
            cov = geo.coverage(sp["test"], sp["train"])
            rows.append({"esquema": f"cv_bloques_buffer{b:g}", "buffer_m": b, "fold": f,
                         "n_train": len(sp["train"]), "n_test": len(sp["test"]),
                         "solape_union_pct": geo.union_overlap(sp["test"], sp["train"]),
                         "cobertura_foto_mediana_pct": float(np.nanmedian(cov)),
                         "cobertura_foto_media_pct": float(np.nanmean(cov)),
                         "dist_min_train_test_m": geo.min_dist(sp["train"], sp["test"]),
                         "solape_guardado_pct": sp.get("overlap_train_test_pct")})
    # resumen por esquema (media de pliegues)
    summary = []
    for esq in dict.fromkeys(r["esquema"] for r in rows):
        rr = [r for r in rows if r["esquema"] == esq]
        summary.append({"esquema": esq, "n_pliegues": len(rr),
                        "n_train_medio": float(np.mean([r["n_train"] for r in rr])),
                        "solape_union_medio_pct": float(np.mean([r["solape_union_pct"] for r in rr])),
                        "solape_union_min_pct": float(min(r["solape_union_pct"] for r in rr)),
                        "solape_union_max_pct": float(max(r["solape_union_pct"] for r in rr)),
                        "cobertura_foto_mediana_pct": float(np.mean([r["cobertura_foto_mediana_pct"] for r in rr]))})
    checks = [abs(r["solape_union_pct"] - r["solape_guardado_pct"]) for r in rows
              if r["solape_guardado_pct"] is not None]
    return {"rows": rows, "summary": summary, "random_test": te, "random_train": tr,
            "max_diff_vs_guardado_pp": max(checks) if checks else None}


# ============================================================================ A2

def per_photo_table(runs: list[Run], geo: Geometry) -> list[dict]:
    """Una fila por (config, foto de test): IoU (media entre semillas), cobertura, pliegue, n_train."""
    acc: dict[tuple, dict] = {}
    for r in runs:
        ious = photo_iou(r.cms)
        cov = geo.coverage(r.names, r.split["train"])
        for n, i, c in zip(r.names, ious, cov):
            d = acc.setdefault((r.config, n), {"config": r.config, "photo": n, "fold": r.fold,
                                               "n_train": len(r.split["train"]), "coverage_pct": float(c),
                                               "ious": []})
            d["ious"].append(float(i))
    out = []
    for d in acc.values():
        out.append({**{k: v for k, v in d.items() if k != "ious"}, "iou": float(np.nanmean(d["ious"])),
                    "n_seeds": len(d["ious"])})
    return out


def analysis_a2(table: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    res = []
    for c in dict.fromkeys(t["config"] for t in table):
        rows = [t for t in table if t["config"] == c]
        x = np.array([t["coverage_pct"] for t in rows]); y = np.array([t["iou"] for t in rows])
        g = np.array([t["fold"] for t in rows])
        res.append({"config": c, "n_fotos": len(rows),
                    "rho": spearman(x, y),
                    "rho_ci": boot_ci(lambda ix: spearman(x[ix], y[ix]), len(x), n_boot, seed),
                    "rho_parcial_pliegue": partial_spearman(x, y, g),
                    "rho_parcial_ci": boot_ci(lambda ix: partial_spearman(x[ix], y[ix], g[ix]), len(x),
                                              n_boot, seed, groups=g),
                    "rho_por_pliegue": {int(f): spearman(x[g == f], y[g == f]) for f in np.unique(g)}})
    return {"correlations": res, "n_boot": n_boot}


def analysis_pair(table: list[dict], a: str = "J0", b: str = "J0-b75", n_boot: int = 2000, seed: int = 0) -> Optional[dict]:
    """Comparación pareada por foto b − a (mismas fotos de test, distinto train)."""
    A = {t["photo"]: t for t in table if t["config"] == a}
    B = {t["photo"]: t for t in table if t["config"] == b}
    common = sorted(set(A) & set(B))
    if not common:
        return None
    rows = []
    for n in common:
        if A[n]["fold"] != B[n]["fold"]:
            raise RuntimeError(f"{n}: pliegue distinto en {a} y {b}")
        rows.append({"photo": n, "fold": A[n]["fold"], "iou_a": A[n]["iou"], "iou_b": B[n]["iou"],
                     "d_iou": B[n]["iou"] - A[n]["iou"], "cov_a": A[n]["coverage_pct"],
                     "cov_b": B[n]["coverage_pct"], "d_cov": B[n]["coverage_pct"] - A[n]["coverage_pct"],
                     "n_train_a": A[n]["n_train"], "n_train_b": B[n]["n_train"],
                     "d_n_train": B[n]["n_train"] - A[n]["n_train"]})
    dy = np.array([r["d_iou"] for r in rows]); dc = np.array([r["d_cov"] for r in rows])
    dn = np.array([r["d_n_train"] for r in rows], float); g = np.array([r["fold"] for r in rows])
    N = len(rows)
    X = np.column_stack([np.ones(N), dc, dn])
    rank = int(np.linalg.matrix_rank(X))

    def coefs(ix):
        if np.linalg.matrix_rank(X[ix]) < 3:
            return np.array([np.nan] * 3)
        return ols(dy[ix], X[ix])
    rng = np.random.default_rng(seed)
    boots = np.array([coefs(rng.integers(0, N, N)) for _ in range(n_boot)])
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] if np.isfinite(v).any() else [float("nan")] * 2
    beta = coefs(np.arange(N))
    with np.errstate(invalid="ignore"):
        r_cn = float(np.corrcoef(dc, dn)[0, 1]) if np.ptp(dc) > 0 and np.ptp(dn) > 0 else float("nan")
    folds = []
    for f in np.unique(g):
        m = g == f
        folds.append({"fold": int(f), "n_fotos": int(m.sum()), "n_train_a": int(rows[np.flatnonzero(m)[0]]["n_train_a"]),
                      "n_train_b": int(rows[np.flatnonzero(m)[0]]["n_train_b"]), "d_n_train": float(dn[m][0]),
                      "d_iou_medio": float(np.nanmean(dy[m])), "d_cov_medio": float(np.nanmean(dc[m]))})
    return {
        "a": a, "b": b, "n_fotos": N, "rows": rows, "folds": folds,
        "d_iou_medio": float(np.nanmean(dy)),
        "d_iou_medio_ci": boot_ci(lambda ix: np.nanmean(dy[ix]), N, n_boot, seed),
        "rho_diou_dcov": spearman(dc, dy), "rho_diou_dcov_ci": boot_ci(lambda ix: spearman(dc[ix], dy[ix]), N, n_boot, seed),
        "rho_diou_dntrain": spearman(dn, dy), "rho_diou_dntrain_ci": boot_ci(lambda ix: spearman(dn[ix], dy[ix]), N, n_boot, seed),
        "rho_dcov_dntrain": spearman(dn, dc), "pearson_dcov_dntrain": r_cn,
        "rho_diou_dcov_dentro_pliegue": partial_spearman(dc, dy, g),
        "rho_diou_dcov_dentro_pliegue_ci": boot_ci(lambda ix: partial_spearman(dc[ix], dy[ix], g[ix]), N, n_boot, seed, groups=g),
        "n_valores_distintos_dntrain": int(len(np.unique(dn))),
        "ols": {"rank": rank, "coef": {"intercepto": float(beta[0]), "d_cov": float(beta[1]), "d_n_train": float(beta[2])},
                "ci": {"intercepto": ci(boots[:, 0]), "d_cov": ci(boots[:, 1]), "d_n_train": ci(boots[:, 2])},
                "unidades": "ΔIoU por punto porcentual de cobertura y por foto de train"},
    }


# ============================================================================ A3

def load_mapping(path: Path) -> dict:
    import yaml
    m = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for k in ("common", "mine", "mauricio"):
        if k not in m:
            raise ValueError(f"{path}: falta '{k}'")
    common = list(m["common"])
    m["unmapped_prediction"] = _policies(m.get("unmapped_prediction", "error"))
    for side in ("mine", "mauricio"):
        bad = {f: t for f, t in m[side].items() if t is not None and t not in common}
        if bad:
            raise ValueError(f"{path}: destinos que no están en 'common' ({side}): {bad}")
    for v in m.get("variants") or []:
        if "name" not in v:
            raise ValueError(f"{path}: variante sin 'name'")
        for side in ("mine", "mauricio"):
            for f, t in (v.get(side) or {}).items():
                if f not in m[side]:
                    raise ValueError(f"{path}: variante {v['name']}: clase desconocida {f!r} ({side})")
                if t is not None and t not in common:
                    raise ValueError(f"{path}: variante {v['name']}: destino {t!r} no está en 'common'")
    return m


def _policies(p) -> dict:
    """``unmapped_prediction`` como escalar (ambos lados) o {mine, mauricio}."""
    out = {"mine": p, "mauricio": p} if isinstance(p, str) else {"mine": "error", "mauricio": "error", **(p or {})}
    for side, v in out.items():
        if v not in ("error", "drop"):
            raise ValueError(f"unmapped_prediction[{side}] debe ser 'error' o 'drop', no {v!r}")
    return out


def apply_variant(mapping: dict, variant: dict) -> dict:
    """Esquema principal con los cambios de una variante (mapeos por lado y política de predicción)."""
    return {**mapping,
            "mine": {**mapping["mine"], **(variant.get("mine") or {})},
            "mauricio": {**mapping["mauricio"], **(variant.get("mauricio") or {})},
            "unmapped_prediction": {**mapping["unmapped_prediction"], **(variant.get("unmapped_prediction") or {})}}


def pred_into_null_pct(cm: np.ndarray, fine: list[str], mapping: dict) -> dict[str, float]:
    """% de píxeles etiquetados (real en una clase con destino) predichos como cada clase sin destino."""
    cm = np.asarray(cm, np.float64)
    if cm.ndim == 3:
        cm = cm.sum(0)
    rows = [i for i, c in enumerate(fine) if mapping[c] is not None]
    tot = cm[rows].sum()
    return {c: (float(100 * cm[rows, j].sum() / tot) if tot else float("nan"))
            for j, c in enumerate(fine) if mapping[c] is None}


def check_mapping_covers(mapping: dict, side: str, fine: list[str]) -> None:
    miss = [c for c in fine if c not in mapping[side]]
    extra = [c for c in mapping[side] if c not in fine]
    if miss or extra:
        raise ValueError(f"Mapeo '{side}': faltan {miss}; sobran {extra}")


def aggregate(cms: np.ndarray, fine: list[str], mapping: dict, common: list[str],
              unmapped_pred: str = "error") -> tuple[np.ndarray, list[str]]:
    """Agrega matrices (…, C, C) fino→grueso.

    Filas (real) de clases mapeadas a ``null``: se descartan (como píxeles ignorados).
    Columnas (predicha) de clases a ``null``: ``unmapped_pred='error'`` las lleva a una clase extra
    ``fuera_de_esquema`` (cuentan como error y no entran en mIoU); ``'drop'`` las descarta.
    """
    if unmapped_pred not in ("error", "drop"):
        raise ValueError("unmapped_pred debe ser 'error' o 'drop'")
    has_null = any(mapping[c] is None for c in fine)
    names = common + ([OUT_CLASS] if has_null and unmapped_pred == "error" else [])
    K = len(names)
    R = np.zeros((len(fine), K))     # filas
    Cm = np.zeros((len(fine), K))    # columnas
    for i, c in enumerate(fine):
        t = mapping[c]
        if t is not None:
            R[i, common.index(t)] = 1
            Cm[i, common.index(t)] = 1
        elif unmapped_pred == "error":
            Cm[i, K - 1] = 1
    cms = np.asarray(cms, np.float64)
    out = np.einsum("ik,...ij,jl->...kl", R, cms, Cm)
    return np.rint(out).astype(np.int64), names


def read_mauricio(xlsx: Path) -> dict:
    """Matriz de confusión agregada (filas real, columnas predicha) y tabla de IoU de Mauricio."""
    import pandas as pd
    cm = pd.read_excel(xlsx, sheet_name="Matriz_confusion", index_col=0)
    classes = [str(c) for c in cm.columns]
    if [str(i) for i in cm.index] != classes:
        raise ValueError("Matriz_confusion: filas y columnas no coinciden")
    iou = pd.read_excel(xlsx, sheet_name="IoU_por_clase")
    iou = iou[iou["Clase"].isin(classes)]
    cfg = pd.read_excel(xlsx, sheet_name="Config")
    return {"classes": classes, "cm": cm.to_numpy(dtype=np.int64),
            "iou_test": dict(zip(iou["Clase"], iou["IoU prueba"].astype(float))),
            "config": dict(zip(cfg.iloc[:, 0].astype(str), cfg.iloc[:, 1].astype(str)))}


def analysis_a3(runs: list[Run], classes: list[str], mapping: dict, mau: Optional[dict], K: int,
                n_boot: int = 2000, seed: int = 0) -> dict:
    """Esquema principal + variantes de sensibilidad (``variants`` del YAML)."""
    if not isinstance(mapping.get("unmapped_prediction"), dict) or set(mapping["unmapped_prediction"]) != {"mine", "mauricio"}:
        mapping = {**mapping, "unmapped_prediction": _policies(mapping.get("unmapped_prediction", "error"))}
    out = _a3_core(runs, classes, mapping, mau, K, n_boot, seed)
    out["variants"] = []
    for v in mapping.get("variants") or []:
        res = _a3_core(runs, classes, apply_variant(mapping, v), mau, K, n_boot, seed)
        out["variants"].append({"name": v["name"], "description": " ".join(str(v.get("description", "")).split()),
                                "result": res})
    return out


def _a3_core(runs: list[Run], classes: list[str], mapping: dict, mau: Optional[dict], K: int,
             n_boot: int, seed: int) -> dict:
    common = list(mapping["common"])
    pol = mapping["unmapped_prediction"]
    check_mapping_covers(mapping, "mine", classes)
    out = {"common": common, "policy": pol, "mine": {}, "mauricio": None,
           "mapping_mine": dict(mapping["mine"]), "mapping_mauricio": dict(mapping["mauricio"])}
    for c in configs_in(runs):
        rr = sorted([r for r in runs if r.config == c], key=lambda r: (r.seed, r.fold))
        seeds = sorted({r.seed for r in rr})
        # una matriz por foto (media entre semillas no aplica a conteos: se usa la primera semilla
        # y se reporta cuántas hay; con una semilla es exacto)
        s0 = [r for r in rr if r.seed == seeds[0]]
        cms = np.concatenate([r.cms for r in s0])
        names = [n for r in s0 for n in r.names]
        agg, cn = aggregate(cms, classes, mapping["mine"], common, pol["mine"])
        rep = M.evaluate(agg, cn, names, min_photos=K, n_boot=n_boot, seed=seed)
        rep["n_seeds_available"] = len(seeds)
        rep["pred_en_clases_sin_destino_pct"] = pred_into_null_pct(cms, classes, mapping["mine"])
        out["mine"][c] = rep
    if mau is not None:
        check_mapping_covers(mapping, "mauricio", mau["classes"])
        # verificación: IoU recalculado de la matriz = hoja IoU_por_clase (prueba)
        iou_fine = M.per_class(mau["cm"])["iou"]
        diffs = {n: abs(float(iou_fine[i]) - mau["iou_test"][n]) for i, n in enumerate(mau["classes"])
                 if n in mau["iou_test"]}
        agg, cn = aggregate(mau["cm"], mau["classes"], mapping["mauricio"], common, pol["mauricio"])
        pc = M.per_class(agg)
        ev = np.array([n != OUT_CLASS and pc["gt_px"][i] > 0 for i, n in enumerate(cn)])
        out["mauricio"] = {"classes": cn, "confusion": agg.tolist(), "summary": M.summarize(agg, ev),
                           "iou": {n: (None if not np.isfinite(pc["iou"][i]) else float(pc["iou"][i]))
                                   for i, n in enumerate(cn)},
                           "gt_px": {n: int(pc["gt_px"][i]) for i, n in enumerate(cn)},
                           "pred_en_clases_sin_destino_pct": pred_into_null_pct(mau["cm"], mau["classes"], mapping["mauricio"]),
                           "check_max_abs_diff_iou_fino": max(diffs.values()) if diffs else None,
                           "config": mau["config"]}
    # mIoU sobre las clases comunes evaluables en AMBOS lados
    if out["mauricio"] is not None:
        mau_ev = {n for n, v in out["mauricio"]["gt_px"].items() if v > 0 and n != OUT_CLASS}
        for c, rep in out["mine"].items():
            both = [x["class"] for x in rep["classes"] if x["evaluable"] and x["class"] in mau_ev]
            rep["mIoU_clases_compartidas"] = float(np.mean([x["iou"] for x in rep["classes"] if x["class"] in both]))
            rep["clases_compartidas"] = both
        out["mauricio"]["mIoU_clases_compartidas_por_config"] = {
            c: float(np.mean([out["mauricio"]["iou"][n] for n in rep["clases_compartidas"]]))
            for c, rep in out["mine"].items()}
    return out


# ============================================================================ costo / pliegues

def cost_table(runs: list[Run]) -> list[dict]:
    rows = []
    for c in configs_in(runs):
        rr = [r for r in runs if r.config == c]
        gpus = sorted({s.get("gpu", {}).get("name", "?") for r in rr for s in r.env.get("sessions", [])
                       if isinstance(s.get("gpu"), dict)})
        rows.append({"config": c, "corridas": len(rr),
                     "pasos_total": int(sum(r.done.get("optimizer_steps_total", 0) for r in rr)),
                     "pasos_por_pliegue": float(np.mean([r.done.get("optimizer_steps_total", 0) for r in rr])),
                     "pasos_al_mejor_por_pliegue": float(np.mean([r.done.get("optimizer_steps_at_best", 0) for r in rr])),
                     "epocas_por_pliegue": float(np.mean([r.done.get("epochs_trained", 0) for r in rr])),
                     "mejor_epoca_por_pliegue": float(np.mean([r.done.get("best_epoch", 0) for r in rr])),
                     "horas_gpu": float(sum(r.done.get("train_hours", 0.0) for r in rr)),
                     "gpu": ";".join(gpus) or "?"})
    return rows


def fold_table(runs: list[Run], classes: list[str], not_evaluable: dict[str, list[str]]) -> list[dict]:
    """mIoU por pliegue con el conjunto de clases evaluables del conjunto agrupado de cada config."""
    rows = []
    for r in sorted(runs, key=lambda r: (CONFIG_ORDER.index(r.config) if r.config in CONFIG_ORDER else 99, r.seed, r.fold)):
        ev = np.array([n not in not_evaluable.get(r.config, []) for n in classes])
        s = M.summarize(r.cms.sum(0), ev)
        rows.append({"config": r.config, "seed": r.seed, "fold": r.fold, "n_test": len(r.names),
                     "n_train": len(r.split["train"]), "mIoU": s["mIoU"], "OA": s["OA"],
                     "mIoU_done_json": r.done.get("test", {}).get("mIoU")})
    return rows


# ============================================================================ salida

def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v) for k, v in r.items()})
    return path


def _f(x, nd=1, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{(100 * x if pct else x):.{nd}f}"


def _ci(c, nd=2, pct=False):
    if not c or any(v is None or not math.isfinite(v) for v in c):
        return "—"
    return f"[{_f(c[0], nd, pct)}, {_f(c[1], nd, pct)}]"


def guard_out(out: Path, study: Path) -> None:
    o, r = out.resolve(), (study / "runs").resolve()
    if o == r or r in o.parents:
        raise SystemExit(f"--out no puede estar dentro de {r}")


def snapshot(d: Path) -> dict:
    return {str(p.relative_to(d)): (p.stat().st_size, p.stat().st_mtime_ns) for p in d.rglob("*") if p.is_file()}


def run(study: Path, manifest: Path, out: Path, dataset_cfg: Path, mapping_path: Optional[Path] = None,
        mauricio: Optional[Path] = None, n_boot: int = 2000, seed: int = 0, source: str = "",
        figures: bool = True, log=print) -> dict:
    from . import report as RP
    from .data_io import load_dataset_config

    guard_out(out, study)
    out.mkdir(parents=True, exist_ok=True)
    tables = out / "tables"
    before = snapshot(study / "runs")
    params = json.loads((study / "study.json").read_text(encoding="utf-8"))["params"]
    classes = params["classes"]
    dcfg = load_dataset_config(dataset_cfg)
    geom = dcfg["geometry"]
    checks = []
    if params.get("camera") and params["camera"] != geom.get("camera"):
        checks.append(("Cámara de study.json = cámara del YAML del dataset", False, f"{params['camera']} vs {geom.get('camera')}"))
    else:
        checks.append(("Cámara de study.json = cámara del YAML del dataset", True, ""))

    runs, inv = load_runs(study)
    if not runs:
        raise SystemExit(f"No hay corridas terminadas en {study / 'runs'}")
    cfgs = configs_in(runs)
    log(f"Corridas leídas: {len(runs)} ({', '.join(cfgs)})")
    geo = load_geometry(manifest, geom)
    test_photos = sorted({n for r in runs for n in r.names})
    missing_geo = [n for n in test_photos if n not in geo.idx]
    checks.append(("Todas las fotos de test están en el manifest", not missing_geo, ", ".join(missing_geo[:5])))
    if missing_geo:
        raise SystemExit(f"Fotos de test que no están en el manifest: {missing_geo[:5]}")

    # --- tabla comparativa recalculada (misma función que el informe de la cola)
    log("Recalculando la tabla comparativa …")
    rep = RP.build(study, cfgs, n_boot=n_boot, seed=seed, out_dir=tables / "comparacion_recalculada")
    K = rep["K"]
    old = study / "results" / "summary.csv"
    if old.exists():
        prev = {r["config"]: float(r["mIoU"]) for r in _read_csv(old)}
        diffs = [abs(prev[r["config"]] - r["mIoU"]) for r in rep["summary"] if r["config"] in prev]
        checks.append(("mIoU recalculado = results/summary.csv", bool(diffs) and max(diffs) < 1e-9,
                       f"máx |Δ| = {max(diffs):.2e} en {len(diffs)} configs" if diffs else "sin configs comunes"))
    not_ev = {r["config"]: r["not_evaluable"] for r in rep["summary"]}

    # --- A1
    log("A1: línea base de fuga …")
    a1 = analysis_a1(geo, geo.names, runs)
    write_csv(tables / "A1_solape_por_pliegue.csv", a1["rows"])
    write_csv(tables / "A1_solape_resumen.csv", a1["summary"])
    write_csv(tables / "A1_split_aleatorio.csv", [{"photo": n, "particion": "train"} for n in a1["random_train"]]
              + [{"photo": n, "particion": "test"} for n in a1["random_test"]])
    if a1["max_diff_vs_guardado_pp"] is not None:
        checks.append(("Solape recalculado = solape guardado en split.json", a1["max_diff_vs_guardado_pp"] < 1e-6,
                       f"máx |Δ| = {a1['max_diff_vs_guardado_pp']:.2e} pp"))

    # --- A2
    log("A2: fuga residual …")
    table = per_photo_table(runs, geo)
    write_csv(tables / "A2_iou_vs_cobertura_por_foto.csv", table)
    a2 = analysis_a2(table, n_boot, seed)
    write_csv(tables / "A2_spearman.csv", a2["correlations"])
    pair = analysis_pair(table, "J0", "J0-b75", n_boot, seed) if {"J0", "J0-b75"} <= set(cfgs) else None
    if pair:
        write_csv(tables / "A2_pareado_J0_vs_J0-b75_por_foto.csv", pair["rows"])
        write_csv(tables / "A2_pareado_J0_vs_J0-b75_por_pliegue.csv", pair["folds"])
        (tables / "A2_pareado_J0_vs_J0-b75.json").write_text(
            json.dumps({k: v for k, v in pair.items() if k != "rows"}, indent=1, ensure_ascii=False), encoding="utf-8")

    # --- A3
    a3, a3_status = None, "sin mapeo (--mapping)"
    if mapping_path:
        mp = load_mapping(mapping_path)
        if not mp.get("approved"):
            a3_status = f"mapeo {mapping_path.name} sin aprobar (approved: false): no se calcula"
        else:
            mau = read_mauricio(mauricio) if mauricio else None
            a3 = analysis_a3(runs, classes, mp, mau, K, n_boot, seed)
            a3_status = "calculado"
            (tables / "A3_esquema_comun.json").write_text(json.dumps(a3, indent=1, ensure_ascii=False), encoding="utf-8")
            write_csv(tables / "A3_esquema_comun_por_clase.csv", _a3_rows(a3))
            for v in a3["variants"]:
                write_csv(tables / f"A3_sensibilidad_{v['name']}_por_clase.csv", _a3_rows(v["result"]))
            if mau is not None and a3["mauricio"]["check_max_abs_diff_iou_fino"] is not None:
                d = a3["mauricio"]["check_max_abs_diff_iou_fino"]
                checks.append(("IoU fino de Mauricio recalculado de su matriz = hoja IoU_por_clase", d < 5e-4,
                               f"máx |Δ| = {d:.1e} (la hoja redondea a 4 decimales)"))
    log(f"A3: {a3_status}")

    costs = cost_table(runs)
    write_csv(tables / "costo.csv", costs)
    folds = fold_table(runs, classes, not_ev)
    write_csv(tables / "mIoU_por_pliegue.csv", folds)

    figs = []
    if figures:
        from . import figures as F
        log("A4: figuras …")
        figs = F.make_all(out / "figures", rep, runs, classes, folds, costs, a1, table, a2, pair, a3)

    inv_rows = write_csv(tables / "inventario_corridas.csv", inv)
    after = snapshot(study / "runs")
    checks.append(("runs/ sin cambios (tamaño y fecha de cada archivo)", before == after, f"{len(after)} archivos"))
    result = {"study": str(study), "source": source, "params": params, "configs": cfgs, "n_runs": len(runs),
              "inventory": inv, "report": rep, "a1": a1, "a2": a2, "pair": pair, "a3": a3, "a3_status": a3_status,
              "costs": costs, "folds": folds, "checks": checks, "figures": figs, "K": K, "n_boot": n_boot,
              "manifest": str(manifest), "mapping": str(mapping_path) if mapping_path else None,
              "mauricio": str(mauricio) if mauricio else None}
    (out / "REPORT.md").write_text(to_report(result, out), encoding="utf-8")
    log(f"Listo: {out / 'REPORT.md'}")
    return result


def _a3_rows(a3: dict) -> list[dict]:
    rows = []
    for c, rep in a3["mine"].items():
        for x in rep["classes"]:
            rows.append({"lado": "propio (Pascual)", "config": c, "clase": x["class"], "evaluable": x["evaluable"],
                         "n_fotos": x["n_photos"], "gt_px": x["gt_px"], "iou": x["iou"],
                         "iou_ci_lo": (x["iou_ci"] or [None, None])[0], "iou_ci_hi": (x["iou_ci"] or [None, None])[1]})
    if a3["mauricio"]:
        for n in a3["mauricio"]["classes"]:
            rows.append({"lado": "Mauricio (otro dataset)", "config": "run05_b3_v2", "clase": n,
                         "evaluable": a3["mauricio"]["gt_px"][n] > 0 and n != OUT_CLASS, "n_fotos": None,
                         "gt_px": a3["mauricio"]["gt_px"][n], "iou": a3["mauricio"]["iou"][n],
                         "iou_ci_lo": None, "iou_ci_hi": None})
    return rows


# ============================================================================ REPORT.md

def _a3_table(a3: dict) -> list[str]:
    cfgs = list(a3["mine"])
    L = ["| Clase común | " + " | ".join(f"{c} (Pascual) [IC 95 %]" for c in cfgs)
         + " | Mauricio run05_b3_v2 (otro dataset, sin IC) |", "|---|" + "---|" * (len(cfgs) + 1)]
    for n in a3["common"]:
        cells = []
        for c in cfgs:
            x = next(x for x in a3["mine"][c]["classes"] if x["class"] == n)
            cells.append(f"{_f(x['iou'], 1, True)} {_ci(x['iou_ci'], 1, True)}" + ("" if x["evaluable"] else "*"))
        mv = a3["mauricio"]["iou"].get(n) if a3["mauricio"] else None
        L.append(f"| {n} | " + " | ".join(cells) + f" | {_f(mv, 1, True)} |")
    L.append("| **mIoU (evaluables)** | " + " | ".join(
        f"{_f(a3['mine'][c]['mIoU'], 1, True)} {_ci((a3['mine'][c]['ci'] or {}).get('mIoU'), 1, True)}" for c in cfgs)
        + f" | {_f(a3['mauricio']['summary']['mIoU'], 1, True) if a3['mauricio'] else '—'} |")
    if a3["mauricio"]:
        L.append("| **mIoU (clases evaluables en ambos lados)** | " + " | ".join(
            _f(a3["mine"][c]["mIoU_clases_compartidas"], 1, True) for c in cfgs) + " | " +
            " / ".join(f"{c}: {_f(v, 1, True)}" for c, v in a3["mauricio"]["mIoU_clases_compartidas_por_config"].items()) + " |")
        L.append(f"| OA | " + " | ".join(_f(a3["mine"][c]["OA"], 1, True) for c in cfgs)
                 + f" | {_f(a3['mauricio']['summary']['OA'], 1, True)} |")
    L += ["", "\\* = no evaluable en el lado propio (< K fotos). Valores en %. En la última fila de mIoU, la columna de "
          "Mauricio se calcula sobre las mismas clases que cada config propia."]
    return L


def to_report(R: dict, out: Path) -> str:
    p, rep = R["params"], R["report"]
    L = ["# Análisis posterior del estudio", "",
         f"Generado: {time.strftime('%Y-%m-%d %H:%M')} con `python -m segpipe.analyze`. "
         "Este documento contiene método y números; no contiene interpretación.", "",
         "## 1. Fuente de datos e inventario", "",
         f"- Estudio: `{R['study']}` (fuente: {R['source'] or '—'}).",
         f"- Manifest (huellas): `{R['manifest']}`.",
         f"- Dataset: {p.get('dataset')} {p.get('dataset_version')}; protocolo {p['protocol']} k={p['k']}, "
         f"bloques {p['block_m']} m, buffer del estudio {p['buffer_m']} m, huella `{p.get('footprint')}`.",
         f"- Clases ({len(p['classes'])}): {', '.join(p['classes'])}; ignoradas: {', '.join(p.get('ignored') or []) or '—'}.",
         f"- Corridas terminadas leídas: {R['n_runs']} ({', '.join(R['configs'])}).",
         f"- Bootstrap: {R['n_boot']} remuestreos de fotos, IC percentil 95 %. K (fotos mínimas por clase evaluable) = {R['K']}.",
         "", "Archivos por corrida (✓ presente):", "",
         "| Config | Semilla | Pliegue | DONE | split | config | epoch_log | test npz | metrics.json | per_class | best.pt | last.pt |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    mk = lambda b: "✓" if b else "✗"
    for i in R["inventory"]:
        L.append(f"| {i['config']} | {i['seed']} | {i['fold']} | {mk(i['DONE.json'])} | {mk(i['split.json'])} | "
                 f"{mk(i['config.json'])} | {mk(i['epoch_log.csv'])} | {mk(i['test/photo_confusion.npz'])} | "
                 f"{mk(i['test/metrics.json'])} | {mk(i['test/metrics_per_class.csv'])} | {mk(i['best.pt'])} | {mk(i['last.pt'])} |")
    L += ["", "(best.pt/last.pt no se usan en el análisis; en el paquete ligero se excluyen.)", "",
          "### Verificaciones automáticas", "", "| Verificación | Resultado | Detalle |", "|---|---|---|"]
    for name, ok, det in R["checks"]:
        L.append(f"| {name} | {'OK' if ok else 'FALLA'} | {det} |")

    # comparación recalculada
    L += ["", "## 2. Tabla comparativa (recalculada con `segpipe.report.build`)", "",
          "Test agrupado de los 5 pliegues (cada foto se evalúa una vez). Valores en %, [IC 95 %].", "",
          "| Config | mIoU | mAcc | mF1 | OA | No evaluables |", "|---|---|---|---|---|---|"]
    for r in rep["summary"]:
        L.append(f"| {r['config']} | " + " | ".join(f"{_f(r[k], 1, True)} {_ci(r[k + '_ci'], 1, True)}" for k in ("mIoU", "mAcc", "mF1", "OA"))
                 + f" | {', '.join(r['not_evaluable']) or '—'} |")
    L += ["", "| De → a | ΔmIoU (pp) | IC 95 % (bootstrap pareado) | P(Δ ≤ 0) |", "|---|---|---|---|"]
    for d in rep["deltas"]:
        L.append(f"| {d['from']} → {d['to']} | {100 * d['delta_mIoU']:+.1f} | [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}] | {d['p_le_0']:.3f} |")

    # A1
    a1 = R["a1"]
    L += ["", "## 3. A1 — Línea base de fuga", "",
          "Método: huella de cada foto = 4 esquinas de la imagen proyectadas a un plano a `RelativeAltitude` "
          "bajo el dron con la cámara calibrada (`segpipe.split.footprint_polygons`), rasterizadas a 2 m. "
          "Solape train→test = % del área de la unión de huellas de test cubierta por la unión de huellas de train "
          "(`FootprintRaster.covered_pct`). Cobertura por foto = % de la huella de cada foto de test cubierta por la "
          "unión de huellas de train. Distancia = mínima entre centroides de huella train y test.", "",
          "Split aleatorio original: nombres de las fotos en orden alfabético, "
          "`sklearn.model_selection.train_test_split(train_size=0.8, random_state=42, shuffle=True)` "
          "(reimplementado sin sklearn: `RandomState(42).permutation`, test = primeros 14, train = 52). "
          "Hecho confirmado por el autor: el notebook original usó estas mismas 66 fotos (segpack_v1).", "",
          "| Esquema | Pliegue | Train | Test | Solape unión (%) | Cobertura por foto mediana (%) | Dist. mín. (m) |",
          "|---|---|---|---|---|---|---|"]
    for r in a1["rows"]:
        L.append(f"| {r['esquema']} | {r['fold']} | {r['n_train']} | {r['n_test']} | {_f(r['solape_union_pct'])} | "
                 f"{_f(r['cobertura_foto_mediana_pct'])} | {_f(r['dist_min_train_test_m'])} |")
    L += ["", "| Esquema | Pliegues | Train medio | Solape unión medio (%) | Mín–máx (%) |", "|---|---|---|---|---|"]
    for s in a1["summary"]:
        L.append(f"| {s['esquema']} | {s['n_pliegues']} | {_f(s['n_train_medio'])} | {_f(s['solape_union_medio_pct'])} | "
                 f"{_f(s['solape_union_min_pct'])}–{_f(s['solape_union_max_pct'])} |")

    # A2
    a2, pair = R["a2"], R["pair"]
    L += ["", "## 4. A2 — Fuga residual", "",
          "Método: para cada config y foto de test, IoU medio de la foto = media del IoU de las clases con unión "
          "(real ∪ predicha) > 0 en esa foto, desde `test/photo_confusion.npz` (las 12 clases, sin el criterio K). "
          "Cobertura = % de la huella de la foto cubierta por la unión de huellas del train **de su pliegue y su config** "
          "(J0-b75 usa su propio train). ρ = Spearman; IC = bootstrap de fotos. ρ parcial por pliegue = Pearson entre "
          "los rangos globales tras restar la media de rango de cada pliegue; su IC remuestrea fotos dentro de cada pliegue.", "",
          "| Config | Fotos | ρ | IC 95 % | ρ parcial (pliegue) | IC 95 % | ρ por pliegue |", "|---|---|---|---|---|---|---|"]
    for c in a2["correlations"]:
        pf = ", ".join(f"{f}: {_f(v, 2)}" for f, v in c["rho_por_pliegue"].items())
        L.append(f"| {c['config']} | {c['n_fotos']} | {_f(c['rho'], 2)} | {_ci(c['rho_ci'])} | "
                 f"{_f(c['rho_parcial_pliegue'], 2)} | {_ci(c['rho_parcial_ci'])} | {pf} |")
    if pair:
        o = pair["ols"]
        L += ["", f"### Pareado por foto: {pair['b']} − {pair['a']} ({pair['n_fotos']} fotos de test, las mismas en ambas)", "",
              "Δ = valor en J0-b75 menos valor en J0. ΔIoU = cambio del IoU medio de la foto; Δcobertura en puntos "
              "porcentuales; Δn_train = cambio del número de fotos de train del pliegue (constante dentro de cada pliegue). "
              "Ambas configs tienen el mismo número de muestras por época y de épocas máximas.", "",
              "| Pliegue | Fotos | Train J0 | Train J0-b75 | Δn_train | ΔIoU medio (pp) | Δcobertura media (pp) |",
              "|---|---|---|---|---|---|---|"]
        for f in pair["folds"]:
            L.append(f"| {f['fold']} | {f['n_fotos']} | {f['n_train_a']} | {f['n_train_b']} | {f['d_n_train']:+.0f} | "
                     f"{100 * f['d_iou_medio']:+.2f} | {f['d_cov_medio']:+.1f} |")
        L += ["", "| Estadístico | Valor | IC 95 % |", "|---|---|---|",
              f"| ΔIoU medio (pp) | {100 * pair['d_iou_medio']:+.2f} | {_ci([100 * v for v in pair['d_iou_medio_ci']])} |",
              f"| ρ(ΔIoU, Δcobertura) | {_f(pair['rho_diou_dcov'], 2)} | {_ci(pair['rho_diou_dcov_ci'])} |",
              f"| ρ(ΔIoU, Δn_train) | {_f(pair['rho_diou_dntrain'], 2)} | {_ci(pair['rho_diou_dntrain_ci'])} |",
              f"| ρ(ΔIoU, Δcobertura) dentro de pliegue (Δn_train fijo) | {_f(pair['rho_diou_dcov_dentro_pliegue'], 2)} | {_ci(pair['rho_diou_dcov_dentro_pliegue_ci'])} |",
              f"| ρ(Δcobertura, Δn_train) | {_f(pair['rho_dcov_dntrain'], 2)} | — |",
              f"| r de Pearson(Δcobertura, Δn_train) | {_f(pair['pearson_dcov_dntrain'], 2)} | — |",
              f"| Valores distintos de Δn_train | {pair['n_valores_distintos_dntrain']} | — |", "",
              f"MCO ΔIoU ~ 1 + Δcobertura + Δn_train (rango de la matriz de diseño = {o['rank']}; IC por bootstrap de fotos):", "",
              "| Término | Coeficiente | IC 95 % |", "|---|---|---|"]
        for k in ("intercepto", "d_cov", "d_n_train"):
            L.append(f"| {k} | {o['coef'][k]:+.5f} | {_ci(o['ci'][k], 5)} |")
        L += ["", "Δn_train solo toma tantos valores como pliegues; su efecto no se distingue de cualquier otra diferencia "
              "entre pliegues. La fila \"dentro de pliegue\" mantiene Δn_train fijo."]

    # A3
    L += ["", "## 5. A3 — Esquema común con Mauricio", "", f"Estado: {R['a3_status']}."]
    a3 = R["a3"]
    if a3:
        pol = a3["policy"]
        L += ["", f"Mapeo: `{R['mapping']}` (aprobado por el autor). Clases comunes: {', '.join(a3['common'])}. "
              "Agregación: suma de filas (real) y columnas (predicha) de la matriz por foto según el mapeo; una clase con "
              "destino `null` no entra: sus píxeles reales se ignoran y lo predicho en ella sobre píxeles etiquetados va a "
              f"`{OUT_CLASS}` (cuenta como error; política `error`) o se descarta (política `drop`). Política: propio "
              f"`{pol['mine']}`, Mauricio `{pol['mauricio']}`. mIoU propio sobre clases evaluables (≥ {R['K']} fotos), "
              "IC por bootstrap de fotos.", "",
              "Verificación del mapeo de Mauricio: por inspección visual de sus máscaras (su `via_pavimentada` incluye vías "
              "pavimentadas, andenes y plazas); no existe una definición escrita de sus clases.", "",
              "**Datasets distintos**: el lado propio es Pascual Guerrero (66 fotos, CV por bloques, test agrupado de 5 pliegues); "
              "el de Mauricio es su dataset v2 (38 fotos de prueba de una zona no vista, una sola partición). "
              "La matriz de Mauricio es agregada (no hay matrices por foto): su columna no tiene IC.", ""]
        L += _a3_table(a3)
        if a3["mauricio"]:
            pct = a3["mauricio"]["pred_en_clases_sin_destino_pct"]
            L += ["", "Píxeles etiquetados (real en una clase con destino) que el modelo de Mauricio predijo en una clase sin destino: "
                  + ", ".join(f"`{c}` {_f(v, 2)} %" for c, v in pct.items()) + ". "
                  "Mis modelos no tienen esa salida: predicen solo las 12 clases de `classes.json` (background es "
                  "`ignore_index` y nunca se predice), y en el esquema principal las 12 tienen destino."]
        for v in a3.get("variants") or []:
            res = v["result"]
            L += ["", f"### Sensibilidad: {v['name']}", "", v["description"], ""]
            L += _a3_table(res)
            mine_pct = {c: rep.get("pred_en_clases_sin_destino_pct") or {} for c, rep in res["mine"].items()}
            if any(mine_pct.values()):
                L += ["", "Píxeles etiquetados que mis modelos predijeron en clases sin destino: " + "; ".join(
                    f"{c}: " + ", ".join(f"`{k}` {_f(x, 2)} %" for k, x in d.items()) for c, d in mine_pct.items()) + "."]
            if res["mauricio"]:
                L += ["", "Mauricio en esta variante: " + ", ".join(
                    f"`{k}` {_f(x, 2)} % de los píxeles etiquetados"
                    for k, x in res["mauricio"]["pred_en_clases_sin_destino_pct"].items())
                    + f" (política `{res['policy']['mauricio']}`)."]

    # costo y pliegues
    L += ["", "## 6. Costo", "", "Horas = suma de los segundos por época de `epoch_log` (entrenamiento + validación), "
          "sin la evaluación final. Pasos = pasos del optimizador.", "",
          "| Config | Corridas | Pasos totales | Pasos por pliegue | Pasos al mejor (media) | Épocas por pliegue | Mejor época (media) | Horas GPU | GPU |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in R["costs"]:
        L.append(f"| {c['config']} | {c['corridas']} | {c['pasos_total']} | {_f(c['pasos_por_pliegue'], 0)} | "
                 f"{_f(c['pasos_al_mejor_por_pliegue'], 0)} | {_f(c['epocas_por_pliegue'])} | {_f(c['mejor_epoca_por_pliegue'])} | "
                 f"{_f(c['horas_gpu'], 2)} | {c['gpu']} |")
    L += ["", "## 7. mIoU por pliegue", "",
          "Clases evaluables = las del test agrupado de cada config (no las del pliegue), para que los pliegues sean comparables.", "",
          "| Config | " + " | ".join(f"Pliegue {f}" for f in sorted({r['fold'] for r in R['folds']})) + " |",
          "|---|" + "---|" * len({r['fold'] for r in R['folds']})]
    for c in R["configs"]:
        L.append(f"| {c} | " + " | ".join(_f(r["mIoU"], 1, True) for r in R["folds"] if r["config"] == c and r["seed"] == min(x["seed"] for x in R["folds"] if x["config"] == c)) + " |")

    L += ["", "## 8. Figuras y tablas", ""]
    for f in R["figures"]:
        L.append(f"- `{Path(f).relative_to(out).as_posix()}` (+ .pdf, .csv)")
    for t in sorted((out / "tables").rglob("*")):
        if t.is_file():
            L.append(f"- `{t.relative_to(out).as_posix()}`")
    return "\n".join(L) + "\n"


# ============================================================================ CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m segpipe.analyze", description=__doc__.split("\n")[0])
    ap.add_argument("--study", help="carpeta del estudio (o SEGPIPE_STUDY)")
    ap.add_argument("--manifest", default=os.environ.get("SEGPIPE_MANIFEST"),
                    help="manifest.csv del dataset (o SEGPIPE_MANIFEST)")
    ap.add_argument("--dataset-config", default=str(REPO / "configs" / "datasets" / "pascual.yaml"))
    ap.add_argument("--out", help="carpeta de salida (por defecto <study>/analysis)")
    ap.add_argument("--mapping", help="YAML del esquema común (A3; requiere approved: true)")
    ap.add_argument("--mauricio", default=os.environ.get("SEGPIPE_MAURICIO_XLSX"), help="xlsx de métricas de Mauricio")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    study, source = find_study(a.study)
    if not (study / "study.json").exists():
        raise SystemExit(f"{study} no tiene study.json")
    if not a.manifest:
        raise SystemExit("Falta --manifest (o SEGPIPE_MANIFEST): se necesita para las huellas")
    print(f"Estudio: {study} (fuente: {source})")
    run(study, Path(a.manifest), Path(a.out) if a.out else study / "analysis", Path(a.dataset_config),
        Path(a.mapping) if a.mapping else None, Path(a.mauricio) if a.mauricio else None,
        a.n_boot, a.seed, source, not a.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
