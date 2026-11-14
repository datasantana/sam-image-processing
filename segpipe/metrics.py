"""Métricas desde matrices de confusión a resolución nativa.

- La matriz se acumula por foto (filas = clase real, columnas = predicha) excluyendo los
  píxeles ignorados (``ignore_index``, p. ej. background sin anotar).
- Una clase presente en menos de ``min_photos`` fotos de la partición evaluada se marca
  "no evaluable" y se excluye de mIoU / mAcc / mF1 (queda explícito en la salida).
- IC por bootstrap remuestreando FOTOS (no píxeles). El conjunto de clases evaluables se fija
  en la muestra original; en cada remuestreo, una clase sin píxeles reales ni predichos aporta NaN
  y se promedia con nanmean.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np


def confusion(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    """Matriz (C, C) int64 con filas = real y columnas = predicha."""
    pred = np.asarray(pred).ravel()
    target = np.asarray(target).ravel()
    keep = target != ignore_index
    t = target[keep].astype(np.int64)
    p = pred[keep].astype(np.int64)
    if t.size and (t.max() >= num_classes or p.max() >= num_classes or p.min() < 0):
        raise ValueError("Índice de clase fuera de rango en confusion()")
    return np.bincount(t * num_classes + p, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def confusion_torch(pred, target, num_classes: int, ignore_index: int = 255):
    """Versión en GPU (torch) de ``confusion``; devuelve tensor int64 (C, C)."""
    import torch
    keep = target != ignore_index
    t = target[keep].long()
    p = pred[keep].long()
    return torch.bincount(t * num_classes + p, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def per_class(cm: np.ndarray) -> dict[str, np.ndarray]:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    gt = cm.sum(1)
    pr = cm.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = tp / (gt + pr - tp)
        recall = tp / gt
        precision = tp / pr
        f1 = 2 * tp / (gt + pr)
    return {"iou": iou, "acc": recall, "precision": precision, "f1": f1, "gt_px": gt, "pred_px": pr}


def summarize(cm: np.ndarray, evaluable: Optional[np.ndarray] = None) -> dict[str, float]:
    """mIoU, mAcc, mF1 sobre clases evaluables; OA sobre todos los píxeles no ignorados."""
    pc = per_class(cm)
    C = cm.shape[0]
    ev = np.ones(C, bool) if evaluable is None else np.asarray(evaluable, bool)
    total = cm.sum()
    return {
        "mIoU": float(np.nanmean(pc["iou"][ev])) if ev.any() else float("nan"),
        "mAcc": float(np.nanmean(pc["acc"][ev])) if ev.any() else float("nan"),
        "mF1": float(np.nanmean(pc["f1"][ev])) if ev.any() else float("nan"),
        "OA": float(np.trace(cm) / total) if total else float("nan"),
    }


def photos_per_class(photo_cms: np.ndarray) -> np.ndarray:
    """Nº de fotos en que cada clase aparece en la verdad de terreno."""
    return (photo_cms.sum(axis=2) > 0).sum(axis=0)


def evaluable_mask(photo_cms: np.ndarray, min_photos: int = 5) -> np.ndarray:
    return photos_per_class(photo_cms) >= min_photos


def bootstrap(photo_cms: np.ndarray, evaluable: np.ndarray, n_boot: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> dict:
    """IC percentil por bootstrap de fotos. ``photo_cms``: (N, C, C)."""
    rng = np.random.default_rng(seed)
    N, C, _ = photo_cms.shape
    keys = ("mIoU", "mAcc", "mF1", "OA")
    stats = {k: np.empty(n_boot) for k in keys}
    cls_iou = np.empty((n_boot, C))
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        cm = photo_cms[idx].sum(0)
        s = summarize(cm, evaluable)
        for k in keys:
            stats[k][b] = s[k]
        cls_iou[b] = per_class(cm)["iou"]
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {k: [float(np.nanpercentile(v, lo)), float(np.nanpercentile(v, hi))] for k, v in stats.items()}
    with np.errstate(all="ignore"):
        out["class_iou"] = [[float(np.nanpercentile(cls_iou[:, c], lo)) if np.isfinite(cls_iou[:, c]).any() else None,
                             float(np.nanpercentile(cls_iou[:, c], hi)) if np.isfinite(cls_iou[:, c]).any() else None]
                            for c in range(C)]
    out["n_boot"] = n_boot
    out["alpha"] = alpha
    return out


def evaluate(photo_cms: np.ndarray, class_names: list[str], photo_names: Optional[list[str]] = None,
             min_photos: int = 5, n_boot: int = 2000, seed: int = 0) -> dict:
    """Informe completo a partir de matrices por foto (N, C, C)."""
    photo_cms = np.asarray(photo_cms, dtype=np.int64)
    cm = photo_cms.sum(0)
    npc = photos_per_class(photo_cms)
    ev = npc >= min_photos
    pc = per_class(cm)
    summ = summarize(cm, ev)
    ci = bootstrap(photo_cms, ev, n_boot=n_boot, seed=seed) if n_boot > 0 else None

    def f(x):
        return None if x is None or not np.isfinite(x) else float(x)

    classes = []
    for c, name in enumerate(class_names):
        classes.append({"class": name, "evaluable": bool(ev[c]), "n_photos": int(npc[c]),
                        "gt_px": int(pc["gt_px"][c]), "iou": f(pc["iou"][c]), "acc": f(pc["acc"][c]),
                        "precision": f(pc["precision"][c]), "f1": f(pc["f1"][c]),
                        "iou_ci": ci["class_iou"][c] if ci else None})
    return {
        **summ,
        "ci": {k: ci[k] for k in ("mIoU", "mAcc", "mF1", "OA")} if ci else None,
        "bootstrap": {"n_boot": n_boot, "alpha": 0.05, "unit": "photo", "seed": seed},
        "min_photos": min_photos,
        "n_photos": int(photo_cms.shape[0]),
        "evaluable_classes": [n for n, e in zip(class_names, ev) if e],
        "not_evaluable_classes": [n for n, e in zip(class_names, ev) if not e],
        "note": (f"mIoU/mAcc/mF1 promedian solo clases presentes en >= {min_photos} fotos; "
                 f"no evaluables: {[n for n, e in zip(class_names, ev) if not e] or 'ninguna'}. "
                 "OA usa todos los píxeles anotados."),
        "classes": classes,
        "confusion": cm.tolist(),
        "photos": photo_names,
    }


def save_report(report: dict, out_dir: str | Path, prefix: str = "metrics") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jp = out / f"{prefix}.json"
    jp.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    cp = out / f"{prefix}_per_class.csv"
    with open(cp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "evaluable", "n_photos", "gt_px", "iou", "iou_ci_lo", "iou_ci_hi",
                    "acc", "precision", "f1"])
        for c in report["classes"]:
            lo, hi = (c["iou_ci"] or [None, None])
            w.writerow([c["class"], c["evaluable"], c["n_photos"], c["gt_px"], c["iou"], lo, hi,
                        c["acc"], c["precision"], c["f1"]])
        for k in ("mIoU", "mAcc", "mF1", "OA"):
            lo, hi = (report["ci"] or {}).get(k, [None, None])
            w.writerow([f"[{k}]", "", report["n_photos"], "", report[k], lo, hi, "", "", ""])
    return jp, cp
