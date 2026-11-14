"""Tabla comparativa entre configs (media ± DE entre semillas, IC por bootstrap de fotos, por clase).

Para cada config y semilla se juntan las matrices de confusión por foto del test de todos los
pliegues (con CV cada foto se evalúa exactamente una vez). Las clases evaluables se determinan
sobre ese conjunto (≥ K fotos). El IC de la media entre semillas se obtiene remuestreando fotos
con los mismos índices para todas las semillas; las diferencias entre configs usan bootstrap
pareado (mismas fotos).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import metrics as M

KEYS = ("mIoU", "mAcc", "mF1", "OA")


def collect(study_dir: str | Path, configs: Iterable[str]) -> dict:
    """{config: {seed: (cms (N,C,C), names)}} solo para semillas con todos los pliegues terminados."""
    study_dir = Path(study_dir)
    params = json.loads((study_dir / "study.json").read_text(encoding="utf-8"))["params"]
    n_folds = params["k"] if params["protocol"] == "cv" else 1
    out, missing = {}, []
    for c in configs:
        cdir = study_dir / "runs" / c
        for sdir in sorted(cdir.glob("seed*")) if cdir.exists() else []:
            seed = int(sdir.name[4:])
            parts, dones = [], []
            for f in range(n_folds):
                p = sdir / f"fold{f}" / "test" / "photo_confusion.npz"
                if not (sdir / f"fold{f}" / "DONE.json").exists() or not p.exists():
                    missing.append(f"{c}/seed{seed}/fold{f}")
                    break
                z = np.load(p, allow_pickle=False)
                parts.append((z["cms"], [str(n) for n in z["names"]]))
                dones.append(json.loads((sdir / f"fold{f}" / "DONE.json").read_text(encoding="utf-8")))
            else:
                cms = np.concatenate([a for a, _ in parts])
                names = [n for _, ns in parts for n in ns]
                order = np.argsort(names)
                out.setdefault(c, {})[seed] = (cms[order], [names[i] for i in order], dones)
    return {"runs": out, "missing": missing, "params": params}


def _metric_seed_mean(cms_by_seed: list[np.ndarray], idx: np.ndarray, ev: np.ndarray) -> dict:
    vals = {k: [] for k in KEYS}
    ious = []
    for cms in cms_by_seed:
        cm = cms[idx].sum(0)
        s = M.summarize(cm, ev)
        for k in KEYS:
            vals[k].append(s[k])
        ious.append(M.per_class(cm)["iou"])
    return {**{k: float(np.mean(v)) for k, v in vals.items()}, "class_iou": np.nanmean(np.stack(ious), 0)}


LADDER = ["J0", "J1", "J2", "J3"]


def default_pairs(configs: list[str], reference: str = "J0") -> list[tuple[str, str]]:
    """Escalera consecutiva (J0→J1→J2→J3), cada extra vs J0 y el control de pasos J0-pasos → J1."""
    lad = [c for c in LADDER if c in configs]
    pairs = list(zip(lad, lad[1:]))
    for c in configs:
        if c not in lad and reference in configs:
            pairs.append((reference, c))
    if "J0-pasos" in configs and "J1" in configs:
        pairs.append(("J0-pasos", "J1"))   # mismo presupuesto de pasos: aísla el efecto de la resolución
    if "M-b3" in configs and "J3" in configs:
        pairs.append(("J3", "M-b3"))
    return list(dict.fromkeys(pairs))


def build(study_dir: str | Path, configs: Iterable[str], reference: Optional[str] = "J0",
          n_boot: int = 2000, seed: int = 0, out_dir: Optional[str | Path] = None,
          pairs: Optional[list[tuple[str, str]]] = None) -> dict:
    study_dir = Path(study_dir)
    configs = list(configs)
    col = collect(study_dir, configs)
    runs, params = col["runs"], col["params"]
    classes = params["classes"]
    K = None
    rows, cls_rows, deltas = [], [], []
    boot_cache = {}
    rng = np.random.default_rng(seed)
    any_cfg = next((c for c in configs if c in runs), None)
    if any_cfg is None:
        raise RuntimeError("No hay configs con todos los pliegues terminados. Faltan: " + ", ".join(col["missing"][:10]))
    names_ref = next(iter(runs[any_cfg].values()))[1]
    N = len(names_ref)
    idx_boot = [rng.integers(0, N, N) for _ in range(n_boot)]

    for c in configs:
        if c not in runs:
            continue
        seeds = sorted(runs[c])
        cms_by_seed = [runs[c][s][0] for s in seeds]
        if runs[c][seeds[0]][1] != names_ref:
            raise RuntimeError(f"{c}: las fotos de test no coinciden con las de {any_cfg}")
        if K is None:
            K = json.loads(next((study_dir / "runs" / c).glob("seed*/fold0/config.json")).read_text(
                encoding="utf-8"))["config"]["eval"]["min_photos"]
        ev = M.photos_per_class(cms_by_seed[0]) >= K
        per_seed = [M.summarize(cms.sum(0), ev) for cms in cms_by_seed]
        per_seed_iou = np.stack([M.per_class(cms.sum(0))["iou"] for cms in cms_by_seed])
        boots = [_metric_seed_mean(cms_by_seed, ix, ev) for ix in idx_boot]
        boot_cache[c] = boots
        dn = [d for s in seeds for d in runs[c][s][2]]
        mean = lambda k: float(np.mean([d[k] for d in dn if d.get(k) is not None])) if any(d.get(k) is not None for d in dn) else None
        row = {"config": c, "n_seeds": len(seeds), "seeds": seeds, "n_photos": N,
               "not_evaluable": [n for n, e in zip(classes, ev) if not e],
               "steps_total": mean("optimizer_steps_total"), "steps_at_best": mean("optimizer_steps_at_best"),
               "epochs_trained": mean("epochs_trained"), "gpu_hours": float(sum(d.get("train_hours", 0) for d in dn)),
               "buffer_m": mean("buffer_m"), "overlap_train_test_pct": mean("overlap_train_test_pct"),
               "min_dist_train_test_m": min((d["min_dist_train_test_m"] for d in dn
                                             if d.get("min_dist_train_test_m") is not None), default=None)}
        for k in KEYS:
            v = np.array([s[k] for s in per_seed])
            b = np.array([bb[k] for bb in boots])
            row[k] = float(v.mean())
            row[f"{k}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            row[f"{k}_ci"] = [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
        rows.append(row)
        bci = np.stack([bb["class_iou"] for bb in boots])
        for j, n in enumerate(classes):
            v = per_seed_iou[:, j]
            with np.errstate(all="ignore"):
                cls_rows.append({"config": c, "class": n, "evaluable": bool(ev[j]),
                                 "n_photos": int(M.photos_per_class(cms_by_seed[0])[j]),
                                 "iou": float(np.nanmean(v)) if np.isfinite(v).any() else None,
                                 "iou_sd": float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else None,
                                 "iou_ci": [float(np.nanpercentile(bci[:, j], 2.5)), float(np.nanpercentile(bci[:, j], 97.5))]
                                 if np.isfinite(bci[:, j]).any() else None})

    done = [r["config"] for r in rows]
    pairs = [(a, b) for a, b in (pairs or default_pairs(done, reference)) if a in done and b in done]
    for a, b in pairs:
        d = np.array([bb["mIoU"] for bb in boot_cache[b]]) - np.array([aa["mIoU"] for aa in boot_cache[a]])
        ra = next(r for r in rows if r["config"] == a); rb = next(r for r in rows if r["config"] == b)
        if ra["not_evaluable"] != rb["not_evaluable"]:
            note = "conjuntos de clases evaluables distintos"
        else:
            note = ""
        deltas.append({"from": a, "to": b, "delta_mIoU": rb["mIoU"] - ra["mIoU"],
                       "ci": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))],
                       "p_le_0": float(np.mean(d <= 0)), "note": note})

    result = {"params": params, "K": K, "n_boot": n_boot, "summary": rows, "per_class": cls_rows,
              "deltas": deltas, "missing": col["missing"]}
    if out_dir is None:
        out_dir = study_dir / "results"
    _save(result, Path(out_dir))
    return result


def _fmt(x, sd=None, ci=None, pct=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    f = 100 if pct else 1
    s = f"{f * x:.1f}"
    if sd is not None and np.isfinite(sd):
        s += f" ± {f * sd:.1f}"
    if ci:
        s += f" [{f * ci[0]:.1f}, {f * ci[1]:.1f}]"
    return s


def to_markdown(result: dict) -> str:
    L = [f"Protocolo: {result['params']['protocol']} (k={result['params']['k']}, bloques "
         f"{result['params']['block_m']} m, buffer {result['params']['buffer_m']} m). "
         f"Valores en %: media ± DE entre semillas [IC 95 % bootstrap de fotos, n={result['n_boot']}]. "
         f"mIoU/mAcc/mF1 excluyen clases con < {result['K']} fotos (no evaluables).", "",
         "| Config | Semillas | mIoU | mAcc | mF1 | OA | No evaluables | Pasos (total / al mejor) | Buffer (m) | Fuga reducida: solape train→test (%) |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["summary"]:
        st = "—" if r.get("steps_total") is None else f"{r['steps_total']:.0f} / {r['steps_at_best']:.0f}"
        L.append(f"| {r['config']} | {r['n_seeds']} | " + " | ".join(_fmt(r[k], r[f'{k}_sd'], r[f'{k}_ci']) for k in KEYS)
                 + f" | {', '.join(r['not_evaluable']) or '—'} | {st} | {_fmt(r.get('buffer_m'), pct=False)} | "
                 + f"{_fmt(r.get('overlap_train_test_pct'), pct=False)} |")
    L += ["", "| De → a | ΔmIoU (pp) | IC 95 % | P(Δ ≤ 0) | Nota |", "|---|---|---|---|---|"]
    for d in result["deltas"]:
        L.append(f"| {d['from']} → {d['to']} | {100 * d['delta_mIoU']:+.1f} | "
                 f"[{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}] | {d['p_le_0']:.3f} | {d['note']} |")
    cfgs = [r["config"] for r in result["summary"]]
    classes = list(dict.fromkeys(r["class"] for r in result["per_class"]))
    L += ["", "IoU por clase (%), media ± DE entre semillas; * = no evaluable", "",
          "| Clase | " + " | ".join(cfgs) + " |", "|---|" + "---|" * len(cfgs)]
    for n in classes:
        cells = []
        for c in cfgs:
            r = next(x for x in result["per_class"] if x["config"] == c and x["class"] == n)
            cells.append(_fmt(r["iou"], r["iou_sd"]) + ("*" if not r["evaluable"] else ""))
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    if result["missing"]:
        L += ["", f"Corridas pendientes (excluidas): {', '.join(result['missing'])}"]
    return "\n".join(L)


def _save(result: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "comparison.md").write_text(to_markdown(result), encoding="utf-8")
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        extra = ["steps_total", "steps_at_best", "epochs_trained", "gpu_hours", "buffer_m",
                 "overlap_train_test_pct", "min_dist_train_test_m"]
        w.writerow(["config", "n_seeds"] + [f"{k}{s}" for k in KEYS for s in ("", "_sd", "_ci_lo", "_ci_hi")]
                   + ["not_evaluable"] + extra)
        for r in result["summary"]:
            w.writerow([r["config"], r["n_seeds"]] + [v for k in KEYS for v in (r[k], r[f"{k}_sd"], *r[f"{k}_ci"])]
                       + [";".join(r["not_evaluable"])] + [r.get(k) for k in extra])
    with open(out / "per_class.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "class", "evaluable", "n_photos", "iou", "iou_sd", "iou_ci_lo", "iou_ci_hi"])
        for r in result["per_class"]:
            lo, hi = r["iou_ci"] or (None, None)
            w.writerow([r["config"], r["class"], r["evaluable"], r["n_photos"], r["iou"], r["iou_sd"], lo, hi])
    with open(out / "deltas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["from", "to", "delta_mIoU", "ci_lo", "ci_hi", "p_le_0", "note"])
        for d in result["deltas"]:
            w.writerow([d["from"], d["to"], d["delta_mIoU"], *d["ci"], d["p_le_0"], d["note"]])
