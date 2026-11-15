"""Figuras del análisis posterior (PNG 300 dpi + PDF + CSV con los datos de cada figura).

Textos en español. Paleta categórica fija (orden constante por entidad; un pliegue o una config
conserva su color en todas las figuras), escala secuencial de un solo tono para mapas de calor.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics as M  # noqa: E402
from .analyze import CONFIG_ORDER, LADDER_FIG  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def config_color(c: str) -> str:
    i = CONFIG_ORDER.index(c) if c in CONFIG_ORDER else len(CONFIG_ORDER)
    return CAT[i % len(CAT)]


def _save(fig, rows: list[dict], out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    png = out / f"{name}.png"
    fig.savefig(png, dpi=300)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    keys = list(dict.fromkeys(k for r in rows for k in r)) if rows else ["vacio"]
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    return png


def _pct(x):
    return np.nan if x is None else 100 * x


# ----------------------------------------------------------------------------- A4

def fig_ladder(rep: dict, out: Path):
    S = {r["config"]: r for r in rep["summary"]}
    cfgs = [c for c in LADDER_FIG if c in S]
    if not cfgs:
        return None
    rows = [{"config": c, "mIoU_pct": 100 * S[c]["mIoU"], "ic_lo_pct": 100 * S[c]["mIoU_ci"][0],
             "ic_hi_pct": 100 * S[c]["mIoU_ci"][1]} for c in cfgs]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(cfgs))
    y = np.array([r["mIoU_pct"] for r in rows])
    lo = y - np.array([r["ic_lo_pct"] for r in rows]); hi = np.array([r["ic_hi_pct"] for r in rows]) - y
    ax.plot(x, y, color=GRID, lw=2, zorder=1)
    for i, c in enumerate(cfgs):
        ax.errorbar(x[i], y[i], yerr=[[lo[i]], [hi[i]]], fmt="o", ms=7, color=config_color(c), ecolor=config_color(c),
                    elinewidth=2, capsize=4, mec="white", mew=1.5, zorder=3)
        ax.annotate(f"{y[i]:.1f}", (x[i], y[i] + hi[i]), textcoords="offset points", xytext=(0, 4),
                    ha="center", color=INK, fontsize=8)
    ax.set_xticks(x, cfgs)
    ax.set_ylabel("mIoU de test (%)")
    ax.set_title("Escalera de ablación: test agrupado de 5 pliegues, IC 95 % (bootstrap de fotos)", loc="left")
    return _save(fig, rows, out, "A4_escalera_miou")


def fig_heatmap(rep: dict, classes: list[str], out: Path):
    cfgs = [c for c in CONFIG_ORDER if c in {r["config"] for r in rep["summary"]}]
    pc = {(r["config"], r["class"]): r for r in rep["per_class"]}
    Z = np.array([[_pct(pc[(c, n)]["iou"]) for n in classes] for c in cfgs], float)
    rows = [{"config": c, "clase": n, "iou_pct": Z[i, j], "evaluable": pc[(c, n)]["evaluable"]}
            for i, c in enumerate(cfgs) for j, n in enumerate(classes)]
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(classes), 0.9 + 0.42 * len(cfgs)))
    ax.grid(False)
    im = ax.imshow(Z, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(len(cfgs)):
        for j, n in enumerate(classes):
            v = Z[i, j]
            star = "" if pc[(cfgs[i], n)]["evaluable"] else "*"
            ax.text(j, i, "—" if not np.isfinite(v) else f"{v:.0f}{star}", ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(v) and v > 60 else INK)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(cfgs)), cfgs)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("IoU (%)")
    ax.set_title("IoU por clase y configuración (test agrupado; * = no evaluable)", loc="left")
    return _save(fig, rows, out, "A4_iou_por_clase")


def fig_folds(folds: list[dict], rep: dict, out: Path):
    cfgs = [c for c in CONFIG_ORDER if c in {r["config"] for r in folds}]
    S = {r["config"]: r for r in rep["summary"]}
    fset = sorted({r["fold"] for r in folds})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for j, f in enumerate(fset):
        xs, ys = [], []
        for i, c in enumerate(cfgs):
            for r in folds:
                if r["config"] == c and r["fold"] == f:
                    xs.append(i + (j - (len(fset) - 1) / 2) * 0.08); ys.append(100 * r["mIoU"])
        ax.scatter(xs, ys, s=30, color=CAT[j % len(CAT)], marker=MARKERS[j % len(MARKERS)], edgecolor="white",
                   linewidth=0.8, label=f"Pliegue {f}", zorder=3)
    for i, c in enumerate(cfgs):
        if c in S:
            ax.hlines(100 * S[c]["mIoU"], i - 0.3, i + 0.3, color=INK, lw=1.5, zorder=2)
    ax.set_xticks(range(len(cfgs)), cfgs)
    ax.set_ylabel("mIoU de test del pliegue (%)")
    ax.set_title("Variabilidad entre pliegues (raya = test agrupado)", loc="left")
    ax.legend(ncol=len(fset), loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    rows = [{k: r[k] for k in ("config", "seed", "fold", "n_train", "n_test", "mIoU", "OA")} for r in folds]
    return _save(fig, rows, out, "A4_variabilidad_pliegues")


def fig_curves(runs, out: Path, key: str, ylabel: str, name: str, title: str, pct: bool):
    cfgs = [c for c in CONFIG_ORDER if c in {r.config for r in runs}]
    n = len(cfgs)
    ncol = min(4, n); nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 2.0 * nrow), sharey=True, squeeze=False)
    rows = []
    for k, c in enumerate(cfgs):
        ax = axes[k // ncol][k % ncol]
        for r in sorted([r for r in runs if r.config == c], key=lambda r: r.fold):
            if not r.log or key not in r.log[0]:
                continue
            ep = [int(h["epoch"]) for h in r.log]
            v = [float(h[key]) * (100 if pct else 1) if h[key] not in ("", "nan") else np.nan for h in r.log]
            ax.plot(ep, v, lw=1.2, color=CAT[r.fold % len(CAT)], label=f"Pliegue {r.fold}")
            best = r.done.get("best_epoch")
            if best and best <= len(v):
                ax.plot(best, v[best - 1], marker=MARKERS[r.fold % len(MARKERS)], ms=5,
                        color=CAT[r.fold % len(CAT)], mec="white", mew=0.8)
            for e, x in zip(ep, v):
                rows.append({"config": c, "seed": r.seed, "fold": r.fold, "epoch": e, key: x,
                             "opt_steps": r.log[e - 1].get("opt_steps"), "mejor_epoca": best})
        ax.set_title(c, loc="left")
        ax.set_xlabel("Época")
        if k % ncol == 0:
            ax.set_ylabel(ylabel)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, rows, out, name)


def fig_confusion(runs, classes: list[str], out: Path, configs=("J0", "J1", "M-b3")):
    cfgs = [c for c in configs if any(r.config == c for r in runs)]
    if not cfgs:
        return None
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.3 * len(cfgs), 4.4), squeeze=False)
    rows = []
    for k, c in enumerate(cfgs):
        rr = [r for r in runs if r.config == c]
        s0 = min(r.seed for r in rr)
        cm = sum(r.cms.sum(0) for r in rr if r.seed == s0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            nm = cm / cm.sum(1, keepdims=True)
        ax = axes[0][k]
        ax.grid(False)
        ax.imshow(np.nan_to_num(nm), cmap="Blues", vmin=0, vmax=1)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = nm[i, j]
                if np.isfinite(v) and v >= 0.01:
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.5,
                            color="white" if v > 0.6 else INK)
                rows.append({"config": c, "real": classes[i], "predicha": classes[j], "pixeles": int(cm[i, j]),
                             "fraccion_fila": None if not np.isfinite(v) else float(v)})
        ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(classes)), classes if k == 0 else [""] * len(classes), fontsize=6.5)
        ax.set_xlabel("Clase predicha")
        if k == 0:
            ax.set_ylabel("Clase real")
        ax.set_title(f"{c} (% de cada fila)", loc="left")
    fig.suptitle("Matrices de confusión normalizadas por fila (test agrupado)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    return _save(fig, rows, out, "A4_matrices_confusion")


def fig_cost(costs: list[dict], out: Path):
    cfgs = [c["config"] for c in costs]
    fig, ax = plt.subplots(figsize=(6, 0.6 + 0.38 * len(cfgs)))
    y = np.arange(len(cfgs))[::-1]
    h = [c["horas_gpu"] for c in costs]
    ax.barh(y, h, color=[config_color(c) for c in cfgs], height=0.6)
    for yi, c in zip(y, costs):
        ax.text(c["horas_gpu"], yi, f"  {c['horas_gpu']:.1f} h · {c['pasos_total']} pasos · "
                f"{c['epocas_por_pliegue']:.0f} épocas/pliegue", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, cfgs)
    ax.set_xlabel("Horas de GPU (suma de los pliegues)")
    ax.set_xlim(0, max(h + [1e-6]) * 1.9)
    ax.grid(axis="y", visible=False)
    ax.set_title("Costo de entrenamiento por configuración", loc="left")
    return _save(fig, costs, out, "A4_costo")


# ----------------------------------------------------------------------------- A1–A3

def fig_a1(a1: dict, out: Path):
    rows = a1["rows"]
    labels = {"aleatorio_80_20": "Aleatorio 80/20"}
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    xt, xl, x = [], [], 0.0
    for k, esq in enumerate(dict.fromkeys(r["esquema"] for r in rows)):
        rr = [r for r in rows if r["esquema"] == esq]
        xs = x + np.arange(len(rr))
        ax.bar(xs, [r["solape_union_pct"] for r in rr], width=0.7, color=CAT[k % len(CAT)])
        for xi, r in zip(xs, rr):
            ax.text(xi, r["solape_union_pct"] + 1, f"{r['solape_union_pct']:.0f}", ha="center", fontsize=7, color=INK)
        xt.append(xs.mean())
        name = labels.get(esq, esq.replace("cv_bloques_buffer", "CV bloques, buffer ") + " m")
        xl.append(name + ("" if len(rr) == 1 else "\n(pliegues " + ", ".join(str(r["fold"]) for r in rr) + ")"))
        x = xs[-1] + 1.6
    ax.set_xticks(xt, xl, fontsize=8)
    ax.set_ylabel("Área de test cubierta por train (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.set_title("Solape de huellas train→test: split aleatorio frente a CV por bloques (fuga reducida)", loc="left")
    return _save(fig, rows, out, "A1_solape_huellas")


def fig_a2_scatter(table: list[dict], a2: dict, out: Path):
    cfgs = [c for c in CONFIG_ORDER if c in {t["config"] for t in table}]
    n = len(cfgs); ncol = min(4, n); nrow = -(-n // ncol)
    R = {c["config"]: c for c in a2["correlations"]}
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.3 * nrow), sharex=True, sharey=True, squeeze=False)
    for k, c in enumerate(cfgs):
        ax = axes[k // ncol][k % ncol]
        rr = [t for t in table if t["config"] == c]
        for f in sorted({t["fold"] for t in rr}):
            q = [t for t in rr if t["fold"] == f]
            ax.scatter([t["coverage_pct"] for t in q], [100 * t["iou"] for t in q], s=16, color=CAT[f % len(CAT)],
                       marker=MARKERS[f % len(MARKERS)], edgecolor="white", linewidth=0.6, label=f"Pliegue {f}")
        r = R[c]
        ax.set_title(f"{c}  ρ = {r['rho']:.2f}" if np.isfinite(r["rho"]) else c, loc="left", fontsize=9)
        if k // ncol == nrow - 1:
            ax.set_xlabel("Huella cubierta por train (%)")
        if k % ncol == 0:
            ax.set_ylabel("IoU medio de la foto (%)")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), loc="lower center", bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.suptitle("Fuga residual: IoU por foto de test frente a la cobertura de su huella por el train",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, table, out, "A2_iou_vs_cobertura")


def fig_a2_pair(pair: dict, out: Path):
    rows = pair["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for f in sorted({r["fold"] for r in rows}):
        q = [r for r in rows if r["fold"] == f]
        kw = dict(s=18, color=CAT[f % len(CAT)], marker=MARKERS[f % len(MARKERS)], edgecolor="white",
                  linewidth=0.6, label=f"Pliegue {f}")
        axes[0].scatter([r["d_cov"] for r in q], [100 * r["d_iou"] for r in q], **kw)
        jit = np.linspace(-0.25, 0.25, len(q)) if len(q) > 1 else [0]
        axes[1].scatter([r["d_n_train"] + j for r, j in zip(q, jit)], [100 * r["d_iou"] for r in q], **kw)
    for ax in axes:
        ax.axhline(0, color=INK2, lw=0.8)
    axes[0].set_xlabel("Δ cobertura de la huella (pp)")
    axes[1].set_xlabel("Δ fotos de train del pliegue")
    axes[0].set_ylabel("Δ IoU de la foto (pp)")
    axes[0].set_title(f"ρ = {pair['rho_diou_dcov']:.2f}", loc="left", fontsize=9)
    axes[1].set_title(f"ρ = {pair['rho_diou_dntrain']:.2f}", loc="left", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=len(l), loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle(f"{pair['b']} − {pair['a']} por foto de test: menos solape frente a menos datos",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, rows, out, "A2_pareado_J0_vs_J0-b75")


def fig_a3(a3: dict, out: Path):
    common = a3["common"]
    cfgs = list(a3["mine"])
    series = [(c, {x["class"]: (x["iou"], x["iou_ci"]) for x in a3["mine"][c]["classes"]},
               {x["class"]: x["evaluable"] for x in a3["mine"][c]["classes"]}) for c in cfgs]
    if a3["mauricio"]:
        series.append(("Mauricio (otro dataset)", {n: (v, None) for n, v in a3["mauricio"]["iou"].items()},
                       dict(a3["mauricio"]["evaluable"])))
    rows = [{"serie": s, "clase": n, "iou_pct": _pct(d.get(n, (None, None))[0]),
             "ic_lo_pct": _pct((d.get(n, (None, None))[1] or [None, None])[0]),
             "ic_hi_pct": _pct((d.get(n, (None, None))[1] or [None, None])[1]), "evaluable": e.get(n)}
            for s, d, e in series for n in common]
    fig, ax = plt.subplots(figsize=(1.2 + 0.95 * len(common), 3.6))
    w = 0.8 / len(series)
    for k, (s, d, e) in enumerate(series):
        xs = np.arange(len(common)) - 0.4 + w * (k + 0.5)
        ys = np.array([_pct(d.get(n, (None, None))[0]) for n in common], float)
        mau = s.startswith("Mauricio")
        ax.bar(xs, np.nan_to_num(ys), width=w * 0.9, color="white" if mau else config_color(s),
               edgecolor=INK if mau else "none", hatch="////" if mau else None, linewidth=0.8, label=s)
        for xi, yi, n in zip(xs, ys, common):
            if not e.get(n, True):
                ax.text(xi, np.nan_to_num(yi) + 1.5, "*", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(common)), common, rotation=30, ha="right")
    ax.set_ylabel("IoU (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=min(4, len(series)), loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=7.5)
    ax.set_title("Esquema común: Pascual frente a Mauricio (datasets distintos; * = no evaluable)", loc="left")
    return _save(fig, rows, out, "A3_esquema_comun")


def make_all(out: Path, rep, runs, classes, folds, costs, a1, table, a2, pair, a3) -> list[Path]:
    figs = [fig_ladder(rep, out), fig_heatmap(rep, classes, out), fig_folds(folds, rep, out),
            fig_curves(runs, out, "val_mIoU", "mIoU de validación (%)", "A4_curvas_val_miou",
                       "Curvas de entrenamiento: mIoU de validación por época (marcador = época elegida)", True),
            fig_curves(runs, out, "train_loss", "Pérdida de entrenamiento", "A4_curvas_perdida",
                       "Curvas de entrenamiento: pérdida por época", False),
            fig_confusion(runs, classes, out), fig_cost(costs, out), fig_a1(a1, out), fig_a2_scatter(table, a2, out)]
    if pair:
        figs.append(fig_a2_pair(pair, out))
    if a3:
        figs.append(fig_a3(a3, out))
    return [f for f in figs if f is not None]
