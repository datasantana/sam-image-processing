"""Estudio = dataset + split por bloques fijo; cola de corridas (config × semilla × pliegue).

Estructura de ``study_dir`` (en Google Drive en Colab):
  study.json                  parámetros del estudio (dataset, protocolo, k, bloques, buffer, semilla del split)
  splits/                     splits.json, fold<k>.csv, split_report.txt (buffer del estudio)
  splits/buffer<b>/           variantes con otro buffer y la MISMA asignación de bloques (p. ej. J0-b75)
  dataset/                    ignored_px_per_photo.csv (% de píxeles ignorados por foto)
  runs/<config>/seed<s>/fold<f>/   una carpeta por corrida (ver segpipe.train)
  results/                    tablas comparativas (segpipe.report)
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import split as S
from .classes import ClassSchema
from .config import diff, load_run_config
from .data import PhotoStore
from .data_io import ensure_dataset, load_dataset_config, read_manifest
from .train import write_json

REPO = Path(__file__).resolve().parents[1]
LADDER = ["J0", "J1", "J2", "J3"]
# Orden pedido: J0 y J3 completos → J1 → J2 → M-b3; después los controles fuera de la escalera.
DEFAULT_QUEUE = ["J0", "J3", "J1", "J2", "M-b3", "J0-pasos", "J0-b75"]
SMOKE_CONFIGS = ["J0", "J3", "J1", "J2", "M-b3"]   # J0-pasos y J0-b75 cuestan por época lo mismo que J0
DEFAULT_SEEDS = (42,)


@dataclass
class Study:
    dir: Path
    params: dict
    data_root: Path
    dataset_cfg: dict
    rows: list
    schema: ClassSchema
    folds: list                      # pliegues usados (todos con CV; solo el 0 con "fixed")
    all_folds: list = field(default_factory=list, repr=False)
    presence: Optional[dict] = field(default=None, repr=False)
    _store: Optional[PhotoStore] = field(default=None, repr=False)
    _variants: dict = field(default_factory=dict, repr=False)

    def store(self, names: Optional[list[str]] = None) -> PhotoStore:
        if self._store is None:
            self._store = PhotoStore(self.data_root, self.rows, self.schema, names)
        return self._store

    def run_dir(self, cfg_name: str, seed: int, fold: int) -> Path:
        return self.dir / "runs" / cfg_name / f"seed{seed}" / f"fold{fold}"

    def folds_for(self, buffer_m: Optional[float], log: Callable = print) -> list:
        """Pliegues con otro buffer, conservando la asignación de bloques (mismos val y test)."""
        if buffer_m is None or float(buffer_m) == float(self.params["buffer_m"]):
            return self.folds
        b = float(buffer_m)
        if b not in self._variants:
            d = self.dir / "splits" / f"buffer{b:g}"
            if (d / "splits.json").exists():
                folds = S.load_folds(d / "splits.json")
            else:
                fp = S.footprints(self.rows, self.dataset_cfg["geometry"])
                names = [r["name"] for r in self.rows]
                folds = S.folds_from_assignment(names, fp, S.fold_assignment(self.all_folds),
                                                len(self.all_folds), b)
                S.save_folds(folds, d, {**self.params, "buffer_m": b}, self.presence, self.schema.names)
                txt = S.format_report([S.split_report(f, self.presence, self.schema.names) for f in folds])
                (d / "split_report.txt").write_text(txt, encoding="utf-8")
                log(f"Split con buffer {b:g} m (misma asignación de bloques):\n" + txt)
            self._variants[b] = folds if self.params["protocol"] == "cv" else folds[:1]
        return self._variants[b]


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() or p.exists() else REPO / p


def prepare_study(study_dir: str | Path, dataset_cfg: str | Path = "configs/datasets/pascual.yaml",
                  protocol: str = "cv", k: int = 5, block_m: float = 150.0, buffer_m: float = 50.0,
                  split_seed: int = 0, data_root: Optional[str | Path] = None, log: Callable = print) -> Study:
    """Obtiene el dataset, construye (o reutiliza) el split y guarda el reporte."""
    study_dir = Path(study_dir)
    dcfg = load_dataset_config(_resolve(dataset_cfg))
    root = Path(data_root) if data_root else ensure_dataset(dcfg, verbose=True)
    lay = dcfg["layout"]
    ccfg = dcfg.get("classes", {}) or {}
    schema = ClassSchema.from_json(root / lay["classes"], ignore=ccfg.get("ignore"),
                                   mapping=ccfg.get("mapping"), target=ccfg.get("target"))
    rows = read_manifest(root, lay["manifest"])
    if protocol not in ("cv", "fixed"):
        raise ValueError("protocol debe ser 'cv' o 'fixed'")
    geom = dcfg["geometry"]
    params = {"dataset": dcfg["name"], "dataset_version": dcfg.get("version"),
              "dataset_sha256": dcfg["archive"]["sha256"], "protocol": protocol, "k": k,
              "block_m": block_m, "buffer_m": buffer_m, "split_seed": split_seed,
              "footprint": "camera_centroid" if geom.get("camera") else "principal_ray",
              "camera": geom.get("camera"), "classes": schema.names, "ignored": schema.ignored}
    sj = study_dir / "study.json"
    if sj.exists():
        old = json.loads(sj.read_text(encoding="utf-8"))["params"]
        if old != params:
            raise RuntimeError(f"{sj} existe con otros parámetros:\n  " + "\n  ".join(diff(old, params))
                               + "\nUsa otro study_dir o borra el estudio a mano.")
    else:
        write_json(sj, {"params": params, "created": time.strftime("%Y-%m-%dT%H:%M:%S")})

    # clases presentes por foto: desde class_stats.csv (nombres fuente) mapeado al esquema de entrenamiento
    stats = root / lay.get("class_stats", "class_stats.csv")
    src_names = [c["name"] for c in schema.source]
    presence = None
    if stats.exists():
        src_presence = S.presence_from_class_stats(stats, src_names)
        presence = {n: {schema.names[schema.source_to_train[src_names.index(c)]] for c in cs
                        if schema.source_to_train[src_names.index(c)] != schema.ignore_index}
                    for n, cs in src_presence.items()}

    names = [r["name"] for r in rows]
    fp = S.footprints(rows, geom)
    mix = S.flight_mix(names, fp["cx"], fp["cy"], block_m, geom.get("flight_regex", r"DJI_(\d{8})"))
    split_path = study_dir / "splits" / "splits.json"
    if split_path.exists():
        all_folds = S.load_folds(split_path)
    else:
        all_folds = S.make_folds(rows, geom, k=k, block_m=block_m, buffer_m=buffer_m,
                                 presence=presence, seed=split_seed)
        S.save_folds(all_folds, study_dir / "splits", params, presence, schema.names, {"flight_mix": mix})
    folds = all_folds[:1] if protocol == "fixed" else all_folds
    reports = [S.split_report(f, presence, schema.names) for f in folds]
    head = []
    if geom.get("camera"):
        hf, vf = S.camera_fov(geom["camera"])
        head.append(f"Huellas: polígono de la cámara (FOV {hf:.1f}° × {vf:.1f}°), centro = centroide.")
    head.append(f"Bloques de {block_m:g} m por huella en tierra (la fecha no interviene): {mix['n_blocks']} bloques, "
                f"{mix['blocks_with_several_groups']} mezclan vuelos {mix['blocks_by_groups']}.")
    head.append("Fuga reducida, no eliminada: el % de solape residual es el área de las huellas de test (o val) "
                "cubierta por huellas de train.")
    txt = "\n".join(head) + "\n" + S.format_report(reports)
    (study_dir / "splits" / "split_report.txt").write_text(txt, encoding="utf-8")
    log(txt)
    if presence and protocol == "cv":
        all_counts = {c: sum(c in presence[r["name"]] for r in rows) for c in schema.names}
        log("fotos por clase en el test agrupado (todos los pliegues): "
            + ", ".join(f"{c} {v}" for c, v in all_counts.items()))
    return Study(study_dir, params, root, dcfg, rows, schema, folds, all_folds, presence)


def write_ignored_report(study: Study) -> Path:
    """% de píxeles ignorados (sin anotar) por foto."""
    out = study.dir / "dataset" / "ignored_px_per_photo.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frac = study.store().ignored_fraction()
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "ignored_pct"])
        for n in sorted(frac):
            w.writerow([n, round(100 * frac[n], 3)])
    vals = sorted(frac.values())
    print(f"píxeles ignorados por foto: mediana {100 * vals[len(vals) // 2]:.1f} %, "
          f"rango {100 * vals[0]:.1f}–{100 * vals[-1]:.1f} %  → {out}")
    return out


def config_paths(names: Iterable[str]) -> list[Path]:
    return [_resolve(f"configs/runs/{n}.yaml") if not str(n).endswith(".yaml") else _resolve(n) for n in names]


def queue_plan(study: Study, configs: Iterable[str] = DEFAULT_QUEUE, seeds: Iterable[int] = DEFAULT_SEEDS,
               folds: Optional[Iterable[int]] = None, overrides: Optional[dict] = None) -> list[tuple]:
    """Lista ordenada (config, semilla, pliegue, cfg) en orden config → semilla → pliegue."""
    cfgs = [load_run_config(p, overrides) for p in config_paths(configs)]
    fold_ids = list(folds) if folds is not None else [f.fold for f in study.folds]
    return [(c["name"], s, fi, c) for c in cfgs for s in seeds for fi in fold_ids]


def run_queue(study: Study, configs: Iterable[str] = DEFAULT_QUEUE, seeds: Iterable[int] = DEFAULT_SEEDS,
              folds: Optional[Iterable[int]] = None, overrides: Optional[dict] = None,
              subset: Optional[dict] = None, stop_after_epochs: Optional[int] = None,
              max_runs: Optional[int] = None, dry_run: bool = False, log: Callable = print) -> list[dict]:
    """Lanza la cola en orden config → semilla → pliegue (cada config completa antes de la siguiente)
    y salta las corridas terminadas.

    ``subset`` (humo): {"train": n, "val": n, "test": n} recorta cada partición a las primeras n fotos.
    ``overrides``: dict aplicado a todas las configs (p. ej. {"train": {"epochs": 2}}).
    Una config puede pedir otro buffer con ``split: {buffer_m: 75}`` (misma asignación de bloques).
    """
    from .train import train_run
    seeds = list(seeds)
    plan = queue_plan(study, configs, seeds, folds, overrides)
    n_cfg = len({p[0] for p in plan})
    log(f"Cola: {n_cfg} configs × {len(seeds)} semillas × {len(plan) // max(1, n_cfg * len(seeds))} pliegues "
        f"= {len(plan)} corridas")
    results, n_run = [], 0
    for name, seed, fi, cfg in plan:
        rd = study.run_dir(name, seed, fi)
        if (rd / "DONE.json").exists():
            results.append(json.loads((rd / "DONE.json").read_text(encoding="utf-8")))
            continue
        if dry_run:
            log(f"[pendiente] {rd}")
            continue
        if max_runs is not None and n_run >= max_runs:
            log(f"Límite max_runs={max_runs} alcanzado; vuelve a ejecutar para continuar.")
            break
        buf = (cfg.get("split") or {}).get("buffer_m")
        f = {x.fold: x for x in study.folds_for(buf, log)}[fi]
        split = {"fold": fi, "train": f.train, "val": f.val, "test": f.test, "excluded": f.excluded,
                 "protocol": study.params["protocol"], "buffer_m": f.buffer_m,
                 "min_dist_train_test_m": f.min_dist_train_test,
                 "overlap_train_test_pct": f.overlap_train_test_pct}
        if subset:
            split = {**split, **{p: split[p][: subset[p]] for p in ("train", "val", "test") if p in subset}}
        results.append(train_run(cfg, study.store(), split, rd, seed, log=log, stop_after_epochs=stop_after_epochs,
                                 extra_env={"study": str(study.dir), "config_name": name}))
        n_run += 1
    return results


# ----------------------------------------------------------------------------- humo y costo

SMOKE_OVERRIDES = {"train": {"epochs": 2}, "data": {"samples_per_epoch": 32}, "eval": {"n_boot": 200}}
SMOKE_SUBSET = {"train": 6, "val": 2, "test": 3}


def run_smoke(smoke: Study, configs: Iterable[str] = SMOKE_CONFIGS, seed: int = 42,
              overrides: Optional[dict] = None, subset: Optional[dict] = None, log: Callable = print) -> list[dict]:
    """Por config: entrena 1 época, interrumpe a propósito, reanuda, termina y evalúa."""
    from .train import Interrupted
    kw = dict(seeds=[seed], folds=[smoke.folds[0].fold], overrides=overrides or SMOKE_OVERRIDES,
              subset=subset or SMOKE_SUBSET, log=log)
    out = []
    for c in configs:
        try:
            run_queue(smoke, [c], stop_after_epochs=1, **kw)
        except Interrupted as e:
            log(f"→ {c}: {e}")
        out += run_queue(smoke, [c], **kw)
    return out


def _proxy_log(smoke_dir: Path, cfg: dict, seed: int, fold: int) -> Optional[tuple[str, dict]]:
    """Última fila del epoch_log del humo para la config o, si no se corrió, para su base."""
    name, base = cfg["name"], cfg.get("base")
    seen = set()
    while name and name not in seen:
        seen.add(name)
        p = smoke_dir / "runs" / name / f"seed{seed}" / f"fold{fold}" / "epoch_log.csv"
        if p.exists():
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            return name, rows[-1]
        if not base:
            break
        c = load_run_config(_resolve(f"configs/runs/{base}"))
        name, base = c["name"], c.get("base")
    return None


def estimate_cost(smoke: Study, study: Study, configs: Iterable[str] = DEFAULT_QUEUE,
                  seeds: Iterable[int] = DEFAULT_SEEDS, log: Callable = print) -> dict:
    """Estimación de horas de GPU de la cola a partir de los tiempos medidos en el humo.

    Por época: s/muestra de entrenamiento × muestras por época + s/foto de val × fotos de val
    (medidos en la última época del humo, misma GPU). Rango de épocas: [paciencia + 1, máximo];
    el early stopping real cae en medio. Se suma la evaluación final (val + test con best.pt).
    """
    seeds = list(seeds)
    rows, tot_lo, tot_hi = [], 0.0, 0.0
    sm_fold = smoke.folds[0].fold
    smoke_seed = 42
    for name, seed, fi, cfg in queue_plan(study, configs, seeds[:1]):
        if seed != seeds[0]:
            continue
        px = _proxy_log(smoke.dir, cfg, smoke_seed, sm_fold)
        if px is None:
            raise RuntimeError(f"No hay humo para {name} ni para su base; corre run_smoke primero")
        proxy, r = px
        sps = float(r["train_seconds"]) / max(1, float(r["n_train_samples"]))
        spv = float(r["val_seconds"]) / max(1, float(r["n_val"]))
        f = {x.fold: x for x in study.folds_for((cfg.get("split") or {}).get("buffer_m"), lambda *a: None)}[fi]
        samples = cfg["data"].get("samples_per_epoch") or len(f.train)
        ep_s = sps * samples + spv * len(f.val)
        e_max = int(cfg["train"]["epochs"])
        e_min = min(e_max, int(cfg["train"]["patience"]) + 1)
        final = spv * (len(f.val) + len(f.test))
        lo, hi = (e_min * ep_s + final) * len(seeds), (e_max * ep_s + final) * len(seeds)
        tot_lo += lo; tot_hi += hi
        rows.append({"config": name, "fold": fi, "proxy": proxy, "s_per_train_sample": sps, "s_per_val_photo": spv,
                     "epoch_s": ep_s, "epochs_min": e_min, "epochs_max": e_max, "h_min": lo / 3600, "h_max": hi / 3600})
    by_cfg: dict[str, dict] = {}
    for r in rows:
        d = by_cfg.setdefault(r["config"], {"proxy": r["proxy"], "epoch_s": [], "h_min": 0.0, "h_max": 0.0, "runs": 0})
        d["epoch_s"].append(r["epoch_s"]); d["h_min"] += r["h_min"]; d["h_max"] += r["h_max"]; d["runs"] += len(seeds)
    lines = [f"Estimación con {len(seeds)} semilla(s); tiempos medidos en el humo (misma GPU):",
             "| Config | Medido en | s/época (media pliegues) | Corridas | Horas GPU (mín–máx) |", "|---|---|---|---|---|"]
    for c, d in by_cfg.items():
        lines.append(f"| {c} | {d['proxy']} | {sum(d['epoch_s']) / len(d['epoch_s']):.0f} | {d['runs']} | "
                     f"{d['h_min']:.1f}–{d['h_max']:.1f} |")
    n_runs = sum(d["runs"] for d in by_cfg.values())
    lines.append(f"| **Total** | | | **{n_runs}** | **{tot_lo / 3600:.1f}–{tot_hi / 3600:.1f}** |")
    lines.append("Mínimo = early stopping lo antes posible (paciencia + 1 épocas); máximo = todas las épocas. "
                 "Multiplica por la tasa de unidades de cómputo/hora de tu GPU en Colab.")
    txt = "\n".join(lines)
    log(txt)
    return {"by_config": by_cfg, "rows": rows, "total_h": [tot_lo / 3600, tot_hi / 3600], "n_runs": n_runs,
            "markdown": txt}


def smoke_epoch_times(smoke: Study, configs: Iterable[str] = SMOKE_CONFIGS, seed: int = 42) -> str:
    """Tabla con el tiempo medido por época en el humo (train y val por separado)."""
    lines = ["| Config | GPU | micro-lote × acum. | s train / muestra | s val / foto | s época (humo) |",
             "|---|---|---|---|---|---|"]
    for c in configs:
        d = smoke.run_dir(c, seed, smoke.folds[0].fold)
        p = d / "epoch_log.csv"
        if not p.exists():
            continue
        r = list(csv.DictReader(open(p, encoding="utf-8")))[-1]
        env = json.loads((d / "env.json").read_text(encoding="utf-8"))["sessions"][-1]
        lines.append(f"| {c} | {env['gpu'].get('name')} | {env['micro_batch']} × {env['accum_steps']} | "
                     f"{float(r['train_seconds']) / max(1, float(r['n_train_samples'])):.3f} | "
                     f"{float(r['val_seconds']) / max(1, float(r['n_val'])):.2f} | {float(r['seconds']):.0f} |")
    return "\n".join(lines)


def describe_ladder(names: Iterable[str] = LADDER + ["M-b3", "J0-pasos", "J0-b75"]) -> str:
    """Texto con lo que cambia cada config respecto a su base (para el artículo)."""
    lines = []
    for p in config_paths(names):
        c = load_run_config(p)
        lines.append(f"{c['name']}: {c.get('description', '')}")
        refname = c.get("reference") or c.get("base")
        ref = load_run_config(_resolve(f"configs/runs/{refname}")) if refname else None
        if ref is not None:
            lines.append(f"    respecto a {ref['name']}:")
            for d in diff(ref, c):
                lines.append(f"      {d}")
    return "\n".join(lines)
