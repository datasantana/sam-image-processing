"""Estudio = dataset + protocolo de split fijo; cola de corridas (config × semilla × pliegue).

Estructura de ``study_dir`` (en Google Drive en Colab):
  study.json                  parámetros del estudio (dataset, protocolo, k, bloques, buffer, semilla del split)
  splits/                     splits.json, fold<k>.csv, split_report.txt
  dataset/                    ignored_px_per_photo.csv (% de píxeles ignorados por foto)
  runs/<config>/seed<s>/fold<f>/   una carpeta por corrida (ver segpipe.train)
  results/                    tablas comparativas (segpipe.report)
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from . import split as S
from .classes import ClassSchema
from .config import deep_merge, diff, load_run_config
from .data import PhotoStore
from .data_io import ensure_dataset, load_dataset_config, read_manifest
from .train import write_json

REPO = Path(__file__).resolve().parents[1]
LADDER = ["J0", "J1", "J2", "J3"]
DEFAULT_QUEUE = LADDER + ["M-b3"]
DEFAULT_SEEDS = (42, 43, 44)


@dataclass
class Study:
    dir: Path
    params: dict
    data_root: Path
    dataset_cfg: dict
    rows: list
    schema: ClassSchema
    folds: list
    _store: Optional[PhotoStore] = field(default=None, repr=False)

    def store(self, names: Optional[list[str]] = None) -> PhotoStore:
        if self._store is None:
            self._store = PhotoStore(self.data_root, self.rows, self.schema, names)
        return self._store

    def run_dir(self, cfg_name: str, seed: int, fold: int) -> Path:
        return self.dir / "runs" / cfg_name / f"seed{seed}" / f"fold{fold}"


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
    params = {"dataset": dcfg["name"], "dataset_version": dcfg.get("version"),
              "dataset_sha256": dcfg["archive"]["sha256"], "protocol": protocol, "k": k,
              "block_m": block_m, "buffer_m": buffer_m, "split_seed": split_seed,
              "classes": schema.names, "ignored": schema.ignored}
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

    split_path = study_dir / "splits" / "splits.json"
    if split_path.exists():
        folds = S.load_folds(split_path)
    else:
        folds = S.make_folds(rows, dcfg["geometry"], k=k, block_m=block_m, buffer_m=buffer_m,
                             presence=presence, seed=split_seed)
        S.save_folds(folds, study_dir / "splits", params, presence, schema.names)
    if protocol == "fixed":
        folds = folds[:1]
    reports = [S.split_report(f, presence, schema.names) for f in folds]
    txt = S.format_report(reports)
    (study_dir / "splits" / "split_report.txt").write_text(txt, encoding="utf-8")
    log(txt)
    if presence and protocol == "cv":
        all_counts = {c: sum(c in presence[r["name"]] for r in rows) for c in schema.names}
        log("fotos por clase en el test agrupado (todos los pliegues): "
            + ", ".join(f"{c} {v}" for c, v in all_counts.items()))
    return Study(study_dir, params, root, dcfg, rows, schema, folds)


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


def run_queue(study: Study, configs: Iterable[str] = DEFAULT_QUEUE, seeds: Iterable[int] = DEFAULT_SEEDS,
              folds: Optional[Iterable[int]] = None, overrides: Optional[dict] = None,
              subset: Optional[dict] = None, stop_after_epochs: Optional[int] = None,
              max_runs: Optional[int] = None, dry_run: bool = False, log: Callable = print) -> list[dict]:
    """Lanza config × semilla × pliegue en orden semilla → pliegue → config; salta las terminadas.

    ``subset`` (humo): {"train": n, "val": n, "test": n} recorta cada partición a las primeras n fotos.
    ``overrides``: dict aplicado a todas las configs (p. ej. {"train": {"epochs": 2}}).
    """
    from .train import train_run
    cfg_files = config_paths(configs)
    cfgs = [load_run_config(p, overrides) for p in cfg_files]
    fold_ids = list(folds) if folds is not None else [f.fold for f in study.folds]
    by_id = {f.fold: f for f in study.folds}
    plan = [(s, fi, c) for s in seeds for fi in fold_ids for c in cfgs]
    log(f"Cola: {len(cfgs)} configs × {len(list(seeds))} semillas × {len(fold_ids)} pliegues = {len(plan)} corridas")
    results, n_run = [], 0
    for seed, fi, cfg in plan:
        rd = study.run_dir(cfg["name"], seed, fi)
        if (rd / "DONE.json").exists():
            results.append(json.loads((rd / "DONE.json").read_text(encoding="utf-8")))
            continue
        if dry_run:
            log(f"[pendiente] {rd}")
            continue
        if max_runs is not None and n_run >= max_runs:
            log(f"Límite max_runs={max_runs} alcanzado; vuelve a ejecutar para continuar.")
            break
        f = by_id[fi]
        split = {"fold": fi, "train": f.train, "val": f.val, "test": f.test, "excluded": f.excluded,
                 "protocol": study.params["protocol"]}
        if subset:
            split = {**split, **{p: split[p][: subset[p]] for p in ("train", "val", "test") if p in subset}}
        store = study.store()
        results.append(train_run(cfg, store, split, rd, seed, log=log, stop_after_epochs=stop_after_epochs,
                                 extra_env={"study": str(study.dir), "config_name": cfg["name"]}))
        n_run += 1
    return results


def describe_ladder(names: Iterable[str] = DEFAULT_QUEUE) -> str:
    """Texto con lo que cambia cada config respecto a la anterior (para el artículo)."""
    paths = config_paths(names)
    cfgs = [load_run_config(p) for p in paths]
    lines = []
    for i, c in enumerate(cfgs):
        lines.append(f"{c['name']}: {c.get('description', '')}")
        if i:
            for d in diff(cfgs[i - 1], c):
                lines.append(f"    {d}")
    return "\n".join(lines)
