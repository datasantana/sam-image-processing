"""Segmentación de todas las fotos del proyecto fotogramétrico con los modelos de la CV (sin entrenar).

Por config (p. ej. J0, J1, M-b3) se cargan los ``best.pt`` de sus 5 pliegues. Cada modelo infiere
exactamente como en su evaluación: ``segpipe.infer.predict_logits`` con la sección ``infer``/``data``
de la config guardada en su corrida (``config.json``), logits a resolución nativa y la misma regla de AMP.

Dos versiones por foto:

- ``ensamble`` (las 961): promedio de las probabilidades softmax de los 5 modelos y argmax.
- ``oof`` (solo las etiquetadas): el modelo del pliegue donde la foto fue de test (fuera de pliegue);
  pred = argmax de sus logits, igual que en la evaluación. Es la que se verifica contra la CV.

Salidas por config (``<out>/<config>/``), con sufijo ``_oof`` en la versión fuera de pliegue:
``masks_index`` (PNG 8 bits, 0..C-1), ``masks_rgb`` (PNG RGB con los colores de classes.json; nunca JPG),
``conf`` (probabilidad máxima, media de bloques 4×4 con INTER_AREA, ``round(p·255)``), ``progress.jsonl``
(una fila por foto y versión; marca de hecho para reanudar), ``predictions_manifest.csv``, ``models.json``,
``verificacion_oof.json`` y los zips.

El modelo predice siempre una de las C clases: no existe "sin anotar", así que cielo y fondo lejano reciben
alguna clase. No se corrige; se reporta por foto la fracción de píxeles con probabilidad máxima < 0.5/0.7/0.9.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np

from . import metrics as M

EXPECTED_N = 961
EXPECTED_WH = (4032, 2268)
CONF_THRESHOLDS = (0.5, 0.7, 0.9)
CONF_FACTOR = 4
VERIFY_TOL_PP = 0.1
VERSIONS = {"ensamble": "", "oof": "_oof"}
KINDS = ("masks_index", "masks_rgb", "conf")
IMG_EXT = {".jpg", ".jpeg"}
EXIF_ORIENTATION = 274


class PredictError(RuntimeError):
    """Algo no cuadra: la corrida se detiene y se reporta."""


# ----------------------------------------------------------------------------- fotos

def list_photos(photo_dir: str | Path) -> dict[str, Path]:
    """JPG directamente en la carpeta (sin subcarpetas): nombre base → ruta."""
    out: dict[str, Path] = {}
    for p in sorted(Path(photo_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            if p.stem in out:
                raise PredictError(f"Nombre base repetido en {photo_dir}: {p.name} y {out[p.stem].name}")
            out[p.stem] = p
    return out


def load_image(path: str | Path) -> np.ndarray:
    """RGB uint8 (H, W, 3), leída igual que en el entrenamiento (PIL, sin rotación EXIF)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def sha256_path(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def stage_photos(src: dict[str, Path], dst_dir: str | Path, log: Callable = print) -> dict[str, Path]:
    """Copia las fotos al disco local de la VM (salta las ya copiadas con el mismo tamaño)."""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    out, copied = {}, 0
    for n, p in src.items():
        d = dst_dir / p.name
        if not d.exists() or d.stat().st_size != p.stat().st_size:
            tmp = d.with_name(d.name + ".part")
            shutil.copyfile(p, tmp)
            os.replace(tmp, d)
            copied += 1
        out[n] = d
    log(f"fotos en disco local: {len(out)} ({copied} copiadas ahora) → {dst_dir}")
    return out


# ----------------------------------------------------------------------------- splits

def load_splits(study_dir: str | Path, config: str, seed: int = 42) -> dict[int, dict]:
    """Pliegue → {"dir", "test"} desde ``runs/<config>/seed<s>/fold<k>/split.json``."""
    base = Path(study_dir) / "runs" / config / f"seed{seed}"
    out = {}
    if not base.is_dir():
        raise PredictError(f"No existe {base}")
    for d in sorted(base.glob("fold*"), key=lambda p: int(p.name[4:])):
        sj = d / "split.json"
        if not sj.exists():
            raise PredictError(f"Falta {sj}")
        out[int(d.name[4:])] = {"dir": d, "test": list(json.loads(sj.read_text(encoding="utf-8"))["test"])}
    if not out:
        raise PredictError(f"Sin pliegues en {base}")
    return out


def test_fold_map(splits: dict[int, dict]) -> dict[str, int]:
    """Foto etiquetada → pliegue donde fue de test (debe ser exactamente uno)."""
    m: dict[str, int] = {}
    for f, s in splits.items():
        for n in s["test"]:
            if n in m:
                raise PredictError(f"La foto {n} es de test en los pliegues {m[n]} y {f}")
            m[n] = f
    return m


# ----------------------------------------------------------------------------- verificación inicial

def preflight(photo_dir: str | Path, labelled_rows: list[dict], splits_by_config: dict[str, dict[int, dict]],
              expected_n: int = EXPECTED_N, expected_wh: tuple[int, int] = EXPECTED_WH, full: bool = True,
              log: Callable = print) -> dict:
    """Comprueba la carpeta de fotos, las etiquetadas y los splits; lanza PredictError con TODO lo que no cuadra.

    ``full=False``: solo conteo y presencia (barato sobre Drive). ``full=True``: además tamaño en píxeles,
    orientación EXIF (debe ser 1 o no estar) y sha256 de las etiquetadas contra el manifest.
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    problems: list[str] = []
    photos = list_photos(photo_dir)
    if len(photos) != expected_n:
        problems.append(f"{len(photos)} fotos en {photo_dir}; se esperaban {expected_n}")
    lab = {r["name"]: r for r in labelled_rows}
    missing = sorted(n for n in lab if n not in photos)
    if missing:
        problems.append(f"{len(missing)} fotos etiquetadas no están en la carpeta: {missing[:10]}")
    wrong_name = sorted(n for n, r in lab.items() if n in photos and Path(r["image"]).name != photos[n].name)
    if wrong_name:
        problems.append(f"Nombre de archivo distinto al del manifest: {wrong_name[:10]}")

    maps = {}
    for cfg, splits in splits_by_config.items():
        try:
            maps[cfg] = test_fold_map(splits)
        except PredictError as e:
            problems.append(f"{cfg}: {e}")
            continue
        not_in_test = sorted(set(lab) - set(maps[cfg]))
        extra = sorted(set(maps[cfg]) - set(lab))
        if not_in_test:
            problems.append(f"{cfg}: {len(not_in_test)} fotos etiquetadas no son de test en ningún pliegue: {not_in_test[:10]}")
        if extra:
            problems.append(f"{cfg}: {len(extra)} fotos de test no están en el manifest: {extra[:10]}")
    ref = next(iter(maps.values()), None)
    for cfg, m in maps.items():
        if m != ref:
            problems.append(f"{cfg}: la asignación foto→pliegue de test difiere de {next(iter(maps))}")

    bad_size, bad_orient, bad_sha = [], [], []
    if full:
        for n, p in photos.items():
            with Image.open(p) as im:
                if im.size != tuple(expected_wh):
                    bad_size.append(f"{n} {im.size[0]}x{im.size[1]}")
                o = im.getexif().get(EXIF_ORIENTATION, 1)
                if o not in (1, None):
                    bad_orient.append(f"{n} (orientación {o})")
        for n, r in lab.items():
            if n in photos and r.get("sha256") and sha256_path(photos[n]) != r["sha256"]:
                bad_sha.append(n)
        if bad_size:
            problems.append(f"{len(bad_size)} fotos no miden {expected_wh[0]}x{expected_wh[1]}: {bad_size[:10]}")
        if bad_orient:
            problems.append(f"{len(bad_orient)} fotos con orientación EXIF ≠ 1: {bad_orient[:10]}")
        if bad_sha:
            problems.append(f"{len(bad_sha)} fotos etiquetadas con sha256 distinto al del manifest: {bad_sha[:10]}")
    rep = {"photo_dir": str(photo_dir), "n_photos": len(photos), "n_labelled": len(lab),
           "labelled_fold": ref or {}, "full": full, "problems": problems}
    if problems:
        raise PredictError("Verificación inicial fallida:\n  - " + "\n  - ".join(problems))
    log(f"verificación inicial OK ({'completa' if full else 'conteo y presencia'}): {len(photos)} fotos, "
        f"{len(lab)} etiquetadas presentes" + (", tamaño, orientación y sha256 correctos" if full else ""))
    return rep


# ----------------------------------------------------------------------------- modelos

@dataclass
class Member:
    fold: int
    model: object
    cfg: dict
    run_dir: Path = Path(".")
    ckpt_sha256: str = ""
    best_epoch: Optional[int] = None
    test: list = field(default_factory=list)


def amp_dtype_for(device):
    """Misma regla que el entrenamiento/evaluación: bf16 si la GPU lo admite, si no fp16; en CPU, sin AMP."""
    import torch
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_members(study_dir: str | Path, config: str, class_names: list[str], device, seed: int = 42,
                 log: Callable = print) -> list[Member]:
    """Los ``best.pt`` de todos los pliegues de una config, con la config guardada en cada corrida."""
    import torch
    from .models import build_model
    members = []
    for f, s in load_splits(study_dir, config, seed).items():
        d = s["dir"]
        cj = json.loads((d / "config.json").read_text(encoding="utf-8"))
        if list(cj["classes"]) != list(class_names):
            raise PredictError(f"{d}: clases {cj['classes']} ≠ esquema {class_names}")
        cfg = cj["config"]
        ck = d / "best.pt"
        if not ck.exists():
            raise PredictError(f"Falta {ck}")
        model = build_model({**cfg["model"], "weights": None}, len(class_names))   # sin descargas
        bk = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(bk["model"])
        model.to(device).eval()
        members.append(Member(f, model, cfg, d, sha256_path(ck), int(bk.get("epoch", -1)) + 1, s["test"]))
    inf = {json.dumps(m.cfg["infer"], sort_keys=True) for m in members}
    if len(inf) != 1:
        raise PredictError(f"{config}: los pliegues tienen distinta inferencia: {inf}")
    log(f"{config}: {len(members)} modelos (pliegues {[m.fold for m in members]}), infer = {members[0].cfg['infer']}")
    return members


def member_logits(member: Member, image: np.ndarray, device, amp_dtype):
    from .infer import predict_logits
    from .models import input_multiple
    return predict_logits(member.model, image, member.cfg["infer"], member.cfg["data"], device, amp_dtype,
                          multiple=input_multiple(member.cfg["model"]))


def predict_photo(members: list[Member], image: np.ndarray, device, amp_dtype,
                  logits_fn: Callable = member_logits) -> tuple[np.ndarray, np.ndarray]:
    """(pred uint8 (H, W), conf float32 (H, W)).

    Un modelo: pred = argmax de los logits (igual que la evaluación), conf = max softmax.
    Varios: media de las probabilidades softmax (no de los logits), pred = argmax, conf = max de la media.
    """
    import torch
    with torch.no_grad():
        if len(members) == 1:
            logits = logits_fn(members[0], image, device, amp_dtype).float()
            pred = logits.argmax(0)
            conf = torch.softmax(logits, 0).amax(0)
            del logits
        else:
            acc = None
            for m in members:
                p = torch.softmax(logits_fn(m, image, device, amp_dtype).float(), 0)
                acc = p if acc is None else acc.add_(p)
                del p
            acc.div_(len(members))
            pred = acc.argmax(0)
            conf = acc.amax(0)
            del acc
        return pred.to(torch.uint8).cpu().numpy(), conf.float().cpu().numpy()


# ----------------------------------------------------------------------------- salidas

def palette_from_schema(schema) -> np.ndarray:
    """(C, 3) uint8 con el color de classes.json de cada clase del modelo, en orden de índice."""
    rgb = {c["name"]: c["rgb"] for c in schema.source}
    return np.array([rgb[n] for n in schema.names], dtype=np.uint8)


def index_to_rgb(pred: np.ndarray, palette: np.ndarray) -> np.ndarray:
    if pred.max(initial=0) >= len(palette):
        raise ValueError("Índice de clase fuera de la paleta")
    return palette[pred]


def conf_to_png(conf: np.ndarray, factor: int = CONF_FACTOR) -> np.ndarray:
    """Media de bloques ``factor``×``factor`` (cv2.INTER_AREA) y ``round(p·255)`` → uint8."""
    import cv2
    h, w = conf.shape
    small = cv2.resize(conf.astype(np.float32), (w // factor, h // factor), interpolation=cv2.INTER_AREA)
    return np.clip(np.round(small * 255.0), 0, 255).astype(np.uint8)


def photo_stats(pred: np.ndarray, conf: np.ndarray, class_names: list[str]) -> dict:
    n = pred.size
    cnt = np.bincount(pred.ravel(), minlength=len(class_names))
    out = {f"pct_{c}": round(100.0 * float(cnt[i]) / n, 4) for i, c in enumerate(class_names)}
    out["conf_media"] = round(float(conf.mean(dtype=np.float64)), 5)
    for t in CONF_THRESHOLDS:
        out[f"frac_conf_lt_{t}"] = round(float((conf < t).mean()), 5)
    return out


def out_paths(out_dir: str | Path, name: str, version: str) -> dict[str, Path]:
    sfx = VERSIONS[version]
    return {k: Path(out_dir) / f"{k}{sfx}" / f"{name}.png" for k in KINDS}


def write_png(arr: np.ndarray, dest: Path, staging: Path) -> None:
    """PNG (L si 2D, RGB si (H, W, 3)) escrito en local y copiado a destino con renombre atómico."""
    from PIL import Image
    staging.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".png", dir=staging)
    os.close(fd)
    try:
        Image.fromarray(np.ascontiguousarray(arr)).save(tmp, format="PNG")
        part = dest.with_name(dest.name + ".part")
        shutil.copyfile(tmp, part)
        os.replace(part, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


class Progress:
    """``progress.jsonl``: una fila por (foto, versión), escrita después de sus 3 archivos."""

    def __init__(self, out_dir: str | Path):
        self.path = Path(out_dir) / "progress.jsonl"
        self.out_dir = Path(out_dir)
        self._lock = threading.Lock()
        self.rows: dict[tuple[str, str], dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:      # última línea a medias si la sesión se cortó
                        continue
                    self.rows[(r["foto"], r["version"])] = r

    def done(self, name: str, version: str) -> bool:
        return (name, version) in self.rows and all(p.exists() for p in out_paths(self.out_dir, name, version).values())

    def append(self, row: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.rows[(row["foto"], row["version"])] = row

    def of(self, version: str) -> list[dict]:
        return [r for (n, v), r in sorted(self.rows.items()) if v == version]


# ----------------------------------------------------------------------------- bucle por config

def model_tag(config: str, members: list[Member]) -> str:
    return f"{config}/" + "+".join(f"fold{m.fold}" for m in members)


def plan_jobs(names: Iterable[str], labelled: dict[str, int], versions: Iterable[str]) -> list[tuple[str, str]]:
    """Orden: primero todas las ``oof`` (etiquetadas), luego ``ensamble`` (etiquetadas primero, después el resto)."""
    names = sorted(names)
    lab = [n for n in names if n in labelled]
    rest = [n for n in names if n not in labelled]
    jobs = []
    for v in versions:
        if v == "oof":
            jobs += [(n, "oof") for n in lab]
        elif v == "ensamble":
            jobs += [(n, "ensamble") for n in lab + rest]
        else:
            raise ValueError(f"Versión desconocida: {v}")
    return jobs


def predict_config(config: str, members: list[Member], photos: dict[str, Path], labelled: dict[str, int],
                   out_dir: str | Path, palette: np.ndarray, class_names: list[str], device, amp_dtype,
                   versions: Iterable[str] = ("oof", "ensamble"), mask_fn: Optional[Callable] = None,
                   staging: Optional[str | Path] = None, limit: Optional[int] = None, writers: int = 3,
                   logits_fn: Callable = member_logits, log: Callable = print) -> list[dict]:
    """Predice y escribe; salta lo ya hecho (reanudable). ``mask_fn(nombre)`` → máscara índice (255 = ignorar)
    para guardar la matriz de confusión de la versión ``oof``. Devuelve las filas nuevas."""
    out_dir = Path(out_dir)
    staging = Path(staging or tempfile.gettempdir()) / f"segpipe_predict_{config}"
    prog = Progress(out_dir)
    by_fold = {m.fold: m for m in members}
    jobs = [j for j in plan_jobs(photos, labelled, versions) if not prog.done(*j)]
    skipped = len(plan_jobs(photos, labelled, versions)) - len(jobs)
    if limit is not None:
        jobs = jobs[:limit]
    log(f"{config}: {len(jobs)} pendientes, {skipped} ya escritas (se saltan)")
    new_rows: list[dict] = []
    C = len(class_names)

    def write_job(name, version, pred, conf, row):
        t = time.perf_counter()
        paths = out_paths(out_dir, name, version)
        write_png(pred, paths["masks_index"], staging)
        write_png(index_to_rgb(pred, palette), paths["masks_rgb"], staging)
        write_png(conf_to_png(conf), paths["conf"], staging)
        row["segundos_escritura"] = round(time.perf_counter() - t, 3)
        prog.append(row)
        return row

    pending = []
    with ThreadPoolExecutor(max_workers=writers) as pool:
        for i, (name, version) in enumerate(jobs):
            t0 = time.perf_counter()
            if version == "oof":
                if name not in labelled or labelled[name] not in by_fold:
                    raise PredictError(f"{name}: sin modelo fuera de pliegue")
                used = [by_fold[labelled[name]]]
            else:
                used = members
            image = load_image(photos[name])
            pred, conf = predict_photo(used, image, device, amp_dtype, logits_fn)
            if pred.shape != image.shape[:2]:
                raise PredictError(f"{name}: predicción {pred.shape} ≠ foto {image.shape[:2]}")
            sfx = VERSIONS[version]
            row = {"foto": name, "version": version, "etiquetada": name in labelled,
                   "pliegue_test": labelled.get(name, ""), "modelos": model_tag(config, used),
                   "archivos": ";".join(f"{k}{sfx}/{name}.png" for k in KINDS),
                   "segundos_inferencia": round(time.perf_counter() - t0, 3),
                   **photo_stats(pred, conf, class_names)}
            if version == "oof" and mask_fn is not None:
                mask = mask_fn(name)
                if mask.shape != pred.shape:
                    raise PredictError(f"{name}: máscara {mask.shape} ≠ predicción {pred.shape}")
                row["cm"] = M.confusion(pred, mask, C).tolist()
            pending.append(pool.submit(write_job, name, version, pred, conf, row))
            while len(pending) > writers + 1:
                new_rows.append(pending.pop(0).result())
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                log(f"  {config}: {i + 1}/{len(jobs)}")
        for f in pending:
            new_rows.append(f.result())
    return new_rows


# ----------------------------------------------------------------------------- verificación fuera de pliegue

def load_cv_cms(study_dir: str | Path, config: str, seed: int = 42) -> dict[str, np.ndarray]:
    """Matrices de confusión por foto de test guardadas por la CV (``test/photo_confusion.npz``)."""
    out = {}
    for f, s in load_splits(study_dir, config, seed).items():
        z = np.load(s["dir"] / "test" / "photo_confusion.npz", allow_pickle=False)
        for n, cm in zip(z["names"].tolist(), z["cms"]):
            out[str(n)] = np.asarray(cm, dtype=np.int64)
    return out


def verify_oof(rows: list[dict], cv_cms: dict[str, np.ndarray], class_names: list[str], min_photos: int,
               tol_pp: float = VERIFY_TOL_PP, cv_summary_miou: Optional[float] = None) -> dict:
    """Compara el mIoU fuera de pliegue con el de la CV; pasa si |ΔmIoU| ≤ ``tol_pp`` pp.

    Discrepancia por foto = Σ|cm_nueva − cm_CV| / 2: cota inferior del nº de píxeles etiquetados con
    predicción distinta (desde las matrices; un intercambio a↔b dentro de la misma clase real no se ve).
    """
    by = {r["foto"]: r for r in rows if r["version"] == "oof"}
    names = sorted(cv_cms)
    missing = [n for n in names if n not in by or "cm" not in by[n]]
    if missing:
        raise PredictError(f"Faltan {len(missing)} fotos fuera de pliegue para verificar: {missing[:10]}")
    new = np.stack([np.asarray(by[n]["cm"], dtype=np.int64) for n in names])
    ref = np.stack([cv_cms[n] for n in names])
    m_new = M.evaluate(new, class_names, names, min_photos=min_photos, n_boot=0)
    m_cv = M.evaluate(ref, class_names, names, min_photos=min_photos, n_boot=0)
    disc = (np.abs(new - ref).sum(axis=(1, 2)) // 2).astype(int)
    delta = 100.0 * (m_new["mIoU"] - m_cv["mIoU"])
    return {"n_fotos": len(names), "mIoU_oof": m_new["mIoU"], "mIoU_cv": m_cv["mIoU"],
            "mIoU_cv_summary": cv_summary_miou, "delta_pp": delta, "tol_pp": tol_pp,
            "passed": bool(abs(delta) <= tol_pp),
            "clases_evaluables_iguales": m_new["evaluable_classes"] == m_cv["evaluable_classes"],
            "px_distintos_max": int(disc.max()) if len(disc) else 0,
            "foto_px_distintos_max": names[int(disc.argmax())] if len(disc) else None,
            "fotos_con_diferencia": int((disc > 0).sum()),
            "px_distintos_por_foto": {n: int(d) for n, d in zip(names, disc)},
            "nota": "px distintos = Σ|Δcm|/2, cota inferior sobre píxeles etiquetados"}


def summary_miou(study_dir: str | Path, config: str) -> Optional[float]:
    p = Path(study_dir) / "results" / "summary.csv"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["config"] == config:
                return float(r["mIoU"])
    return None


# ----------------------------------------------------------------------------- manifiesto, zips

def write_manifest(out_dir: str | Path, prog: Progress, class_names: list[str]) -> Path:
    """``predictions_manifest.csv``: una fila por (foto, versión) escrita, con modelos y archivos."""
    rows = sorted(prog.rows.values(), key=lambda r: (r["foto"], r["version"]))
    versions_of: dict[str, list[str]] = {}
    for r in rows:
        versions_of.setdefault(r["foto"], []).append(r["version"])
    cols = (["foto", "version", "etiquetada", "pliegue_test", "versiones_disponibles", "modelos", "archivos",
             "conf_media"] + [f"frac_conf_lt_{t}" for t in CONF_THRESHOLDS] + [f"pct_{c}" for c in class_names]
            + ["segundos_inferencia", "segundos_escritura"])
    p = Path(out_dir) / "predictions_manifest.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "versiones_disponibles": ";".join(sorted(versions_of[r["foto"]]))})
    return p


def write_models_json(out_dir: str | Path, config: str, members: list[Member], extra: Optional[dict] = None) -> Path:
    p = Path(out_dir) / "models.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"config": config, **(extra or {}), "modelos": [
        {"pliegue": m.fold, "run_dir": str(m.run_dir), "best_pt_sha256": m.ckpt_sha256, "mejor_epoca": m.best_epoch,
         "infer": m.cfg.get("infer"), "fotos_test": m.test} for m in members]}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    return p


def make_zips(out_dir: str | Path, n_ensemble: int = EXPECTED_N, n_oof: int = 66,
              staging: Optional[str | Path] = None) -> dict[str, Path]:
    """``masks_rgb.zip`` y ``masks_index.zip`` (ensamble, ``n_ensemble`` fotos) y ``oof.zip`` (las tres
    carpetas ``_oof``, ``n_oof`` fotos). Comprueba los conteos antes; ZIP sin compresión (los PNG ya lo están)."""
    out_dir = Path(out_dir)
    plan = {"masks_rgb.zip": [("masks_rgb", n_ensemble)], "masks_index.zip": [("masks_index", n_ensemble)],
            "oof.zip": [(f"{k}_oof", n_oof) for k in KINDS]}
    problems = []
    for z, parts in plan.items():
        for folder, n in parts:
            k = len(list((out_dir / folder).glob("*.png"))) if (out_dir / folder).is_dir() else 0
            if k != n:
                problems.append(f"{folder}: {k} PNG, se esperaban {n}")
    if problems:
        raise PredictError("No se crean los zips:\n  - " + "\n  - ".join(problems))
    staging = Path(staging or tempfile.gettempdir())
    staging.mkdir(parents=True, exist_ok=True)
    out = {}
    for z, parts in plan.items():
        tmp = staging / f"{out_dir.name}_{z}"
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            for folder, _ in parts:
                for p in sorted((out_dir / folder).glob("*.png")):
                    zf.write(p, f"{folder}/{p.name}")
        part = out_dir / (z + ".part")
        shutil.copyfile(tmp, part)
        os.replace(part, out_dir / z)
        tmp.unlink()
        out[z] = out_dir / z
    return out


# ----------------------------------------------------------------------------- humo y estimación

def estimate_hours(rows: list[dict], n_ensemble: int = EXPECTED_N, n_oof: int = 66, writers: int = 3) -> dict:
    """Horas estimadas por config con las medianas de la corrida de humo (sin la primera foto: calentamiento).

    Por foto cuenta max(inferencia, escritura / nº de escritores), porque la escritura va en paralelo."""
    def med(v, key):
        xs = [r[key] for r in rows if r["version"] == v]
        xs = xs[1:] if len(xs) > 1 else xs
        return float(np.median(xs)) if xs else float("nan")
    per = {}
    for v, n in (("ensamble", n_ensemble), ("oof", n_oof)):
        inf, wr = med(v, "segundos_inferencia"), med(v, "segundos_escritura")
        per[v] = {"s_inferencia": inf, "s_escritura": wr, "s_por_foto": max(inf, wr / writers), "n": n}
    hours = sum(p["s_por_foto"] * p["n"] for p in per.values()) / 3600
    return {"por_version": per, "horas": hours}


# ----------------------------------------------------------------------------- hoja de contacto

def pick_contact_photos(labelled: dict[str, int], all_names: Iterable[str], n: int = 4, seed: int = 42) -> dict:
    """``n`` etiquetadas de pliegues distintos y ``n`` sin etiquetar, al azar con ``seed``."""
    rng = np.random.default_rng(seed)
    folds = sorted({int(f) for f in labelled.values()})
    if len(folds) < n:
        raise PredictError(f"Solo hay {len(folds)} pliegues para {n} fotos etiquetadas de pliegues distintos")
    chosen = sorted(int(f) for f in rng.choice(folds, n, replace=False))
    lab = [str(rng.choice(sorted(k for k, f in labelled.items() if f == fo))) for fo in chosen]
    unl_pool = sorted(set(all_names) - set(labelled))
    unl = sorted(str(x) for x in rng.choice(unl_pool, n, replace=False))
    return {"etiquetadas": lab, "pliegues": chosen, "sin_etiquetar": unl, "semilla": seed}


def contact_sheet(pred_root: str | Path, configs: list[str], photos: dict[str, Path], labelled: dict[str, int],
                  gt_rgb_fn: Callable[[str], np.ndarray], legend: list[tuple[str, tuple]], out_base: str | Path,
                  seed: int = 42, n: int = 4, step: int = 8) -> dict:
    """Figura de 2n filas: foto | real (CVAT) | una columna por config. Etiquetadas con la versión ``_oof``;
    sin etiquetar con el ensamble. Reducción ×``step`` (máscaras por vecino más cercano)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image
    sel = pick_contact_photos(labelled, photos, n, seed)
    rows = [(x, "oof") for x in sel["etiquetadas"]] + [(x, "ensamble") for x in sel["sin_etiquetar"]]
    cols = ["Foto", "Real (CVAT)"] + configs
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.3 * len(cols), 1.45 * len(rows) + 0.9))

    def small(arr, nearest):
        im = Image.fromarray(arr)
        return np.asarray(im.resize((im.width // step, im.height // step),
                                    Image.NEAREST if nearest else Image.BILINEAR))
    records = []
    for i, (name, ver) in enumerate(rows):
        panels = [small(load_image(photos[name]), False)]
        panels.append(small(gt_rgb_fn(name), True) if ver == "oof" else None)
        for c in configs:
            p = out_paths(Path(pred_root) / c, name, ver)["masks_rgb"]
            with Image.open(p) as im:
                panels.append(small(np.asarray(im.convert("RGB")), True))
            records.append({"fila": i, "foto": name, "version": ver, "config": c,
                            "pliegue_test": labelled.get(name, ""), "archivo": str(p)})
        for j, pnl in enumerate(panels):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if pnl is None:
                ax.text(0.5, 0.5, "sin máscara", ha="center", va="center", fontsize=7, color="#666666",
                        transform=ax.transAxes)
            else:
                ax.imshow(pnl)
            if i == 0:
                ax.set_title(cols[j], fontsize=9)
        tag = (f"etiquetada, pliegue {labelled[name]}\n(fuera de pliegue)" if ver == "oof"
               else "sin etiquetar\n(ensamble de 5)")
        axes[i][0].set_ylabel(f"{name}\n{tag}", fontsize=5.5, rotation=0, ha="right", va="center")
    fig.legend(handles=[Patch(color=np.array(rgb) / 255, label=nm) for nm, rgb in legend],
               loc="lower center", ncol=7, fontsize=6.5, frameon=False)
    fig.suptitle("Hoja de contacto: foto, máscara real y predicción por configuración", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)
    with open(out_base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    return {"seleccion": sel, "png": out_base.with_suffix(".png"), "pdf": out_base.with_suffix(".pdf")}


# ----------------------------------------------------------------------------- orquestación de una config

def run_config(config: str, study_dir: str | Path, photos: dict[str, Path], labelled: dict[str, int],
               out_root: str | Path, schema, device, mask_fn: Callable, min_photos: Optional[int] = None, seed: int = 42,
               tol_pp: float = VERIFY_TOL_PP, staging: Optional[str | Path] = None, n_oof: Optional[int] = None,
               members: Optional[list[Member]] = None, log: Callable = print) -> dict:
    """Una config completa: modelos → fuera de pliegue → verificación (se detiene si falla) → ensamble →
    manifiesto → zips."""
    out_dir = Path(out_root) / config
    out_dir.mkdir(parents=True, exist_ok=True)
    names = schema.names
    palette = palette_from_schema(schema)
    members = members or load_members(study_dir, config, names, device, seed, log)
    amp = amp_dtype_for(device)
    if min_photos is None:                     # el mismo K que usó la evaluación de la CV
        min_photos = int(members[0].cfg.get("eval", {}).get("min_photos", 5))
    write_models_json(out_dir, config, members, {"amp": str(amp), "n_fotos": len(photos)})
    predict_config(config, members, photos, labelled, out_dir, palette, names, device, amp, ("oof",),
                   mask_fn=mask_fn, staging=staging, log=log)
    prog = Progress(out_dir)
    ver = verify_oof(prog.of("oof"), load_cv_cms(study_dir, config, seed), names, min_photos, tol_pp,
                     summary_miou(study_dir, config))
    (out_dir / "verificacion_oof.json").write_text(json.dumps(ver, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"{config}: mIoU fuera de pliegue {100 * ver['mIoU_oof']:.2f} % vs CV {100 * ver['mIoU_cv']:.2f} % "
        f"(Δ {ver['delta_pp']:+.3f} pp, tolerancia {tol_pp} pp); px distintos máx {ver['px_distintos_max']} "
        f"({ver['foto_px_distintos_max']}), fotos con alguna diferencia {ver['fotos_con_diferencia']}/{ver['n_fotos']}")
    if not ver["passed"]:
        raise PredictError(f"{config}: la verificación fuera de pliegue no pasa (Δ {ver['delta_pp']:+.3f} pp)")
    predict_config(config, members, photos, labelled, out_dir, palette, names, device, amp, ("ensamble",),
                   staging=staging, log=log)
    prog = Progress(out_dir)
    man = write_manifest(out_dir, prog, names)
    zips = make_zips(out_dir, len(photos), n_oof if n_oof is not None else len(labelled), staging)
    return {"config": config, "verificacion": ver, "manifest": man, "zips": zips}
