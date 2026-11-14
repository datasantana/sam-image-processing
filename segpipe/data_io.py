"""Obtención, verificación y validación del dataset empaquetado (segpack).

Orden de búsqueda (``ensure_dataset``):
  1. Carpeta ya descomprimida y validada en <root>/<extract_dir> (marca ``.segpipe_validated.json``).
  2. Colab: descarga por file ID con la cuenta del usuario (Drive API, ``files().get_media``),
     por rangos, con reanudación y barra de progreso. Funciona con archivos privados.
  3. Drive montado en /content/drive (ruta configurable al zip): se copia al disco local.
  4. Local (pruebas): variable de entorno SEGPACK_DIR (carpeta) o SEGPACK_ZIP (zip).

El zip se verifica (tamaño + sha256) antes de descomprimir al disco local de la VM y la carpeta
resultante se valida contra manifest.csv. Durante el entrenamiento nunca se lee desde Drive.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml

MARKER = ".segpipe_validated.json"
CHUNK = 32 * 1024 * 1024  # 32 MiB por petición de rango


class DatasetError(RuntimeError):
    """Error de obtención o validación del dataset."""


# --------------------------------------------------------------------------- config

def load_dataset_config(path: str | os.PathLike) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ("name", "archive", "local", "layout"):
        if key not in cfg:
            raise DatasetError(f"{path}: falta la sección '{key}'")
    return cfg


def in_colab() -> bool:
    """True en un kernel de Colab (navegador o extensión de VS Code: ambos corren en la VM)."""
    if "google.colab" in sys.modules:
        return True
    import importlib.util
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        return False


def local_root(cfg: dict) -> Path:
    env = os.environ.get("SEGPACK_ROOT")
    if env:
        return Path(env)
    return Path(cfg["local"]["root_colab"] if in_colab() else cfg["local"]["root_other"])


# --------------------------------------------------------------------------- hashing / verify

def _progress(total: int, desc: str, initial: int = 0):
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, initial=initial, unit="B", unit_scale=True, unit_divisor=1024, desc=desc)
    except ImportError:  # pragma: no cover
        class _Dummy:
            def update(self, n): pass
            def close(self): pass
        return _Dummy()


def sha256_file(path: str | os.PathLike, show_progress: bool = False) -> str:
    h = hashlib.sha256()
    size = os.path.getsize(path)
    bar = _progress(size, f"sha256 {Path(path).name}") if show_progress else None
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            if bar:
                bar.update(len(chunk))
    if bar:
        bar.close()
    return h.hexdigest()


def verify_archive(path: str | os.PathLike, size_bytes: int, sha256: str, show_progress: bool = True) -> None:
    """Falla con DatasetError si el zip no tiene el tamaño o el sha256 esperados."""
    p = Path(path)
    if not p.is_file():
        raise DatasetError(f"No existe el zip: {p}")
    actual = p.stat().st_size
    if actual != size_bytes:
        raise DatasetError(f"Tamaño incorrecto de {p}: {actual} bytes, se esperaban {size_bytes}")
    digest = sha256_file(p, show_progress=show_progress)
    if digest != sha256.lower():
        raise DatasetError(f"sha256 incorrecto de {p}: {digest}, se esperaba {sha256}")


# --------------------------------------------------------------------------- extract

def safe_extract(zip_path: str | os.PathLike, dest: str | os.PathLike) -> Path:
    """Descomprime a una carpeta temporal y la renombra a ``dest`` (atómico a nivel de carpeta).

    Rechaza entradas con rutas absolutas o '..' (zip slip).
    """
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".extracting")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        base = tmp.resolve()
        for info in z.infolist():
            target = (tmp / info.filename).resolve()
            if base != target and base not in target.parents:
                raise DatasetError(f"Entrada insegura en el zip: {info.filename}")
        z.extractall(tmp)
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return dest


# --------------------------------------------------------------------------- validate

@dataclass
class ValidationReport:
    n_pairs: int
    problems: list

    @property
    def ok(self) -> bool:
        return not self.problems


def read_manifest(root: str | os.PathLike, manifest_name: str = "manifest.csv") -> list[dict]:
    with open(Path(root) / manifest_name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_extracted(root: str | os.PathLike, cfg: dict, check_image_hashes: bool = False) -> ValidationReport:
    """Valida la carpeta contra manifest.csv: parejas completas, tamaños de imagen y máscara."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    root = Path(root)
    lay = cfg["layout"]
    problems: list[str] = []
    for key in ("classes", "manifest"):
        if not (root / lay[key]).is_file():
            problems.append(f"falta {lay[key]}")
    if problems:
        return ValidationReport(0, problems)

    rows = read_manifest(root, lay["manifest"])
    names = set()
    for r in rows:
        name = r["name"]
        if name in names:
            problems.append(f"{name}: nombre duplicado en manifest")
        names.add(name)
        ip, mp = root / r["image"], root / r["mask"]
        if not ip.is_file():
            problems.append(f"{name}: falta imagen {r['image']}")
            continue
        if not mp.is_file():
            problems.append(f"{name}: falta máscara {r['mask']}")
            continue
        if r.get("bytes") and ip.stat().st_size != int(r["bytes"]):
            problems.append(f"{name}: imagen con {ip.stat().st_size} bytes, manifest dice {r['bytes']}")
        w, h = int(r["width"]), int(r["height"])
        with Image.open(ip) as im:
            isz = im.size
        with Image.open(mp) as im:
            msz = im.size
        if isz != (w, h):
            problems.append(f"{name}: imagen {isz} != manifest {(w, h)}")
        if msz != (w, h):
            problems.append(f"{name}: máscara {msz} != manifest {(w, h)}")
        if check_image_hashes and r.get("sha256") and sha256_file(ip) != r["sha256"]:
            problems.append(f"{name}: sha256 de imagen no coincide con manifest")
    # archivos sobrantes (imágenes o máscaras que no están en el manifest)
    listed = {Path(r["image"]).as_posix() for r in rows} | {Path(r["mask"]).as_posix() for r in rows}
    for sub in ("images", "masks"):
        d = root / sub
        if d.is_dir():
            for p in d.iterdir():
                rel = f"{sub}/{p.name}"
                if rel not in listed:
                    problems.append(f"archivo sin entrada en manifest: {rel}")
    return ValidationReport(len(rows), problems)


def _write_marker(root: Path, cfg: dict, report: ValidationReport, source: str) -> None:
    data = {"name": cfg["name"], "version": cfg.get("version"), "sha256": cfg["archive"]["sha256"],
            "n_pairs": report.n_pairs, "source": source, "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (root / MARKER).write_text(json.dumps(data, indent=1), encoding="utf-8")


def _marker_ok(root: Path, cfg: dict) -> bool:
    m = root / MARKER
    if not m.is_file():
        return False
    try:
        data = json.loads(m.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return data.get("sha256") == cfg["archive"]["sha256"]


# --------------------------------------------------------------------------- download

def download_ranges(fetch: Callable[[int, int], bytes], total: int, out_path: str | os.PathLike,
                    chunk: int = CHUNK, retries: int = 5, desc: str = "descarga") -> Path:
    """Descarga ``total`` bytes en rangos a ``out_path`` vía ``<out>.part`` con reanudación.

    ``fetch(start, end_inclusive)`` devuelve los bytes de ese rango. Si existe un ``.part``
    previo (sesión cortada) se continúa desde su tamaño.
    """
    out_path = Path(out_path)
    part = out_path.with_name(out_path.name + ".part")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = part.stat().st_size if part.exists() else 0
    if done > total:
        part.unlink()
        done = 0
    bar = _progress(total, desc, initial=done)
    try:
        with open(part, "ab") as f:
            while done < total:
                end = min(done + chunk, total) - 1
                for attempt in range(retries + 1):
                    try:
                        data = fetch(done, end)
                        break
                    except Exception as e:  # red inestable: reintento con espera exponencial
                        if attempt == retries:
                            raise DatasetError(f"Descarga fallida en bytes {done}-{end}: {e}") from e
                        time.sleep(min(2 ** attempt, 30))
                if len(data) != end - done + 1:
                    raise DatasetError(f"Rango {done}-{end}: se recibieron {len(data)} bytes")
                f.write(data)
                f.flush()
                done += len(data)
                bar.update(len(data))
    finally:
        bar.close()
    os.replace(part, out_path)
    return out_path


def drive_service():  # pragma: no cover - requiere Colab
    """Autentica con la cuenta del usuario en Colab y devuelve un cliente de Drive API v3."""
    from google.colab import auth
    auth.authenticate_user()
    from googleapiclient.discovery import build
    return build("drive", "v3", cache_discovery=False)


def download_from_drive(file_id: str, out_path: str | os.PathLike, expected_size: Optional[int] = None,
                        service=None) -> Path:
    """Descarga un archivo (privado o no) de Drive por file ID con ``files().get_media``."""
    service = service or drive_service()
    meta = service.files().get(fileId=file_id, fields="name,size", supportsAllDrives=True).execute()
    total = int(meta["size"])
    if expected_size is not None and total != expected_size:
        raise DatasetError(f"El archivo de Drive '{meta.get('name')}' mide {total} bytes, se esperaban {expected_size}")

    def fetch(start: int, end: int) -> bytes:
        req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        req.headers["Range"] = f"bytes={start}-{end}"
        return req.execute()

    return download_ranges(fetch, total, out_path, desc=meta.get("name", file_id))


# --------------------------------------------------------------------------- main entry

def _extract_and_validate(zip_path: Path, dest: Path, cfg: dict, source: str, log) -> Path:
    arc = cfg["archive"]
    log(f"Verificando {zip_path} (tamaño + sha256)…")
    verify_archive(zip_path, int(arc["size_bytes"]), arc["sha256"])
    log(f"Descomprimiendo en {dest}…")
    safe_extract(zip_path, dest)
    rep = validate_extracted(dest, cfg)
    if not rep.ok:
        raise DatasetError("Validación contra manifest fallida:\n  " + "\n  ".join(rep.problems[:20]))
    _write_marker(dest, cfg, rep, source)
    log(f"Dataset listo: {dest} ({rep.n_pairs} parejas, origen: {source})")
    return dest


def ensure_dataset(cfg: dict | str | os.PathLike, drive_mounted_path: Optional[str] = None,
                   try_api: bool = True, try_mount: bool = True, verbose: bool = True) -> Path:
    """Devuelve la carpeta descomprimida y validada del dataset, obteniéndola si hace falta."""
    if not isinstance(cfg, dict):
        cfg = load_dataset_config(cfg)
    log = print if verbose else (lambda *a, **k: None)
    arc, loc = cfg["archive"], cfg["local"]
    root = local_root(cfg)
    dest = root / loc["extract_dir"]
    zip_local = root / arc["filename"]
    errors: list[str] = []

    # 1. ya existe y validado
    if dest.is_dir() and _marker_ok(dest, cfg):
        rep = validate_extracted(dest, cfg)
        if rep.ok:
            log(f"Dataset ya presente y validado: {dest} ({rep.n_pairs} parejas)")
            return dest
        log("La carpeta existente no pasa la validación; se vuelve a obtener.")

    # 1b. zip ya descargado en disco local (p. ej. descarga previa sin descomprimir)
    if zip_local.is_file():
        try:
            return _extract_and_validate(zip_local, dest, cfg, "zip local previo", log)
        except DatasetError as e:
            errors.append(f"zip local previo: {e}")
            zip_local.unlink()

    colab = in_colab()
    # 2. Colab + Drive API con la cuenta del usuario
    if colab and try_api and arc.get("drive_file_id"):
        try:
            log("Descargando de Google Drive con tu cuenta (Drive API)…")
            download_from_drive(arc["drive_file_id"], zip_local, int(arc["size_bytes"]))
            return _extract_and_validate(zip_local, dest, cfg, "drive_api", log)
        except Exception as e:  # p. ej. auth no disponible en la extensión de VS Code
            errors.append(f"Drive API: {e}")
            log(f"Drive API no disponible ({e}); se prueba Drive montado.")

    # 3. Drive montado
    mounted = Path(drive_mounted_path or arc.get("drive_mounted_path") or "")
    if colab and try_mount and str(mounted) and not mounted.is_file() and not Path("/content/drive/MyDrive").exists():
        try:
            from google.colab import drive  # pragma: no cover
            drive.mount("/content/drive")
        except Exception as e:  # pragma: no cover
            errors.append(f"drive.mount: {e}")
    if str(mounted) not in ("", ".") and mounted.is_file():
        try:
            log(f"Copiando {mounted} al disco local…")
            zip_local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mounted, zip_local)
            return _extract_and_validate(zip_local, dest, cfg, "drive_montado", log)
        except DatasetError as e:
            errors.append(f"Drive montado: {e}")

    # 4. Local por variables de entorno (solo pruebas)
    env_dir = os.environ.get(loc.get("env_dir", "SEGPACK_DIR"))
    if env_dir:
        d = Path(env_dir)
        rep = validate_extracted(d, cfg, check_image_hashes=True)
        if rep.ok:
            log(f"Usando carpeta local {d} ({rep.n_pairs} parejas)")
            return d
        errors.append(f"SEGPACK_DIR={d}: " + "; ".join(rep.problems[:5]))
    env_zip = os.environ.get(loc.get("env_zip", "SEGPACK_ZIP"))
    if env_zip:
        try:
            return _extract_and_validate(Path(env_zip), dest, cfg, "SEGPACK_ZIP", log)
        except DatasetError as e:
            errors.append(f"SEGPACK_ZIP={env_zip}: {e}")

    raise DatasetError("No se pudo obtener el dataset. Intentos:\n  " + ("\n  ".join(errors) or
                       "ninguna fuente disponible (fuera de Colab define SEGPACK_ZIP o SEGPACK_DIR)"))
