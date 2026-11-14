"""Entorno de ejecución: GPU, batch por micro-lotes, workers, versiones y commit de git."""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Optional


def gpu_info() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {"device": "cpu", "name": None, "mem_gb": 0.0, "bf16": False}
    p = torch.cuda.get_device_properties(0)
    return {"device": "cuda", "name": p.name, "mem_gb": round(p.total_memory / 1024 ** 3, 1),
            "bf16": bool(torch.cuda.is_bf16_supported())}


def micro_batch(run_cfg: dict, gpu: dict) -> int:
    """Micro-lote que cabe en la GPU; el batch efectivo (train.batch) se mantiene con acumulación.

    Heurística conservadora (AMP activo); se puede fijar con ``train.micro_batch``.
    """
    eff = int(run_cfg["train"]["batch"])
    if run_cfg["train"].get("micro_batch"):
        return min(eff, int(run_cfg["train"]["micro_batch"]))
    if gpu["device"] != "cuda":
        return min(eff, 2)
    side = run_cfg["data"]["resize"] if run_cfg["data"]["mode"] == "resize" else run_cfg["data"]["crop"]
    px = (side / 1024) ** 2  # coste relativo a un recorte de 1024
    mem = gpu["mem_gb"]
    budget = 8 if mem >= 38 else 4 if mem >= 20 else 2   # recortes de 1024 por paso
    mb = max(1, int(budget / max(px, 1e-6)))
    return max(1, min(eff, mb))


def num_workers() -> int:
    import torch
    if not torch.cuda.is_available():
        return 0
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def versions() -> dict:
    out = {"python": platform.python_version()}
    for mod in ("torch", "torchvision", "transformers", "segmentation_models_pytorch", "timm",
                "albumentations", "numpy", "cv2", "PIL"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = None
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
    except Exception:
        pass
    return out


def git_commit(repo_dir: Optional[str | Path] = None) -> dict:
    repo_dir = str(repo_dir or Path(__file__).resolve().parents[1])
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=repo_dir, text=True, stderr=subprocess.DEVNULL).strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}
