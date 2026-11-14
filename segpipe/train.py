"""Entrenamiento de una corrida (config × semilla × pliegue) con checkpoint por época y reanudación.

Estructura de ``run_dir`` (normalmente en Google Drive):
  config.json     config resuelta          split.json   partición usada (train/val/test)
  env.json        versiones, GPU, commit (una entrada por sesión)
  last.pt         estado completo al final de la última época (modelo, optimizador, scheduler,
                  scaler, época, mejor métrica, contador de paciencia, historial, estado RNG)
  best.pt         pesos del mejor checkpoint según la métrica de selección
  epoch_log.csv   pérdida, métricas de val, lr y segundos por época
  test/           matrices de confusión por foto + metrics.json / metrics_per_class.csv
  val/            ídem para val
  DONE.json       resumen; si existe, la corrida se salta
Si la sesión se corta, volver a llamar ``train_run`` con el mismo ``run_dir`` continúa en la época
siguiente a la del último ``last.pt``.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from . import metrics as M
from .classes import IGNORE_INDEX, ClassSchema
from .data import PhotoStore, build_train_dataset
from .env import git_commit, gpu_info, micro_batch, num_workers, versions
from .infer import predict_logits
from .losses import build_loss, class_weights
from .models import build_model, input_multiple


class Interrupted(RuntimeError):
    """Interrupción simulada (pruebas de reanudación)."""


# ----------------------------------------------------------------------------- utilidades

def atomic_save(obj, dst: Path) -> None:
    """torch.save a un temporal local y luego copia+rename en destino (seguro en Drive)."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(obj, tmp_path)
        part = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(tmp_path, part)
        os.replace(part, dst)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def rng_state() -> dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)


def build_optim(params, cfg: dict):
    o = cfg["optim"]
    if o["name"] == "adam":
        return torch.optim.Adam(params, lr=float(o["lr"]), weight_decay=float(o.get("weight_decay", 0)))
    if o["name"] == "adamw":
        return torch.optim.AdamW(params, lr=float(o["lr"]), weight_decay=float(o.get("weight_decay", 0.01)))
    raise ValueError(f"optim.name desconocido: {o['name']}")


def build_sched(opt, cfg: dict, iters_per_epoch: int):
    s = cfg["sched"]
    if s["name"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=float(s["factor"]),
                                                          patience=int(s["patience"]))
    if s["name"] == "poly":  # por iteración, sobre epochs × iters_per_epoch
        total = max(1, int(cfg["train"]["epochs"]) * iters_per_epoch)
        power = float(s.get("power", 1.0))
        return torch.optim.lr_scheduler.LambdaLR(opt, lambda it: max(0.0, 1 - it / total) ** power)
    if s["name"] == "none":
        return None
    raise ValueError(f"sched.name desconocido: {s['name']}")


def _select_value(rep: dict, how: str) -> float:
    return rep["min_iou"] if how == "maxmin" else rep["mIoU"]


# ----------------------------------------------------------------------------- evaluación

def evaluate_photos(model, store: PhotoStore, names: list[str], cfg: dict, device, amp_dtype, criterion=None,
                    num_classes: Optional[int] = None):
    """Matrices de confusión por foto (N, C, C) a resolución nativa y pérdida media."""
    C = num_classes or store.schema.num_classes
    cms, losses = [], []
    mult = input_multiple(cfg["model"])
    for n in names:
        p = store[n]
        logits = predict_logits(model, p.image, cfg["infer"], cfg["data"], device, amp_dtype, multiple=mult)
        target = torch.from_numpy(p.mask).to(device).long()
        if criterion is not None:
            with torch.no_grad():
                losses.append(float(criterion(logits[None], target[None])))
        pred = logits.argmax(0)
        cms.append(M.confusion_torch(pred, target, C, IGNORE_INDEX).cpu().numpy())
        del logits
    return np.stack(cms), (float(np.mean(losses)) if losses else float("nan"))


def val_summary(cms: np.ndarray, min_photos: int) -> dict:
    ev = M.photos_per_class(cms) >= min_photos
    cm = cms.sum(0)
    s = M.summarize(cm, ev)
    iou = M.per_class(cm)["iou"]
    s["min_iou"] = float(np.nanmin(iou[ev])) if ev.any() else float("nan")
    s["class_iou"] = [None if not np.isfinite(v) else float(v) for v in iou]
    return s


# ----------------------------------------------------------------------------- corrida

def train_run(cfg: dict, store: PhotoStore, split: dict, run_dir: str | Path, seed: int,
              device: Optional[torch.device] = None, log: Callable = print,
              stop_after_epochs: Optional[int] = None, keep_last: bool = False,
              extra_env: Optional[dict] = None) -> dict:
    """Entrena (o reanuda) una corrida y evalúa el mejor checkpoint en val y test."""
    run_dir = Path(run_dir)
    done = run_dir / "DONE.json"
    if done.exists():
        log(f"[skip] {run_dir} ya terminada")
        return json.loads(done.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schema: ClassSchema = store.schema
    C = schema.num_classes
    gpu = gpu_info()

    # --- metadatos de la corrida
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old.get("config") != json.loads(json.dumps(cfg, default=str)):
            raise RuntimeError(f"{run_dir}: la config guardada no coincide con la actual; usa otro run_dir")
    write_json(cfg_path, {"config": cfg, "seed": seed, "classes": schema.names, "ignored": schema.ignored})
    write_json(run_dir / "split.json", split)
    env_path = run_dir / "env.json"
    sessions = json.loads(env_path.read_text(encoding="utf-8"))["sessions"] if env_path.exists() else []
    mb = micro_batch(cfg, gpu)
    eff = int(cfg["train"]["batch"])
    accum = max(1, eff // mb)
    sessions.append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "gpu": gpu, "versions": versions(),
                     "git": git_commit(), "micro_batch": mb, "accum_steps": accum,
                     "num_workers": num_workers(), **(extra_env or {})})
    write_json(env_path, {"sessions": sessions})

    # --- datos
    train_names, val_names, test_names = split["train"], split["val"], split["test"]
    counts = store.pixel_counts(train_names)
    ds, rare = build_train_dataset(store, train_names, cfg["data"], seed, schema.names)
    if rare:
        log("clases raras para el muestreo: " + ", ".join(schema.names[c] for c in rare))
    nw = num_workers()
    drop_last = len(ds) % mb == 1  # evita un lote de 1 (BatchNorm)
    iters_per_epoch = (len(ds) // mb if drop_last else -(-len(ds) // mb))
    opt_steps_per_epoch = -(-iters_per_epoch // accum)

    # --- modelo y optimización
    seed_everything(seed)
    model = build_model(cfg["model"], C).to(device)
    w = class_weights(counts, cfg["loss"].get("weights"))
    criterion = build_loss(cfg["loss"], w.to(device) if w is not None else None).to(device)
    opt = build_optim(model.parameters(), cfg)
    sched = build_sched(opt, cfg, opt_steps_per_epoch)
    amp_on = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = (torch.bfloat16 if gpu.get("bf16") else torch.float16) if amp_on else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on and amp_dtype == torch.float16)
    select = cfg["train"].get("select", "miou")
    sel_min = int(cfg["train"].get("select_min_photos", 1))

    start_epoch, best, bad, history = 0, -float("inf"), 0, []
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched is not None and ck.get("scheduler") is not None:
            sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, bad, history = ck["epoch"] + 1, ck["best"], ck["bad_epochs"], ck["history"]
        set_rng_state(ck["rng"])
        log(f"[resume] {run_dir.name}: continúa en la época {start_epoch + 1} (mejor {select}={best:.4f})")
    opt_steps = int(history[-1]["opt_steps"]) if history else 0   # pasos de optimización acumulados

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["patience"])
    loader_kw = dict(batch_size=mb, num_workers=nw, pin_memory=device.type == "cuda", drop_last=drop_last)

    for epoch in range(start_epoch, epochs):
        if bad >= patience:
            break
        t0 = time.time()
        model.train()
        ds.set_epoch(epoch)
        g = torch.Generator().manual_seed(int(np.random.SeedSequence([seed, epoch]).generate_state(1)[0]))
        loader = torch.utils.data.DataLoader(ds, shuffle=cfg["data"]["mode"] == "resize", generator=g, **loader_kw)
        lr = opt.param_groups[0]["lr"]
        tot, nb, n_samples = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_on):
                out = model(x)
            loss = criterion(out.float(), y)
            scaler.scale(loss / accum).backward()
            last_in_step = (it + 1) % accum == 0 or it + 1 == iters_per_epoch
            if last_in_step:
                if cfg["train"].get("grad_clip"):
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"]["grad_clip"]))
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                opt_steps += 1
                if sched is not None and cfg["sched"]["name"] == "poly":
                    sched.step()
            tot += float(loss.detach()); nb += 1; n_samples += len(x)
        train_loss = tot / max(1, nb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - t0

        cms, val_loss = evaluate_photos(model, store, val_names, cfg, device, amp_dtype, criterion, C)
        vs = val_summary(cms, sel_min)
        if sched is not None and cfg["sched"]["name"] == "plateau":
            sched.step(val_loss)
        cur = _select_value(vs, select)
        improved = np.isfinite(cur) and cur > best
        if improved:
            best, bad = float(cur), 0
            atomic_save({"model": model.state_dict(), "epoch": epoch, "select": select, "value": best,
                         "classes": schema.names}, run_dir / "best.pt")
        else:
            bad += 1
        rec = {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "val_mIoU": vs["mIoU"],
               "val_min_iou": vs["min_iou"], "val_OA": vs["OA"], "lr": lr, "seconds": time.time() - t0,
               "train_seconds": t_train, "val_seconds": time.time() - t0 - t_train,
               "n_train_samples": n_samples, "n_val": len(val_names), "opt_steps": opt_steps,
               "best": best, "improved": bool(improved),
               **{f"val_iou_{n}": v for n, v in zip(schema.names, vs["class_iou"])}}
        history.append(rec)
        atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict() if sched is not None else None,
                     "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "bad_epochs": bad,
                     "history": history, "rng": rng_state()}, last)
        _write_log(run_dir / "epoch_log.csv", history)
        log(f"[{run_dir.parent.parent.name}/{run_dir.parent.name}/{run_dir.name}] época {epoch + 1}/{epochs} "
            f"loss {train_loss:.4f} | val loss {val_loss:.4f} mIoU {vs['mIoU']:.4f} min {vs['min_iou']:.4f} | "
            f"lr {lr:.2e} | pasos {opt_steps} | {rec['seconds']:.0f}s (train {t_train:.0f}s)" + (" *" if improved else ""))
        if stop_after_epochs is not None and epoch + 1 >= stop_after_epochs and epoch + 1 < epochs:
            raise Interrupted(f"interrupción simulada tras la época {epoch + 1}")

    # --- evaluación final con el mejor checkpoint
    if not (run_dir / "best.pt").exists():
        raise RuntimeError(f"{run_dir}: no hay best.pt (¿métrica de selección siempre NaN?)")
    bk = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(bk["model"])
    ev = cfg["eval"]
    result = {"run_dir": str(run_dir), "seed": seed, "best_epoch": bk["epoch"] + 1, "select": select,
              "select_value": bk["value"], "epochs_trained": len(history),
              "optimizer_steps_total": int(history[-1]["opt_steps"]) if history else 0,
              "optimizer_steps_at_best": int(history[bk["epoch"]]["opt_steps"]) if history else 0,
              "train_hours": sum(float(h["seconds"]) for h in history) / 3600,
              "buffer_m": split.get("buffer_m"),
              "min_dist_train_test_m": split.get("min_dist_train_test_m"),
              "overlap_train_test_pct": split.get("overlap_train_test_pct")}
    for part, names in (("val", val_names), ("test", test_names)):
        cms, _ = evaluate_photos(model, store, names, cfg, device, amp_dtype, None, C)
        (run_dir / part).mkdir(exist_ok=True)
        np.savez_compressed(run_dir / part / "photo_confusion.npz", cms=cms, names=np.array(names))
        rep = M.evaluate(cms, schema.names, names, min_photos=int(ev["min_photos"]),
                         n_boot=int(ev["n_boot"]), seed=seed)
        M.save_report(rep, run_dir / part)
        result[part] = {k: rep[k] for k in ("mIoU", "mAcc", "mF1", "OA", "ci", "not_evaluable_classes")}
    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    write_json(done, result)
    if not keep_last:
        last.unlink(missing_ok=True)
    log(f"[done] {run_dir}: test mIoU {result['test']['mIoU']:.4f} (mejor época {result['best_epoch']})")
    return result


def _write_log(path: Path, history: list[dict]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(history[0].keys()), lineterminator="\n")
    w.writeheader()
    w.writerows(history)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8", newline="")
    os.replace(tmp, path)
