import json

import numpy as np
import pytest
import torch

from segpipe import data_io
from segpipe.classes import ClassSchema
from segpipe.config import load_run_config
from segpipe.data import PhotoStore, build_train_dataset
from segpipe.infer import predict_logits
from segpipe.models import build_model
from segpipe.runner import REPO
from segpipe.train import Interrupted, train_run

TINY = {
    "model": {"encoder": "resnet18", "weights": None},
    "data": {"resize": 32, "crop": 32, "samples_per_epoch": 8},
    "infer": {"window": 32, "overlap": 0.5},
    "train": {"batch": 4, "micro_batch": 2, "epochs": 3, "patience": 10, "amp": False},
    "eval": {"min_photos": 1, "n_boot": 20},
}


@pytest.fixture
def store(pack, tmp_path):
    zpath, cfg = pack
    root = data_io.safe_extract(zpath, tmp_path / "ds")
    schema = ClassSchema.from_json(root / "classes.json", ignore=["background"])
    rows = data_io.read_manifest(root)
    return PhotoStore(root, rows, schema, verbose=False)


def split_of(store):
    names = sorted(store.photos)
    return {"fold": 0, "train": names[:6], "val": names[6:9], "test": names[9:12], "excluded": []}


def tiny_cfg(name, **extra):
    from segpipe.config import deep_merge
    return load_run_config(REPO / "configs" / "runs" / f"{name}.yaml", deep_merge(TINY, extra))


def test_deeplab_forward_backward():
    m = build_model({"arch": "deeplabv3plus", "encoder": "resnet50", "weights": None}, 12)
    x = torch.randn(2, 3, 64, 64)
    out = m(x)
    assert out.shape == (2, 12, 64, 64)
    out.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_segformer_b3_forward_backward():
    m = build_model({"arch": "segformer", "hf_name": "nvidia/mit-b3", "weights": None}, 12)
    assert sum(p.numel() for p in m.parameters()) > 40e6  # tamaño de mit-b3 (~47 M)
    x = torch.randn(2, 3, 64, 64)
    out = m(x)
    assert out.shape == (2, 12, 64, 64)
    loss = torch.nn.functional.cross_entropy(out, torch.randint(0, 12, (2, 64, 64)))
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("infer", [
    {"mode": "resize"},
    {"mode": "sliding", "window": 32, "overlap": 0.5, "scale": 1.0},
    {"mode": "sliding", "window": 32, "overlap": 0.5, "scale": 0.75},
])
def test_inference_returns_native_resolution(store, infer):
    m = build_model({"arch": "deeplabv3plus", "encoder": "resnet18", "weights": None}, 3).eval()
    p = next(iter(store.photos.values()))
    data_cfg = tiny_cfg("J0")["data"]
    out = predict_logits(m, p.image, infer, data_cfg, torch.device("cpu"), multiple=16)
    assert out.shape == (3,) + p.mask.shape


def test_crop_dataset_and_rare_sampler(store):
    cfg = tiny_cfg("J2")
    names = split_of(store)["train"]
    ds, rare = build_train_dataset(store, names, cfg["data"], 0, store.schema.names)
    assert store.schema.names[rare[0]] == "Fence" if rare else True
    x, y = ds[0]
    assert x.shape == (3, 32, 32) and y.shape == (32, 32)
    ds.set_epoch(1); a = ds[3][0]
    ds.set_epoch(1); b = ds[3][0]
    assert torch.equal(a, b)  # determinista por (semilla, época, índice)


@pytest.mark.parametrize("name", ["J0", "J1"])
def test_resume_gives_same_metrics(store, tmp_path, name):
    cfg = tiny_cfg(name)
    split = split_of(store)
    a = train_run(cfg, store, split, tmp_path / "a" / "x" / "y", seed=7, log=lambda *a: None)
    with pytest.raises(Interrupted):
        train_run(cfg, store, split, tmp_path / "b" / "x" / "y", seed=7, log=lambda *a: None, stop_after_epochs=1)
    rb = tmp_path / "b" / "x" / "y"
    assert (rb / "last.pt").exists() and not (rb / "DONE.json").exists()
    b = train_run(cfg, store, split, rb, seed=7, log=lambda *a: None)
    assert a["test"]["mIoU"] == pytest.approx(b["test"]["mIoU"], abs=1e-9)
    la = (tmp_path / "a" / "x" / "y" / "epoch_log.csv").read_text().splitlines()
    lb = (rb / "epoch_log.csv").read_text().splitlines()
    def strip(lines):  # sin columnas de tiempo
        head = lines[0].split(",")
        keep = [i for i, h in enumerate(head) if h not in ("seconds", "train_seconds", "val_seconds")]
        return [[r.split(",")[i] for i in keep] for r in lines]
    assert strip(la) == strip(lb)
    wa = torch.load(tmp_path / "a" / "x" / "y" / "best.pt", weights_only=False)["model"]
    wb = torch.load(rb / "best.pt", weights_only=False)["model"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    # una corrida terminada se salta
    assert train_run(cfg, store, split, rb, seed=7, log=lambda *a: None)["test"] == b["test"]
    rep = json.loads((rb / "test" / "metrics.json").read_text(encoding="utf-8"))
    assert rep["n_photos"] == 3 and len(rep["classes"]) == 3


def test_changed_config_in_same_run_dir_is_rejected(store, tmp_path):
    cfg = tiny_cfg("J0")
    split = split_of(store)
    with pytest.raises(Interrupted):
        train_run(cfg, store, split, tmp_path / "r" / "a" / "b", seed=1, log=lambda *a: None, stop_after_epochs=1)
    cfg2 = tiny_cfg("J0", optim={"lr": 1e-3})
    with pytest.raises(RuntimeError, match="no coincide"):
        train_run(cfg2, store, split, tmp_path / "r" / "a" / "b", seed=1, log=lambda *a: None)


def test_step_budget_is_logged_and_j0_pasos_matches_j1(store, tmp_path):
    split = split_of(store)
    over = {"data": {"samples_per_epoch": 16}, "train": {"epochs": 2}}
    j0 = train_run(tiny_cfg("J0", data={"samples_per_epoch": None}, train={"epochs": 2}), store, split,
                   tmp_path / "a" / "b" / "j0", 3, log=lambda *a: None)
    jp = train_run(tiny_cfg("J0-pasos", **over), store, split, tmp_path / "a" / "b" / "jp", 3, log=lambda *a: None)
    j1 = train_run(tiny_cfg("J1", **over), store, split, tmp_path / "a" / "b" / "j1", 3, log=lambda *a: None)
    # batch 4 = 2 micro-lotes de 2. J0: 6 fotos → 3 micro-lotes → 2 pasos/época.
    assert j0["optimizer_steps_total"] == 2 * 2
    # J0-pasos y J1: 16 muestras/época → 8 micro-lotes → 4 pasos/época: mismo presupuesto
    assert jp["optimizer_steps_total"] == j1["optimizer_steps_total"] == 2 * 4
    log = (tmp_path / "a" / "b" / "jp" / "epoch_log.csv").read_text().splitlines()
    head = log[0].split(",")
    for col in ("opt_steps", "train_seconds", "val_seconds", "n_train_samples", "n_val"):
        assert col in head
    assert log[-1].split(",")[head.index("n_train_samples")] == "16"


def test_mb3_recipe_runs_and_selects_by_maxmin(store, tmp_path):
    # SegFormer pequeño (arquitectura mit-b0, sin descargar pesos) con la receta M-b3 completa:
    # escala aleatoria [0.5, 1], AdamW + poly, CE sin pesos, inferencia a escala 0.75, selección max-min.
    cfg = tiny_cfg("M-b3", model={"hf_name": "nvidia/mit-b0", "weights": None},
                   train={"epochs": 2, "patience": 15})
    assert cfg["data"]["scale_range"] == [0.5, 1.0] and cfg["infer"]["scale"] == 0.75
    assert cfg["train"]["select"] == "maxmin" and cfg["sched"]["name"] == "poly"
    r = train_run(cfg, store, split_of(store), tmp_path / "a" / "b" / "m", 42, log=lambda *a: None)
    assert r["select"] == "maxmin" and r["optimizer_steps_total"] == 2 * 2
    import csv
    rows = list(csv.DictReader(open(tmp_path / "a" / "b" / "m" / "epoch_log.csv", encoding="utf-8")))
    best = max(float(x["val_min_iou"]) for x in rows)
    assert r["select_value"] == pytest.approx(best)
    lrs = [float(x["lr"]) for x in rows]
    assert lrs[1] < lrs[0]  # poly decae por iteración


def test_scale_range_crops_cover_larger_native_area(store):
    cfg = tiny_cfg("M-b3")
    names = split_of(store)["train"]
    ds, _ = build_train_dataset(store, names, {**cfg["data"], "augment": "none"}, 0, store.schema.names)
    import numpy as np
    sides = []
    orig = ds._scaled_crop
    ds._scaled_crop = lambda img, mask, cy, cx, side, rng: (sides.append(side), orig(img, mask, cy, cx, side, rng))[1]
    for i in range(40):
        x, y = ds[i]
        assert x.shape == (3, 32, 32)
    s = np.array(sides) / 32  # side nativo = crop / escala → escala = 32 / side
    assert (s >= 1.0 - 1e-9).all() and (s <= 2.0 + 1e-9).all() and s.std() > 0.1
