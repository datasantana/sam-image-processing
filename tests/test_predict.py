"""Pruebas en CPU con datos sintéticos de segpipe.predict: ensamble, regla fuera de pliegue, formatos,
reanudación, verificación inicial, verificación contra la CV, zips y hoja de contacto."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from segpipe import metrics as M
from segpipe import predict as P
from segpipe.classes import ClassSchema

SOURCE = [{"index": 0, "name": "Road", "rgb": [89, 134, 179]},
          {"index": 1, "name": "Vegetation", "rgb": [36, 179, 83]},
          {"index": 2, "name": "Fence", "rgb": [184, 61, 245]},
          {"index": 3, "name": "background", "rgb": [0, 0, 0]}]
SCHEMA = ClassSchema.from_list(SOURCE, ignore=["background"])
NAMES = SCHEMA.names                      # Road, Vegetation, Fence
PAL = P.palette_from_schema(SCHEMA)
W, H = 96, 64
CPU = torch.device("cpu")


def const_logits(values):
    """logits_fn falso: cada pliegue da logits constantes ``values[fold]`` en toda la foto."""
    def fn(member, image, device, amp_dtype):
        v = torch.tensor(values[member.fold], dtype=torch.float32)
        return v[:, None, None].expand(len(v), image.shape[0], image.shape[1]).clone()
    return fn


def members(k=5):
    return [P.Member(f, None, {"infer": {"mode": "resize"}}) for f in range(k)]


def write_photos(d: Path, names, size=(W, H), seed=0):
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out = {}
    for n in names:
        p = d / f"{n}.JPG"
        Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(p, quality=95)
        out[n] = p
    return out


# ----------------------------------------------------------------------------- ensamble

def test_ensemble_averages_probabilities_not_logits():
    # media de logits → clase 1; media de probabilidades → clase 0
    vals = {0: [0.0, 10.0, -50.0], 1: [2.0, 0.0, -50.0], 2: [2.0, 0.0, -50.0]}
    img = np.zeros((4, 6, 3), np.uint8)
    pred, conf = P.predict_photo(members(3), img, CPU, None, const_logits(vals))
    probs = np.mean([torch.softmax(torch.tensor(v), 0).numpy() for v in vals.values()], 0)
    assert np.mean(list(vals.values()), 0).argmax() == 1 and probs.argmax() == 0
    assert pred.dtype == np.uint8 and pred.shape == (4, 6) and (pred == 0).all()
    assert np.allclose(conf, probs.max(), atol=1e-6)


def test_single_model_is_argmax_of_logits_and_softmax_conf():
    vals = {3: [0.5, 2.0, 1.0]}
    pred, conf = P.predict_photo([P.Member(3, None, {})], np.zeros((2, 3, 3), np.uint8), CPU, None,
                                 const_logits(vals))
    assert (pred == 1).all()
    assert np.allclose(conf, torch.softmax(torch.tensor(vals[3]), 0).max().item())


# ----------------------------------------------------------------------------- formatos

def test_conf_png_is_block_mean_quantized():
    rng = np.random.default_rng(1)
    conf = rng.random((H, W)).astype(np.float32)
    q = P.conf_to_png(conf)
    assert q.dtype == np.uint8 and q.shape == (H // 4, W // 4)
    ref = np.round(conf.reshape(H // 4, 4, W // 4, 4).mean((1, 3)) * 255)
    assert np.abs(q.astype(int) - ref).max() <= 1          # INTER_AREA en float32 vs media exacta


def test_photo_stats_columns():
    pred = np.array([[0, 0, 1, 2]], np.uint8)
    conf = np.array([[0.4, 0.6, 0.8, 0.95]], np.float32)
    s = P.photo_stats(pred, conf, NAMES)
    assert s["pct_Road"] == 50.0 and s["pct_Vegetation"] == 25.0 and s["pct_Fence"] == 25.0
    assert s["frac_conf_lt_0.5"] == 0.25 and s["frac_conf_lt_0.7"] == 0.5 and s["frac_conf_lt_0.9"] == 0.75
    assert abs(s["conf_media"] - 0.6875) < 1e-6


def run_predict(tmp_path, labelled, versions=("oof", "ensamble"), limit=None, mask_fn=None, photos=None):
    photos = photos or write_photos(tmp_path / "fotos", ["a", "b", "c"])
    vals = {f: [0.0] * 3 for f in range(5)}
    for f in range(5):
        vals[f][f % 3] = 5.0 + f                         # pliegue f → clase f % 3
    rows = P.predict_config("J1", members(), photos, labelled, tmp_path / "out" / "J1", PAL, NAMES, CPU, None,
                            versions=versions, mask_fn=mask_fn, staging=tmp_path / "stg", limit=limit,
                            logits_fn=const_logits(vals), log=lambda *a: None)
    return photos, rows


def test_oof_rule_and_output_formats(tmp_path):
    labelled = {"a": 2, "b": 4}
    photos, rows = run_predict(tmp_path, labelled, mask_fn=lambda n: np.full((H, W), 1, np.uint8))
    out = tmp_path / "out" / "J1"
    by = {(r["foto"], r["version"]): r for r in rows}
    assert set(by) == {("a", "oof"), ("b", "oof"), ("a", "ensamble"), ("b", "ensamble"), ("c", "ensamble")}
    assert by[("a", "oof")]["modelos"] == "J1/fold2" and by[("b", "oof")]["modelos"] == "J1/fold4"
    assert by[("c", "ensamble")]["modelos"] == "J1/fold0+fold1+fold2+fold3+fold4"
    assert by[("a", "oof")]["archivos"] == "masks_index_oof/a.png;masks_rgb_oof/a.png;conf_oof/a.png"
    assert "cm" in by[("a", "oof")] and "cm" not in by[("a", "ensamble")]
    assert np.array(by[("a", "oof")]["cm"])[1, 2] == H * W          # real 1, predicho 2 (pliegue 2)

    idx = Image.open(out / "masks_index_oof" / "a.png")
    assert idx.mode == "L" and idx.size == (W, H) and set(np.unique(np.asarray(idx))) == {2}
    rgb = Image.open(out / "masks_rgb" / "c.png")
    assert rgb.mode == "RGB" and rgb.size == (W, H)
    colors = {tuple(c) for c in np.asarray(rgb).reshape(-1, 3)}
    assert colors <= {tuple(c) for c in PAL.tolist()}
    cf = Image.open(out / "conf" / "c.png")
    assert cf.mode == "L" and cf.size == (W // 4, H // 4)
    assert not list(out.rglob("*.part")) and not list(out.rglob("*.jpg"))
    # el ensamble de las etiquetadas va a las carpetas normales, junto con el resto
    assert sorted(p.stem for p in (out / "masks_rgb").glob("*.png")) == ["a", "b", "c"]
    assert sorted(p.stem for p in (out / "masks_rgb_oof").glob("*.png")) == ["a", "b"]


def test_resume_skips_written_and_redoes_missing(tmp_path):
    labelled = {"a": 0}
    photos, rows1 = run_predict(tmp_path, labelled, limit=2)
    assert [(r["foto"], r["version"]) for r in rows1] == [("a", "oof"), ("a", "ensamble")]
    _, rows2 = run_predict(tmp_path, labelled, photos=photos)
    assert [(r["foto"], r["version"]) for r in rows2] == [("b", "ensamble"), ("c", "ensamble")]
    _, rows3 = run_predict(tmp_path, labelled, photos=photos)
    assert rows3 == []
    (tmp_path / "out" / "J1" / "conf" / "b.png").unlink()       # archivo perdido → se rehace
    with open(tmp_path / "out" / "J1" / "progress.jsonl", "a", encoding="utf-8") as f:
        f.write('{"foto": "c", "vers')                         # última línea a medias
    _, rows4 = run_predict(tmp_path, labelled, photos=photos)
    assert [(r["foto"], r["version"]) for r in rows4] == [("b", "ensamble")]


def test_manifest_lists_versions_and_models(tmp_path):
    run_predict(tmp_path, {"a": 1})
    out = tmp_path / "out" / "J1"
    p = P.write_manifest(out, P.Progress(out), NAMES)
    import csv
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert [(r["foto"], r["version"]) for r in rows] == [("a", "ensamble"), ("a", "oof"), ("b", "ensamble"),
                                                          ("c", "ensamble")]
    assert rows[0]["versiones_disponibles"] == "ensamble;oof" and rows[2]["versiones_disponibles"] == "ensamble"
    assert rows[1]["modelos"] == "J1/fold1" and rows[1]["pliegue_test"] == "1"
    for col in ("frac_conf_lt_0.5", "frac_conf_lt_0.7", "frac_conf_lt_0.9", "conf_media", "pct_Fence"):
        assert col in rows[0]


def test_zips_check_counts_and_contents(tmp_path):
    run_predict(tmp_path, {"a": 1, "b": 2})
    out = tmp_path / "out" / "J1"
    with pytest.raises(P.PredictError, match="se esperaban 4"):
        P.make_zips(out, n_ensemble=4, n_oof=2, staging=tmp_path / "stg")
    z = P.make_zips(out, n_ensemble=3, n_oof=2, staging=tmp_path / "stg")
    assert sorted(zipfile.ZipFile(z["masks_rgb.zip"]).namelist()) == [f"masks_rgb/{n}.png" for n in "abc"]
    assert len(zipfile.ZipFile(z["masks_index.zip"]).namelist()) == 3
    names = zipfile.ZipFile(z["oof.zip"]).namelist()
    assert len(names) == 6 and all(n.split("/")[0].endswith("_oof") for n in names)


# ----------------------------------------------------------------------------- verificación inicial

def preflight_setup(tmp_path, n=6):
    names = [f"DJI_{i:04d}_D" for i in range(n)]
    photos = write_photos(tmp_path / "fotos", names)
    lab_rows = [{"name": x, "image": f"images/{x}.JPG", "sha256": P.sha256_path(photos[x])} for x in names[:3]]
    splits = {f: {"test": [names[f]], "dir": tmp_path} for f in range(3)}
    return names, photos, lab_rows, splits


def test_preflight_ok(tmp_path):
    names, photos, lab, splits = preflight_setup(tmp_path)
    rep = P.preflight(tmp_path / "fotos", lab, {"J0": splits, "J1": splits}, expected_n=6, expected_wh=(W, H),
                      log=lambda *a: None)
    assert rep["labelled_fold"] == {names[0]: 0, names[1]: 1, names[2]: 2}


@pytest.mark.parametrize("case", ["count", "missing", "size", "sha", "orientation", "dup_test", "cfg_mismatch"])
def test_preflight_detects_problems(tmp_path, case):
    names, photos, lab, splits = preflight_setup(tmp_path)
    by_cfg = {"J0": splits, "J1": splits}
    n = 6
    if case == "count":
        n = 7
    elif case == "missing":
        photos[names[1]].unlink(); n = 5
    elif case == "size":
        Image.new("RGB", (W + 1, H)).save(photos[names[4]])
    elif case == "sha":
        Image.new("RGB", (W, H)).save(photos[names[0]])
    elif case == "orientation":
        im = Image.open(photos[names[5]]).copy()
        ex = Image.Exif(); ex[P.EXIF_ORIENTATION] = 6
        im.save(photos[names[5]], exif=ex)
    elif case == "dup_test":
        by_cfg = {"J0": {**splits, 3: {"test": [names[0]], "dir": tmp_path}}}
    elif case == "cfg_mismatch":
        by_cfg = {"J0": splits, "J1": {0: {"test": [names[1]]}, 1: {"test": [names[0]]}, 2: {"test": [names[2]]}}}
    expect = {"count": "6 fotos .*se esperaban 7","missing": "no están en la carpeta", "size": "no miden",
              "sha": "sha256 distinto", "orientation": "orientación EXIF", "dup_test": "pliegues 0 y 3",
              "cfg_mismatch": "difiere"}[case]
    with pytest.raises(P.PredictError, match=expect):
        P.preflight(tmp_path / "fotos", lab, by_cfg, expected_n=n, expected_wh=(W, H), log=lambda *a: None)


# ----------------------------------------------------------------------------- verificación contra la CV

def cms_for(n_photos, seed=0):
    rng = np.random.default_rng(seed)
    return {f"p{i}": rng.integers(0, 1000, (3, 3)) * np.eye(3, dtype=int) * 5 + rng.integers(0, 50, (3, 3))
            for i in range(n_photos)}


def test_verify_oof_passes_on_identical_and_reports_discrepancy():
    ref = cms_for(8)
    rows = [{"foto": n, "version": "oof", "cm": cm.tolist()} for n, cm in ref.items()]
    v = P.verify_oof(rows, ref, NAMES, min_photos=5)
    assert v["passed"] and v["delta_pp"] == 0 and v["px_distintos_max"] == 0 and v["fotos_con_diferencia"] == 0
    cm = np.array(rows[3]["cm"]); cm[0, 0] -= 7; cm[0, 1] += 7; rows[3]["cm"] = cm.tolist()
    v = P.verify_oof(rows, ref, NAMES, min_photos=5)
    assert v["px_distintos_max"] == 7 and v["foto_px_distintos_max"] == "p3" and v["fotos_con_diferencia"] == 1
    assert abs(v["delta_pp"]) <= 0.1 and v["passed"]


def test_verify_oof_fails_beyond_tolerance_and_on_missing():
    ref = cms_for(8)
    rows = [{"foto": n, "version": "oof", "cm": cm.tolist()} for n, cm in ref.items()]
    for r in rows:                                   # mueve un 20 % de Road a Vegetation en todas las fotos
        cm = np.array(r["cm"]); k = cm[0, 0] // 5; cm[0, 0] -= k; cm[0, 1] += k; r["cm"] = cm.tolist()
    v = P.verify_oof(rows, ref, NAMES, min_photos=5)
    assert not v["passed"] and v["delta_pp"] < -0.1
    with pytest.raises(P.PredictError, match="Faltan 1"):
        P.verify_oof(rows[1:], ref, NAMES, min_photos=5)


# ----------------------------------------------------------------------------- hoja de contacto

def test_pick_contact_photos_distinct_folds_and_deterministic():
    lab = {f"L{i}": i % 5 for i in range(20)}
    all_names = list(lab) + [f"U{i}" for i in range(30)]
    a = P.pick_contact_photos(lab, all_names, 4, 42)
    assert a == P.pick_contact_photos(lab, all_names, 4, 42)
    assert len(set(lab[x] for x in a["etiquetadas"])) == 4 and a["pliegues"] == sorted(lab[x] for x in a["etiquetadas"])
    assert all(x.startswith("U") for x in a["sin_etiquetar"]) and len(set(a["sin_etiquetar"])) == 4


# ----------------------------------------------------------------------------- de punta a punta con modelos reales

def tiny_study(tmp_path, infer):
    """Estudio con 5 pliegues de un DeepLabv3+ resnet18 sin preentrenar (pesos al azar, fijos por pliegue)."""
    from segpipe.models import build_model
    names = [f"DJI_{i:04d}_D" for i in range(12)]
    lab = names[:10]
    model_cfg = {"arch": "deeplabv3plus", "encoder": "resnet18", "weights": None}
    cfg = {"model": model_cfg, "data": {"mode": "resize", "resize": 64, "mean": [0.485, 0.456, 0.406],
                                       "std": [0.229, 0.224, 0.225]}, "infer": infer}
    for f in range(5):
        d = tmp_path / "study" / "runs" / "J0" / "seed42" / f"fold{f}"
        d.mkdir(parents=True)
        torch.manual_seed(f)
        m = build_model(model_cfg, 3).eval()
        torch.save({"model": m.state_dict(), "epoch": 9}, d / "best.pt")
        (d / "config.json").write_text(json.dumps({"config": cfg, "classes": NAMES}), encoding="utf-8")
        (d / "split.json").write_text(json.dumps({"test": lab[2 * f: 2 * f + 2]}), encoding="utf-8")
    return names, lab


@pytest.mark.parametrize("infer", [{"mode": "resize"},
                                   {"mode": "sliding", "window": 32, "overlap": 0.5, "scale": 0.75}],
                         ids=["resize_J0", "sliding_Mb3"])
def test_end_to_end_with_real_models(tmp_path, infer):
    from segpipe.infer import predict_logits
    names, lab = tiny_study(tmp_path, infer)
    study = tmp_path / "study"
    photos = write_photos(tmp_path / "fotos", names)
    rng = np.random.default_rng(3)
    masks = {n: rng.integers(0, 3, (H, W)).astype(np.uint8) for n in lab}
    for n in lab[:3]:
        masks[n][:5] = 255                                   # píxeles sin anotar
    members = P.load_members(study, "J0", NAMES, CPU, log=lambda *a: None)
    assert [m.fold for m in members] == list(range(5)) and all(m.best_epoch == 10 for m in members)
    labelled = P.test_fold_map(P.load_splits(study, "J0"))
    # "CV": matrices como las de evaluate_photos (predict_logits → argmax → confusion)
    for m in members:
        cms = []
        for n in m.test:
            lg = predict_logits(m.model, P.load_image(photos[n]), m.cfg["infer"], m.cfg["data"], CPU, None, multiple=16)
            cms.append(M.confusion(lg.argmax(0).numpy(), masks[n], 3))
        (m.run_dir / "test").mkdir()
        np.savez_compressed(m.run_dir / "test" / "photo_confusion.npz", cms=np.stack(cms), names=np.array(m.test))
    res = P.run_config("J0", study, photos, labelled, tmp_path / "pred", SCHEMA, CPU, mask_fn=masks.__getitem__,
                       min_photos=1, staging=tmp_path / "stg", members=members, log=lambda *a: None)
    v = res["verificacion"]
    assert v["passed"] and v["delta_pp"] == 0 and v["px_distintos_max"] == 0
    out = tmp_path / "pred" / "J0"
    assert len(list((out / "masks_rgb").glob("*.png"))) == 12 and len(list((out / "masks_rgb_oof").glob("*.png"))) == 10
    assert set(res["zips"]) == {"masks_rgb.zip", "masks_index.zip", "oof.zip"}
    mj = json.loads((out / "models.json").read_text(encoding="utf-8"))
    assert len(mj["modelos"]) == 5 and all(len(x["best_pt_sha256"]) == 64 for x in mj["modelos"])

    # hoja de contacto con 4 etiquetadas (pliegues distintos) y 2 sin etiquetar
    gt = {n: PAL[np.where(masks[n] == 255, 0, masks[n])] for n in lab}
    cs = P.contact_sheet(tmp_path / "pred", ["J0"], photos, labelled, gt.__getitem__,
                         [(n, tuple(c)) for n, c in zip(NAMES, PAL.tolist())], tmp_path / "pred" / "hoja", n=2, step=2)
    assert cs["png"].exists() and cs["pdf"].exists() and (tmp_path / "pred" / "hoja.csv").exists()


def test_end_to_end_stops_when_verification_fails(tmp_path):
    names, lab = tiny_study(tmp_path, {"mode": "resize"})
    study = tmp_path / "study"
    photos = write_photos(tmp_path / "fotos", names)
    masks = {n: np.zeros((H, W), np.uint8) for n in lab}
    for f in range(5):                                        # "CV" falsa: todo acertado
        d = study / "runs" / "J0" / "seed42" / f"fold{f}" / "test"
        d.mkdir()
        t = lab[2 * f: 2 * f + 2]
        np.savez_compressed(d / "photo_confusion.npz", cms=np.stack([np.diag([H * W, 0, 0])] * 2), names=np.array(t))
    labelled = P.test_fold_map(P.load_splits(study, "J0"))
    with pytest.raises(P.PredictError, match="no pasa"):
        P.run_config("J0", study, photos, labelled, tmp_path / "pred", SCHEMA, CPU, mask_fn=masks.__getitem__,
                     min_photos=1, staging=tmp_path / "stg", log=lambda *a: None)
    out = tmp_path / "pred" / "J0"
    assert (out / "verificacion_oof.json").exists()
    assert not (out / "masks_rgb").exists()                   # el ensamble no arrancó


# ----------------------------------------------------------------------------- GPU

@pytest.mark.parametrize("name,cap", [("NVIDIA A100-SXM4-40GB", (8, 0)), ("NVIDIA A100 80GB PCIe", (8, 0)),
                                      ("NVIDIA H100 80GB HBM3", (9, 0)), ("NVIDIA L4", (8, 9))])
def test_check_gpu_accepts_a100_h100_l4(name, cap):
    info = P.check_gpu(name, cap)
    assert info["gpu"] == name and info["bf16_nativo"] and info["compute_capability"] == f"{cap[0]}.{cap[1]}"


@pytest.mark.parametrize("name,cap,msg", [("Tesla T4", (7, 5), "no admitida"), ("NVIDIA T4", (8, 0), "no admitida"),
                                          ("NVIDIA L40S", (8, 9), "no admitida"), ("Tesla V100-SXM2-16GB", (7, 0), "no admitida"),
                                          ("NVIDIA A100-SXM4-40GB", (7, 5), "sin bf16 nativo"),
                                          (None, None, "No hay GPU")])
def test_check_gpu_blocks_t4_and_non_native_bf16(name, cap, msg):
    with pytest.raises(P.PredictError, match=msg):
        P.check_gpu(name, cap)


def test_gpu_recorded_in_manifest_and_models_json(tmp_path):
    photos = write_photos(tmp_path / "fotos", ["a", "b"])
    vals = {f: [5.0, 0.0, 0.0] for f in range(5)}
    out = tmp_path / "out" / "J1"
    P.predict_config("J1", members(), photos, {"a": 0}, out, PAL, NAMES, CPU, None, staging=tmp_path / "stg",
                     logits_fn=const_logits(vals), gpu="NVIDIA L4", log=lambda *a: None)
    import csv
    rows = list(csv.DictReader(open(P.write_manifest(out, P.Progress(out), NAMES), encoding="utf-8")))
    assert {r["gpu"] for r in rows} == {"NVIDIA L4"} and len(rows) == 3
    g1 = P.check_gpu("NVIDIA L4", (8, 9))
    P.write_models_json(out, "J1", members(), {"amp": "torch.bfloat16"}, g1)
    P.write_models_json(out, "J1", members(), {"amp": "torch.bfloat16"}, P.check_gpu("NVIDIA A100-SXM4-40GB", (8, 0)))
    mj = json.loads((out / "models.json").read_text(encoding="utf-8"))
    assert [s["gpu"] for s in mj["sesiones"]] == ["NVIDIA L4", "NVIDIA A100-SXM4-40GB"]
    assert all(s["bf16_nativo"] for s in mj["sesiones"]) and len(mj["modelos"]) == 5


# ----------------------------------------------------------------------------- por pliegue

def run_per_fold(tmp_path, labelled, limit=None, photos=None, oof_dir=None):
    photos = photos or write_photos(tmp_path / "fotos", ["a", "b", "c"])
    vals = {f: [0.0] * 3 for f in range(5)}
    for f in range(5):
        vals[f][f % 3] = 5.0 + f                         # pliegue f → clase f % 3
    rows = P.predict_per_fold("M-b3", members(), photos, tmp_path / "out" / "per_fold", NAMES, CPU, None,
                              labelled=labelled, oof_dir=oof_dir, staging=tmp_path / "stg", limit=limit,
                              logits_fn=const_logits(vals), log=lambda *a: None)
    return photos, rows


def test_per_fold_outputs_one_model_each(tmp_path):
    photos, rows = run_per_fold(tmp_path, {"a": 2})
    out = tmp_path / "out" / "per_fold"
    assert len(rows) == 15 and {r["version"] for r in rows} == {f"fold{f}" for f in range(5)}
    for f in range(5):
        idx = np.asarray(Image.open(out / f"fold{f}" / "masks_index" / "b.png"))
        assert set(np.unique(idx)) == {f % 3}
        cf = Image.open(out / f"fold{f}" / "conf" / "b.png")
        assert cf.size == (W // 4, H // 4)
        assert not (out / f"fold{f}" / "masks_rgb").exists()
    by = {(r["foto"], r["pliegue"]): r for r in rows}
    assert by[("a", 2)]["es_test_de_este_pliegue"] and not by[("a", 1)]["es_test_de_este_pliegue"]
    assert by[("a", 2)]["modelos"] == "M-b3/fold2"
    assert [r["foto"] for r in rows[:5]] == ["a"] * 5               # etiquetadas primero


def test_per_fold_matches_oof_and_compares(tmp_path):
    labelled = {"a": 2, "b": 4}
    photos, _ = run_predict(tmp_path, labelled, versions=("oof",))
    _, rows = run_per_fold(tmp_path, labelled, photos=photos, oof_dir=tmp_path / "out" / "J1")
    by = {(r["foto"], r["pliegue"]): r for r in rows}
    assert by[("a", 2)]["px_distintos_vs_oof"] == 0 and by[("b", 4)]["px_distintos_vs_oof"] == 0
    assert "px_distintos_vs_oof" not in by[("a", 1)] and "px_distintos_vs_oof" not in by[("c", 0)]
    out = tmp_path / "out"
    a_fold = np.asarray(Image.open(out / "per_fold" / "fold2" / "masks_index" / "a.png"))
    a_oof = np.asarray(Image.open(out / "J1" / "masks_index_oof" / "a.png"))
    assert (a_fold == a_oof).all()
    rep = P.verify_per_fold(out / "per_fold", range(5), photos)
    assert rep["pliegues"][2]["px_distintos_vs_oof_max"] == 0 and rep["pliegues"][0]["fotos"] == 3


def test_per_fold_resume_and_verify_missing(tmp_path):
    photos, rows1 = run_per_fold(tmp_path, {}, limit=1)
    assert {r["foto"] for r in rows1} == {"a"} and len(rows1) == 5
    _, rows2 = run_per_fold(tmp_path, {}, photos=photos)
    assert sorted({r["foto"] for r in rows2}) == ["b", "c"] and len(rows2) == 10
    _, rows3 = run_per_fold(tmp_path, {}, photos=photos)
    assert rows3 == []
    out = tmp_path / "out" / "per_fold"
    (out / "fold3" / "conf" / "c.png").unlink()                     # un archivo perdido → solo ese par
    with pytest.raises(P.PredictError, match="fold3: faltan 1"):
        P.verify_per_fold(out, range(5), photos)
    with open(out / "fold1" / "progress.jsonl", "a", encoding="utf-8") as f:
        f.write('{"foto": "c", "vers')                              # última línea a medias
    _, rows4 = run_per_fold(tmp_path, {}, photos=photos)
    assert [(r["foto"], r["pliegue"]) for r in rows4] == [("c", 3)]
    assert P.verify_per_fold(out, range(5), photos)["problemas"] == []


def test_verify_per_fold_flags_oof_mismatch(tmp_path):
    labelled = {"a": 2}
    photos, _ = run_predict(tmp_path, labelled, versions=("oof",))
    oof = tmp_path / "out" / "J1" / "masks_index_oof" / "a.png"
    arr = np.asarray(Image.open(oof)).copy()
    arr[:10, :10] = 0
    Image.fromarray(arr).save(oof)
    run_per_fold(tmp_path, labelled, photos=photos, oof_dir=tmp_path / "out" / "J1")
    with pytest.raises(P.PredictError, match="difieren de _oof"):
        P.verify_per_fold(tmp_path / "out" / "per_fold", range(5), photos, max_px_vs_oof=50)


def test_estimate_per_fold_hours():
    rows = [{"segundos_inferencia": 9.0, "segundos_escritura": 9.0}] * 5 + \
           [{"segundos_inferencia": 1.0, "segundos_escritura": 0.6}] * 10
    e = P.estimate_per_fold_hours(rows, n_photos=961, n_folds=5)
    assert e["s_por_foto_y_pliegue"] == 1.0 and abs(e["horas"] - 961 * 5 / 3600) < 1e-9
