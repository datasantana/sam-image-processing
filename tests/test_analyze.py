"""Pruebas del análisis posterior con un estudio sintético (sin torch ni GPU)."""
import csv
import json
import math

import numpy as np
import pytest
import yaml

from segpipe import analyze as A
from segpipe import split as S
from tests.conftest import MANIFEST_COLS

CAMERA = {"focal_mm": 6.72, "pixel_mm": 0.0026187585746188587, "width": 4032, "height": 2268, "max_range_m": 400}
GEOM = {"lat": "GpsLatitude", "lon": "GpsLongitude", "height": "RelativeAltitude", "pitch": "GimbalPitchDegree",
        "yaw": "GimbalYawDegree", "roll": "GimbalRollDegree", "camera": CAMERA}
CLASSES = ["Road", "Vegetation", "Fence"]


def manifest_rows(positions, yaw=0.0):
    lat0, lon0 = 3.43, -76.54
    rows = []
    for i, (dx, dy) in enumerate(positions):
        lat = lat0 + dy / 6371000 * 180 / math.pi
        lon = lon0 + dx / (6371000 * math.cos(math.radians(lat0))) * 180 / math.pi
        rows.append({"name": f"P{i:03d}", "image": "", "mask": "", "width": 4032, "height": 2268,
                     "mask_size_ok": True, "bytes": 0, "sha256": "", "GpsLatitude": f"{lat:.9f}",
                     "GpsLongitude": f"{lon:.9f}", "AbsoluteAltitude": "1080", "RelativeAltitude": "58.7",
                     "GimbalYawDegree": str(yaw), "GimbalPitchDegree": "-45", "GimbalRollDegree": "0",
                     "FlightYawDegree": "0", "FlightPitchDegree": "0", "FlightRollDegree": "0"})
    return rows


def write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)
    return path


def write_dataset_cfg(path):
    path.write_text(yaml.safe_dump({"name": "synthetic", "archive": {}, "local": {}, "layout": {},
                                    "geometry": GEOM}), encoding="utf-8")
    return path


def make_study(tmp_path, configs=("J0", "J1", "J0-b75"), k=3):
    pos = [(x * 100.0, y * 100.0) for y in range(3) for x in range(5)]
    rows = manifest_rows(pos)
    man = write_manifest(tmp_path / "manifest.csv", rows)
    dcfg = write_dataset_cfg(tmp_path / "ds.yaml")
    folds0 = S.make_folds(rows, GEOM, k=k, block_m=150, buffer_m=0)
    fp = S.footprints(rows, GEOM)
    names = [r["name"] for r in rows]
    folds75 = S.folds_from_assignment(names, fp, S.fold_assignment(folds0), k, 75)
    study = tmp_path / "study"
    params = {"dataset": "synthetic", "dataset_version": "t", "protocol": "cv", "k": k, "block_m": 150,
              "buffer_m": 0, "split_seed": 0, "footprint": "camera_centroid", "camera": CAMERA,
              "classes": CLASSES, "ignored": ["background"]}
    study.mkdir()
    (study / "study.json").write_text(json.dumps({"params": params}), encoding="utf-8")
    rng = np.random.default_rng(0)
    for ci, c in enumerate(configs):
        folds = folds75 if c == "J0-b75" else folds0
        for f in folds:
            d = study / "runs" / c / "seed42" / f"fold{f.fold}"
            (d / "test").mkdir(parents=True)
            cms = []
            for _ in f.test:
                m = rng.integers(0, 50, (3, 3))
                m[np.diag_indices(3)] = rng.integers(200, 400, 3) + 30 * ci
                cms.append(m)
            np.savez_compressed(d / "test" / "photo_confusion.npz", cms=np.array(cms), names=np.array(f.test))
            sp = {"fold": f.fold, "train": f.train, "val": f.val, "test": f.test, "excluded": f.excluded,
                  "protocol": "cv", "buffer_m": f.buffer_m, "min_dist_train_test_m": f.min_dist_train_test,
                  "overlap_train_test_pct": f.overlap_train_test_pct}
            (d / "split.json").write_text(json.dumps(sp), encoding="utf-8")
            (d / "config.json").write_text(json.dumps({"config": {"eval": {"min_photos": 1}}}), encoding="utf-8")
            (d / "DONE.json").write_text(json.dumps({
                "best_epoch": 2, "epochs_trained": 3, "optimizer_steps_total": 30, "optimizer_steps_at_best": 20,
                "train_hours": 0.5, "buffer_m": f.buffer_m, "test": {"mIoU": 0.5}}), encoding="utf-8")
            with open(d / "epoch_log.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["epoch", "train_loss", "val_mIoU", "opt_steps"])
                for e in range(1, 4):
                    w.writerow([e, 1 / e, 0.3 + 0.1 * e, 10 * e])
            (d / "test" / "metrics.json").write_text("{}", encoding="utf-8")
            (d / "test" / "metrics_per_class.csv").write_text("", encoding="utf-8")
            (d / "best.pt").write_bytes(b"no-es-un-checkpoint")
    return study, man, dcfg


# ----------------------------------------------------------------------------- A1

def test_random_split_sizes_and_determinism():
    names = [f"F{i:02d}" for i in range(66)]
    tr, te = A.random_split(names[::-1], 0.8, 42)
    assert len(tr) == 52 and len(te) == 14 and not set(tr) & set(te)
    assert (tr, te) == A.random_split(names, 0.8, 42)     # el orden de entrada no importa (se ordena)


def test_random_split_matches_sklearn():
    sk = pytest.importorskip("sklearn.model_selection")
    names = [f"DJI_{i:04d}" for i in range(66)]
    tr, te = sk.train_test_split(sorted(names), train_size=0.8, random_state=42, shuffle=True)
    assert A.random_split(names, 0.8, 42) == (tr, te)


def test_coverage_identical_and_far_footprints(tmp_path):
    rows = manifest_rows([(0, 0), (0, 0), (5000, 0)])
    geo = A.load_geometry(write_manifest(tmp_path / "m.csv", rows), GEOM)
    cov = geo.coverage(["P000", "P002"], ["P001"])
    assert cov[0] == pytest.approx(100) and cov[1] == pytest.approx(0)
    assert geo.union_overlap(["P000"], ["P001"]) == pytest.approx(100)
    assert geo.min_dist(["P000"], ["P001"]) == pytest.approx(0, abs=1e-6)


def test_a1_reports_random_and_cv(tmp_path):
    study, man, _ = make_study(tmp_path)
    runs, _ = A.load_runs(study)
    geo = A.load_geometry(man, GEOM)
    a1 = A.analysis_a1(geo, geo.names, runs)
    esq = [s["esquema"] for s in a1["summary"]]
    assert esq == ["aleatorio_80_20", "cv_bloques_buffer0", "cv_bloques_buffer75"]
    assert a1["max_diff_vs_guardado_pp"] < 1e-9           # mismo método que segpipe.split
    s = {x["esquema"]: x for x in a1["summary"]}
    assert s["cv_bloques_buffer75"]["solape_union_medio_pct"] <= s["cv_bloques_buffer0"]["solape_union_medio_pct"]


# ----------------------------------------------------------------------------- A2

def test_photo_iou_by_hand():
    cm = np.array([[[2, 1, 0], [0, 3, 0], [0, 0, 0]]])   # clase 2 sin unión → fuera
    # IoU0 = 2/(3+2-2)=2/3, IoU1 = 3/(3+4-3)=3/4
    assert A.photo_iou(cm)[0] == pytest.approx((2 / 3 + 3 / 4) / 2)


def test_spearman_and_partial_by_fold():
    x = np.arange(12, dtype=float)
    assert A.spearman(x, x ** 2) == pytest.approx(1)
    g = np.repeat([0, 1, 2], 4)
    within = np.tile([0, 1, 2, 3], 3).astype(float)
    y = within - 10 * g                                    # dentro del pliegue sube; entre pliegues baja
    xx = within + 0.0 * g
    assert A.spearman(xx, y) < 0.5
    assert A.partial_spearman(xx, y, g) == pytest.approx(1)
    ci = A.boot_ci(lambda ix: A.partial_spearman(xx[ix], y[ix], g[ix]), len(x), 200, 0, groups=g)
    assert 0.5 < ci[0] <= ci[1] <= 1 + 1e-9


def test_a2_detects_monotone_leakage():
    rng = np.random.default_rng(1)
    table = [{"config": "J0", "photo": f"p{i}", "fold": i % 3, "n_train": 10, "coverage_pct": c,
              "iou": 0.3 + 0.004 * c} for i, c in enumerate(rng.uniform(0, 100, 30))]
    r = A.analysis_a2(table, n_boot=200)["correlations"][0]
    assert r["rho"] == pytest.approx(1) and r["rho_parcial_pliegue"] == pytest.approx(1)
    assert r["rho_ci"][0] > 0.9


def test_pair_j0_vs_b75_separates_within_fold():
    rows = []
    for f, dn in enumerate([-3, -5, -8]):
        for j in range(6):
            dc = -5.0 * j
            for cfg, cov, nt, iou in (("J0", 60.0, 30, 0.6), ("J0-b75", 60.0 + dc, 30 + dn, 0.6 + 0.001 * dc)):
                rows.append({"config": cfg, "photo": f"f{f}p{j}", "fold": f, "n_train": nt,
                             "coverage_pct": cov, "iou": iou})
    p = A.analysis_pair(rows, n_boot=200)
    assert p["n_fotos"] == 18 and p["n_valores_distintos_dntrain"] == 3
    assert p["rho_diou_dcov_dentro_pliegue"] == pytest.approx(1)
    assert p["ols"]["coef"]["d_cov"] == pytest.approx(0.001, abs=1e-9)
    assert p["ols"]["coef"]["d_n_train"] == pytest.approx(0, abs=1e-9)


# ----------------------------------------------------------------------------- A3

MAP = {"approved": True, "unmapped_prediction": "error", "common": ["paved", "green"],
       "mine": {"Road": "paved", "Vegetation": "green", "Fence": None},
       "mauricio": {"sin_clase": None, "via": "paved", "veg_alta": "green", "veg_baja": "green"},
       "variants": [{"name": "fence_a_paved", "description": "Fence → paved", "mine": {"Fence": "paved"}},
                    {"name": "sin_clase_descartado", "description": "drop",
                     "unmapped_prediction": {"mauricio": "drop"}}]}


def test_aggregate_fine_to_coarse_by_hand():
    cm = np.array([[5, 1, 2],     # real Road
                   [0, 7, 3],     # real Vegetation
                   [4, 4, 9]])    # real Fence → fila ignorada
    agg, names = A.aggregate(cm, CLASSES, MAP["mine"], MAP["common"], "error")
    assert names == ["paved", "green", A.OUT_CLASS]
    assert agg.tolist() == [[5, 1, 2], [0, 7, 3], [0, 0, 0]]
    agg2, names2 = A.aggregate(cm, CLASSES, MAP["mine"], MAP["common"], "drop")
    assert names2 == ["paved", "green"] and agg2.tolist() == [[5, 1], [0, 7]]
    # batch (N, C, C) y suma de clases
    m2 = {"Road": "paved", "Vegetation": "paved", "Fence": "green"}
    agg3, _ = A.aggregate(np.stack([cm, cm]), CLASSES, m2, ["paved", "green"])
    assert agg3[1].tolist() == [[13, 5], [8, 9]]


def test_mapping_validation(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({**MAP, "mine": {"Road": "asfalto"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="destinos"):
        A.load_mapping(p)
    with pytest.raises(ValueError, match="faltan"):
        A.check_mapping_covers({"mine": {"Road": "paved"}}, "mine", CLASSES)


def test_repo_mapping_is_approved_with_sensitivity_variants():
    m = A.load_mapping(A.REPO / "configs" / "analysis" / "common_scheme.yaml")
    assert m["approved"] is True
    assert len(m["mine"]) == 12 and len(m["mauricio"]) == 10
    assert m["mine"]["Sports Fields"] == "vegetacion" and m["mauricio"]["sin_clase"] is None
    assert m["unmapped_prediction"] == {"mine": "error", "mauricio": "error"}
    v = {x["name"]: A.apply_variant(m, x) for x in m["variants"]}
    assert v["sports_fields_fuera"]["mine"]["Sports Fields"] is None
    assert v["sports_fields_fuera"]["mine"]["Vegetation"] == "vegetacion"
    assert v["sin_clase_descartado"]["unmapped_prediction"] == {"mine": "error", "mauricio": "drop"}
    # regla única: ≥ 5 fotos, sin umbral de píxeles, sin exclusiones; agua evaluable en la tabla principal
    assert m["evaluability"] == {"min_photos": 5, "min_pixel_share_pct": None, "exclude": {}}
    assert A.evaluability_rule(m, K=99)["min_photos"] == 5
    assert A.evaluability_rule({}, K=99) == {"min_photos": 5, "min_pixel_share_pct": None, "exclude": {},
                                             "mauricio_exclude": {}}
    assert "no concluyente" in m["footnotes"]["mauricio"]["agua"] and "sin IC" in m["footnotes"]["mauricio"]["agua"]
    assert set(v) == {"sports_fields_fuera", "sin_clase_descartado", "umbral_pixeles_0_1", "sin_agua"}
    assert v["umbral_pixeles_0_1"]["evaluability"]["min_pixel_share_pct"] == 0.1
    assert v["umbral_pixeles_0_1"]["evaluability"]["min_photos"] == 5
    assert any("después de ver los datos" in " ".join(t.split()) for t in m["report_notes"])
    assert list(v["sin_agua"]["evaluability"]["exclude"]) == ["agua"]
    assert v["sin_agua"]["evaluability"]["min_photos"] == 5
    assert "LEEME" in next(x for x in m["variants"] if x["name"] == "sin_agua")["description"]


def test_pred_into_null_pct_by_hand():
    cm = np.array([[50, 5, 5, 0], [10, 100, 0, 0], [0, 0, 80, 20], [0, 5, 10, 60]])
    pct = A.pred_into_null_pct(cm, ["sin_clase", "via", "veg_alta", "veg_baja"], MAP["mauricio"])
    # filas etiquetadas = 1..3 (285 px); columna sin_clase en esas filas = 10
    assert pct == {"sin_clase": pytest.approx(100 * 10 / 285)}


def test_variant_with_unknown_class_is_rejected(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump({**MAP, "variants": [{"name": "x", "mine": {"Nope": None}}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="clase desconocida"):
        A.load_mapping(p)


def write_mauricio(path, leeme=None):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")
    cls = ["sin_clase", "via", "veg_alta", "veg_baja"]
    cm = np.array([[50, 5, 5, 0], [10, 100, 0, 0], [0, 0, 80, 20], [0, 5, 10, 60]])
    iou = M_iou(cm)
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame(cm, index=cls, columns=cls).to_excel(xw, sheet_name="Matriz_confusion")
        pd.DataFrame({"Clase": cls, "IoU prueba": np.round(iou, 4)}).to_excel(xw, sheet_name="IoU_por_clase", index=False)
        pd.DataFrame({"Parametro": ["modelo"], "Valor": ["nvidia/mit-b3"]}).to_excel(xw, sheet_name="Config", index=False)
        if leeme is not None:
            pd.DataFrame(leeme).to_excel(xw, sheet_name="LEEME", index=False, header=False)
    return cm


def M_iou(cm):
    from segpipe import metrics as M
    return M.per_class(cm)["iou"]


def test_read_mauricio_and_aggregate(tmp_path):
    cm = write_mauricio(tmp_path / "mau.xlsx")
    mau = A.read_mauricio(tmp_path / "mau.xlsx")
    assert mau["classes"][0] == "sin_clase" and (mau["cm"] == cm).all()
    agg, names = A.aggregate(mau["cm"], mau["classes"], MAP["mauricio"], MAP["common"])
    # green = veg_alta+veg_baja: tp = 80+20+10+60 = 170
    assert agg[1, 1] == 170 and agg[:, 2].sum() == 10      # 10 px reales predichos como sin_clase


# ----------------------------------------------------------------------------- extremo a extremo

def test_end_to_end_reads_only_and_writes_report(tmp_path):
    study, man, dcfg = make_study(tmp_path)
    write_mauricio(tmp_path / "mau.xlsx")
    mp = tmp_path / "map.yaml"
    mp.write_text(yaml.safe_dump(MAP), encoding="utf-8")
    out = tmp_path / "analysis"
    before = A.snapshot(study / "runs")
    res = A.run(study, man, out, dcfg, mp, tmp_path / "mau.xlsx", n_boot=50, log=lambda *a: None)
    assert A.snapshot(study / "runs") == before
    assert all(ok for _, ok, _ in res["checks"]), res["checks"]
    assert res["a3_status"] == "calculado"
    a3 = res["a3"]
    assert [v["name"] for v in a3["variants"]] == ["fence_a_paved", "sin_clase_descartado"]
    # principal: Fence sin destino → la columna fuera_de_esquema existe en mis matrices
    assert a3["mine"]["J0"]["classes"][-1]["class"] == A.OUT_CLASS
    drop = a3["variants"][1]["result"]["mauricio"]
    assert A.OUT_CLASS not in drop["classes"] and A.OUT_CLASS in a3["mauricio"]["classes"]
    # descartar sin_clase no puede bajar el IoU de Mauricio
    assert all(drop["iou"][n] >= a3["mauricio"]["iou"][n] - 1e-12 for n in MAP["common"])
    assert (out / "tables" / "A3_sensibilidad_fence_a_paved_por_clase.csv").exists()
    txt = (out / "REPORT.md").read_text(encoding="utf-8")
    for s in ("A1", "A2", "A3", "Datasets distintos", "Costo", "Sensibilidad: fence_a_paved",
              "Mis modelos no tienen esa salida", "inspección visual", "Hecho confirmado", "sin IC"):
        assert s in txt
    assert "sin fuga" not in txt.lower()
    for f in res["figures"]:
        assert f.exists() and f.with_suffix(".pdf").exists() and f.with_suffix(".csv").exists()
    names = {f.stem for f in res["figures"]}
    assert {"A4_escalera_miou", "A4_iou_por_clase", "A4_variabilidad_pliegues", "A4_curvas_val_miou",
            "A4_matrices_confusion", "A4_costo", "A1_solape_huellas", "A2_iou_vs_cobertura",
            "A2_pareado_J0_vs_J0-b75", "A3_esquema_comun"} <= names


def test_unapproved_mapping_is_not_computed(tmp_path):
    study, man, dcfg = make_study(tmp_path, configs=("J0",))
    mp = tmp_path / "map.yaml"
    mp.write_text(yaml.safe_dump({**MAP, "approved": False}), encoding="utf-8")
    res = A.run(study, man, tmp_path / "out", dcfg, mp, None, n_boot=20, figures=False, log=lambda *a: None)
    assert res["a3"] is None and "sin aprobar" in res["a3_status"]


def test_out_inside_runs_is_refused(tmp_path):
    study, man, dcfg = make_study(tmp_path, configs=("J0",))
    with pytest.raises(SystemExit):
        A.run(study, man, study / "runs" / "x", dcfg, n_boot=10, figures=False, log=lambda *a: None)


def test_find_study_order(tmp_path, monkeypatch):
    monkeypatch.setenv("SEGPIPE_STUDY", str(tmp_path / "env"))
    assert A.find_study(str(tmp_path / "arg")) == (tmp_path / "arg", "argumento --study")
    assert A.find_study(None) == (tmp_path / "env", "variable SEGPIPE_STUDY")


# ----------------------------------------------------------------------------- A3: criterio de evaluable

def _a3_inputs(tmp_path, leeme=None, **rule):
    study, man, dcfg = make_study(tmp_path, configs=("J0",))
    runs, _ = A.load_runs(study)
    write_mauricio(tmp_path / "mau.xlsx", leeme)
    mau = A.read_mauricio(tmp_path / "mau.xlsx")
    mp = {**MAP, "unmapped_prediction": {"mine": "error", "mauricio": "error"}, "variants": [],
          "evaluability": {"min_photos": 1, **rule}}
    return runs, mau, mp


def test_leeme_photo_count_is_parsed(tmp_path):
    write_mauricio(tmp_path / "m.xlsx", ["Título", "via: solo 1.000 px y 3 fotos en prueba; no concluyente.",
                                         "Otra nota con 7 fotos que no empieza por una clase"])
    assert A.read_mauricio(tmp_path / "m.xlsx")["photos_stated"] == {"via": 3}


def test_photo_rule_on_mauricio_and_intersection(tmp_path):
    runs, mau, mp = _a3_inputs(tmp_path, ["via: solo 115 px y 3 fotos en prueba"])
    mp["evaluability"]["min_photos"] = 5
    for r in runs:                                   # que el lado propio sea evaluable con K=5
        r.cms = np.concatenate([r.cms] * 5)
        r.names = r.names * 5
    out = A.analysis_a3(runs, CLASSES, mp, mau, K=5, n_boot=30)
    m = out["mauricio"]
    assert m["photos_stated"] == {"paved": 3, "green": None}       # green viene de 2 clases finas
    assert m["evaluable"] == {"paved": False, "green": True, A.OUT_CLASS: False}
    assert "3 fotos < 5" in m["reason"]["paved"]
    j0 = out["mine"]["J0"]
    assert j0["evaluable_classes"] == ["paved", "green"]
    assert out["clases_interseccion"] == ["green"]
    # (b) intersección = IoU de green en ambos lados
    green = next(x for x in j0["classes"] if x["class"] == "green")
    assert j0["mIoU_interseccion"] == pytest.approx(green["iou"])
    assert m["mIoU_interseccion"] == pytest.approx(m["iou"]["green"])
    assert m["summary"]["mIoU"] == pytest.approx(m["iou"]["green"])   # (a) de Mauricio: solo green


def test_pixel_share_rule_applies_to_both_sides(tmp_path):
    runs, mau, mp = _a3_inputs(tmp_path, None, min_pixel_share_pct=45.0)
    out = A.analysis_a3(runs, CLASSES, mp, mau, K=1, n_boot=30)
    m = out["mauricio"]
    sh = m["gt_share_pct"]
    assert sh["paved"] + sh["green"] == pytest.approx(100)
    assert m["evaluable"]["paved"] == (sh["paved"] >= 45) and m["evaluable"]["green"] == (sh["green"] >= 45)
    for x in out["mine"]["J0"]["classes"]:
        if x["class"] != A.OUT_CLASS:
            assert x["evaluable"] == (x["gt_share_pct"] >= 45)
            assert ("% de los píxeles" in x["reason"]) == (not x["evaluable"])
    # variante que relaja el umbral
    mp["variants"] = [{"name": "sin_umbral", "description": "x", "evaluability": {"min_pixel_share_pct": None}}]
    out = A.analysis_a3(runs, CLASSES, A.load_mapping_dict(mp), mau, K=1, n_boot=30)
    assert out["variants"][0]["result"]["mauricio"]["evaluable"]["paved"]


def test_mauricio_agua_rule_exclude_variant_and_footnote():
    """Con ≥ 5 fotos, una clase con 6 fotos declaradas es evaluable; 'exclude' la quita en ambos lados."""
    cls = ["sin_clase", "via", "agua"]
    mau = {"classes": cls, "cm": np.array([[5, 1, 0], [2, 100, 0], [1, 3, 4]]), "iou_test": {},
           "photos_stated": {"agua": 6}, "config": {}}
    fine = ["Road", "Pools"]
    mp = A.load_mapping_dict({
        "approved": True, "common": ["paved", "agua"],
        "mine": {"Road": "paved", "Pools": "agua"},
        "mauricio": {"sin_clase": None, "via": "paved", "agua": "agua"},
        "evaluability": {"min_photos": 5, "min_pixel_share_pct": None, "exclude": {}},
        "footnotes": {"mauricio": {"agua": "6 fotos de prueba; el autor la reporta como no concluyente; sin IC"}},
        "variants": [{"name": "sin_agua", "description": "LEEME",
                      "evaluability": {"exclude": {"agua": "variante sin_agua"}}}]})

    class R:
        config, seed, fold = "J0", 42, 0
        names = [f"p{i}" for i in range(6)]
        cms = np.stack([np.array([[50, 2], [1, 9]])] * 6)
    out = A.analysis_a3([R()], fine, mp, mau, K=5, n_boot=20)
    m, j0 = out["mauricio"], out["mine"]["J0"]
    assert m["evaluable"]["agua"] and j0["evaluable_classes"] == ["paved", "agua"]
    assert out["clases_interseccion"] == ["paved", "agua"]                 # agua entra en (b)
    sin = out["variants"][0]["result"]
    assert not sin["mauricio"]["evaluable"]["agua"] and sin["mine"]["J0"]["evaluable_classes"] == ["paved"]
    assert sin["clases_interseccion"] == ["paved"]
    txt = "\n".join(A._a3_table(out))
    assert "| agua |" in txt and "†" in txt
    assert "† agua (Mauricio): 6 fotos de prueba; el autor la reporta como no concluyente; sin IC." in txt
    assert "(b) intersección (2): paved, agua." in txt
