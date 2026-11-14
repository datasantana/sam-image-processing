import math

import numpy as np
import pytest

from segpipe import split as S
from tests.conftest import build_pack

GEOM = {"lat": "GpsLatitude", "lon": "GpsLongitude", "height": "RelativeAltitude",
        "pitch": "GimbalPitchDegree", "yaw": "GimbalYawDegree"}


def rows_from_xy(xy, yaw=0.0, pitch=-45.0, h=58.7):
    lat0, lon0 = 3.43, -76.54
    rows = []
    for i, (x, y) in enumerate(xy):
        rows.append({"name": f"p{i:03d}",
                     "GpsLatitude": lat0 + y / S.EARTH_R * 180 / math.pi,
                     "GpsLongitude": lon0 + x / (S.EARTH_R * math.cos(math.radians(lat0))) * 180 / math.pi,
                     "RelativeAltitude": h, "GimbalPitchDegree": pitch, "GimbalYawDegree": yaw})
    return rows


def test_footprint_offset_follows_gimbal_yaw():
    # dron en el mismo punto; yaw 90° (este) → huella desplazada h/tan(45°)=h metros al este
    rows = rows_from_xy([(0, 0), (0, 0)])
    rows[1]["GimbalYawDegree"] = 90.0
    fx, fy, dx, dy = S.footprint_centers(rows, GEOM)
    assert fy[0] - dy[0] == pytest.approx(58.7, abs=0.05)   # yaw 0 → norte
    assert fx[1] - dx[1] == pytest.approx(58.7, abs=0.05)   # yaw 90 → este
    assert fy[1] - dy[1] == pytest.approx(0, abs=0.05)


def test_no_train_photo_within_buffer_of_val_or_test():
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 700, size=(80, 2))
    rows = rows_from_xy(xy)
    for i, r in enumerate(rows):
        r["GimbalYawDegree"] = float(rng.uniform(-180, 180))
    fx, fy, _, _ = S.footprint_centers(rows, GEOM)
    pos = {r["name"]: (x, y) for r, x, y in zip(rows, fx, fy)}
    folds = S.make_folds(rows, GEOM, k=5, block_m=150, buffer_m=60, seed=0)
    all_names = {r["name"] for r in rows}
    assert {n for f in folds for n in f.test} == all_names  # cada foto es test exactamente una vez
    assert sum(len(f.test) for f in folds) == len(rows)
    for f in folds:
        assert not set(f.train) & set(f.val) and not set(f.train) & set(f.test) and not set(f.val) & set(f.test)
        held = [pos[n] for n in f.val + f.test]
        for n in f.train:
            d = min(math.hypot(pos[n][0] - a, pos[n][1] - b) for a, b in held)
            assert d >= 60
        assert set(f.train) | set(f.val) | set(f.test) | set(f.excluded) == all_names
        assert f.min_dist_train_heldout >= 60


def test_blocks_are_not_split_across_folds():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 600, size=(60, 2))
    rows = rows_from_xy(xy)
    fx, fy, _, _ = S.footprint_centers(rows, GEOM)
    fold_of = S.assign_blocks([r["name"] for r in rows], fx, fy, 4, 150.0)
    bx = np.floor((fx - fx.min()) / 150).astype(int); by = np.floor((fy - fy.min()) / 150).astype(int)
    seen = {}
    for r, a, b in zip(rows, bx, by):
        assert seen.setdefault((a, b), fold_of[r["name"]]) == fold_of[r["name"]]


def test_too_few_blocks_raises():
    rows = rows_from_xy([(0, 0), (10, 10), (20, 0)])
    with pytest.raises(ValueError, match="bloques"):
        S.make_folds(rows, GEOM, k=5, block_m=150)


def test_save_and_report(tmp_path):
    rows = rows_from_xy([(x * 160, y * 160) for x in range(4) for y in range(3)])
    presence = {r["name"]: ({"Fence"} if i % 3 == 0 else set()) | {"Road"} for i, r in enumerate(rows)}
    folds = S.make_folds(rows, GEOM, k=4, block_m=150, buffer_m=50, presence=presence)
    p = S.save_folds(folds, tmp_path, {"k": 4}, presence, ["Road", "Fence"])
    back = S.load_folds(p)
    assert [f.test for f in back] == [f.test for f in folds]
    assert (tmp_path / "fold0.csv").exists()
    txt = S.format_report([S.split_report(f, presence, ["Road", "Fence"]) for f in folds])
    assert "fotos por clase en test" in txt


CAM = {"focal_mm": 6.72, "pixel_mm": 0.0026187585746188587, "width": 4032, "height": 2268, "max_range_m": 400}


def test_camera_fov_and_footprint_geometry():
    hf, vf = S.camera_fov(CAM)
    assert hf == pytest.approx(76.3, abs=0.1) and vf == pytest.approx(47.7, abs=0.1)
    rows = rows_from_xy([(0, 0)], yaw=0.0, pitch=-45.0, h=58.7)
    for r in rows:
        r["GimbalRollDegree"] = 0
    poly = S.footprint_polygons(rows, {**GEOM, "camera": CAM})[0]
    # yaw 0 → la huella mira al norte; borde cercano a h/tan(45+vf/2), lejano a h/tan(45-vf/2)
    near = 58.7 / math.tan(math.radians(45 + vf / 2))
    far = 58.7 / math.tan(math.radians(45 - vf / 2))
    assert poly[2, 1] == pytest.approx(near, abs=0.5) and poly[0, 1] == pytest.approx(far, abs=0.5)
    assert abs(poly[0, 0] - poly[1, 0]) > abs(poly[3, 0] - poly[2, 0])  # trapecio: más ancho lejos
    cx, cy = S.polygon_centroid(poly)
    assert cx == pytest.approx(0, abs=1e-6) and cy > 58.7  # centroide más lejos que el rayo central
    fx, fy, _, _ = S.footprint_centers(rows, {**GEOM, "camera": CAM})
    assert fy[0] == pytest.approx(cy, abs=0.5)


def test_overlap_raster():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    polys = np.stack([sq, sq + [5, 0], sq + [100, 0]])
    r = S.FootprintRaster(polys, res_m=0.5)
    assert r.covered_pct([0], [0]) == pytest.approx(100)
    assert r.covered_pct([0], [1]) == pytest.approx(50, abs=3)
    assert r.covered_pct([0], [2]) == 0


def test_folds_report_leakage_and_buffer_variant_keeps_test():
    rng = np.random.default_rng(5)
    rows = rows_from_xy(rng.uniform(0, 800, size=(60, 2)))
    for r in rows:
        r["GimbalYawDegree"] = float(rng.uniform(-180, 180)); r["GimbalRollDegree"] = 0
    geom = {**GEOM, "camera": CAM}
    f50 = S.make_folds(rows, geom, k=4, block_m=150, buffer_m=50)
    fp = S.footprints(rows, geom)
    f75 = S.folds_from_assignment([r["name"] for r in rows], fp, S.fold_assignment(f50), 4, 75)
    for a, b in zip(f50, f75):
        assert a.test == b.test and a.val == b.val
        assert set(b.train) <= set(a.train)
        assert b.min_dist_train_test >= 75 and a.min_dist_train_test >= 50
        assert 0 <= a.overlap_train_test_pct <= 100 and b.overlap_train_test_pct <= a.overlap_train_test_pct + 1e-9


def test_blocks_ignore_flight_date():
    # dos "vuelos" sobre el mismo terreno: mismas posiciones, fechas distintas en el nombre
    xy = [(x * 200.0, y * 200.0) for x in range(3) for y in range(2)]
    rows = rows_from_xy(xy + xy)
    for i, r in enumerate(rows):
        r["name"] = f"DJI_{'20250522' if i < 6 else '20250611'}1111{i:02d}_D"
    fx, fy, _, _ = S.footprint_centers(rows, GEOM)
    mix = S.flight_mix([r["name"] for r in rows], fx, fy, 150)
    assert mix["n_blocks"] == 6 and mix["blocks_with_several_groups"] == 6
    folds = S.make_folds(rows, GEOM, k=3, block_m=150, buffer_m=0)
    fa = S.fold_assignment(folds)
    for i in range(6):
        assert fa[rows[i]["name"]] == fa[rows[i + 6]["name"]]
