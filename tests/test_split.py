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
