"""Fixtures con un segpack sintético pequeño (mismo formato que segpack_v1)."""
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

CLASSES = [
    {"index": 0, "name": "Road", "rgb": [89, 134, 179]},
    {"index": 1, "name": "Vegetation", "rgb": [36, 179, 83]},
    {"index": 2, "name": "Fence", "rgb": [184, 61, 245]},
    {"index": 3, "name": "background", "rgb": [0, 0, 0]},
]
W, H = 64, 48

MANIFEST_COLS = ["name", "image", "mask", "width", "height", "mask_size_ok", "bytes", "sha256",
                 "GpsLatitude", "GpsLongitude", "AbsoluteAltitude", "RelativeAltitude",
                 "GimbalYawDegree", "GimbalPitchDegree", "GimbalRollDegree",
                 "FlightYawDegree", "FlightPitchDegree", "FlightRollDegree"]


def make_mask(rng, n_classes=4):
    m = np.full((H, W), 3, np.uint8)
    m[: H // 2] = 0
    m[H // 2:, : W // 2] = 1
    if rng.random() < 0.5:
        m[:, W // 2: W // 2 + 2] = 2  # clase delgada
    rgb = np.zeros((H, W, 3), np.uint8)
    for c in CLASSES:
        rgb[m == c["index"]] = c["rgb"]
    return m, rgb


def build_pack(dirpath: Path, n=12, seed=0, positions=None):
    """Crea segpack sintético; devuelve (zip_path, cfg_dict)."""
    rng = np.random.default_rng(seed)
    dirpath.mkdir(parents=True, exist_ok=True)
    zpath = dirpath / "segpack_test.zip"
    rows = []
    lat0, lon0 = 3.43, -76.54
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        z.writestr("classes.json", json.dumps(CLASSES))
        for i in range(n):
            name = f"DJI_2025052211{i:04d}_{i:04d}_D"
            img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
            b = io.BytesIO(); Image.fromarray(img).save(b, format="JPEG"); jb = b.getvalue()
            _, rgb = make_mask(rng)
            mb = io.BytesIO(); Image.fromarray(rgb).save(mb, format="PNG")
            z.writestr(f"images/{name}.JPG", jb)
            z.writestr(f"masks/{name}.png", mb.getvalue())
            if positions is not None:
                dx, dy = positions[i]
            else:
                dx, dy = (i % 4) * 100.0, (i // 4) * 100.0
            lat = lat0 + dy / 6371000 * 180 / math.pi
            lon = lon0 + dx / (6371000 * math.cos(math.radians(lat0))) * 180 / math.pi
            rows.append({"name": name, "image": f"images/{name}.JPG", "mask": f"masks/{name}.png",
                         "width": W, "height": H, "mask_size_ok": True, "bytes": len(jb),
                         "sha256": hashlib.sha256(jb).hexdigest(),
                         "GpsLatitude": f"{lat:.9f}", "GpsLongitude": f"{lon:.9f}",
                         "AbsoluteAltitude": "1080", "RelativeAltitude": "58.7",
                         "GimbalYawDegree": "0", "GimbalPitchDegree": "-45", "GimbalRollDegree": "0",
                         "FlightYawDegree": "0", "FlightPitchDegree": "0", "FlightRollDegree": "0"})
        s = io.StringIO(); w = csv.DictWriter(s, fieldnames=MANIFEST_COLS); w.writeheader(); w.writerows(rows)
        z.writestr("manifest.csv", s.getvalue())
    data = zpath.read_bytes()
    cfg = {
        "name": "synthetic", "version": "test",
        "archive": {"filename": zpath.name, "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(), "drive_file_id": "FAKEID",
                    "drive_mounted_path": ""},
        "local": {"root_colab": str(dirpath / "content"), "root_other": str(dirpath / "root"),
                  "extract_dir": "segpack_test", "env_zip": "SEGPACK_ZIP", "env_dir": "SEGPACK_DIR"},
        "layout": {"classes": "classes.json", "manifest": "manifest.csv", "class_stats": "class_stats.csv"},
        "classes": {"ignore": ["background"]},
        "geometry": {"lat": "GpsLatitude", "lon": "GpsLongitude", "height": "RelativeAltitude",
                     "pitch": "GimbalPitchDegree", "yaw": "GimbalYawDegree", "roll": "GimbalRollDegree",
                     "camera": {"focal_mm": 6.72, "pixel_mm": 0.0026187585746188587, "width": 4032,
                                "height": 2268, "max_range_m": 400}},
    }
    return zpath, cfg


@pytest.fixture
def pack(tmp_path, monkeypatch):
    for v in ("SEGPACK_ZIP", "SEGPACK_DIR", "SEGPACK_ROOT"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("segpipe.data_io.in_colab", lambda: False)
    return build_pack(tmp_path / "pack")
