import json

import numpy as np
import pytest

from segpipe.classes import ClassSchema, IGNORE_INDEX
from tests.conftest import CLASSES


@pytest.fixture
def cj(tmp_path):
    p = tmp_path / "classes.json"
    p.write_text(json.dumps(CLASSES))
    return p


def test_read_classes_json_with_ignore(cj):
    s = ClassSchema.from_json(cj, ignore=["background"])
    assert s.names == ["Road", "Vegetation", "Fence"]
    assert s.num_classes == 3
    assert s.source_to_train.tolist() == [0, 1, 2, IGNORE_INDEX]


def test_classes_json_must_be_ordered(tmp_path):
    bad = [dict(c) for c in CLASSES]
    bad[0]["index"], bad[1]["index"] = 1, 0
    p = tmp_path / "c.json"; p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="ordenado"):
        ClassSchema.from_json(p)


def test_rgb_to_index_vectorized(cj):
    s = ClassSchema.from_json(cj, ignore=["background"])
    m = np.zeros((2, 3, 3), np.uint8)
    m[0, 0] = [89, 134, 179]   # Road
    m[0, 1] = [36, 179, 83]    # Vegetation
    m[0, 2] = [184, 61, 245]   # Fence
    # fila 1: negro = background → ignorado
    out = s.rgb_to_index(m)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1, 2], [255, 255, 255]]


def test_rgb_to_index_unknown_color_fails_clearly(cj):
    s = ClassSchema.from_json(cj)
    m = np.zeros((2, 2, 3), np.uint8)
    m[1, 1] = [1, 2, 3]
    with pytest.raises(ValueError, match=r"Colores desconocidos.*\(1,2,3\): 1 px"):
        s.rgb_to_index(m, where="foto_x")


def test_fine_to_coarse_mapping(cj):
    s = ClassSchema.from_json(cj, ignore=["background"],
                              mapping={"Road": "paved", "Fence": "objects", "Vegetation": "veg"},
                              target=["paved", "veg", "objects"])
    assert s.names == ["paved", "veg", "objects"]
    assert s.source_to_train.tolist() == [0, 1, 2, 255]
    with pytest.raises(ValueError, match="sin mapeo"):
        ClassSchema.from_json(cj, ignore=["background"], mapping={"Road": "paved"})
