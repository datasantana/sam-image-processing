import numpy as np
import pytest

from segpipe import metrics as M


def hand_case():
    target = np.array([0, 0, 1, 1, 2, 255])
    pred = np.array([0, 1, 1, 1, 0, 2])
    return pred, target


def test_confusion_excludes_ignore():
    pred, target = hand_case()
    cm = M.confusion(pred, target, 3)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]


def test_metrics_against_hand_computation():
    pred, target = hand_case()
    cm = M.confusion(pred, target, 3)
    pc = M.per_class(cm)
    np.testing.assert_allclose(pc["iou"], [1 / 3, 2 / 3, 0])
    np.testing.assert_allclose(pc["f1"], [0.5, 0.8, 0])
    s = M.summarize(cm)
    assert s["mIoU"] == pytest.approx(1 / 3)
    assert s["OA"] == pytest.approx(3 / 5)
    assert s["mAcc"] == pytest.approx(0.5)
    assert s["mF1"] == pytest.approx(1.3 / 3)


def test_not_evaluable_class_excluded_from_miou():
    # 6 fotos; clase 2 aparece solo en 1 foto → no evaluable con K=5
    cms = []
    for i in range(6):
        t = np.array([0, 0, 1, 1] + ([2] if i == 0 else [0]))
        p = np.array([0, 0, 1, 0] + ([0] if i == 0 else [0]))
        cms.append(M.confusion(p, t, 3))
    rep = M.evaluate(np.stack(cms), ["a", "b", "c"], min_photos=5, n_boot=50)
    assert rep["not_evaluable_classes"] == ["c"]
    assert rep["evaluable_classes"] == ["a", "b"]
    cm = np.stack(cms).sum(0)
    iou = M.per_class(cm)["iou"]
    assert rep["mIoU"] == pytest.approx(np.mean(iou[:2]))
    assert rep["classes"][2]["evaluable"] is False
    assert "no evaluables" in rep["note"]


def test_bootstrap_resamples_photos():
    rng = np.random.default_rng(0)
    cms = []
    for i in range(30):
        t = rng.integers(0, 3, 200)
        p = np.where(rng.random(200) < 0.7, t, rng.integers(0, 3, 200))
        cms.append(M.confusion(p, t, 3))
    cms = np.stack(cms)
    ev = np.ones(3, bool)
    a = M.bootstrap(cms, ev, n_boot=300, seed=1)
    b = M.bootstrap(cms, ev, n_boot=300, seed=1)
    assert a == b  # reproducible
    point = M.summarize(cms.sum(0), ev)["mIoU"]
    lo, hi = a["mIoU"]
    assert lo < point < hi and hi - lo < 0.2
    # con fotos idénticas el IC colapsa al punto
    same = np.repeat(cms[:1], 10, axis=0)
    c = M.bootstrap(same, ev, n_boot=100)
    assert c["mIoU"][0] == pytest.approx(c["mIoU"][1])


def test_save_report(tmp_path):
    pred, target = hand_case()
    cms = np.stack([M.confusion(pred, target, 3)] * 6)
    rep = M.evaluate(cms, ["a", "b", "c"], n_boot=20)
    jp, cp = M.save_report(rep, tmp_path)
    assert jp.exists() and cp.read_text().startswith("class,evaluable")
