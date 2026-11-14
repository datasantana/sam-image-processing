import json

import pytest

from segpipe import data_io, report
from segpipe.runner import prepare_study, run_queue, write_ignored_report
from tests.test_train import TINY


def test_study_queue_and_comparison_table(pack, tmp_path):
    zpath, _ = pack
    root = data_io.safe_extract(zpath, tmp_path / "ds")
    import yaml
    _, cfg = pack
    dcfg_path = tmp_path / "ds.yaml"
    dcfg_path.write_text(yaml.safe_dump(cfg))
    study = prepare_study(tmp_path / "study", dcfg_path, protocol="cv", k=3, block_m=150, buffer_m=0,
                          data_root=root, log=lambda *a: None)
    assert (tmp_path / "study" / "splits" / "splits.json").exists()
    names = {r["name"] for r in study.rows}
    assert {n for f in study.folds for n in f.test} == names
    write_ignored_report(study)
    assert (tmp_path / "study" / "dataset" / "ignored_px_per_photo.csv").exists()

    over = {**TINY, "train": {**TINY["train"], "epochs": 2}}
    res = run_queue(study, ["J0", "J1"], seeds=[1, 2], overrides=over, log=lambda *a: None)
    assert len(res) == 2 * 2 * 3
    # segunda vez: todo se salta
    again = run_queue(study, ["J0", "J1"], seeds=[1, 2], overrides=over, log=lambda *a: None)
    assert [r["run_dir"] for r in again] == [r["run_dir"] for r in res]

    out = report.build(tmp_path / "study", ["J0", "J1"], n_boot=50)
    rows = {r["config"]: r for r in out["summary"]}
    assert rows["J0"]["n_seeds"] == 2 and rows["J0"]["n_photos"] == len(names)
    lo, hi = rows["J1"]["mIoU_ci"]
    assert lo <= hi
    assert out["deltas"][0]["from"] == "J0" and out["deltas"][0]["to"] == "J1"
    md = (tmp_path / "study" / "results" / "comparison.md").read_text(encoding="utf-8")
    assert "| J0 |" in md and "ΔmIoU" in md

    # parámetros distintos sobre el mismo estudio → error explícito
    with pytest.raises(RuntimeError, match="otros parámetros"):
        prepare_study(tmp_path / "study", dcfg_path, protocol="cv", k=3, block_m=100, buffer_m=0,
                      data_root=root, log=lambda *a: None)
