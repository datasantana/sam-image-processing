import pytest
import yaml

from segpipe import data_io, report
from segpipe.runner import estimate_cost, prepare_study, run_queue, run_smoke, smoke_epoch_times, write_ignored_report
from tests.test_train import TINY


def _study(pack, tmp_path, name, **kw):
    zpath, cfg = pack
    root = tmp_path / "ds"
    if not root.exists():
        data_io.safe_extract(zpath, root)
    dcfg_path = tmp_path / "ds.yaml"
    dcfg_path.write_text(yaml.safe_dump(cfg))
    args = dict(protocol="cv", k=3, block_m=150, buffer_m=0)
    args.update(kw)
    return prepare_study(tmp_path / name, dcfg_path, data_root=root, log=lambda *a: None, **args), dcfg_path, root


def test_study_queue_and_comparison_table(pack, tmp_path):
    study, dcfg_path, root = _study(pack, tmp_path, "study")
    assert (tmp_path / "study" / "splits" / "splits.json").exists()
    rep_txt = (tmp_path / "study" / "splits" / "split_report.txt").read_text(encoding="utf-8")
    assert "solape residual" in rep_txt and "Fuga reducida" in rep_txt and "mezclan vuelos" in rep_txt
    names = {r["name"] for r in study.rows}
    assert {n for f in study.folds for n in f.test} == names
    write_ignored_report(study)
    assert (tmp_path / "study" / "dataset" / "ignored_px_per_photo.csv").exists()

    over = {**TINY, "train": {**TINY["train"], "epochs": 2}}
    cfgs = ["J0", "J1", "J0-pasos", "J0-b75"]
    res = run_queue(study, cfgs, seeds=[1], overrides=over, log=lambda *a: None)
    assert len(res) == 4 * 3
    # orden config → semilla → pliegue
    assert [r["run_dir"].split("runs")[1][1:3] for r in res[:3]] == ["J0"] * 3
    again = run_queue(study, cfgs, seeds=[1], overrides=over, log=lambda *a: None)
    assert [r["run_dir"] for r in again] == [r["run_dir"] for r in res]
    # J0-b75: mismo test que J0, train ⊆ train de J0
    import json
    from pathlib import Path
    for f in range(3):
        a = json.loads(Path(study.run_dir("J0", 1, f) / "split.json").read_text())
        b = json.loads(Path(study.run_dir("J0-b75", 1, f) / "split.json").read_text())
        assert a["test"] == b["test"] and set(b["train"]) <= set(a["train"]) and b["buffer_m"] == 75

    out = report.build(tmp_path / "study", cfgs, n_boot=50)
    rows = {r["config"]: r for r in out["summary"]}
    assert rows["J0"]["n_seeds"] == 1 and rows["J0"]["n_photos"] == len(names)
    assert rows["J0-pasos"]["steps_total"] == rows["J1"]["steps_total"]
    assert rows["J0-b75"]["buffer_m"] == 75 and rows["J0"]["buffer_m"] == 0
    assert ("J0-pasos", "J1") in [(d["from"], d["to"]) for d in out["deltas"]]
    md = (tmp_path / "study" / "results" / "comparison.md").read_text(encoding="utf-8")
    assert "| J0 |" in md and "ΔmIoU" in md and "Pasos" in md and "Fuga reducida" in md

    with pytest.raises(RuntimeError, match="otros parámetros"):
        prepare_study(tmp_path / "study", dcfg_path, protocol="cv", k=3, block_m=100, buffer_m=0,
                      data_root=root, log=lambda *a: None)


def test_smoke_then_cost_estimate(pack, tmp_path):
    study, _, _ = _study(pack, tmp_path, "study")
    smoke, _, _ = _study(pack, tmp_path, "smoke", protocol="fixed")
    over = {**TINY, "train": {**TINY["train"], "epochs": 2}, "data": {**TINY["data"], "samples_per_epoch": 8}}
    res = run_smoke(smoke, ["J0", "J1"], overrides=over, subset={"train": 4, "val": 2, "test": 2}, log=lambda *a: None)
    assert len(res) == 2 and all(r["epochs_trained"] == 2 for r in res)
    assert "| J0 |" in smoke_epoch_times(smoke, ["J0", "J1"])
    cost = estimate_cost(smoke, study, ["J0", "J1", "J0-pasos", "J0-b75"], seeds=[42], log=lambda *a: None)
    assert cost["n_runs"] == 4 * 3
    assert cost["by_config"]["J0-pasos"]["proxy"] == "J0" and cost["by_config"]["J0-b75"]["proxy"] == "J0"
    lo, hi = cost["total_h"]
    assert 0 < lo <= hi
    # J2 (mismo coste por muestra que J1) se estima con J1; M-b3 sin humo propio no se estima con nadie
    assert estimate_cost(smoke, study, ["J2"], seeds=[42], log=lambda *a: None)["by_config"]["J2"]["proxy"] == "J1"
    with pytest.raises(RuntimeError, match="M-b3"):
        estimate_cost(smoke, study, ["M-b3"], seeds=[42], log=lambda *a: None)
