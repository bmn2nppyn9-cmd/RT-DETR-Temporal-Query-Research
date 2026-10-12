import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tqinfra.config import load_config
from tqinfra.demo import run_demo
from tqinfra.report import load_evidence
from tqinfra.runner import run_worker, workspace_lock
from tqinfra.runtime import ROOT


def config(tmp_path, **kwargs):
    return {
        "workspace": str(tmp_path / "workspace"),
        "backend": str(tmp_path / "backend"),
        "data_root": str(tmp_path / "data"),
        "initial_weights": str(tmp_path / "baseline.pt"),
        "timeout_seconds": 5,
        **kwargs,
    }


def test_config_resolution_and_validation(tmp_path):
    path = tmp_path / "cfg.json"
    cfg = {k: "relative/path with spaces" for k in ("workspace", "backend", "data_root", "initial_weights")}
    path.write_text(json.dumps(cfg))
    assert load_config(path)["workspace"] == str((tmp_path / "relative/path with spaces").resolve())
    cfg["epochs"] = True
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_config(path)
    cfg["epochs"] = 1
    cfg["unexpected"] = 2
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_config(path)


def test_concurrent_process_cannot_enter_workspace(tmp_path):
    code = "from tqinfra.runner import workspace_lock; import sys; ctx=workspace_lock(sys.argv[1]); ctx.__enter__()"
    with workspace_lock(tmp_path):
        p = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT, capture_output=True)
        assert p.returncode != 0
        assert b"Workspace busy" in p.stderr
    assert not (tmp_path / ".worker.lock").exists()


@pytest.mark.parametrize("exit_code", [0, 7])
def test_worker_exit_status_and_logs(tmp_path, exit_code):
    code, directory = run_worker(
        [sys.executable, "-c", f"print('worker-output'); raise SystemExit({exit_code})"], config(tmp_path)
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert code == exit_code
    assert manifest["status"] == ("succeeded" if exit_code == 0 else "failed")
    assert "worker-output" in (directory / "worker.log").read_text(encoding="utf-8")
    assert not (tmp_path / "workspace/.worker.lock").exists()


def test_timeout_kills_child_tree_and_records_failure(tmp_path):
    marker = tmp_path / "orphan-marker.txt"
    child = f"import time; from pathlib import Path; time.sleep(4); Path({str(marker)!r}).write_text('orphan')"
    parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c'," + repr(child) + "]); time.sleep(20)"
    code, directory = run_worker([sys.executable, "-c", parent], config(tmp_path, timeout_seconds=1))
    assert code == 124
    assert json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["status"] == "timed_out"
    time.sleep(4)
    assert not marker.exists()


def test_launch_failure_is_durable(tmp_path):
    code, directory = run_worker([str(tmp_path / "missing-executable")], config(tmp_path))
    data = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert code != 0 and data["status"] == "failed" and data["error"]


def test_bundled_evidence_and_fixture_demo(tmp_path):
    data = load_evidence(ROOT)
    assert (len(data["static"]), len(data["latency_only"]), len(data["training"])) == (30, 27, 5)
    result = run_demo(tmp_path / "demo with spaces")
    manifest = json.loads(Path(result["run_manifest"]).read_text(encoding="utf-8"))
    assert result["worker_exit_code"] == 0
    assert manifest["artifacts"][0]["sha256"] == result["fixture_sha256"]
    assert Path(result["report"]).exists()


def test_reject_corrupted_evidence(tmp_path):
    import shutil

    for file in ("ablation_2d/out_grid_2d.json", "ablation_3d/timing_3d.json", "b1a_pair/out_full_metrics_b1a_e3.json"):
        target = tmp_path / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / file, target)
    path = tmp_path / "ablation_2d/out_grid_2d.json"
    grid = json.loads(path.read_text(encoding="utf-8"))
    next(iter(grid["results"].values()))["p50_ms_f"] = 99999
    path.write_text(json.dumps(grid))
    with pytest.raises(ValueError, match="Quantiles"):
        load_evidence(tmp_path)
