"""Run the same isolated process lifecycle used by model workers, without GPU assets."""

import json
import sys
from pathlib import Path

from .report import render
from .runner import run_worker, sha256, write_json
from .runtime import ROOT


def run_demo(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = {
        "workspace": str(output / "workspace"),
        "data_root": str(output / "data"),
        "backend": str(output / "backend"),
        "initial_weights": str(output / "weights.pt"),
        "timeout_seconds": 30,
    }
    result_path = output / "workspace/evaluation/fixture.json"
    worker = ROOT / "tqinfra/demo_worker.py"
    code, run_dir = run_worker(
        [sys.executable, str(worker), str(result_path)], cfg, task="control-plane-fixture", inputs=[worker]
    )
    if code:
        raise RuntimeError(f"Demo worker failed: {run_dir}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["sum"] != 108 or result["count"] != 6:
        raise ValueError("Worker fixture integrity mismatch")
    page = render(ROOT, output / "report.html")
    summary = {
        "scope": "CPU control-plane demo + historical research evidence",
        "worker_exit_code": code,
        "run_manifest": str(run_dir / "manifest.json"),
        "fixture_sha256": sha256(result_path),
        "report": str(page),
    }
    write_json(output / "summary.json", summary)
    return summary
