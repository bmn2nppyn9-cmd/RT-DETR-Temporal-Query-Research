import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .config import load_config
from .report import load_evidence, render
from .runner import TASKS, command_for, environment, run_worker
from .runtime import ROOT


def doctor(cfg, probe=False):
    checks = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_available": bool(shutil.which("git")),
        "paths": {
            k: {"path": cfg[k], "exists": Path(cfg[k]).exists()} for k in ("data_root", "initial_weights", "backend")
        },
        "modules": {
            m: importlib.util.find_spec(m) is not None for m in ("numpy", "torch", "cv2", "scipy", "pycocotools")
        },
    }
    metadata = json.loads((ROOT / "patches/upstream.json").read_text(encoding="utf-8"))
    if shutil.which("git") and Path(cfg["backend"]).exists():
        p = subprocess.run(["git", "-C", cfg["backend"], "rev-parse", "HEAD"], capture_output=True, text=True)
        checks["backend_commit"] = p.stdout.strip() if p.returncode == 0 else None
    checks["expected_research_commit"] = metadata["research_commit"]
    checks["backend_source_matches"] = all(
        (Path(cfg["backend"]) / rel).is_file()
        and (Path(cfg["backend"]) / rel).read_bytes().replace(b"\r\n", b"\n")
        == (ROOT / "upstream_snapshot" / rel).read_bytes().replace(b"\r\n", b"\n")
        for rel in metadata["files"]
    )
    if probe:
        script = "from tqinfra.runtime import activate_backend; activate_backend(); import json,torch,ultralytics; from ultralytics.nn.modules.head import RTDETRDecoder; print(json.dumps({'torch':torch.__version__,'cuda':torch.cuda.is_available(),'backend':ultralytics.__file__,'pair_api':hasattr(RTDETRDecoder,'forward_train_pair')}))"
        p = subprocess.run(
            [sys.executable, "-c", script], env=environment(cfg), cwd=ROOT, capture_output=True, text=True, timeout=60
        )
        checks["probe"] = {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    ready = (
        all(x["exists"] for x in checks["paths"].values())
        and all(checks["modules"].values())
        and checks["git_available"]
        and checks["backend_source_matches"]
    )
    if probe:
        ready = ready and checks["probe"]["returncode"] == 0
        if checks["probe"]["returncode"] == 0:
            ready = ready and json.loads(checks["probe"]["stdout"].splitlines()[-1])["pair_api"]
    checks["ready_for_ml"] = bool(ready)
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tqinfra", description="RT-DETR local ML experiment control plane")
    sub = parser.add_subparsers(dest="action", required=True)
    d = sub.add_parser("doctor", help="Check paths/dependencies; optional isolated model probe")
    d.add_argument("--config", default=str(ROOT / "configs/local.example.json"))
    d.add_argument("--probe", action="store_true")
    report = sub.add_parser("report", help="Build an offline evidence dashboard without Torch")
    report.add_argument("--output", default=str(ROOT / "artifacts/report.html"))
    demo = sub.add_parser("demo", help="Run an isolated CPU fixture worker and build the evidence dashboard")
    demo.add_argument("--output", default=str(ROOT / "artifacts/demo"))
    sub.add_parser("validate-results", help="Validate bundled metric ranges, quantiles and provenance")
    run = sub.add_parser("run", help="Launch a managed research worker")
    run.add_argument("task", choices=TASKS)
    run.add_argument("--config", required=True)
    run.add_argument("--dry-run", action="store_true", help="Show command/config; do not check assets or launch")
    args = parser.parse_args(argv)
    try:
        if args.action == "demo":
            from .demo import run_demo

            print(json.dumps(run_demo(args.output), indent=2))
        elif args.action == "report":
            print(render(ROOT, args.output))
        elif args.action == "validate-results":
            data = load_evidence(ROOT)
            print(json.dumps({k: len(data[k]) for k in ("static", "latency_only", "training")}, indent=2))
        elif args.action == "doctor":
            checks = doctor(load_config(args.config), args.probe)
            print(json.dumps(checks, indent=2, ensure_ascii=False))
            return 0 if checks["ready_for_ml"] else 1
        else:
            cfg = load_config(args.config)
            command = command_for(args.task, cfg)
            if args.dry_run:
                print(json.dumps({"command": command, "config": cfg}, indent=2, ensure_ascii=False))
            else:
                code, folder = run_worker(command, cfg, args.task)
                print(f"Run artifacts: {folder}")
                print(f"Worker exit code: {code}")
                return code
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired, KeyError) as exc:
        print(f"tqinfra: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
