"""Isolated local workers, exclusive workspace, and durable run provenance."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .runtime import ROOT

TASKS = ("prepare-pairs", "train", "record", "evaluate", "regression")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


@contextmanager
def workspace_lock(workspace):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    lock = workspace / ".worker.lock"
    try:
        with lock.open("x", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "host": platform.node()}))
    except FileExistsError:
        raise RuntimeError(f"Workspace busy: {lock}. Verify owner before removing a stale lock.") from None
    try:
        yield
    finally:
        lock.unlink()


def environment(cfg):
    env = os.environ.copy()
    for key in ("data_root", "initial_weights", "backend", "workspace"):
        env["TQ_" + key.upper()] = cfg[key]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), cfg["backend"], env.get("PYTHONPATH")]))
    env["PYTHONIOENCODING"] = "utf-8"
    env["YOLO_AUTOINSTALL"] = "false"
    return env


def command_for(task, cfg):
    workspace = Path(cfg["workspace"])
    commands = {
        "prepare-pairs": ["b1a_pair/make_pair_subset.py"],
        "train": [
            "b1a_pair/train_b1a.py",
            "--configs",
            *cfg["configs"],
            "--epochs",
            str(cfg["epochs"]),
            "--batch",
            str(cfg["batch"]),
            "--device",
            cfg["device"],
            "--workers",
            str(cfg["workers"]),
        ],
        "record": [
            "b1a_pair/record_b1a.py",
            "--configs",
            *cfg["configs"],
            "--run-dir",
            str(workspace / "checkpoints"),
            "--out",
            str(workspace / "records"),
            "--limit-scenes",
            str(cfg["limit_scenes"]),
            "--device",
            cfg["device"],
            "--limit-frames",
            str(cfg["limit_frames"]),
        ],
        "evaluate": [
            "b1a_pair/eval_b1a.py",
            "--records",
            str(workspace / "records"),
            "--tags",
            *cfg["configs"],
            "--suffix",
            "run",
        ],
        "regression": ["b1a_pair/test_b1a_regression.py"],
    }
    if task not in commands:
        raise ValueError(f"Unknown task: {task}")
    script, *args = commands[task]
    return [sys.executable, str(ROOT / script), *args]


def preflight(task, cfg):
    data = Path(cfg["data_root"])
    work = Path(cfg["workspace"])
    required = []
    if task in ("train", "record", "regression"):
        required += [Path(cfg["backend"]) / "ultralytics/nn/modules/head.py"]
    if task in ("train", "regression"):
        required += [
            Path(cfg["initial_weights"]),
            work / "pairs/pairs_train.json",
            work / "pairs/all_frames_train.txt",
            work / "pairs/t_frames_train.txt",
            work / "pairs/val_sub20.txt",
        ]
    if task == "prepare-pairs":
        required += [data / "argoverse-hd/stream_index_train.json", data / "argoverse-hd-yolo/train_sub_val.txt"]
    if task in ("record", "evaluate"):
        required += [data / "argoverse-hd/stream_index.json", data / "argoverse-hd/annotations_val.json"]
        required += [
            (work / "checkpoints" / tag / "weights/best.pt")
            if task == "record"
            else (work / "records" / f"preds_{tag}.npz")
            for tag in cfg["configs"]
        ]
    if task == "evaluate" and "b1a0" not in cfg["configs"]:
        raise ValueError("evaluate requires b1a0 as its matched baseline")
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing prerequisites:\n" + "\n".join(missing))
    outputs = {"prepare-pairs": work / "pairs", "record": work / "records", "evaluate": work / "evaluation"}
    if task in outputs and outputs[task].exists() and any(outputs[task].iterdir()):
        raise FileExistsError(
            f"Stage output exists: {outputs[task]}; use a fresh workspace or explicitly move old outputs"
        )
    if task == "train" and any((work / "checkpoints" / tag).exists() for tag in cfg["configs"]):
        raise FileExistsError("Training output exists; use a fresh workspace to avoid overwriting checkpoints")
    return required


def run_worker(command, cfg, task="custom", inputs=()):
    """No shell expansion. Record failure/timeout even if the worker exits abruptly."""
    workspace = Path(cfg["workspace"])
    with workspace_lock(workspace):
        # Check under the lock so concurrent launches cannot both pass preflight.
        if task in TASKS:
            inputs = preflight(task, cfg)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = workspace / "runs" / run_id
        run_dir.mkdir(parents=True)
        manifest_path = run_dir / "manifest.json"

        def git(*args):
            p = subprocess.run(
                ["git", "-C", str(ROOT), *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            return p.stdout.strip() if p.returncode == 0 else None

        code_paths = sorted(list((ROOT / "tqinfra").glob("*.py")) + list((ROOT / "b1a_pair").glob("*.py")))
        code_paths += [
            ROOT / p
            for p in (
                "eval_sap.py",
                "eval_full_metrics.py",
                "query_depth_probe/eval_qd_t.py",
                "query_prop_b0/eval_b0.py",
                "cfg/rtdetr-resnet18.yaml",
                "patches/ultralytics_temporal_query.patch",
            )
        ]
        meta = {
            "schema_version": 1,
            "run_id": run_id,
            "task": task,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "config": cfg,
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
            "backend_code": {
                rel: sha256(Path(cfg["backend"]) / rel)
                for rel in (
                    "ultralytics/nn/modules/head.py",
                    "ultralytics/nn/modules/transformer.py",
                    "ultralytics/nn/tasks.py",
                    "ultralytics/models/utils/loss.py",
                )
                if (Path(cfg["backend"]) / rel).is_file()
            },
            "inputs": [{"path": str(p), "sha256": sha256(p)} for p in inputs],
            "code": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code_paths if p.is_file()},
        }
        write_json(manifest_path, meta)
        started = time.monotonic()
        code = 1
        process = None
        try:
            with (run_dir / "worker.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment(cfg),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name != "nt"),
                )
                try:
                    code = process.wait(timeout=cfg.get("timeout_seconds", 0) or None)
                    meta["status"] = "succeeded" if code == 0 else "failed"
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                    else:
                        import signal

                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    meta["status"] = "timed_out" if isinstance(exc, subprocess.TimeoutExpired) else "cancelled"
                    code = 124 if meta["status"] == "timed_out" else 130
        except OSError as exc:
            meta["status"] = "failed"
            meta["error"] = str(exc)
        finally:
            meta.update(
                returncode=code,
                duration_seconds=round(time.monotonic() - started, 3),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            # Checkpoints can be large; hash them once at stage completion for lineage.
            meta["artifacts"] = []
            for dirname in ("pairs", "checkpoints", "records", "evaluation"):
                for p in sorted((workspace / dirname).rglob("*")):
                    if p.is_file() and p.suffix in (".json", ".pt", ".npz", ".txt", ".yaml"):
                        meta["artifacts"].append(
                            {
                                "path": p.relative_to(workspace).as_posix(),
                                "bytes": p.stat().st_size,
                                "sha256": sha256(p),
                            }
                        )
            write_json(manifest_path, meta)
        return code, run_dir
