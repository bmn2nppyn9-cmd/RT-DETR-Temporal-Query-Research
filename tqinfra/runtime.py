"""Shared paths for the portable B1a worker chain."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("TQ_DATA_ROOT", ROOT / "data")).resolve()
WORKSPACE = Path(os.environ.get("TQ_WORKSPACE", ROOT / "artifacts" / "workspace")).resolve()
PAIR_DIR = WORKSPACE / "pairs"
TRAIN_DIR = WORKSPACE / "checkpoints"
RECORD_DIR = WORKSPACE / "records"
EVAL_DIR = WORKSPACE / "evaluation"
MODEL_YAML = ROOT / "cfg" / "rtdetr-resnet18.yaml"
INITIAL_WEIGHTS = Path(os.environ.get("TQ_INITIAL_WEIGHTS", ROOT / "weights" / "baseline.pt")).resolve()
BACKEND = Path(os.environ.get("TQ_BACKEND", ROOT / "vendor" / "ultralytics")).resolve()


def activate_backend():
    """Prefer the explicitly configured backend; never silently use the old PC path."""
    if not (BACKEND / "ultralytics" / "nn" / "modules" / "head.py").is_file():
        raise RuntimeError(f"Patched backend missing: {BACKEND}; see docs/quickstart.md")
    sys.path.insert(0, str(BACKEND))
