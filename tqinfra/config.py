import json
from pathlib import Path

TAGS = ("b1a0", "b1ac25", "b1at31", "b1at23", "b1at33")
PATH_KEYS = ("data_root", "initial_weights", "backend", "workspace")
DEFAULTS = {
    "configs": ["b1a0", "b1at33"],
    "epochs": 1,
    "batch": 4,
    "device": "0",
    "workers": 3,
    "limit_scenes": 0,
    "limit_frames": 0,
    "timeout_seconds": 0,
}


def load_config(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be an object")
    unknown = set(data) - set(PATH_KEYS) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **data}
    for key in PATH_KEYS:
        value = cfg.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing path: {key}")
        p = Path(value).expanduser()
        cfg[key] = str((p if p.is_absolute() else path.parent / p).resolve())
    if (
        not isinstance(cfg["configs"], list)
        or not cfg["configs"]
        or any(not isinstance(t, str) or t not in TAGS for t in cfg["configs"])
        or len(set(cfg["configs"])) != len(cfg["configs"])
    ):
        raise ValueError(f"configs must be unique members of {TAGS}")
    for key in ("epochs", "batch", "workers", "limit_scenes", "limit_frames", "timeout_seconds"):
        minimum = 1 if key in ("epochs", "batch") else 0
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f"Invalid {key}: expected integer >= {minimum}")
    if not isinstance(cfg["device"], str) or not cfg["device"].strip():
        raise ValueError("device must be a nonempty string")
    return cfg
