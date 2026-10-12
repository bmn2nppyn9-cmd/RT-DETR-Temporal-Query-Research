"""Dependency-free fixture worker for demonstrating orchestration, not inference."""

import json
import sys
from pathlib import Path


def main():
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [4, 8, 15, 16, 23, 42]
    result = {
        "scope": "deterministic control-plane fixture; not a model benchmark",
        "input": values,
        "sum": sum(values),
        "count": len(values),
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("fixture result committed", flush=True)


if __name__ == "__main__":
    main()
