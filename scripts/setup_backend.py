"""Install the pinned research patch into a new, dedicated backend checkout."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", default=str(ROOT / "vendor/ultralytics"))
    args = parser.parse_args()
    destination = Path(args.destination).resolve()
    if destination.exists():
        raise SystemExit("Destination already exists; choose a new directory to protect local modifications.")
    metadata = json.loads((ROOT / "patches/upstream.json").read_text(encoding="utf-8"))
    subprocess.run(["git", "clone", metadata["repository"], str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", metadata["base_commit"]], check=True)
    patch = str(ROOT / "patches/ultralytics_temporal_query.patch")
    subprocess.run(["git", "-C", str(destination), "apply", "--check", patch], check=True)
    subprocess.run(["git", "-C", str(destination), "apply", patch], check=True)
    print(f"Backend ready: {destination}. Install it with: python -m pip install -e {destination}")


if __name__ == "__main__":
    main()
