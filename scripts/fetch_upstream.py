#!/usr/bin/env python3
"""Fetch the pinned upstream source without installing its extra dependencies."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "configs/upstream.json").read_text())
    target = ROOT / manifest["checkout_directory"]
    if target.exists():
        revision = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"], text=True)
        if revision != manifest["commit"] or dirty:
            raise SystemExit("Existing upstream checkout differs from the pin or has changes; preserve it and use a fresh checkout.")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", manifest["url"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", manifest["commit"]], check=True)
    print(f"Pinned upstream ready: {manifest['commit']}")


if __name__ == "__main__":
    main()
