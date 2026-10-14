#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
uv run --locked --extra inference python -m kv_cache_lab run --config configs/fullkv-pilot.json "$@"
