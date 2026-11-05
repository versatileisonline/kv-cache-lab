#!/usr/bin/env bash
# Submit every remaining official run: final passkey, Qasper, and speed sets (one Slurm array each).
# Usage: bash scripts/submit_all.sh   (on a login node, after exporting the variables in docs/arc.md)
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
: "${KV_LAB_SLURM_ACCOUNT:?Export KV_LAB_SLURM_ACCOUNT first (see docs/arc.md).}"
: "${KV_LAB_WORKDIR:?Export KV_LAB_WORKDIR first (see docs/arc.md).}"

# Download the pinned Qasper data here; compute nodes may not reach the internet.
uv run --locked --extra inference python -m kv_cache_lab fetch-data

bash scripts/submit_set.sh configs/final --time=01:30:00
bash scripts/submit_set.sh configs/qasper --time=01:00:00
bash scripts/submit_set.sh configs/speed --time=01:30:00
echo
echo "Three array jobs submitted. When squeue -u \"\$USER\" shows none of them, run each Table: command above."
