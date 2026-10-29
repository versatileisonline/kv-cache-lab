#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
uv sync --locked --extra inference
uv run --locked --extra inference python scripts/fetch_upstream.py
uv run --locked --extra inference python -c 'import torch, transformers; print("PyTorch:", torch.__version__); print("Transformers:", transformers.__version__); print("CUDA available:", torch.cuda.is_available())'
