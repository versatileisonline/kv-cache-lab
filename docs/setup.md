# Setup

ARC is the intended compute platform. Class resource access is pending the professor's response. Continue local work now; confirm ARC GPU, allocation, modules, and scheduler settings after access is granted. No account names or scheduler directives are assumed here.

## uv workflow

Install [uv using its official instructions](https://docs.astral.sh/uv/getting-started/installation/). The committed `.python-version` selects Python 3.11; `pyproject.toml` declares dependencies and `uv.lock` pins their resolved versions and artifact hashes. uv 0.8.22 was used to generate and validate this lock. Use `--locked` to reject changes that would require updating it. The supported environments are Apple Silicon macOS for development and Linux x86_64 for GPU experiments.

```bash
uv sync --locked
```

The default environment installs the project without ML dependencies. Configuration, scoring, and previews use the standard library:

```bash
uv run --locked python -m kv_cache_lab validate --config configs/fullkv-pilot.json
uv run --locked python -m kv_cache_lab preview --config configs/fullkv-pilot.json --output results/runs/offline-preview
uv run --locked python -m unittest discover -s tests -v
```

The optional `inference` extra adds PyTorch, Transformers, Accelerate, Hugging Face Hub, Safetensors, and NumPy. Transformers 4.44.2 follows the pinned upstream baseline; the remaining versions form a candidate compatible stack, awaiting validation on ARC. Tokenizers and other transitive dependencies are pinned by `uv.lock`. No requirements file is needed.

```bash
bash scripts/setup_env.sh
```

On Linux, PyTorch is explicitly sourced from its CUDA 12.1 index; on macOS it comes from PyPI. The custom index is restricted to PyTorch. This follows [uv's documented PyTorch index configuration](https://docs.astral.sh/uv/guides/integration/pytorch/). macOS inference dependencies enable CPU correctness checks; the experiment runner still requires CUDA and never reports CPU/MPS results as GPU measurements.

## After ARC access

Work inside an assigned GPU allocation, following the class/ARC instructions. Record GPU name, UUID, VRAM, driver, and other processes with `nvidia-smi`. Confirm CUDA 12.1 driver compatibility and model fit before the pilot. A different ARC software stack requires an intentionally updated lock and a new campaign. Do not run GPU experiments on a login node.

```bash
bash scripts/setup_env.sh
uv run --locked python scripts/fetch_upstream.py
```

Upstream is pinned in `configs/upstream.json`, downloaded into ignored `third_party/`, and reserved for later compression adapters. The initial FullKV runner uses standard Hugging Face attention with caching; no upstream monkey patches, quantization libraries, or FlashAttention builds are needed.

The candidate [Llama-3.2-3B-Instruct model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) requires approved Hugging Face access. Request access yourself and authenticate in the GPU environment:

```bash
uv run --locked --extra inference huggingface-cli login
```

Use the platform's secret support; keep credentials out of Git. If the team changes the model, pin its revision and repeat all paired comparisons.

## First GPU pilot

```bash
bash scripts/run_baseline.sh
```

The runner uses `cuda:0` within the scheduler's assignment and preserves `CUDA_VISIBLE_DEVICES`. It writes a unique directory under ignored `results/runs/`, with predictions, metrics, prompts, a summary, and metadata. Downloads/model loading are outside inference timing. Setup or inference failure returns nonzero. Quality-pilot timing is diagnostic; the fixed-length performance campaign is separate.

## Dependency changes

Change `pyproject.toml` deliberately, run `uv lock`, review the lock diff, then repeat the environment checks. Routine installation uses `uv sync --locked`, never an unconstrained upgrade. Each GPU run records the lockfile hash, actual installed packages, Python, model revision, Git revision, GPU, driver, and CUDA runtime. A lockfile cannot make different hardware or kernels equivalent.

References: [uv locking and syncing](https://docs.astral.sh/uv/concepts/projects/sync/), [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory), [official PyTorch version installation commands](https://pytorch.org/get-started/previous-versions/).
