# kv-cache-lab

Benchmarking and optimizing LLM KV caches to evaluate tradeoffs between GPU memory usage, generation speed and answer quality.

The repository provides a FullKV passkey-retrieval pilot. See the [generic ARC setup guide](docs/arc.md) to run it using your own Virginia Tech account and allocation.

## Local setup

Use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python 3.11 and the committed dependency lock:

```bash
uv sync --locked
uv run --locked python -m kv_cache_lab validate --config configs/fullkv-pilot.json
uv run --locked python -m kv_cache_lab preview --config configs/fullkv-pilot.json
uv run --locked python -m unittest discover -s tests -v
```

The default environment needs no ML dependencies. Previews contain no measured results. See [setup instructions](docs/setup.md) for inference dependencies and ARC preparation.

## First GPU pilot

Inside your assigned NVIDIA GPU allocation, after model access is approved:

```bash
bash scripts/setup_env.sh
uv run --locked --extra inference huggingface-cli login
bash scripts/run_baseline.sh
```

The runner requires CUDA, keeps FP16 weights on one GPU, checks exact input token length, and saves predictions, cache bytes, phase memory peaks, timing diagnostics, and environment metadata. Runs go into ignored `results/runs/`. CPU/MPS execution cannot substitute for a GPU experiment.

## Project layout

| Path | Purpose |
| --- | --- |
| `configs/` | Pilot settings and pinned upstream revision |
| `src/kv_cache_lab/` | Configuration, prompts, scoring, and baseline runner |
| `scripts/` | uv setup, upstream checkout, and pilot command |
| `docs/` | Setup and [experiment specification](docs/experiment-spec.md) |
| `tests/` | Offline correctness checks; optional tiny-model CPU checks |
| `results/` | [Artifact meanings](results/README.md) and ignored run output |
| `pyproject.toml`, `uv.lock` | Dependency declarations and resolved lock |

The upstream [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory) revision is pinned for later method adapters. The first FullKV runner uses standard Hugging Face attention without compression patches. The pilot tests integration; its short-answer timings are not the planned performance benchmark.

## Scoring

The runner reports strict response accuracy and a separate passkey accuracy allowing one final period. Both require the complete response to match the documented form. See [scoring fields and offline rescoring](results/README.md#scoring-fields). Saved predictions can be rescored into a new directory without rerunning the model or changing the original run.
