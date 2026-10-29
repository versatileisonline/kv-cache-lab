# KV-cache experiment specification

## Research question

How much resident KV-cache memory can we save while preserving answer quality, and what does the saving cost in generation time? We compare a FullKV baseline against token eviction (H2O) and low-bit cache quantization (KIVI-style) on the same model, prompts, and hardware. The FullKV pilot is an integration check, not a final comparison.

This specification is the Week 2 freeze. Values marked *provisional* may change only at the pilot gate (below), and any change gets recorded here before the final split runs.

## Hardware

All official measurements run on **one NVIDIA A100 on Virginia Tech ARC (TinkerCliffs `a100_normal_q`)**, requested through Slurm by each researcher with their own allocation; see the [ARC setup guide](arc.md). Configurations set `hardware.gpu_name_contains` to `A100`, and the runner refuses any other GPU. Kaggle, Colab, T4, CPU, and MPS runs are for development and the demo only. They never appear in comparison tables. If the A100 variant (40 GB vs 80 GB) differs between runs, report it; paired comparisons use a single variant.

## Controlled settings

| Setting | Value |
| --- | --- |
| Model | `meta-llama/Llama-3.2-3B-Instruct` (28 layers, 8 KV heads, head dim 128) |
| Model/tokenizer revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Weights | FP16, all on `cuda:0`; never offloaded or quantized |
| Attention | Hugging Face SDPA |
| Batch and decoding | Batch 1, greedy, inference mode, stop at model/tokenizer EOS IDs |
| Raw FullKV size | 114,688 bytes per token: about 0.22 / 0.44 / 0.875 / 1.75 / 3.5 GiB at 2K / 4K / 8K / 16K / 32K |

`configs/*.json` (schema version 2) are the executable source of these settings. If the model is replaced, pin a new revision, regenerate the prompts, and repeat every comparison.

## Methods

| Config name | Method | Parameters |
| --- | --- | --- |
| `fullkv` | Complete FP16 cache (baseline) | none |
| `h2o` | Keep recent and high-attention prompt positions per layer | `budget_fraction` 0.25 and 0.125 of the input length (512 and 256 positions at 2K, 1,024 and 512 at 4K, 2,048 and 1,024 at 8K); `recent_fraction` 0.5 of the budget (*provisional*) |
| `kivi` | KIVI-style quantized K/V | `bits` 4, then 2 after 4-bit validation; `group_size` 32 and `residual_length` 128 full-precision recent tokens (*provisional*) |

PyramidKV is an optional later comparison. The Week 8–10 hybrid (H2O eviction plus quantizing the retained states is the leading candidate) is specified once its components pass the pilot gate. Implementation notes, recorded so results are read correctly:

- **H2O** uses the SDPA H2O attention from the pinned [KVCache-Factory](../configs/upstream.json) commit, patched onto this model's attention layers only. The cache is stored per KV head (8 heads, `kv_cache_granularity=kv_head`), with scores averaged over each head's 3 query heads, so cache bytes compare directly with FullKV. Eviction happens once, at the end of prefill: the prompt keeps `budget` positions per layer (the most recent `budget × recent_fraction` plus the highest-scoring rest), and generated tokens are added without further eviction. Prefill attention itself uses the full prompt. Scoring builds the full prompt-length attention matrix, a large temporary allocation that may not fit at 16K–32K on a 40 GB A100.
- **KIVI** uses the Transformers 4.44 `HQQQuantizedCache` with keys quantized per channel (HQQ `axis=0`) and values per token (`axis=1`). KVCache-Factory's `kivi` setting uses the reverse axes (key 1, value 0), which puts keys per token, so we don't copy it. Differences from the KIVI paper: each key group shares one channel but its 32 elements are strided across tokens (and, for long prompts, heads) instead of 32 consecutive tokens; HQQ fits scales and zero-points by optimization instead of plain min/max; and the whole prompt is quantized at prefill. Only tokens generated afterwards stay in full precision, until `residual_length` of them accumulate. The cache is dequantized on every decode step, so KIVI saves resident cache memory but not decode time. Reported cache bytes include the packed values plus scales and zero-points.
- Pilot configurations for every method are in `configs/pilot/`. They share the seed and lengths, so all methods see identical prompts. Their 2K prompts are the same ones as `configs/fullkv-pilot.json`.

## Workloads

**Passkey retrieval.** Deterministic seeded prose filler with a single marked random eight-digit code (leading zeroes kept), placed at about 20%, 50%, or 80% of the full input. Inputs are exactly 2,048, 4,096, or 8,192 token IDs, including the chat template and question. The GPU path renders the pinned chat template with a fixed date, reserves space for the marker and question, fills the remaining slots, and stores the exact input IDs; prompts are never retokenized for evaluation.

- *Pilot split:* seed 42, 10 examples (2,048 tokens; extended to 4K/8K before the gate if needed).
- *Final split:* 180 fresh examples = 3 lengths × 3 positions × 20, using a seed never used for the pilot. The same 180 prompts are run for every method. The final split is frozen only after the pilot gate passes.

**Qasper.** 50 follow-up cases from the LongBench Qasper subset, truncated to at most 8,192 input tokens, up to 128 new tokens, scored with token-level F1. These are in addition to the 180 passkey prompts. (*Not yet implemented.*)

## Scoring

Passkey scores use the entire response. `strict-v1` accepts either the eight-digit code or `Answer: <code>`, with whitespace allowed around the optional prefix/code. `single-final-period-v1` also permits one period immediately after the code. Both reject explanations, multiple codes, other punctuation, Unicode digits, and incorrect digits. Leading zeroes are significant, and the prefix is case-sensitive.

`correct` and `accuracy` use `strict-v1`. `passkey_correct` and `passkey_accuracy` use `single-final-period-v1`. Record `scoring_rules` with each run. Keep incorrect predictions and runtime failures. Changing a scoring rule creates a new named version; it doesn't replace an earlier run's score. Freeze the rules before evaluating the final split.

## Pilot gate

Validate FullKV and each method on the 10 pilot cases before freezing the final split:

- FullKV inaccurate: fix the setup before going further.
- Every method nearly perfect: make the pilot harder (longer inputs, smaller budgets, or decoy codes) and record the change here.
- Checks pass: freeze the 180 final prompts and the provisional parameters.

The draft final configurations are in `configs/final/` (seed 1000, 60 prompts per length, 20 per length and position; no seed or passkey overlaps the pilot). `scripts/submit_set.sh` refuses to submit them until the gate passes. To freeze: apply any parameter changes from the gate to `configs/final/`, record them here, then commit an empty `configs/final/FROZEN` file together with them. After that, the final configurations must not change.

## Metrics and boundaries

Quality runs save exact-match correctness, input/output counts, output text, and failures per prompt. Record raw stored K/V bytes and shapes after prefill and after the last cache update, separately from per-process CUDA allocated/reserved peaks. Count unique physical tensor storage to avoid double-counting aliases. Compressed methods must also count their metadata (quantization scales/zero-points, eviction scores).

CUDA synchronization surrounds each timed phase. TTFT starts with input already on the GPU and ends after the first greedy token is selected. Decode timing covers later forward passes and token selection. Loading, downloads, tokenization, cache inspection, scoring, and file writing are outside these intervals. Memory peaks reset at phase boundaries.

Passkey answers are only a few tokens long, so their timings are diagnostic. Speed claims come from a separate **speed run** (*not yet implemented*): 256 forced new tokens (EOS ignored), 2,048 to 32,768 input tokens, 2 warmups, 5 repeats, and the median reported. Quality runs stay at 2K–8K. Speed and memory runs add 16K and 32K, where cache savings dominate.

## Reproducibility and output

Use the committed `pyproject.toml` and `uv.lock` with `uv sync --locked`. Record the lock hash and installed packages, Git revision and dirty state, model revision, seed, configuration, GPU name/UUID/VRAM/driver, CUDA runtime, and visible-device assignment. Sample GPU processes before and after each run.

Each run produces `metadata.json`, `prompts.jsonl`, `predictions.jsonl`, `metrics.csv`, and `summary.json` under an ignored, unique `results/runs/` directory. Persist completed prompts immediately. On runtime failure, record the failure phase/type, write a failed summary, print the traceback to the terminal only, and return nonzero. Curated examples can go under `results/examples/` with provenance.

References: [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory), [H2O](https://arxiv.org/abs/2306.14048), [KIVI](https://arxiv.org/abs/2402.02750), [model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
