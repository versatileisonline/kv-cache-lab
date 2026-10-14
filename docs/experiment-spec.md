# KV-cache experiment specification

## Question and initial milestone

How much resident KV-cache memory can we save while preserving answer quality, and what does the saving cost in generation time? Build a working FullKV pilot first. The pilot is an integration check, not a final comparison or evidence that compression improves performance.

ARC is the intended platform; class allocation access is pending. GPU model, VRAM, driver, scheduler settings, and model fit remain unconfirmed. This specification records proposed choices without claiming experimental results.

## Controlled settings

| Setting | Initial pilot |
| --- | --- |
| Model | `meta-llama/Llama-3.2-3B-Instruct` |
| Model/tokenizer revision | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Weights and cache | FP16, full cache |
| Attention | Hugging Face SDPA, no compression patches |
| Device | One assigned NVIDIA GPU, all weights on `cuda:0` |
| Batch and decoding | Batch 1, greedy, evaluation/inference mode |
| Prompt length | Exactly 2,048 input token IDs including template/question |
| Retrieval key | Random eight-digit string, including leading zeroes |
| Key positions | Approximately 20%, 50%, or 80% of the full input |
| Pilot size | 10 fixed-seed examples, positions cycled across examples |
| Output limit | 32 generated tokens, stop at model/tokenizer EOS IDs |
| Warmup | 2 unmeasured runs on the first pilot prompt |

`configs/fullkv-pilot.json` is the executable source of these settings. Model access and compatibility are pending the real environment check. If the model is replaced, pin a new revision, regenerate the prompts, and repeat every comparison. Do not silently offload weights or quantize them to fit one method.

## Prompt construction and scoring

Use deterministic seeded prose filler and a single marked passkey. The GPU path renders the pinned chat template with a fixed date, reserves space for the marker/question, then fills the remaining token slots. Store the exact input IDs and decoded prompt; do not retokenize the decoded prompt for evaluation. Record the actual marker index and position, and verify the code survives construction. The offline preview shows passage text and intended positions; it does not claim tokenizer-verified lengths.

Score the entire response as either the eight-digit code or `Answer: <code>`, allowing whitespace only around the optional prefix/code. Explanations, multiple codes, punctuation, and incorrect digits fail exact match. Preserve incorrect predictions and runtime failures in output.

Use pilot examples only to debug. After validation, propose 20 new examples per length/position at 2,048/4,096/8,192 tokens (180 prompts total), followed by 50 fixed, fitting LongBench v1 Qasper examples using its own scorer. Those datasets and method integrations are later work.

## Metrics and boundaries

Save exact-match correctness, input/output counts, output text, and failures per prompt. Record raw stored K/V bytes and shapes after prefill and after the last cache update, separately from per-process CUDA allocated/reserved peaks. Count unique physical tensor storage to avoid double-counting aliases. FullKV has no quantization metadata; quantized adapters must account for their metadata later.

CUDA synchronization surrounds each timed phase. TTFT starts with input already on the GPU and ends after the first greedy token is selected. Decode timing covers later forward passes/token selection. Loading, downloads, tokenization, cache inspection, scoring, and file writing are outside these intervals. Memory peaks reset at phase boundaries. Cache size at the end covers input plus generated tokens already processed; the last selected output token has not necessarily been added to the cache.

Pilot measurements are single-request diagnostics and include Python/EOS-check overhead. They are not stable throughput estimates. Before the main performance study, implement a separate 128-token workload, two warmups and five repetitions, randomized method ordering, median/IQR reporting, and separate instrumentation passes. Flag competing GPU processes using a rule agreed before the main campaign. Different GPU/session/model environments are separate campaigns.

## Reproducibility and output

Use committed `pyproject.toml` and `uv.lock` with `uv sync --locked`. Record the lock hash and installed packages, Git revision/dirty state, model revision, seed, configuration, GPU UUID/VRAM/driver, CUDA runtime, and visible-device assignment. Sample GPU processes before and after the pilot. This documents conditions; it does not guarantee bit-identical execution across GPU architectures.

Each run produces `metadata.json`, `prompts.jsonl`, `predictions.jsonl`, `metrics.csv`, and `summary.json` under an ignored, unique `results/runs/` directory. Persist completed prompts immediately. On runtime failure, preserve the failure phase/type, write a failed summary, and return nonzero. Curated examples can go under `results/examples/` with provenance.

## Method progression after the baseline

1. FullKV baseline and cache-reuse verification.
2. H2O and PyramidKV adapters against the pinned KVCache-Factory revision.
3. Separate 4-bit KV quantization comparison, optionally 2-bit after validation.

For retention methods, propose half/quarter prompt budgets and match total measured retained states across layers. Quantization instead changes bits per stored number; report measured bytes rather than equating bits with token counts. Keep prompts, weights, hardware, attention backend, and decoding constant. Later, investigate a controller choosing budget/precision under a measured byte cap using held-out examples.

## Pending decisions

- ARC allocation access, actual hardware, driver compatibility, and quiet testing periods.
- Hugging Face access, model fit, and compatibility of each planned method.
- Main-run dataset sizes, scoring choices, contamination rule, and GPU-hour estimate.
- Team owners and quantization group/residual settings.

The current scope stops at the FullKV setup/pilot runner. No compression method, issue board, or CI workflow is included.

References: [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory), [candidate model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), [LongBench](https://github.com/THUDM/LongBench).
