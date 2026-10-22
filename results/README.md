# Run artifacts

`results/runs/` is ignored. Every preview or pilot creates a new directory; an existing directory is refused to protect prior results.

Offline previews contain `config.json`, `prompts.jsonl`, and `summary.json`. They use prose/word positions, have no verified model token length, and contain no measured accuracy or GPU metrics.

GPU pilots additionally write `metadata.json`, `predictions.jsonl`, and `metrics.csv`. Schema 2 summaries also record the `method` and per-length counts under `by_input_tokens`. A failed GPU setup also writes a failed summary and error type/phase. The output is experimental evidence only when `mode` is `gpu_pilot`, the environment is recorded, and actual inference succeeds. Incorrect answers remain in the results. A completed pilot is not proof of model quality or compression performance.

Byte counts are raw bytes (divide by 1,048,576 for MiB). `ttft_ms` covers prefill and first greedy selection. `decode_tokens_per_second` excludes the first selected token and is null if no decode forward pass occurred. Peaks are per-process CUDA allocated/reserved bytes, including weights and temporary allocations; cache bytes count the unique K/V storage separately. Short quality answers make speed diagnostics noisy.

The final cache contains states through the last processed token, which is normally one fewer than the number of selected output tokens. The selected last/EOS token need not be fed back through the model. See the recorded cache shapes and `decode_forward_calls`.

Only intentionally reviewed small examples belong in `results/examples/`. Include their provenance and label synthetic/offline fixtures explicitly. Keep measured run artifacts private unless explicitly reviewed for publication.

## Scoring fields

`correct`, `extracted_code`, and summary `accuracy` retain `strict-v1`: an eight-digit code with optional `Answer:` prefix and surrounding whitespace. A final period fails this strict rule.

`passkey_correct`, `passkey_extracted_code`, and summary `passkey_accuracy` use `single-final-period-v1`: the same full-response rule with one optional period immediately after the code. Explanations, multiple answers, other punctuation, and wrong digits still fail. This is not arbitrary code extraction from prose.

New GPU runs record `scoring_rules` in metadata and summaries, and both correctness fields in prediction/CSV records. Accuracies use completed predictions as the denominator; runtime failures are reported separately. Older files remain valid and do not acquire new fields automatically. Do not overwrite them when applying a later scoring rule. Ten pilot examples do not establish general model quality.

## Rescore saved predictions

From the repository root, with `run_dir` set to an existing run directory:

```bash
.venv/bin/python -m kv_cache_lab rescore \
  --predictions "$run_dir/predictions.jsonl" \
  --output "${run_dir}-rescored"
```

This uses only the standard library and performs no model inference. On ARC, use an allocated compute shell. The output must be a new directory outside the original run directory. It contains `scores.jsonl` and `summary.json`; input predictions, original summaries, and GPU measurements are unchanged. A repeated command with the same output path is refused.

Rescoring reports both metrics with rule versions, source-file and scoring-code hashes, and `analysis_type=post_hoc`. This additional analysis is not a new independent evaluation. `strict_score_mismatches` flags disagreement with an original `correct` field, when present; investigate any nonzero value. Failed predictions are counted separately and excluded from both accuracy denominators; a failed-only file reports null accuracies. Malformed, empty, or duplicate-ID inputs are rejected.
