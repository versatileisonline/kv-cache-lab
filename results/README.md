# Run artifacts

`results/runs/` is ignored. Every preview or pilot creates a new directory; an existing directory is refused to protect prior results.

Offline previews contain `config.json`, `prompts.jsonl`, and `summary.json`. They use prose/word positions, have no verified model token length, and contain no measured accuracy or GPU metrics.

GPU pilots additionally write `metadata.json`, `predictions.jsonl`, and `metrics.csv`. A failed GPU setup also writes a failed summary and error type/phase. The output is experimental evidence only when `mode` is `gpu_pilot`, the environment is recorded, and actual inference succeeds. Incorrect answers remain in the results. A completed pilot is not proof of model quality or compression performance.

Byte counts are raw bytes (divide by 1,048,576 for MiB). `ttft_ms` covers prefill and first greedy selection. `decode_tokens_per_second` excludes the first selected token and is null if no decode forward pass occurred. Peaks are per-process CUDA allocated/reserved bytes, including weights and temporary allocations; cache bytes count the unique K/V storage separately. Short quality answers make speed diagnostics noisy.

The final cache contains states through the last processed token, which is normally one fewer than the number of selected output tokens. The selected last/EOS token need not be fed back through the model. See the recorded cache shapes and `decode_forward_calls`.

Only intentionally reviewed small examples belong in `results/examples/`. Include their provenance and label synthetic/offline fixtures explicitly. No measured GPU sample is committed yet because ARC access is pending.
