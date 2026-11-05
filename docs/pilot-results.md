# Pilot results

**Status: pilot, not final.** 10 prompts per method and length, so each accuracy value moves in steps of 0.1. These numbers check that the setup works and choose parameters. They are not the comparison we report.

## Setup

- Array job 8288864, one NVIDIA A100-SXM4-80GB per method, commit `e82aa51`.
- Llama-3.2-3B-Instruct, FP16, greedy decoding.
- Passkey retrieval: one 8-digit code hidden at 20%, 50% or 80% of the prompt. Prompt lengths 2K, 4K and 8K tokens, 10 prompts each, seed 42.
- Every method saw identical prompts.

## Accuracy

The headline score is passkey accuracy (lenient: the code, optionally followed by one period). Strict accuracy, in brackets, also rejects the period.

| Method | Cache kept | 2K | 4K | 8K |
| --- | --- | --- | --- | --- |
| FullKV | 100% | **1.0** (0.4) | **1.0** (0.9) | **1.0** (1.0) |
| KIVI 4-bit | 31% | **1.0** (0.5) | **1.0** (0.7) | **1.0** (1.0) |
| KIVI 2-bit | 19% | 0.8 (0.7) | 0.9 (0.7) | 0.7 (0.7) |
| H2O 25% budget | 25% | 0.0 (0.0) | 0.0 (0.0) | 0.1 (0.1) |
| H2O 12.5% budget | 12.5% | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |

## Memory

KV cache stored after reading an 8K prompt. All lengths show the same ratios.

| Method | Cache | vs FullKV | Peak GPU memory during prefill |
| --- | --- | --- | --- |
| FullKV | 896 MiB | 1.00 | 12.8 GiB |
| KIVI 4-bit | 280 MiB | 0.31 | 12.2 GiB |
| KIVI 2-bit | 168 MiB | 0.19 | 12.1 GiB |
| H2O 25% | 224 MiB | 0.25 | 18.7 GiB |
| H2O 12.5% | 112 MiB | 0.125 | 18.6 GiB |

Every cache size matches the value predicted from the method's design exactly. KIVI's figure includes its scales and zero-points.

## Speed (diagnostic only)

These are short answers on a shared cluster, so they show direction only. Proper speed measurements come later, from a separate speed run.

| Method | Time to first token, 8K | Decode tokens/s |
| --- | --- | --- |
| FullKV | 381 ms | 44 |
| H2O | 1,476 ms | 41–43 |
| KIVI 4-bit | 1,154 ms | 35 |
| KIVI 2-bit | 1,379 ms | 31.5 |

## Findings

1. **The harness works.** All 150 prompts completed with no failures, and FullKV finds every code at every length.
2. **Strict scoring measures punctuation, not memory.** At 2K, FullKV's strict score is 0.4 only because it often ends with a period. Passkey accuracy is now the headline metric, with strict kept as a secondary column.
3. **KIVI 4-bit is lossless here at about a third of the memory.** 2-bit saves more and starts losing codes (70–90% correct).
4. **H2O fails, and the way it fails is informative.** All 12 H2O answers we inspected (25% budget) got the first three digits right and made up the other five, for example `Answer: 85879124` when the code was `85822412`. Llama stores numbers in chunks of up to 3 digits, so H2O keeps the first chunk of the code and evicts the rest. The answers are well formatted and confident, so the model gives a wrong code instead of saying it doesn't know. This is the expected weakness of evicting by total attention: one-off facts get little attention compared with the prompt's first tokens and the filler. It is not a bug, because a position bug would produce garbage. One caveat: the first output token is computed from the full prompt before eviction takes effect, so answers that begin directly with digits got their first chunk "for free". Answers starting with `Answer:` show the first chunk really survived in the cache.
5. **Compression is not free in time.** H2O builds the full attention matrix to score tokens, which adds about 6 GiB of temporary memory at 8K and roughly quadruples time to first token. KIVI decodes 20–30% slower, because it unpacks the cache on every step. These costs are what the later speed run will measure properly.

## Pilot gate

| Check | Result |
| --- | --- |
| FullKV accurate | Pass |
| Not every method near-perfect | Pass (H2O fails, KIVI 2-bit loses some codes) |
| Parameters final | **Not yet.** H2O is at zero at both budgets, so it needs one follow-up pilot. |

The follow-up pilot, `configs/pilot-h2o/` (same prompts), tests whether any H2O setting retrieves the code:

| Config | Budget | Recent window |
| --- | --- | --- |
| `h2o-b50` | 50% | half of budget |
| `h2o-b25-r10` | 25% | 10% of budget |
| `h2o-b12-r10` | 12.5% | 10% of budget |

Scoring also gained `leading-digits-v1`, the number of correct leading digits (0–8). It shows how much of the code each H2O setting keeps instead of a flat zero. The aggregate command computes it from saved answers, so the pilot above gets it without rerunning.

## Next steps

1. Submit `configs/pilot-h2o/`, then aggregate it together with job 8288864.
2. Pick the H2O settings for the final split, using the follow-up results. If none works, keep 25% and 12.5% and report the failure.
3. Copy the chosen settings into `configs/final/`, record them in the spec, and commit `configs/final/FROZEN`.
4. Run the final split: 180 prompts per method, the same prompts for every method.
5. Build the remaining pieces: the speed run (2K–32K, 256 forced tokens), the Qasper task with F1, and the charts and tables for the report.
