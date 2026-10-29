"""Combine run directories into one CSV row per experiment and input length."""
import csv
import json
import statistics
from pathlib import Path

FIELDS = [
    "experiment_id", "method", "method_params", "split", "seed", "input_tokens", "run_status",
    "prompts", "failures", "accuracy", "passkey_accuracy",
    "prefill_cache_bytes", "final_cache_bytes", "prefill_cache_vs_fullkv",
    "median_ttft_ms", "median_decode_tokens_per_second",
    "max_prefill_peak_allocated_bytes", "max_decode_peak_allocated_bytes",
    "gpu_name", "git_revision", "run_directory",
]


def find_runs(paths):
    """Accept run directories or folders of them; a run has config.json and metrics.csv."""
    runs = []
    for path in map(Path, paths):
        if (path / "config.json").is_file() and (path / "metrics.csv").is_file():
            runs.append(path)
        elif path.is_dir():
            runs.extend(sorted(child for child in path.iterdir() if (child / "config.json").is_file() and (child / "metrics.csv").is_file()))
        else:
            raise ValueError(f"{path} is not a run directory or a folder of runs.")
    if not runs:
        raise ValueError("No run directories found.")
    return runs


def _number(value):
    return float(value) if value not in (None, "") else None


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _maximum(values):
    values = [value for value in values if value is not None]
    return max(values) if values else None


def run_rows(run):
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8")) if (run / "summary.json").is_file() else {}
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8")) if (run / "metadata.json").is_file() else {}
    with (run / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        metrics = list(csv.DictReader(stream))
    common = {
        "experiment_id": config["experiment_id"], "method": config["method"]["name"],
        "method_params": json.dumps(config["method"]["params"], sort_keys=True),
        "split": config["dataset"]["split"], "seed": config["seed"],
        "run_status": summary.get("status"),
        "gpu_name": metadata.get("selected_gpu_name"), "git_revision": metadata.get("git_revision"),
        "run_directory": str(run),
    }
    rows = []
    for length in config["dataset"]["input_tokens"]:
        group = [row for row in metrics if row.get("input_tokens") == str(length)]
        completed = [row for row in group if row["status"] == "completed"]
        failures = sum(row["status"] == "failed" for row in group)
        # A failure before any prompt (setup, model loading) has no input length; charge it to every length.
        failures += sum(row["status"] == "failed" and not row.get("input_tokens") for row in metrics)
        rows.append({
            **common, "input_tokens": length, "prompts": len(completed), "failures": failures,
            "accuracy": _mean(float(row["correct"] == "True") for row in completed),
            "passkey_accuracy": _mean(float(row["passkey_correct"] == "True") for row in completed),
            "prefill_cache_bytes": _mean(_number(row["prefill_cache_bytes"]) for row in completed),
            "final_cache_bytes": _mean(_number(row["final_cache_bytes"]) for row in completed),
            "median_ttft_ms": _median(_number(row["ttft_ms"]) for row in completed),
            "median_decode_tokens_per_second": _median(_number(row["decode_tokens_per_second"]) for row in completed),
            "max_prefill_peak_allocated_bytes": _maximum(_number(row["prefill_peak_allocated_bytes"]) for row in completed),
            "max_decode_peak_allocated_bytes": _maximum(_number(row["decode_peak_allocated_bytes"]) for row in completed),
        })
    return rows


def aggregate(paths, output):
    output = Path(output)
    if output.exists():
        raise ValueError(f"{output} already exists; choose a new file.")
    rows = [row for run in find_runs(paths) for row in run_rows(run)]
    keys = [(row["experiment_id"], row["input_tokens"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Two runs share an experiment_id; pass only one run per configuration.")
    fullkv = {(row["split"], row["seed"], row["input_tokens"]): row["prefill_cache_bytes"] for row in rows if row["method"] == "fullkv"}
    for row in rows:
        baseline = fullkv.get((row["split"], row["seed"], row["input_tokens"]))
        row["prefill_cache_vs_fullkv"] = row["prefill_cache_bytes"] / baseline if baseline and row["prefill_cache_bytes"] is not None else None
    rows.sort(key=lambda row: (row["split"], row["input_tokens"], row["method"] != "fullkv", row["experiment_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows
