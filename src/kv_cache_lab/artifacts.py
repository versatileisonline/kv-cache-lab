"""Small, explicit artifact formats with overwrite protection."""
import csv
import hashlib
import json
from pathlib import Path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append_jsonl(path, data):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data, allow_nan=False) + "\n")


def create_run_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def append_metrics(path, row):
    fields = [
        "prompt_id", "status", "correct", "input_tokens", "output_tokens",
        "actual_position", "cache_reused", "prefill_cache_bytes", "final_cache_bytes",
        "ttft_ms", "decode_ms", "decode_tokens_per_second",
        "prefill_peak_allocated_bytes", "prefill_peak_reserved_bytes",
        "decode_peak_allocated_bytes", "decode_peak_reserved_bytes", "failure_phase", "error_type",
    ]
    path = Path(path)
    fresh = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
