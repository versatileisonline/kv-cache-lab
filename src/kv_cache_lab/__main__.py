"""Run with uv run --locked python -m kv_cache_lab."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .artifacts import append_jsonl, create_run_directory, write_json
from .config import load_config
from .prompts import cases, preview_case


def main(argv=None):
    parser = argparse.ArgumentParser(description="KV-cache experiment setup, execution, and analysis")
    parser.add_argument("command", choices=("validate", "preview", "run", "rescore", "aggregate"))
    parser.add_argument("--config", type=Path, default=Path("configs/fullkv-pilot.json"))
    parser.add_argument("--predictions", type=Path, help="Saved predictions.jsonl for the rescore command")
    parser.add_argument("--runs", type=Path, nargs="+", help="Run directories, or folders of them, for the aggregate command")
    parser.add_argument("--output", type=Path, help="New directory (or CSV file for aggregate); existing results are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.command == "aggregate":
            if not args.runs or args.output is None:
                raise ValueError("aggregate requires --runs and --output.")
            from .aggregate import aggregate
            rows = aggregate(args.runs, args.output)
            print(f"Wrote {len(rows)} rows to {args.output.resolve()}")
            return 0
        if args.command == "rescore":
            if args.predictions is None:
                raise ValueError("rescore requires --predictions.")
            from .rescore import rescore_predictions
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            output = args.output or Path("results/runs") / f"rescored-{stamp}-{uuid4().hex[:8]}"
            summary = rescore_predictions(args.predictions, output)
            print(json.dumps({"output_directory": str(output.resolve()), **summary}, indent=2))
            return 0
        config = load_config(args.config)
        if args.command == "validate":
            print(f"Valid configuration: {config['experiment_id']} ({config['method']['name']})")
            return 0
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = create_run_directory(args.output or Path("results/runs") / f"{config['experiment_id']}-{args.command}-{stamp}-{uuid4().hex[:8]}")
        write_json(output / "config.json", config)
        if args.command == "preview":
            for case in cases(config):
                append_jsonl(output / "prompts.jsonl", preview_case(case))
            summary = {"mode": "offline_preview", "status": "completed", "examples": config["dataset"]["num_examples"] * len(config["dataset"]["input_tokens"]), "tokenizer_verified": False, "gpu_measurements": None, "accuracy": None}
            write_json(output / "summary.json", summary)
        else:
            from .baseline import run_pilot
            summary = run_pilot(config, output)
        print(json.dumps({"output_directory": str(output.resolve()), **summary}, indent=2))
        return 0 if summary["status"] == "completed" else 1
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Setup error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
