"""Rescore saved predictions into a separate directory, without inference."""
import hashlib
import json
from pathlib import Path

from .artifacts import append_jsonl, create_run_directory, file_hash, write_json
from .scoring import SCORING_RULES, score_passkey


def rescore_predictions(predictions, output):
    source = Path(predictions).resolve()
    destination = Path(output).resolve()
    if destination == source.parent or source.parent in destination.parents:
        raise ValueError("Choose an output directory outside the original run directory.")
    payload = source.read_bytes()
    scores = []
    seen = set()
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("status") not in ("completed", "failed"):
                raise ValueError("Each prediction must have completed or failed status.")
            prompt_id = row.get("prompt_id")
            if prompt_id is not None and (not isinstance(prompt_id, str) or not prompt_id):
                raise ValueError("prompt_id must be a nonempty string or null for setup failure.")
            if row["status"] == "completed" and prompt_id is None:
                raise ValueError("Completed predictions require prompt_id.")
            if prompt_id in seen:
                raise ValueError("Duplicate prompt_id.")
            seen.add(prompt_id)
            result = {"prompt_id": prompt_id, "status": row["status"]}
            if row["status"] == "completed":
                result.update(score_passkey(row.get("response"), row.get("expected_code")))
                if "correct" in row:
                    if type(row["correct"]) is not bool:
                        raise ValueError("Recorded correct must be boolean.")
                    result["recorded_correct"] = row["correct"]
                    result["strict_score_matches_recorded"] = result["correct"] == row["correct"]
            scores.append(result)
        except (ValueError, TypeError) as error:
            raise ValueError(f"Prediction line {line_number}: {error}") from error
    if not scores:
        raise ValueError("No predictions found.")
    completed = [row for row in scores if row["status"] == "completed"]
    strict_correct = sum(row["correct"] for row in completed)
    passkey_correct = sum(row["passkey_correct"] for row in completed)
    summary = {
        "schema_version": 1, "mode": "rescoring", "status": "completed",
        "analysis_type": "post_hoc", "inference_performed": False,
        "scoring_rules": SCORING_RULES,
        "source_predictions_file": source.name,
        "source_predictions_sha256": hashlib.sha256(payload).hexdigest(),
        "scorer_sha256": file_hash(Path(__file__).with_name("scoring.py")),
        "rescoring_code_sha256": file_hash(Path(__file__)),
        "rows": len(scores), "completed": len(completed),
        "failures": len(scores) - len(completed),
        "correct": strict_correct,
        "accuracy": strict_correct / len(completed) if completed else None,
        "passkey_correct": passkey_correct,
        "passkey_accuracy": passkey_correct / len(completed) if completed else None,
        "strict_score_mismatches": sum(row.get("strict_score_matches_recorded") is False for row in completed),
    }
    create_run_directory(destination)
    for row in scores:
        append_jsonl(destination / "scores.jsonl", row)
    write_json(destination / "summary.json", summary)
    return summary
