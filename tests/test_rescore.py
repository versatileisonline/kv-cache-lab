import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from kv_cache_lab.__main__ import main
from kv_cache_lab.rescore import rescore_predictions


class RescoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / "original"
        self.run.mkdir()
        self.predictions = self.run / "predictions.jsonl"

    def save(self, rows):
        self.predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def prediction(self, prompt_id, response, correct=False):
        return {"prompt_id": prompt_id, "status": "completed", "expected_code": "00123456", "response": response, "correct": correct}

    def test_separate_metrics_preserve_source_and_report_failures(self):
        self.save([
            self.prediction("p0", "00123456", True),
            self.prediction("p1", "Answer: 00123456."),
            self.prediction("p2", "87654321."),
            {"prompt_id": "p3", "status": "failed", "failure_phase": "decode"},
        ])
        source = self.predictions.read_bytes()
        summary_file = self.run / "summary.json"
        summary_file.write_text('{"accuracy": 0.3333333333333333}\n')
        summary_before = summary_file.read_bytes()
        result = rescore_predictions(self.predictions, self.root / "rescored")
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["passkey_correct"], 2)
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertAlmostEqual(result["passkey_accuracy"], 2 / 3)
        self.assertEqual(result["completed"], 3)
        self.assertEqual(result["failures"], 1)
        self.assertEqual(result["strict_score_mismatches"], 0)
        self.assertEqual(result["source_predictions_sha256"], hashlib.sha256(source).hexdigest())
        self.assertFalse(result["inference_performed"])
        self.assertEqual(result["analysis_type"], "post_hoc")
        self.assertEqual(self.predictions.read_bytes(), source)
        self.assertEqual(summary_file.read_bytes(), summary_before)

    def test_invalid_input_is_rejected_before_creating_output(self):
        invalid = [
            [],
            [self.prediction("p0", "00123456"), self.prediction("p0", "00123456")],
            [{"prompt_id": "p0", "status": "completed", "response": "00123456"}],
            [{"prompt_id": "p0", "status": "unknown"}],
            [self.prediction("p0", None)],
        ]
        for index, rows in enumerate(invalid):
            with self.subTest(rows=rows):
                self.save(rows)
                output = self.root / f"invalid-{index}"
                with self.assertRaises(ValueError):
                    rescore_predictions(self.predictions, output)
                self.assertFalse(output.exists())
        self.predictions.write_text("not json\n")
        with self.assertRaisesRegex(ValueError, "Prediction line 1"):
            rescore_predictions(self.predictions, self.root / "bad-json")

    def test_existing_output_and_original_run_descendants_are_protected(self):
        self.save([self.prediction("p0", "00123456", True)])
        for output in (self.run, self.run / "new-scores"):
            with self.assertRaises(ValueError):
                rescore_predictions(self.predictions, output)
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("keep")
        with self.assertRaises(FileExistsError):
            rescore_predictions(self.predictions, output)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_failed_only_runs_have_no_accuracy(self):
        self.save([{"prompt_id": None, "status": "failed"}])
        result = rescore_predictions(self.predictions, self.root / "failed-analysis")
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["passkey_accuracy"])
        self.assertEqual(result["failures"], 1)

    def test_recorded_score_disagreement_is_flagged(self):
        self.save([self.prediction("p0", "00123456", False)])
        result = rescore_predictions(self.predictions, self.root / "mismatch")
        self.assertEqual(result["strict_score_mismatches"], 1)

    def test_cli_needs_no_config_and_keeps_input_unchanged(self):
        self.save([self.prediction("p0", "00123456.")])
        source = self.predictions.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["rescore", "--predictions", str(self.predictions), "--output", str(self.root / "cli"), "--config", str(self.root / "missing.json")]), 0)
            self.assertEqual(main(["rescore"]), 1)
        self.assertEqual(self.predictions.read_bytes(), source)
