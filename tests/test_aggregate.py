import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from kv_cache_lab.aggregate import aggregate
from kv_cache_lab.artifacts import append_metrics, write_json

ROOT = Path(__file__).resolve().parents[1]


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = json.loads((ROOT / "configs/pilot/fullkv.json").read_text())

    def make_run(self, name, method, rows, status="completed"):
        run = self.root / "runs" / name
        run.mkdir(parents=True)
        config = copy.deepcopy(self.config)
        config["experiment_id"] = name
        config["method"] = method
        write_json(run / "config.json", config)
        write_json(run / "summary.json", {"status": status})
        write_json(run / "metadata.json", {"selected_gpu_name": "NVIDIA A100-SXM4-40GB", "git_revision": "abc"})
        for row in rows:
            append_metrics(run / "metrics.csv", row)
        return run

    @staticmethod
    def row(length, correct, cache, ttft):
        return {"prompt_id": f"p-{length}-{ttft}", "status": "completed", "correct": correct, "passkey_correct": correct, "input_tokens": length, "prefill_cache_bytes": cache, "final_cache_bytes": cache + 10, "ttft_ms": ttft, "decode_tokens_per_second": 50.0, "prefill_peak_allocated_bytes": ttft * 100, "decode_peak_allocated_bytes": 1}

    def test_groups_by_length_and_compares_cache_with_fullkv(self):
        self.make_run("pilot-fullkv", {"name": "fullkv", "params": {}}, [self.row(2048, True, 400, 10), self.row(2048, False, 400, 30), self.row(4096, True, 800, 20)])
        self.make_run("pilot-h2o-b25", {"name": "h2o", "params": {"budget_fraction": 0.25, "recent_fraction": 0.5}}, [self.row(2048, True, 100, 12), self.row(4096, False, 200, 22), {"prompt_id": "p-4096-x", "status": "failed", "input_tokens": 4096, "failure_phase": "inference"}], status="failed")
        rows = aggregate([self.root / "runs"], self.root / "results.csv")
        table = {(row["experiment_id"], row["input_tokens"]): row for row in rows}
        self.assertEqual(table[("pilot-fullkv", 2048)]["accuracy"], 0.5)
        self.assertEqual(table[("pilot-fullkv", 2048)]["median_ttft_ms"], 20)
        self.assertEqual(table[("pilot-fullkv", 2048)]["max_prefill_peak_allocated_bytes"], 3000)
        self.assertEqual(table[("pilot-h2o-b25", 2048)]["prefill_cache_vs_fullkv"], 0.25)
        self.assertEqual(table[("pilot-h2o-b25", 4096)]["failures"], 1)
        self.assertEqual(table[("pilot-h2o-b25", 4096)]["run_status"], "failed")
        # Lengths with no prompts still get a row, with empty metrics.
        self.assertEqual(table[("pilot-fullkv", 8192)]["prompts"], 0)
        self.assertIsNone(table[("pilot-fullkv", 8192)]["accuracy"])
        with (self.root / "results.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 6)

    def test_setup_failure_counts_against_every_length(self):
        self.make_run("pilot-kivi-4bit", {"name": "kivi", "params": {"bits": 4, "group_size": 32, "residual_length": 128}}, [{"prompt_id": None, "status": "failed", "failure_phase": "model_loading"}], status="failed")
        rows = aggregate([self.root / "runs" / "pilot-kivi-4bit"], self.root / "results.csv")
        self.assertEqual([row["failures"] for row in rows], [1, 1, 1])

    def test_refuses_duplicates_and_existing_output(self):
        self.make_run("pilot-fullkv", {"name": "fullkv", "params": {}}, [self.row(2048, True, 400, 10)])
        with self.assertRaisesRegex(ValueError, "share an experiment_id"):
            aggregate([self.root / "runs", self.root / "runs" / "pilot-fullkv"], self.root / "a.csv")
        (self.root / "b.csv").write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            aggregate([self.root / "runs"], self.root / "b.csv")
        self.assertEqual((self.root / "b.csv").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
