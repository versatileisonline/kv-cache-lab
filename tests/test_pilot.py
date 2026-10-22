import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from kv_cache_lab.__main__ import main
from kv_cache_lab.config import load_config, validate_config
from kv_cache_lab.methods import validate_method
from kv_cache_lab.prompts import cases, tokenized_case
from kv_cache_lab.scoring import score_passkey

ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    """Deterministic test tokenizer; not a model tokenizer or benchmark fixture."""

    def apply_chat_template(self, messages, **kwargs):
        return "<chat>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>"

    def encode(self, text, **kwargs):
        return list(text.encode("ascii"))

    def decode(self, ids, **kwargs):
        return bytes(ids).decode("ascii")


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/fullkv-pilot.json")

    def test_strict_scoring_preserves_leading_zeroes(self):
        for response in ("00123456", "  Answer: 00123456\n", "Answer:\n00123456"):
            self.assertTrue(score_passkey(response, "00123456")["correct"])
        for response in ("00123457", "The answer is 00123456", "Answer: 00123456.", "00123456 00123456", "１２３４５６７８", "00123456\nExplanation"):
            self.assertFalse(score_passkey(response, "00123456")["correct"])

    def test_invalid_method_revision_hardware_and_sampling_are_rejected(self):
        for section, key, value in ((None, "method", "fullkv"), (None, "method", {"name": "streamingllm", "params": {}}), (None, "method", {"name": "fullkv", "params": {"bits": 4}}), ("hardware", "gpu_name_contains", "T4"), ("dataset", "input_tokens", 2048), ("dataset", "input_tokens", [2048, 2048]), ("model", "revision", "main"), ("generation", "do_sample", True), ("dataset", "positions", [False]), ("measurement", "performance_claims", True)):
            config = copy.deepcopy(self.config)
            (config if section is None else config[section])[key] = value
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_planned_methods_validate_parameters_but_are_not_runnable_yet(self):
        planned = (
            {"name": "h2o", "params": {"budget_fraction": 0.25, "recent_fraction": 0.5}},
            {"name": "kivi", "params": {"bits": 4, "group_size": 32, "residual_length": 128}},
        )
        for method in planned:
            with self.assertRaisesRegex(ValueError, "not implemented"):
                validate_method(method)
        for method in ({"name": "h2o", "params": {"budget_fraction": 1.5, "recent_fraction": 0.5}}, {"name": "kivi", "params": {"bits": 3, "group_size": 32, "residual_length": 128}}):
            with self.assertRaisesRegex(ValueError, "must be"):
                validate_method(method)

    def test_multiple_lengths_get_unique_codes_ids_and_exact_budgets(self):
        config = copy.deepcopy(self.config)
        config["dataset"].update({"split": "final", "num_examples": 3, "input_tokens": [1024, 2048]})
        validate_config(config)
        generated = list(cases(config))
        self.assertEqual(len(generated), 6)
        self.assertEqual(len({case["prompt_id"] for case in generated}), 6)
        self.assertEqual(len({case["seed"] for case in generated}), 6)
        self.assertTrue(all(case["prompt_id"].startswith("final-") for case in generated))
        for case in generated:
            prompt = tokenized_case(case, config, CharacterTokenizer())
            self.assertEqual(len(prompt["input_ids"]), case["target_input_tokens"])

    def test_prompts_are_deterministic_and_match_total_token_budget(self):
        first = list(cases(self.config))
        self.assertEqual(first, list(cases(self.config)))
        self.assertEqual(len({case["prompt_id"] for case in first}), 10)
        for case in first:
            prompt = tokenized_case(case, self.config, CharacterTokenizer())
            self.assertEqual(len(prompt["input_ids"]), 2048)
            self.assertIn(case["expected_code"], prompt["prompt"])
            self.assertLessEqual(abs(prompt["actual_position"] - case["requested_position"]), 1 / 2048)

    def test_preview_has_no_measured_results_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            output = Path(directory) / "preview"
            args = ["preview", "--config", str(ROOT / "configs/fullkv-pilot.json"), "--output", str(output)]
            self.assertEqual(main(args), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["mode"], "offline_preview")
            self.assertIsNone(summary["accuracy"])
            self.assertIsNone(summary["gpu_measurements"])
            self.assertFalse(summary["tokenizer_verified"])
            self.assertEqual(len((output / "prompts.jsonl").read_text().splitlines()), 10)
            self.assertEqual(main(args), 1)


if __name__ == "__main__":
    unittest.main()
