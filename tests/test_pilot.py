import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from kv_cache_lab.__main__ import main
from kv_cache_lab.config import load_config, validate_config
from kv_cache_lab.methods import h2o_budget, validate_method
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

    def test_method_parameters_are_validated(self):
        for method in ({"name": "h2o", "params": {"budget_fraction": 1.5, "recent_fraction": 0.5}}, {"name": "h2o", "params": {"budget_fraction": 0.25, "recent_fraction": 1}}, {"name": "kivi", "params": {"bits": 3, "group_size": 32, "residual_length": 128}}):
            with self.assertRaisesRegex(ValueError, "must be"):
                validate_method(method)
        with self.assertRaisesRegex(ValueError, "too small"):
            h2o_budget({"budget_fraction": 0.0005, "recent_fraction": 0.5}, 2048)
        self.assertEqual(h2o_budget({"budget_fraction": 0.125, "recent_fraction": 0.5}, 8192), (1024, 512))

    def test_pilot_configs_share_prompts_across_methods(self):
        configs = {path.stem: load_config(path) for path in sorted((ROOT / "configs/pilot").glob("*.json"))}
        self.assertEqual(set(configs), {"fullkv", "h2o-b25", "h2o-b12", "kivi-4bit", "kivi-2bit"})
        prompts = {name: list(cases(config)) for name, config in configs.items()}
        self.assertTrue(all(cases_ == prompts["fullkv"] for cases_ in prompts.values()))
        # The 2K pilot prompts are unchanged from the original FullKV pilot.
        self.assertEqual(prompts["fullkv"][:10], list(cases(self.config)))

    def test_final_configs_match_pilot_methods_with_fresh_prompts(self):
        pilot = {path.name: load_config(path) for path in (ROOT / "configs/pilot").glob("*.json")}
        final = {path.name: load_config(path) for path in (ROOT / "configs/final").glob("*.json")}
        self.assertEqual(set(final), set(pilot))
        final_cases = list(cases(final["fullkv.json"]))
        pilot_cases = list(cases(pilot["fullkv.json"]))
        self.assertEqual(len(final_cases), 180)
        self.assertEqual(len({(case["target_input_tokens"], case["requested_position"]) for case in final_cases}), 9)
        self.assertFalse({case["seed"] for case in final_cases} & {case["seed"] for case in pilot_cases})
        self.assertFalse({case["expected_code"] for case in final_cases} & {case["expected_code"] for case in pilot_cases})
        for name, config in final.items():
            self.assertEqual(config["dataset"]["split"], "final")
            self.assertEqual(config["method"], pilot[name]["method"])
            self.assertEqual(list(cases(config)), final_cases)

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
