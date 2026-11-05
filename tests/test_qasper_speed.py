import copy
import json
import unittest
from pathlib import Path

from kv_cache_lab.config import length_groups, load_config, validate_config
from kv_cache_lab.prompts import cases
from kv_cache_lab.qasper import select_cases, score_qasper, token_f1, tokenized_case

ROOT = Path(__file__).resolve().parents[1]


class WordTokenizer:
    """Deterministic test tokenizer: one token per character."""

    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "<assistant>"

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class QasperTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/qasper/fullkv.json")
        self.rows = [{"_id": f"id{index:03d}", "input": f"Question {index}?", "context": f"Article {index}. " * 50, "answers": [f"answer {index}"]} for index in range(120)]

    def test_f1_matches_longbench_normalization(self):
        self.assertEqual(token_f1("The BERT model.", "bert model"), 1.0)
        self.assertEqual(token_f1("unanswerable", "Yes"), 0.0)
        self.assertAlmostEqual(token_f1("a large pretrained model", "pretrained model"), 0.8)
        self.assertEqual(score_qasper("Yes.", ["no", "yes"])["f1"], 1.0)
        with self.assertRaises(ValueError):
            score_qasper("yes", [])

    def test_selection_is_seeded_and_independent_of_file_order(self):
        first = select_cases(self.rows, self.config)
        self.assertEqual(len(first), 50)
        self.assertEqual(len({case["source_id"] for case in first}), 50)
        self.assertEqual(first, select_cases(list(reversed(self.rows)), self.config))
        self.assertTrue(all(case["length_group"] == 8192 for case in first))

    def test_long_articles_are_truncated_from_the_middle_to_the_cap(self):
        config = copy.deepcopy(self.config)
        config["dataset"]["max_input_tokens"] = 1024
        case = {**select_cases(self.rows, config)[0], "context": "BEGIN " + "x" * 5000 + " END"}
        prompt = tokenized_case(case, config, WordTokenizer())
        self.assertEqual(prompt["input_tokens"], 1024)
        self.assertTrue(prompt["article_truncated"])
        self.assertIn("BEGIN", prompt["prompt"])
        self.assertIn(" END", prompt["prompt"])
        self.assertTrue(prompt["prompt"].rstrip("<assistant>").endswith("Answer:"))
        short = tokenized_case({**case, "context": "Short article."}, config, WordTokenizer())
        self.assertFalse(short["article_truncated"])
        self.assertIn("Short article.", short["prompt"])

    def test_qasper_config_rules(self):
        self.assertEqual(length_groups(self.config), [8192])
        for key, value in (("split", "pilot"), ("num_examples", 201), ("max_input_tokens", 512), ("input_tokens", [2048])):
            config = copy.deepcopy(self.config)
            config["dataset"][key] = value
            with self.assertRaises(ValueError):
                validate_config(config)


class RunSetTests(unittest.TestCase):
    def sets(self, folder):
        return {path.stem: load_config(path) for path in sorted((ROOT / "configs" / folder).glob("*.json"))}

    def test_speed_configs_force_fixed_output_up_to_32k(self):
        speed = self.sets("speed")
        self.assertEqual(set(speed), {"fullkv", "h2o-b25", "h2o-b12", "kivi-4bit", "kivi-2bit"})
        prompts = list(cases(speed["fullkv"]))
        for config in speed.values():
            self.assertEqual(config["dataset"]["input_tokens"], [2048, 4096, 8192, 16384, 32768])
            self.assertTrue(config["generation"]["ignore_eos"])
            self.assertEqual(config["generation"]["max_new_tokens"], 256)
            self.assertTrue(config["measurement"]["performance_claims"])
            self.assertEqual(list(cases(config)), prompts)
        quality = copy.deepcopy(speed["fullkv"])
        quality["generation"]["ignore_eos"] = False
        with self.assertRaises(ValueError):
            validate_config(quality)

    def test_qasper_and_final_sets_cover_the_same_methods_and_are_frozen(self):
        final, qasper = self.sets("final"), self.sets("qasper")
        self.assertEqual(set(final), set(qasper))
        self.assertEqual(len(final), 8)
        for name in final:
            self.assertEqual(final[name]["method"], qasper[name]["method"])
        self.assertEqual(len({json.dumps(config["dataset"], sort_keys=True) for config in qasper.values()}), 1)
        for folder in ("final", "qasper"):
            self.assertTrue((ROOT / "configs" / folder / "FROZEN").is_file())


if __name__ == "__main__":
    unittest.main()
