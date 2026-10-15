import csv
import tempfile
import unittest
from pathlib import Path

from kv_cache_lab.artifacts import append_metrics
from kv_cache_lab.scoring import SCORING_RULES, score_passkey


class ScoringTests(unittest.TestCase):
    def test_csv_keeps_strict_and_passkey_results_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            append_metrics(path, {"prompt_id": "p0", "status": "completed", **score_passkey("00123456.", "00123456")})
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["correct"], "False")
            self.assertEqual(rows[0]["passkey_correct"], "True")

    def test_period_changes_only_the_separate_passkey_score(self):
        for response in ("00123456.", "Answer: 00123456.", " \nAnswer:\n00123456.\n"):
            with self.subTest(response=response):
                score = score_passkey(response, "00123456")
                self.assertFalse(score["correct"])
                self.assertIsNone(score["extracted_code"])
                self.assertTrue(score["passkey_correct"])
                self.assertEqual(score["passkey_extracted_code"], "00123456")

    def test_unpunctuated_responses_pass_both_rules(self):
        for response in ("00123456", "Answer: 00123456", " 00123456\n"):
            score = score_passkey(response, "00123456")
            self.assertTrue(score["correct"])
            self.assertTrue(score["passkey_correct"])

    def test_neither_rule_accepts_explanations_multiple_codes_or_other_punctuation(self):
        invalid = [
            "00123456..", "00123456 .", "00123456!", "00123456,", "00123456;",
            "The answer is 00123456.", "Answer: 00123456. Explanation follows.",
            "00123456 00123456", "00123456\n87654321", "001234567", "0123456",
            "answer: 00123456", "００１２３４５６", "Answer: 00123456.\nMore text", "",
        ]
        for response in invalid:
            with self.subTest(response=response):
                score = score_passkey(response, "00123456")
                self.assertFalse(score["correct"])
                self.assertFalse(score["passkey_correct"])

    def test_well_formatted_wrong_code_fails_both_rules(self):
        score = score_passkey("Answer: 87654321.", "00123456")
        self.assertFalse(score["correct"])
        self.assertFalse(score["passkey_correct"])
        self.assertEqual(score["passkey_extracted_code"], "87654321")

    def test_invalid_expected_values_and_nontext_responses_are_rejected(self):
        for expected in (12345678, "1234567", "１２３４５６７８", None):
            with self.assertRaises(ValueError):
                score_passkey("12345678", expected)
        with self.assertRaises(ValueError):
            score_passkey(None, "12345678")
        self.assertEqual(SCORING_RULES, {"strict": "strict-v1", "passkey": "single-final-period-v1"})
