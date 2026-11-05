"""Versioned passkey scoring: strict, punctuation-tolerant, and leading-digit partial credit."""
import re

SCORING_RULES = {"strict": "strict-v1", "passkey": "single-final-period-v1", "leading_digits": "leading-digits-v1"}
ANSWER = re.compile(r"\s*(?:Answer:\s*)?([0-9]{8})\s*\Z")
PASSKEY_ANSWER = re.compile(r"\s*(?:Answer:\s*)?([0-9]{8})\.?\s*\Z")
DIGITS = re.compile(r"[0-9]+")


def leading_digits(response, expected):
    """leading-digits-v1: how many of the expected code's first digits the response's first ASCII digit run gets right (0-8)."""
    match = DIGITS.search(response)
    count = 0
    for given, wanted in zip(match.group(0) if match else "", expected):
        if given != wanted:
            break
        count += 1
    return count


def score_passkey(response, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9]{8}", expected):
        raise ValueError("Expected passkey must be an eight-digit string.")
    if not isinstance(response, str):
        raise ValueError("Response must be a string.")
    match = ANSWER.fullmatch(response)
    extracted = match.group(1) if match else None
    passkey_match = PASSKEY_ANSWER.fullmatch(response)
    passkey_extracted = passkey_match.group(1) if passkey_match else None
    return {
        "extracted_code": extracted, "correct": extracted == expected,
        "passkey_extracted_code": passkey_extracted,
        "passkey_correct": passkey_extracted == expected,
        "leading_digits": leading_digits(response, expected),
    }
