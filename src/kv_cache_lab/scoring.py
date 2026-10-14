"""Strict exact-match scoring without accepting explanatory answers."""
import re

ANSWER = re.compile(r"\s*(?:Answer:\s*)?([0-9]{8})\s*\Z")


def score_passkey(response, expected):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9]{8}", expected):
        raise ValueError("Expected passkey must be an eight-digit string.")
    match = ANSWER.fullmatch(response)
    extracted = match.group(1) if match else None
    return {"extracted_code": extracted, "correct": extracted == expected}
