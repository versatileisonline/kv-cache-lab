"""Seeded passkeys and exactly sized, tokenizer-verified GPU prompts."""
import random

PLACEHOLDER = "<<PASSAGE_BODY>>"
SYSTEM = "Retrieve the marked passkey from the passage. Reply only with Answer: followed by the eight-digit code."
QUESTION = "\n\nWhat is the marked passkey? Reply only with Answer: <code>."
FILLER = (
    "The archive describes a quiet garden and the changing seasons. ",
    "A researcher recorded observations about clouds above the valley. ",
    "The shelves hold essays about rivers, forests, and distant villages. ",
    "Students compared ordinary notes about birds and the afternoon weather. ",
    "A small library collected stories about bridges and mountain paths. ",
)


def cases(config):
    """num_examples per input length; codes and filler seeds never repeat across lengths."""
    rng = random.Random(config["seed"])
    dataset = config["dataset"]
    positions = dataset["positions"]
    count = 0
    for length in dataset["input_tokens"]:
        for index in range(dataset["num_examples"]):
            yield {
                "prompt_id": f"{dataset['split']}-{config['seed']}-{length}-{index:04d}",
                "split": dataset["split"],
                "seed": config["seed"] + count,
                "target_input_tokens": length,
                "expected_code": f"{rng.randrange(100_000_000):08d}",
                "requested_position": positions[index % len(positions)],
            }
            count += 1


def filler_text(seed):
    rng = random.Random(seed)
    return "".join(rng.choice(FILLER) for _ in range(32))


def preview_case(case):
    text = filler_text(case["seed"])
    words = text.split()
    split = round(len(words) * case["requested_position"])
    passage = " ".join(words[:split]) + marker(case) + " ".join(words[split:])
    return {**case, "mode": "offline_preview", "tokenizer_verified": False, "passage": passage, "question": QUESTION.strip()}


def marker(case):
    return f"\nThe marked passkey is: {case['expected_code']}. Remember this code.\n"


def tokenized_case(case, config, tokenizer):
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PLACEHOLDER + QUESTION}],
        tokenize=False, add_generation_prompt=True,
        date_string=config["dataset"]["template_date"],
    )
    if rendered.count(PLACEHOLDER) != 1:
        raise ValueError("The chat template must preserve exactly one passage placeholder.")
    prefix, suffix = rendered.split(PLACEHOLDER)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    marker_ids = tokenizer.encode(marker(case), add_special_tokens=False)
    target = case["target_input_tokens"]
    available = target - len(prefix_ids) - len(suffix_ids) - len(marker_ids)
    if available < 0:
        raise ValueError("The template, marker, and question exceed the input budget.")
    before = round(target * case["requested_position"]) - len(prefix_ids)
    if not 0 <= before <= available:
        raise ValueError("Requested marker position does not fit; adjust length/positions.")
    filler_ids = tokenizer.encode(filler_text(case["seed"]), add_special_tokens=False)
    if not filler_ids:
        raise ValueError("Tokenizer produced empty filler.")
    filler_ids = (filler_ids * (available // len(filler_ids) + 1))[:available]
    input_ids = prefix_ids + filler_ids[:before] + marker_ids + filler_ids[before:] + suffix_ids
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    if len(input_ids) != target or case["expected_code"] not in decoded:
        raise ValueError("Prompt length/passkey verification failed.")
    return {
        **case, "tokenizer_verified": True, "input_ids": input_ids,
        "input_tokens": len(input_ids), "prompt": decoded,
        "marker_token_index": len(prefix_ids) + before,
        "actual_position": (len(prefix_ids) + before) / len(input_ids),
    }
