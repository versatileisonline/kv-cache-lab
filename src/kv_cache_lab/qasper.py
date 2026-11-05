"""LongBench Qasper: pinned data, LongBench prompt and middle truncation, token-level F1."""
import hashlib
import json
import random
import re
import string
import zipfile
from collections import Counter

# Pinned LongBench release (THUDM/LongBench now redirects to zai-org/LongBench).
SOURCE = {
    "repo_id": "zai-org/LongBench", "repo_type": "dataset",
    "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
    "filename": "data.zip", "member": "data/qasper.jsonl",
    "sha256": "cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64",
}
SCORING_RULE = "longbench-qa-f1-v1"
PLACEHOLDER = "<<ARTICLE_BODY>>"
INSTRUCTION = (
    'Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. '
    'If the question cannot be answered based on the information in the article, write "unanswerable". '
    'If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.'
)
# LongBench's qasper prompt, with the article swapped for a placeholder so it is tokenized separately.
TEMPLATE = (
    'You are given a scientific article and a question. ' + INSTRUCTION.replace("based on the above article ", "")
    + "\n\nArticle: " + PLACEHOLDER + "\n\n " + INSTRUCTION + "\n\nQuestion: {question}\n\nAnswer:"
)


def fetch_data():
    """Download (or reuse from the Hugging Face cache) the pinned archive and verify its hash."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SOURCE["repo_id"], SOURCE["filename"], repo_type=SOURCE["repo_type"], revision=SOURCE["revision"])
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != SOURCE["sha256"]:
        raise ValueError("LongBench data.zip does not match the pinned SHA-256.")
    return path


def load_rows(archive):
    with zipfile.ZipFile(archive) as bundle:
        text = bundle.read(SOURCE["member"]).decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def select_cases(rows, config):
    """A seeded sample of num_examples questions, in a stable order."""
    dataset = config["dataset"]
    ordered = sorted(rows, key=lambda row: row["_id"])
    if dataset["num_examples"] > len(ordered):
        raise ValueError(f"Qasper has only {len(ordered)} questions.")
    chosen = random.Random(config["seed"]).sample(ordered, dataset["num_examples"])
    return [{
        "prompt_id": f"qasper-{config['seed']}-{index:04d}", "split": dataset["split"], "seed": config["seed"],
        "source_id": row["_id"], "question": row["input"], "context": row["context"], "answers": row["answers"],
        "length_group": dataset["max_input_tokens"],
    } for index, row in enumerate(chosen)]


def tokenized_case(case, config, tokenizer):
    """Truncate the article from the middle (as LongBench does) so the whole input fits max_input_tokens."""
    content = TEMPLATE.format(question=case["question"])
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
        date_string=config["dataset"]["template_date"],
    )
    if rendered.count(PLACEHOLDER) != 1:
        raise ValueError("The chat template must preserve exactly one article placeholder.")
    prefix, suffix = rendered.split(PLACEHOLDER)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    article_ids = tokenizer.encode(case["context"], add_special_tokens=False)
    available = config["dataset"]["max_input_tokens"] - len(prefix_ids) - len(suffix_ids)
    if available <= 0:
        raise ValueError("The prompt template exceeds max_input_tokens.")
    truncated = len(article_ids) > available
    if truncated:
        half = available // 2
        article_ids = article_ids[:half] + article_ids[len(article_ids) - (available - half):]
    input_ids = prefix_ids + article_ids + suffix_ids
    record = {key: value for key, value in case.items() if key != "context"}
    return {
        **record, "tokenizer_verified": True, "input_ids": input_ids, "input_tokens": len(input_ids),
        "article_truncated": truncated, "prompt": tokenizer.decode(input_ids, skip_special_tokens=False),
    }


def normalize_answer(text):
    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction, reference):
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall)


def score_qasper(response, answers):
    if not isinstance(response, str) or not answers or not all(isinstance(answer, str) for answer in answers):
        raise ValueError("Qasper scoring needs a text response and a list of reference answers.")
    return {"f1": max(token_f1(response, answer) for answer in answers)}
