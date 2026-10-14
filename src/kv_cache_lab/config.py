"""Validate the deliberately small initial pilot configuration."""
import json
import re
from pathlib import Path


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def integer(value, minimum=1):
        return type(value) is int and value >= minimum

    require(isinstance(config, dict), "Configuration must be a JSON object.")
    require(config.get("schema_version") == 1, "Unsupported schema_version.")
    require(isinstance(config.get("experiment_id"), str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]*", config["experiment_id"]), "experiment_id must be a safe lowercase identifier.")
    require(config.get("method") == "FullKV", "Only FullKV is implemented in the initial setup.")
    require(config.get("platform") == "arc", "The intended experiment platform is ARC.")
    require(integer(config.get("seed"), 0), "seed must be a nonnegative integer.")
    model = config.get("model", {})
    dataset = config.get("dataset", {})
    generation = config.get("generation", {})
    measurement = config.get("measurement", {})
    require(all(isinstance(section, dict) for section in (model, dataset, generation, measurement)), "Configuration sections must be objects.")
    require(isinstance(model.get("id"), str) and bool(model["id"].strip()), "model.id is required.")
    require(isinstance(model.get("revision"), str) and re.fullmatch(r"[0-9a-f]{40}", model["revision"]), "Pin model.revision to a full commit SHA.")
    require(model.get("dtype") == "float16", "The initial baseline uses FP16 weights/cache.")
    require(model.get("attention_backend") == "sdpa", "The initial baseline uses SDPA.")
    require(dataset.get("split") == "pilot", "This runner implements the pilot split only.")
    require(integer(dataset.get("num_examples")) and dataset["num_examples"] <= 1000, "num_examples must be between 1 and 1000.")
    require(integer(dataset.get("input_tokens"), 256), "input_tokens must be an integer >= 256.")
    positions = dataset.get("positions")
    require(isinstance(positions, list) and bool(positions), "dataset.positions must be a nonempty list.")
    require(all(type(p) in (int, float) and 0 < p < 1 for p in positions), "Positions must be numeric fractions strictly between 0 and 1.")
    require(isinstance(dataset.get("template_date"), str) and bool(dataset["template_date"]), "Fix template_date to keep the chat template reproducible.")
    require(integer(generation.get("max_new_tokens")) and generation["max_new_tokens"] <= 256, "max_new_tokens must be between 1 and 256.")
    require(generation.get("batch_size") == 1 and type(generation["batch_size"]) is int, "Only batch size 1 is implemented.")
    require(generation.get("do_sample") is False, "Use greedy decoding: do_sample=false.")
    require(integer(measurement.get("warmup_runs"), 0), "warmup_runs must be a nonnegative integer.")
    require(measurement.get("performance_claims") is False, "Pilot timings cannot be marked as performance claims.")
