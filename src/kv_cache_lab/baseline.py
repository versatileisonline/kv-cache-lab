"""One KV-cache method on one task (passkey quality, Qasper, or speed). CUDA failures are recorded, never replaced by CPU runs."""
import gc
import importlib.metadata
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import append_jsonl, append_metrics, file_hash, write_json
from .inference import greedy_decode
from .methods import METHODS, cache_factory
from .prompts import cases, tokenized_case
from .config import length_groups
from .scoring import SCORING_RULES, score_passkey

ROOT = Path(__file__).resolve().parents[2]


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_metadata(config):
    return {
        "schema_version": 3, "config": config,
        "scoring_rules": SCORING_RULES,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "hostname": platform.node(), "system": platform.platform(),
        "git_revision": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "uv_lock_sha256": file_hash(ROOT / "uv.lock"),
        "upstream_manifest_sha256": file_hash(ROOT / "configs/upstream.json"),
        "upstream_revision": command_output(["git", "-C", str(ROOT / "third_party/KVCache-Factory"), "rev-parse", "HEAD"]),
        "upstream_used_for_inference": METHODS[config["method"]["name"]]["upstream"],
        "packages": sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()}),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_inventory_before": command_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"]),
        "gpu_processes_before": command_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"]),
        "measurement_note": "Speed run: fixed output length, EOS ignored, median over repeated prompts." if config["dataset"]["split"] == "speed" else "Quality run: timings are diagnostic only.",
    }


def eos_token_ids(model, tokenizer):
    values = []
    for source in (model.generation_config, model.config, tokenizer):
        value = getattr(source, "eos_token_id", None)
        if isinstance(value, int):
            values.append(value)
        elif value is not None:
            values.extend(value)
    return sorted(set(values))


# Score totals and the summary mean computed from each.
MEANS = {"correct": "accuracy", "passkey_correct": "passkey_accuracy", "leading_digits": "mean_leading_digits", "f1": "mean_f1"}


def build_prompts(config, tokenizer):
    if config["dataset"]["task"] == "qasper":
        from . import qasper
        rows = qasper.load_rows(qasper.fetch_data())
        return [qasper.tokenized_case(case, config, tokenizer) for case in qasper.select_cases(rows, config)]
    return [{**tokenized_case(case, config, tokenizer), "length_group": case["target_input_tokens"]} for case in cases(config)]


def score_response(config, prompt, response):
    """Returns the scores and the reference fields saved next to the response."""
    if config["dataset"]["task"] == "qasper":
        from .qasper import score_qasper
        return score_qasper(response, prompt["answers"]), {"answers": prompt["answers"], "source_id": prompt["source_id"]}
    return score_passkey(response, prompt["expected_code"]), {"expected_code": prompt["expected_code"]}


def tally(target, score):
    target["completed"] += 1
    for key in MEANS:
        if key in score:
            target[key] = target.get(key, 0) + score[key]


def run_pilot(config, output):
    metadata = environment_metadata(config)
    write_json(output / "metadata.json", metadata)
    task = config["dataset"]["task"]
    scores = ("f1",) if task == "qasper" else ("correct", "passkey_correct", "leading_digits")
    summary = {
        "schema_version": 3, "mode": "gpu_speed" if config["dataset"]["split"] == "speed" else "gpu_quality",
        "task": task, "status": "running", "method": config["method"], "by_input_tokens": {},
        "attempted": 0, "completed": 0, "failures": 0,
        "performance_claims": config["measurement"]["performance_claims"],
        **{key: 0 for key in scores}, **{MEANS[key]: None for key in scores},
        "scoring_rules": {"qa_f1": "longbench-qa-f1-v1"} if task == "qasper" else SCORING_RULES,
    }
    write_json(output / "summary.json", summary)
    phase = "environment"
    current_prompt = None
    try:
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("GPU experiments require the Linux x86_64 ARC allocation. Use preview for local checks.")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Obtain a GPU allocation; this runner does not fall back to CPU.")
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError("Use the locked Python 3.11 environment.")
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        properties = torch.cuda.get_device_properties(0)
        required_gpu = config["hardware"]["gpu_name_contains"]
        if required_gpu not in properties.name:
            raise RuntimeError(f"Official runs require an {required_gpu}; cuda:0 is {properties.name}. Request the A100 partition.")
        metadata.update({
            "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "selected_device": "cuda:0", "selected_gpu_name": properties.name,
            "selected_gpu_uuid": str(getattr(properties, "uuid", "Unavailable; see inventory")),
            "selected_gpu_total_bytes": properties.total_memory,
            "tf32_enabled": False, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        })
        write_json(output / "metadata.json", metadata)
        phase = "model_loading"
        model_settings = config["model"]
        tokenizer = AutoTokenizer.from_pretrained(model_settings["id"], revision=model_settings["revision"], trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_settings["id"], revision=model_settings["revision"],
            torch_dtype=torch.float16, attn_implementation="sdpa",
            low_cpu_mem_usage=True, trust_remote_code=False,
        ).to("cuda:0").eval()
        if any(parameter.device != torch.device("cuda:0") or parameter.dtype != torch.float16 for parameter in model.parameters()):
            raise RuntimeError("All model parameters must remain FP16 on cuda:0.")
        if max(length_groups(config)) + config["generation"]["max_new_tokens"] > model.config.max_position_embeddings:
            raise ValueError("Input plus output exceeds the model context limit.")
        # Speed runs ignore EOS so every request generates exactly max_new_tokens.
        stop_ids = [] if config["generation"].get("ignore_eos") else eos_token_ids(model, tokenizer)
        new_cache = cache_factory(config["method"], model)
        metadata.update({"resolved_model_revision": getattr(model.config, "_commit_hash", None), "eos_token_ids": stop_ids})
        write_json(output / "metadata.json", metadata)
        phase = "prompt_construction"
        prompts = build_prompts(config, tokenizer)
        for prompt in prompts:
            append_jsonl(output / "prompts.jsonl", prompt)
        groups = {}
        for prompt in prompts:
            groups.setdefault(prompt["length_group"], []).append(prompt)

        def record_failure(prompt, failure_phase, error):
            # Running out of GPU memory at one length is a result (e.g. H2O scoring at 32K); later lengths still run.
            failure = {"prompt_id": prompt["prompt_id"], "status": "failed", "failure_phase": failure_phase, "error_type": type(error).__name__, "input_tokens": prompt["input_tokens"], "length_group": prompt["length_group"]}
            append_jsonl(output / "predictions.jsonl", failure)
            append_metrics(output / "metrics.csv", failure)
            summary["failures"] += 1
            group = summary["by_input_tokens"].setdefault(str(prompt["length_group"]), {"completed": 0, "failures": 0})
            group["failures"] += 1
            write_json(output / "summary.json", summary)
            print(f"{prompt['prompt_id']}: out of GPU memory during {failure_phase}", flush=True)

        for group_prompts in groups.values():
            phase = "warmup"
            current_prompt = None
            gc.collect()
            torch.cuda.empty_cache()
            try:
                warmup = max(group_prompts, key=lambda prompt: prompt["input_tokens"])
                warmup_ids = torch.tensor([warmup["input_ids"]], dtype=torch.long, device="cuda:0")
                for _ in range(config["measurement"]["warmup_runs"]):
                    greedy_decode(model, warmup_ids, config["generation"]["max_new_tokens"], stop_ids, cache=new_cache(warmup_ids.shape[-1]))
                del warmup_ids
            except torch.cuda.OutOfMemoryError as error:
                for prompt in group_prompts:
                    summary["attempted"] += 1
                    record_failure(prompt, "warmup", error)
                continue
            for prompt in group_prompts:
                current_prompt = prompt["prompt_id"]
                summary["attempted"] += 1
                phase = "input_transfer"
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device="cuda:0")
                    phase = "inference"
                    result = greedy_decode(model, input_ids, config["generation"]["max_new_tokens"], stop_ids, cache=new_cache(input_ids.shape[-1]), measure_cuda=True)
                    del input_ids
                except torch.cuda.OutOfMemoryError as error:
                    record_failure(prompt, getattr(error, "kv_phase", phase), error)
                    continue
                phase = "scoring"
                response = tokenizer.decode(result["output_ids"], skip_special_tokens=True)
                score, reference = score_response(config, prompt, response)
                diagnostics = result["diagnostic_metrics"]
                row = {
                    "prompt_id": current_prompt, "status": "completed", **score,
                    "input_tokens": prompt["input_tokens"], "length_group": prompt["length_group"],
                    "output_tokens": result["output_tokens"],
                    "actual_position": prompt.get("actual_position"), "cache_reused": result["cache_reused"],
                    "prefill_cache_bytes": result["prefill_cache"]["storage_bytes"],
                    "final_cache_bytes": result["final_cache"]["storage_bytes"],
                    "ttft_ms": diagnostics["ttft_ms"], "decode_ms": diagnostics["decode_ms"],
                    "decode_tokens_per_second": diagnostics["decode_tokens_per_second"],
                    **{f"{stage}_peak_{kind}": count for stage in ("prefill", "decode") for kind, count in diagnostics[f"{stage}_peak"].items()},
                }
                append_jsonl(output / "predictions.jsonl", {**row, "response": response, **reference, "generation": result})
                append_metrics(output / "metrics.csv", row)
                tally(summary, score)
                for key in scores:
                    summary[MEANS[key]] = summary[key] / summary["completed"]
                tally(summary["by_input_tokens"].setdefault(str(prompt["length_group"]), {"completed": 0, "failures": 0}), score)
                write_json(output / "summary.json", summary)
                print(f"{current_prompt}: " + ", ".join(f"{key}={score[key]}" for key in scores) + f", input_tokens={prompt['input_tokens']}", flush=True)
        summary["status"] = "completed"
        return summary
    except Exception as error:
        phase = getattr(error, "kv_phase", phase)
        # Terminal only: artifacts omit raw exception text in case it contains credentials.
        traceback.print_exc(file=sys.stderr)
        failure = {"prompt_id": current_prompt, "status": "failed", "failure_phase": phase, "error_type": type(error).__name__}
        append_jsonl(output / "predictions.jsonl", failure)
        append_metrics(output / "metrics.csv", failure)
        summary.update({"status": "failed", "failures": summary["failures"] + 1, "failure_phase": phase, "error_type": type(error).__name__})
        if phase == "environment" and isinstance(error, RuntimeError):
            summary["error_message"] = str(error)
        else:
            summary["error_message"] = "Check model access, the pinned environment, and allocated GPU memory. Raw exception text is omitted to keep credentials out of artifacts."
        return summary
    finally:
        metadata["gpu_processes_after"] = command_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"])
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "metadata.json", metadata)
        write_json(output / "summary.json", summary)
