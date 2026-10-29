"""Passkey quality pilot for one KV-cache method. CUDA failures are recorded, never replaced by CPU runs."""
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
from .scoring import SCORING_RULES, score_passkey

ROOT = Path(__file__).resolve().parents[2]


def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_metadata(config):
    return {
        "schema_version": 2, "mode": "gpu_pilot", "config": config,
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
        "measurement_note": "Single-pass pilot diagnostics; not the fixed-length performance campaign.",
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


def run_pilot(config, output):
    metadata = environment_metadata(config)
    write_json(output / "metadata.json", metadata)
    summary = {"schema_version": 2, "mode": "gpu_pilot", "status": "running", "method": config["method"], "by_input_tokens": {}, "attempted": 0, "completed": 0, "correct": 0, "failures": 0, "performance_claims": False, "passkey_correct": 0, "passkey_accuracy": None, "scoring_rules": SCORING_RULES}
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
        if max(config["dataset"]["input_tokens"]) + config["generation"]["max_new_tokens"] > model.config.max_position_embeddings:
            raise ValueError("Input plus output exceeds the model context limit.")
        stop_ids = eos_token_ids(model, tokenizer)
        new_cache = cache_factory(config["method"], model)
        metadata.update({"resolved_model_revision": getattr(model.config, "_commit_hash", None), "eos_token_ids": stop_ids})
        write_json(output / "metadata.json", metadata)
        phase = "prompt_construction"
        prompts = [tokenized_case(case, config, tokenizer) for case in cases(config)]
        for prompt in prompts:
            append_jsonl(output / "prompts.jsonl", prompt)
        phase = "warmup"
        warmup_prompts = {prompt["target_input_tokens"]: prompt for prompt in reversed(prompts)}
        for prompt in warmup_prompts.values():
            warmup_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device="cuda:0")
            for _ in range(config["measurement"]["warmup_runs"]):
                greedy_decode(model, warmup_ids, config["generation"]["max_new_tokens"], stop_ids, cache=new_cache(warmup_ids.shape[-1]))
            del warmup_ids
        for prompt in prompts:
            current_prompt = prompt["prompt_id"]
            summary["attempted"] += 1
            phase = "input_transfer"
            gc.collect()
            torch.cuda.empty_cache()
            input_ids = torch.tensor([prompt["input_ids"]], dtype=torch.long, device="cuda:0")
            phase = "inference"
            result = greedy_decode(model, input_ids, config["generation"]["max_new_tokens"], stop_ids, cache=new_cache(input_ids.shape[-1]), measure_cuda=True)
            del input_ids
            phase = "scoring"
            response = tokenizer.decode(result["output_ids"], skip_special_tokens=True)
            score = score_passkey(response, prompt["expected_code"])
            diagnostics = result["diagnostic_metrics"]
            row = {
                "prompt_id": current_prompt, "status": "completed", **score,
                "input_tokens": prompt["input_tokens"], "output_tokens": result["output_tokens"],
                "actual_position": prompt["actual_position"], "cache_reused": result["cache_reused"],
                "prefill_cache_bytes": result["prefill_cache"]["storage_bytes"],
                "final_cache_bytes": result["final_cache"]["storage_bytes"],
                "ttft_ms": diagnostics["ttft_ms"], "decode_ms": diagnostics["decode_ms"],
                "decode_tokens_per_second": diagnostics["decode_tokens_per_second"],
                **{f"{stage}_peak_{kind}": count for stage in ("prefill", "decode") for kind, count in diagnostics[f"{stage}_peak"].items()},
            }
            append_jsonl(output / "predictions.jsonl", {**row, "response": response, "expected_code": prompt["expected_code"], "generation": result})
            append_metrics(output / "metrics.csv", row)
            summary["completed"] += 1
            summary["correct"] += int(score["correct"])
            summary["passkey_correct"] += int(score["passkey_correct"])
            summary["accuracy"] = summary["correct"] / summary["completed"]
            summary["passkey_accuracy"] = summary["passkey_correct"] / summary["completed"]
            group = summary["by_input_tokens"].setdefault(str(prompt["input_tokens"]), {"completed": 0, "correct": 0, "passkey_correct": 0})
            group["completed"] += 1
            group["correct"] += int(score["correct"])
            group["passkey_correct"] += int(score["passkey_correct"])
            write_json(output / "summary.json", summary)
            print(f"{current_prompt}: strict_correct={score['correct']}, passkey_correct={score['passkey_correct']}, input_tokens={prompt['input_tokens']}", flush=True)
        summary["status"] = "completed"
        return summary
    except Exception as error:
        phase = getattr(error, "kv_phase", phase)
        # Terminal only: artifacts omit raw exception text in case it contains credentials.
        traceback.print_exc(file=sys.stderr)
        failure = {"prompt_id": current_prompt, "status": "failed", "failure_phase": phase, "error_type": type(error).__name__}
        append_jsonl(output / "predictions.jsonl", failure)
        append_metrics(output / "metrics.csv", failure)
        summary.update({"status": "failed", "failures": 1, "failure_phase": phase, "error_type": type(error).__name__})
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
