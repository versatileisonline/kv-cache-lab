"""KV-cache method registry: parameter validation, model setup, and per-request caches.

H2O runs the pinned KVCache-Factory attention (third_party/KVCache-Factory, see
configs/upstream.json). KIVI uses the Transformers HQQ quantized cache.
FullKV uses unmodified Hugging Face attention.
"""
import sys
import types
from pathlib import Path

UPSTREAM_DIR = Path(__file__).resolve().parents[2] / "third_party" / "KVCache-Factory"
# KIVI quantizes keys per channel and values per token. In HQQ, axis 0 groups
# elements of one channel (strided across tokens) and axis 1 groups consecutive
# channels of one token. KVCache-Factory's kivi setting (key 1, value 0) is the reverse.
KIVI_AXIS_KEY = 0
KIVI_AXIS_VALUE = 1


def _number(value, low, high):
    return type(value) in (int, float) and low < value <= high


def _no_params(params):
    if params:
        raise ValueError("fullkv takes no parameters.")


def _h2o_params(params):
    if set(params) != {"budget_fraction", "recent_fraction"}:
        raise ValueError("h2o requires exactly budget_fraction and recent_fraction.")
    if not _number(params["budget_fraction"], 0, 1):
        raise ValueError("h2o budget_fraction must be in (0, 1].")
    if not _number(params["recent_fraction"], 0, 1) or params["recent_fraction"] == 1:
        raise ValueError("h2o recent_fraction must be in (0, 1).")


def _kivi_params(params):
    if set(params) != {"bits", "group_size", "residual_length"}:
        raise ValueError("kivi requires exactly bits, group_size, and residual_length.")
    if params["bits"] not in (2, 4) or type(params["bits"]) is not int:
        raise ValueError("kivi bits must be 2 or 4.")
    for key in ("group_size", "residual_length"):
        if type(params[key]) is not int or params[key] < 1:
            raise ValueError(f"kivi {key} must be a positive integer.")


def h2o_budget(params, input_tokens):
    """Retained prompt positions per layer and KV head: (total budget, recent window)."""
    budget = round(input_tokens * params["budget_fraction"])
    recent = round(budget * params["recent_fraction"])
    if not 0 < recent < budget:
        raise ValueError(f"h2o budget {budget} and recent window {recent} are too small for {input_tokens} tokens.")
    return budget, recent


def _dynamic_cache(model, params, input_tokens):
    from transformers import DynamicCache
    return DynamicCache()


def _install_h2o(model, params):
    if not (UPSTREAM_DIR / "pyramidkv").is_dir():
        raise RuntimeError("KVCache-Factory is missing; run scripts/fetch_upstream.py.")
    if str(UPSTREAM_DIR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_DIR))
    from pyramidkv.llama_model import llama_sdpa_attn_forward_H2O
    # Cache only the 8 KV heads (no copy per query head), so cache bytes compare fairly with FullKV.
    model.config.kv_cache_granularity = "kv_head"
    model.config.gqa_score_agg = "mean"
    for layer in model.model.layers:
        if type(layer.self_attn).__name__ != "LlamaSdpaAttention":
            raise RuntimeError("H2O requires Llama SDPA attention.")
        # Patch instances, not the class, so other models in the process keep stock attention.
        layer.self_attn.forward = types.MethodType(llama_sdpa_attn_forward_H2O, layer.self_attn)


def _h2o_cache(model, params, input_tokens):
    from transformers import DynamicCache
    budget, recent = h2o_budget(params, input_tokens)
    model.config.max_capacity_prompt = budget
    model.config.window_size = recent
    for layer in model.model.layers:
        # Upstream resets this inside generate(); our decode loop calls the model directly.
        layer.self_attn.kv_seq_len = 0
    return DynamicCache()


def _kivi_cache(model, params, input_tokens):
    from transformers import HQQQuantizedCache, QuantizedCacheConfig
    return HQQQuantizedCache(QuantizedCacheConfig(
        backend="HQQ", nbits=params["bits"], axis_key=KIVI_AXIS_KEY, axis_value=KIVI_AXIS_VALUE,
        q_group_size=params["group_size"], residual_length=params["residual_length"],
        compute_dtype=model.dtype, device=str(model.device),
    ))


# new_cache is None until a method is implemented; its parameters still validate.
METHODS = {
    "fullkv": {"validate": _no_params, "install": None, "new_cache": _dynamic_cache, "upstream": False},
    "h2o": {"validate": _h2o_params, "install": _install_h2o, "new_cache": _h2o_cache, "upstream": True},
    "kivi": {"validate": _kivi_params, "install": None, "new_cache": _kivi_cache, "upstream": False},
}


def validate_method(method):
    if not isinstance(method, dict) or set(method) != {"name", "params"}:
        raise ValueError("method must be an object with name and params.")
    entry = METHODS.get(method["name"])
    if entry is None:
        raise ValueError(f"Unknown method {method['name']!r}; choose from {sorted(METHODS)}.")
    if not isinstance(method["params"], dict):
        raise ValueError("method.params must be an object.")
    entry["validate"](method["params"])
    if entry["new_cache"] is None:
        raise ValueError(f"Method {method['name']!r} is specified but not implemented yet.")


def cache_factory(method, model):
    """Prepare the model once; return a function creating a fresh cache for one request of input_tokens."""
    entry = METHODS[method["name"]]
    if entry["install"] is not None:
        entry["install"](model, method["params"])
    return lambda input_tokens: entry["new_cache"](model, method["params"], input_tokens)
