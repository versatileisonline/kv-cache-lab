"""Cache accounting and greedy decoding with a per-method cache, with optional CUDA diagnostics."""
import time


def _tensors(item):
    if hasattr(item, "untyped_storage"):
        yield item
    elif isinstance(item, dict):
        for value in item.values():
            yield from _tensors(value)
    elif isinstance(item, (tuple, list)):
        for value in item:
            yield from _tensors(value)


def cache_stats(cache):
    """Unique physical storage bytes (quantization scales/zero-points included), shapes, and tokens per layer."""
    if hasattr(cache, "_quantized_key_cache"):
        layers = []
        for parts in zip(cache._quantized_key_cache, cache._quantized_value_cache, cache.key_cache, cache.value_cache):
            (quantized_key, key_meta), _, residual_key, _ = parts
            residual_tokens = residual_key.shape[-2] if residual_key.dim() == 4 else 0
            shape = {"quantized_key": list(quantized_key.shape), "quantized_dtype": str(quantized_key.dtype), "residual_key": list(residual_key.shape), "dtype": str(residual_key.dtype)}
            layers.append((parts, shape, key_meta["shape"][-2] + residual_tokens))
    else:
        pairs = zip(cache.key_cache, cache.value_cache) if hasattr(cache, "key_cache") else cache
        layers = [((key, value), {"key": list(key.shape), "value": list(value.shape), "dtype": str(key.dtype)}, key.shape[-2]) for key, value in pairs]
    seen = set()
    storage_bytes = 0
    for parts, _, _ in layers:
        for tensor in _tensors(parts):
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr())
            if identity not in seen:
                seen.add(identity)
                storage_bytes += storage.nbytes()
    if not layers:
        raise ValueError("Model did not return populated KV-cache tensors.")
    return {"storage_bytes": storage_bytes, "layer_shapes": [shape for _, shape, _ in layers], "tokens_per_layer": [tokens for _, _, tokens in layers]}


def greedy_decode(model, input_ids, max_new_tokens, eos_ids, *, cache=None, measure_cuda=False):
    """Pass a fresh method cache per request; the default is an uncompressed DynamicCache."""
    import torch
    from transformers import DynamicCache

    if measure_cuda and input_ids.device.type != "cuda":
        raise ValueError("CUDA measurement requires CUDA input tensors.")
    eos_ids = set(eos_ids)
    if cache is None:
        cache = DynamicCache()
    mask = torch.ones_like(input_ids)
    generated = []
    diagnostics = {}
    device = input_ids.device

    def sync():
        if measure_cuda:
            torch.cuda.synchronize(device)

    def peaks():
        return {
            "allocated_bytes": torch.cuda.max_memory_allocated(device),
            "reserved_bytes": torch.cuda.max_memory_reserved(device),
        }

    def forward(phase, tokens, position=None):
        # Decode positions are explicit: an evicting cache (H2O) holds fewer entries than tokens seen.
        positions = {} if position is None else {
            "position_ids": torch.tensor([[position]], device=device),
            "cache_position": torch.tensor([position], device=device),
        }
        try:
            return model(input_ids=tokens, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True, **positions)
        except Exception as error:
            error.kv_phase = phase
            raise

    with torch.inference_mode():
        sync()
        if measure_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        outputs = forward("prefill", input_ids)
        cache = outputs.past_key_values
        token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sync()
        prefill_seconds = time.perf_counter() - start
        if measure_cuda:
            diagnostics["ttft_ms"] = prefill_seconds * 1000
            diagnostics["prefill_peak"] = peaks()
        del outputs
        prefill_cache = cache_stats(cache)
        generated.append(token)
        first_is_eos = token.item() in eos_ids
        sync()
        if measure_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        if not first_is_eos:
            for step in range(max_new_tokens - 1):
                mask = torch.cat((mask, mask.new_ones((1, 1))), dim=-1)
                outputs = forward("decode", token, input_ids.shape[-1] + step)
                cache = outputs.past_key_values
                token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                del outputs
                generated.append(token)
                if token.item() in eos_ids:
                    break
        sync()
        decode_seconds = time.perf_counter() - start
        decode_steps = len(generated) - 1
        if measure_cuda:
            diagnostics.update({
                "decode_ms": decode_seconds * 1000,
                "decode_tokens_per_second": decode_steps / decode_seconds if decode_steps else None,
                "decode_peak": peaks(),
            })
        final_cache = cache_stats(cache)
        tokens = torch.cat(generated, dim=-1).squeeze(0).cpu().tolist()
    return {
        "output_ids": tokens, "output_tokens": len(tokens), "decode_forward_calls": decode_steps,
        "cache_reused": decode_steps > 0,
        "prefill_cache": prefill_cache, "final_cache": final_cache,
        "diagnostic_metrics": diagnostics,
    }
