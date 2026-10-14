"""Cache accounting and greedy FullKV decoding, with optional CUDA diagnostics."""
import time


def cache_stats(cache):
    if hasattr(cache, "key_cache"):
        pairs = zip(cache.key_cache, cache.value_cache)
    else:
        pairs = cache
    seen = set()
    storage_bytes = 0
    shapes = []
    token_counts = []
    for key, value in pairs:
        shapes.append({"key": list(key.shape), "value": list(value.shape), "dtype": str(key.dtype)})
        token_counts.append(key.shape[-2])
        for tensor in (key, value):
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage.data_ptr())
            if identity not in seen:
                seen.add(identity)
                storage_bytes += storage.nbytes()
    if not shapes:
        raise ValueError("Model did not return populated KV-cache tensors.")
    return {"storage_bytes": storage_bytes, "layer_shapes": shapes, "tokens_per_layer": token_counts}


def greedy_decode(model, input_ids, max_new_tokens, eos_ids, *, measure_cuda=False):
    import torch
    from transformers import DynamicCache

    if measure_cuda and input_ids.device.type != "cuda":
        raise ValueError("CUDA measurement requires CUDA input tensors.")
    eos_ids = set(eos_ids)
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

    def forward(phase, tokens):
        try:
            return model(input_ids=tokens, attention_mask=mask, past_key_values=cache, use_cache=True, return_dict=True)
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
            for _ in range(max_new_tokens - 1):
                mask = torch.cat((mask, mask.new_ones((1, 1))), dim=-1)
                outputs = forward("decode", token)
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
