"""Optional offline numerical checks, using random tiny Llama weights on CPU."""
import importlib.util
import unittest

from kv_cache_lab.inference import cache_stats, greedy_decode
from kv_cache_lab.methods import UPSTREAM_DIR, cache_factory

HAS_INFERENCE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))
HAS_UPSTREAM = (UPSTREAM_DIR / "pyramidkv").is_dir()
HAS_HQQ = importlib.util.find_spec("hqq") is not None


@unittest.skipUnless(HAS_INFERENCE, "Run with uv run --locked --extra inference for tiny-model CPU checks")
class InferenceTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        self.torch = torch
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256, eos_token_id=63)
        config._attn_implementation = "sdpa"
        self.model = LlamaForCausalLM(config).eval()
        self.inputs = torch.tensor([[1, 7, 12, 4, 19]])

    def test_cached_greedy_tokens_match_uncached_forward_and_cache_growth(self):
        result = greedy_decode(self.model, self.inputs, 5, [])
        sequence = self.inputs.clone()
        expected = []
        with self.torch.inference_mode():
            for _ in range(5):
                token = self.model(input_ids=sequence, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                expected.append(token.item())
                sequence = self.torch.cat((sequence, token), dim=-1)
        self.assertEqual(result["output_ids"], expected)
        self.assertEqual(result["prefill_cache"]["tokens_per_layer"], [5, 5])
        self.assertEqual(result["final_cache"]["tokens_per_layer"], [9, 9])
        self.assertTrue(result["cache_reused"])
        self.assertEqual(result["diagnostic_metrics"], {})
        # Two layers, K+V, two KV heads, five tokens, head dim eight, FP32 bytes.
        self.assertEqual(result["prefill_cache"]["storage_bytes"], 2 * 2 * 2 * 5 * 8 * 4)

    def test_first_eos_stops_without_a_decode_pass(self):
        first = greedy_decode(self.model, self.inputs, 1, [])["output_ids"][0]
        result = greedy_decode(self.model, self.inputs, 5, [first])
        self.assertEqual(result["output_tokens"], 1)
        self.assertFalse(result["cache_reused"])
        self.assertEqual(result["final_cache"]["tokens_per_layer"], [5, 5])

    def test_registered_fullkv_cache_matches_default_decoding(self):
        new_cache = cache_factory({"name": "fullkv", "params": {}}, self.model)
        default = greedy_decode(self.model, self.inputs, 5, [])
        explicit = greedy_decode(self.model, self.inputs, 5, [], cache=new_cache(self.inputs.shape[-1]))
        self.assertEqual(explicit["output_ids"], default["output_ids"])
        self.assertEqual(explicit["final_cache"], default["final_cache"])

    def long_inputs(self, length=32):
        return self.torch.randint(0, 63, (1, length), generator=self.torch.Generator().manual_seed(3))

    @unittest.skipUnless(HAS_UPSTREAM, "Run scripts/fetch_upstream.py for H2O checks")
    def test_h2o_full_budget_matches_fullkv(self):
        inputs = self.long_inputs()
        fullkv = greedy_decode(self.model, inputs, 6, [])
        new_cache = cache_factory({"name": "h2o", "params": {"budget_fraction": 1.0, "recent_fraction": 0.5}}, self.model)
        h2o = greedy_decode(self.model, inputs, 6, [], cache=new_cache(inputs.shape[-1]))
        self.assertEqual(h2o["output_ids"], fullkv["output_ids"])
        self.assertEqual(h2o["prefill_cache"]["storage_bytes"], fullkv["prefill_cache"]["storage_bytes"])

    @unittest.skipUnless(HAS_UPSTREAM, "Run scripts/fetch_upstream.py for H2O checks")
    def test_h2o_keeps_budget_per_kv_head_and_resets_between_requests(self):
        inputs = self.long_inputs()
        fullkv = greedy_decode(self.model, inputs, 6, [])
        new_cache = cache_factory({"name": "h2o", "params": {"budget_fraction": 0.25, "recent_fraction": 0.5}}, self.model)
        first = greedy_decode(self.model, inputs, 6, [], cache=new_cache(inputs.shape[-1]))
        second = greedy_decode(self.model, inputs, 6, [], cache=new_cache(inputs.shape[-1]))
        self.assertEqual(first["prefill_cache"]["tokens_per_layer"], [8, 8])
        self.assertEqual(first["final_cache"]["tokens_per_layer"], [13, 13])
        # Cached at KV-head granularity (2 heads), not copied per query head (4).
        self.assertEqual(first["prefill_cache"]["layer_shapes"][0]["key"], [1, 2, 8, 8])
        self.assertEqual(first["prefill_cache"]["storage_bytes"] * 4, fullkv["prefill_cache"]["storage_bytes"])
        self.assertEqual(second["output_ids"], first["output_ids"])
        self.assertEqual(second["final_cache"]["tokens_per_layer"], [13, 13])

    @unittest.skipUnless(HAS_UPSTREAM, "Run scripts/fetch_upstream.py for H2O checks")
    def test_h2o_patch_leaves_other_models_unchanged(self):
        from transformers import LlamaForCausalLM
        other = LlamaForCausalLM(self.model.config).eval()
        other.load_state_dict(self.model.state_dict())
        inputs = self.long_inputs()
        before = greedy_decode(other, inputs, 6, [])
        cache_factory({"name": "h2o", "params": {"budget_fraction": 0.25, "recent_fraction": 0.5}}, self.model)
        after = greedy_decode(other, inputs, 6, [])
        self.assertEqual(after, before)

    @unittest.skipUnless(HAS_HQQ, "Install hqq for KIVI checks")
    def test_kivi_counts_quantized_storage_and_metadata(self):
        inputs = self.long_inputs()
        fullkv = greedy_decode(self.model, inputs, 6, [])
        sizes = {}
        for bits in (4, 2):
            new_cache = cache_factory({"name": "kivi", "params": {"bits": bits, "group_size": 8, "residual_length": 128}}, self.model)
            result = greedy_decode(self.model, inputs, 6, [], cache=new_cache(inputs.shape[-1]))
            self.assertEqual(result["output_tokens"], 6)
            self.assertEqual(result["prefill_cache"]["tokens_per_layer"], [32, 32])
            self.assertEqual(result["final_cache"]["tokens_per_layer"], [37, 37])
            sizes[bits] = result["prefill_cache"]["storage_bytes"]
        self.assertLess(sizes[2], sizes[4])
        self.assertLess(sizes[4], fullkv["prefill_cache"]["storage_bytes"])

    def test_aliases_count_physical_storage_once(self):
        tensor = self.torch.ones((1, 2, 3, 4))
        stats = cache_stats([(tensor, tensor.view_as(tensor))])
        self.assertEqual(stats["storage_bytes"], tensor.untyped_storage().nbytes())


if __name__ == "__main__":
    unittest.main()
