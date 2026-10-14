"""Optional offline numerical checks, using random tiny Llama weights on CPU."""
import importlib.util
import unittest

from kv_cache_lab.inference import cache_stats, greedy_decode

HAS_INFERENCE = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers"))


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

    def test_aliases_count_physical_storage_once(self):
        tensor = self.torch.ones((1, 2, 3, 4))
        stats = cache_stats([(tensor, tensor.view_as(tensor))])
        self.assertEqual(stats["storage_bytes"], tensor.untyped_storage().nbytes())


if __name__ == "__main__":
    unittest.main()
