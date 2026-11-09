"""CPU reproductions of historical RouterBERT behaviour; no pretrained downloads.

These tests deliberately retain the original implementation. They distinguish
its token train/eval mismatch from all-keep and sample-routing controls, verify
the corrected query implementation, and document the limits of sparse work.
"""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import AutoModel, BertConfig

from benchmarks.hdc_runtime import HDCBertRuntime
from models.router_tuning_bert import RouterTuningBERTClassifier, ste_binarize


class FixedRouter(nn.Module):
    def __init__(self, keep):
        super().__init__()
        self.register_buffer("keep", keep.float())

    def forward(self, hidden, attention_mask=None):
        return self.keep.to(hidden.dtype).unsqueeze(-1)


class RouterAuditTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        self.config = BertConfig(
            vocab_size=41, hidden_size=16, intermediate_size=32,
            num_attention_heads=4, num_hidden_layers=3,
            max_position_embeddings=32, hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        with patch("models.router_tuning_bert.AutoModel.from_pretrained",
                   side_effect=lambda *_a, **_k: AutoModel.from_config(
                       self.config, attn_implementation="eager")):
            self.model = RouterTuningBERTClassifier(
                "offline", 3, dropout=0.0, target_keep_ratio=0.5
            )
        self.ids = torch.tensor([[2, 5, 7, 9, 0, 0], [2, 4, 6, 8, 10, 0]])
        self.mask = self.ids.ne(0).long()
        self.hidden = self.model.bert.embeddings(input_ids=self.ids)
        self.extended = self.model.bert.get_extended_attention_mask(
            self.mask, self.mask.shape
        )
        self.layer = self.model.bert.encoder.layer[0]

    def test_mixed_token_train_eval_mismatch_and_query_fix(self):
        keep = torch.tensor([[1, 0, 1, 0, 0, 0], [1, 1, 0, 0, 1, 0]])
        router = FixedRouter(keep)
        with torch.inference_mode():
            self.model.train()
            train, _, _ = self.model._routed_layer(
                self.layer, self.hidden, self.extended, self.mask, router
            )
            self.model.eval()
            legacy, _, _ = self.model._routed_layer(
                self.layer, self.hidden, self.extended, self.mask, router
            )
            query_attn = HDCBertRuntime._query_attention(
                self.model, self.layer, self.hidden, self.extended, keep.bool()
            )
            fixed = self.model._run_post_attention(self.layer, self.hidden, query_attn)
        valid = self.mask.bool()
        self.assertGreater((train[valid] - legacy[valid]).abs().max().item(), 1e-5)
        torch.testing.assert_close(train[valid], fixed[valid], rtol=1e-5, atol=1e-6)

    def test_all_valid_tokens_kept_is_a_matched_control(self):
        router = FixedRouter(self.mask)
        with torch.inference_mode():
            self.model.train()
            train, _, _ = self.model._routed_layer(
                self.layer, self.hidden, self.extended, self.mask, router
            )
            self.model.eval()
            legacy, _, _ = self.model._routed_layer(
                self.layer, self.hidden, self.extended, self.mask, router
            )
        torch.testing.assert_close(train, legacy, rtol=1e-5, atol=1e-6)

    def test_sample_router_keeps_train_eval_function(self):
        self.model.routing_mode = "sample"
        router = FixedRouter(torch.tensor([1, 0]))
        with torch.inference_mode():
            self.model.train()
            train, _, _ = self.model._routed_layer(
                self.layer, self.hidden, self.extended, self.mask, router
            )
            self.model.eval()
            actual, _, _ = self.model._routed_layer(
                self.layer, self.hidden, self.extended, self.mask, router
            )
        torch.testing.assert_close(train, actual, rtol=1e-5, atol=1e-6)

    def test_all_skipped_still_executes_full_padded_ffn(self):
        self.model.eval()
        shapes = []
        handle = self.layer.intermediate.register_forward_pre_hook(
            lambda _m, inputs: shapes.append(tuple(inputs[0].shape))
        )
        try:
            with torch.inference_mode():
                actual, _, _ = self.model._routed_layer(
                    self.layer, self.hidden, self.extended, self.mask,
                    FixedRouter(torch.zeros_like(self.mask)),
                )
                expected = self.model._run_post_attention(
                    self.layer, self.hidden, torch.zeros_like(self.hidden)
                )
        finally:
            handle.remove()
        self.assertEqual(shapes, [(2, 6, 16), (2, 6, 16)])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_initialization_keeps_every_token_and_budget_scales_with_batch(self):
        prob = self.model.routers[0](self.hidden)
        mask, hard = ste_binarize(prob, self.model.tau)
        self.assertTrue(torch.equal(hard, torch.ones_like(hard)))
        _, loss = self.model._compute_budget_loss(mask, hard, self.mask)
        _, doubled = self.model._compute_budget_loss(
            mask.repeat(2, 1, 1), hard.repeat(2, 1, 1), self.mask.repeat(2, 1)
        )
        torch.testing.assert_close(doubled, 2 * loss, rtol=0, atol=0)

    def test_historical_bfloat16_scatter_reproduces_dtype_failure(self):
        self.model.eval()
        with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
            with self.assertRaisesRegex(RuntimeError, "dtypes match"):
                self.model._token_attn_skip_inference(
                    self.layer, self.hidden, self.mask, self.mask.unsqueeze(-1)
                )


if __name__ == "__main__":
    unittest.main()
