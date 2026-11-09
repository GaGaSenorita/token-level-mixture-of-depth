"""Offline checks for comparator-only training and checkpoint compatibility."""
import copy
import tempfile
from pathlib import Path
import unittest

import torch
from transformers import BertConfig

from benchmarks.hdc_runtime import HDCBertRuntime
from benchmarks.train_comparators import configure, frozen_hash, router_validation, save_state


class ComparatorTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(91)
        self.config = BertConfig(vocab_size=45, hidden_size=16, num_hidden_layers=3,
            num_attention_heads=4, intermediate_size=32, hidden_dropout_prob=.1,
            attention_probs_dropout_prob=.1)
        self.ids = torch.tensor([[1, 4, 5, 6, 0], [1, 6, 8, 0, 0]])
        self.mask = self.ids.ne(0).long()
        self.labels = torch.tensor([0, 1])

    def test_router_gradient_and_frozen_backbone(self):
        model = HDCBertRuntime(config=self.config, split_layer=0, num_labels=2)
        before = frozen_hash(model)
        configure(model, "router")
        self.assertTrue(model.training)
        self.assertFalse(model.bert.training)
        self.assertFalse(model.dropout.training)
        opt = torch.optim.AdamW(model.routers.parameters(), lr=.01)
        logits, _, budget = model.forward_with_routing(self.ids, self.mask)
        (torch.nn.functional.cross_entropy(logits, self.labels) + .001 * budget).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.routers.parameters()))
        self.assertTrue(all(p.grad is None for p in model.bert.parameters()))
        opt.step()
        self.assertEqual(before, frozen_hash(model))

    def test_router_validation_matches_sparse_with_padding(self):
        model = HDCBertRuntime(config=self.config, split_layer=0, num_labels=2)
        # Nonzero routers produce heterogeneous token masks.
        for router in model.routers:
            torch.nn.init.normal_(router.proj.weight, std=.4)
        configure(model, "router")
        with torch.no_grad():
            expected = model.forward_with_routing(self.ids, self.mask)[0]
            model.eval()
            actual = model.forward_dynamic(self.ids, self.mask, -1, "query")[0]
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        data = {"input_ids": self.ids, "attention_mask": self.mask, "labels": self.labels}
        result = router_validation(model, data, device="cpu")
        self.assertEqual(result["correct"], int((actual.argmax(-1) == self.labels).sum()))
        self.assertFalse(model.training)

    def test_deebert_only_intermediate_exits_train(self):
        model = HDCBertRuntime(config=self.config, split_layer=3, num_labels=2)
        model.offramp_classifiers[-1].load_state_dict(model.classifier.state_dict())
        before = frozen_hash(model)
        last_before = copy.deepcopy(model.offramp_classifiers[-1].state_dict())
        configure(model, "deebert")
        self.assertFalse(model.training)
        self.assertTrue(model.dropout.training)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        logits = model.offramp_classifiers[0](model.dropout(torch.randn(2, 16)))
        torch.nn.functional.cross_entropy(logits, self.labels).backward()
        opt.step()
        self.assertEqual(before, frozen_hash(model))
        for name, value in last_before.items():
            self.assertTrue(torch.equal(value, model.offramp_classifiers[-1].state_dict()[name]))
        model.eval()
        with torch.inference_mode():
            dynamic = model.forward_dynamic(self.ids, self.mask, -1)[0]
            dense = model.forward_dense(self.ids, self.mask)
        torch.testing.assert_close(dynamic, dense)

    def test_atomic_checkpoint_full_and_router_subset(self):
        model = HDCBertRuntime(config=self.config, split_layer=0, num_labels=2)
        with tempfile.TemporaryDirectory() as temp:
            full, part = Path(temp) / "full.pt", Path(temp) / "part.pt"
            save_state(model, full)
            save_state(model, part, "routers.")
            loaded = torch.load(full, mmap=True, weights_only=True)
            model.load_state_dict(loaded, strict=True)
            router = torch.load(part, mmap=True, weights_only=True)
            self.assertTrue(router)
            self.assertTrue(all(k.startswith("routers.") for k in router))


if __name__ == "__main__":
    unittest.main()
