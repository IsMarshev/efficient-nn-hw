"""Focused correctness checks; no network, GPU or training dataset required."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import json
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from run_zero import load_notebook


class ZerOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.m = load_notebook()

    def test_formula_and_residuals(self):
        m = self.m
        h = m.hadamard_matrix(8)
        self.assertTrue(torch.equal(h @ h.T, 8 * torch.eye(8)))
        w = torch.empty(5, 3, 3, 3)
        m.zero_init_(w)
        self.assertTrue(torch.equal(w[:, :, 1, 1], h[:5, :3] * 0.5))
        self.assertEqual(torch.count_nonzero(w).item(), 15)
        for cls in (m.ResNet18, m.ResNet50):
            torch.manual_seed(0)
            a = cls(num_classes=10, init="zero", cifar_stem=True)
            torch.manual_seed(123)
            b = cls(num_classes=10, init="zero", cifar_stem=True)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values())))
            for block in a.modules():
                if isinstance(block, (m.ResBlock, m.Bottleneck)):
                    conv = block.conv2 if isinstance(block, m.ResBlock) else block.conv3
                    bn = block.bn2 if isinstance(block, m.ResBlock) else block.bn3
                    self.assertEqual(torch.count_nonzero(conv.weight).item(), 0)
                    self.assertTrue(torch.all(bn.weight == 1))
            del a, b

    def test_schedule_and_presets(self):
        m = self.m
        cfg = m.h100_config
        self.assertEqual(cfg.methods, ("zero", "kaiming"))
        self.assertEqual(cfg.seeds, (0,))
        self.assertEqual(cfg.val_size, 0)
        self.assertEqual(cfg.epochs, 200)
        self.assertEqual(m.paper_config.precision, "fp32")
        rates = [m.learning_rate_at(i, 2000, 100, .1) for i in range(2000)]
        self.assertAlmostEqual(rates[0], .01)
        self.assertAlmostEqual(rates[99], .1)
        self.assertAlmostEqual(rates[100], .1)
        self.assertAlmostEqual(rates[-1], 0)
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(rates[:99], rates[1:100])))
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(rates[100:-1], rates[101:])))

    def test_budget_plan_is_shared_and_insufficient_budget_rejected(self):
        m = self.m
        cfg = replace(m.h100_config, device="cpu", precision="fp32", min_epochs=50)
        # Контролируемые timings проверяют планировщик без привязки к скорости CI.
        fake_epoch = {"seconds": 1.0, "batches": 10, "images": 1280}
        loaders = (list(range(100)), None, list(range(100)))
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(m, "make_loaders", return_value=loaders), \
                 patch.object(m, "create_model", return_value=torch.nn.Linear(2, 2)), \
                 patch.object(m, "run_epoch", return_value=fake_epoch), \
                 patch.object(m, "stable_ranks", return_value={}):
                planned = m.benchmark_runtime(cfg, Path(tmp), m.time.monotonic() + 6000)
                self.assertGreaterEqual(planned.epochs, 50)
                self.assertLess(planned.epochs, 200)
                self.assertEqual(planned.epochs % 10, 0)
                self.assertEqual(planned.methods, cfg.methods)
                plan = json.loads((Path(tmp) / "runtime_plan.json").read_text())
                self.assertEqual(plan["chosen_epochs"], planned.epochs)
                with self.assertRaises(RuntimeError):
                    m.benchmark_runtime(cfg, Path(tmp), m.time.monotonic() + 60)

    def test_interruption_resume_and_last_epoch_selection(self):
        m = self.m
        cfg = replace(m.smoke_config, batch_size=4, epochs=2, log_every=0, channels_last=False, device="cpu")
        torch.manual_seed(17)
        dataset = TensorDataset(torch.randn(8, 3, 32, 32), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]))
        def loaders():
            return (DataLoader(dataset, batch_size=4, shuffle=True, generator=torch.Generator()),
                    None, DataLoader(dataset, batch_size=4))
        with tempfile.TemporaryDirectory() as tmp:
            whole, interrupted = Path(tmp) / "whole", Path(tmp) / "interrupted"
            reference = m.train_one(cfg, "resnet18", "zero", 0, whole, loaders=loaders())
            original_run_epoch = m.run_epoch
            def interrupt(*args, **kwargs):
                if len(args) > 5 and args[3] is not None and args[5] == 1:
                    raise TimeoutError("simulated time limit")
                return original_run_epoch(*args, **kwargs)
            m.run_epoch = interrupt
            try:
                with self.assertRaises(TimeoutError):
                    m.train_one(cfg, "resnet18", "zero", 0, interrupted, loaders=loaders())
            finally:
                m.run_epoch = original_run_epoch
            self.assertFalse((interrupted / "summary.json").exists())
            checkpoint = torch.load(interrupted / "last.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            resumed = m.train_one(cfg, "resnet18", "zero", 0, interrupted, loaders=loaders())
            self.assertEqual(reference["test_accuracy"], resumed["test_accuracy"])
            a = torch.load(whole / "last.pt", weights_only=True)
            b = torch.load(interrupted / "last.pt", weights_only=True)
            self.assertTrue(all(torch.equal(a["model_state"][k], b["model_state"][k]) for k in a["model_state"]))
            self.assertEqual(a["epoch"], 2)
            self.assertTrue(a["optimizer_state"]["state"])
            self.assertEqual(resumed["checkpoint_selection"], "last_epoch")
            history = json.loads((interrupted / "history.json").read_text())
            self.assertEqual([r["epoch"] for r in history], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
