import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import FixedMaskedSpanDataset, FixedValidationExample, TrainConfig, Trainer


def _make_fake_trainer(early_stopping_enabled=True, patience=3, validation_every=10000,
                        min_delta=0.0, best_val_loss=0.5):
    """A minimal stand-in exposing just what Trainer._update_early_stopping touches, so the
    patience logic can be unit-tested without building a real (GPU/EnCodec-backed) Trainer."""
    cfg = SimpleNamespace(
        early_stopping_enabled=early_stopping_enabled,
        early_stopping_patience=patience,
        early_stopping_min_delta=min_delta,
        validation_every=validation_every,
    )
    return SimpleNamespace(cfg=cfg, _val_no_improve_count=0, best_val_loss=best_val_loss)


def _validation_tick(fake, combined_loss, step):
    """Mirrors Trainer.run_validation's own contract: compute `improved`, update
    `best_val_loss` on a strict improvement, *then* call `_update_early_stopping`."""
    improved = combined_loss < fake.best_val_loss
    if improved:
        fake.best_val_loss = combined_loss
    return Trainer._update_early_stopping(fake, improved=improved, combined_loss=combined_loss, step=step)


class TestEarlyStoppingPatience(unittest.TestCase):
    def test_disabled_never_stops(self):
        fake = _make_fake_trainer(early_stopping_enabled=False, patience=1)
        for step in range(1, 20):
            stop = _validation_tick(fake, combined_loss=1.0, step=step)
            self.assertFalse(stop)

    def test_stops_exactly_on_patience_th_non_improvement(self):
        fake = _make_fake_trainer(patience=3)
        # Non-improving calls 1 and 2 should not stop; the 3rd should.
        self.assertFalse(_validation_tick(fake, combined_loss=1.0, step=1))
        self.assertEqual(fake._val_no_improve_count, 1)
        self.assertFalse(_validation_tick(fake, combined_loss=1.0, step=2))
        self.assertEqual(fake._val_no_improve_count, 2)
        self.assertTrue(_validation_tick(fake, combined_loss=1.0, step=3))
        self.assertEqual(fake._val_no_improve_count, 3)

    def test_improvement_resets_counter(self):
        fake = _make_fake_trainer(patience=3, best_val_loss=1.0)
        _validation_tick(fake, combined_loss=1.5, step=1)
        _validation_tick(fake, combined_loss=1.5, step=2)
        self.assertEqual(fake._val_no_improve_count, 2)
        # An improvement anywhere before the patience threshold resets the streak.
        stop = _validation_tick(fake, combined_loss=0.5, step=3)
        self.assertFalse(stop)
        self.assertEqual(fake._val_no_improve_count, 0)
        self.assertEqual(fake.best_val_loss, 0.5)
        # Needs a fresh run of `patience` non-improvements after the reset.
        self.assertFalse(_validation_tick(fake, combined_loss=1.2, step=4))
        self.assertFalse(_validation_tick(fake, combined_loss=1.2, step=5))
        self.assertEqual(fake._val_no_improve_count, 2)
        stop = _validation_tick(fake, combined_loss=1.2, step=6)
        self.assertTrue(stop)
        self.assertEqual(fake._val_no_improve_count, 3)

    def test_patience_one_stops_on_first_non_improvement(self):
        fake = _make_fake_trainer(patience=1)
        stop = Trainer._update_early_stopping(fake, improved=False, combined_loss=1.0, step=1)
        self.assertTrue(stop)


class TestEarlyStoppingMinDelta(unittest.TestCase):
    """min_delta gives a tolerance band around the best loss: a check that's worse than the
    best but still within min_delta doesn't count against patience (treated as noise), but
    doesn't reset the counter either."""

    def test_within_tolerance_does_not_count_against_patience(self):
        fake = _make_fake_trainer(patience=2, min_delta=0.05, best_val_loss=5.0806)
        # Mirrors the real songs04 step-20000 case: 5.0847 vs best 5.0806, diff=0.0041 << 0.05.
        stop = _validation_tick(fake, combined_loss=5.0847, step=20000)
        self.assertFalse(stop)
        self.assertEqual(fake._val_no_improve_count, 0)

    def test_beyond_tolerance_counts_against_patience(self):
        fake = _make_fake_trainer(patience=2, min_delta=0.05, best_val_loss=5.0806)
        # Mirrors the real songs04 step-30000 case: 5.1381 vs best 5.0806, diff=0.0575 > 0.05.
        stop = _validation_tick(fake, combined_loss=5.1381, step=30000)
        self.assertFalse(stop)
        self.assertEqual(fake._val_no_improve_count, 1)
        stop = _validation_tick(fake, combined_loss=5.1570, step=40000)
        self.assertTrue(stop)
        self.assertEqual(fake._val_no_improve_count, 2)

    def test_min_delta_zero_reduces_to_strict_comparison(self):
        # Default min_delta=0.0 must reproduce the original strict-improvement behavior exactly.
        fake = _make_fake_trainer(patience=1, min_delta=0.0, best_val_loss=5.0)
        stop = _validation_tick(fake, combined_loss=5.0001, step=1)
        self.assertTrue(stop)

    def test_exact_tie_at_best_does_not_count(self):
        fake = _make_fake_trainer(patience=1, min_delta=0.0, best_val_loss=5.0)
        stop = _validation_tick(fake, combined_loss=5.0, step=1)
        self.assertFalse(stop)
        self.assertEqual(fake._val_no_improve_count, 0)


class TestNaNGuard(unittest.TestCase):
    def test_torch_isfinite_detects_nan_and_inf(self):
        self.assertFalse(bool(torch.isfinite(torch.tensor(float("nan")))))
        self.assertFalse(bool(torch.isfinite(torch.tensor(float("inf")))))
        self.assertFalse(bool(torch.isfinite(torch.tensor(float("-inf")))))
        self.assertTrue(bool(torch.isfinite(torch.tensor(1.2345))))


# --- Full-wiring integration tests: does Trainer.train() actually stop when it should? ---
# Same hand-built-Trainer pattern as tests/test_validation_integration.py (bypasses __init__ via
# Trainer.__new__ so no GPU/EnCodec is needed), reused here to test the NaN-guard and
# early-stopping *break* wiring itself, not just the isolated decision logic above.


class DummyWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), int(step)))

    def close(self):
        pass


class TinyTrainDataset(Dataset):
    def __init__(self):
        self.recent_metrics = deque()

    def __len__(self):
        return 8

    def __getitem__(self, idx):
        x = torch.zeros((2, 6), dtype=torch.long)
        y = torch.zeros((2, 6), dtype=torch.long)
        loss_mask = torch.zeros(6, dtype=torch.bool)
        loss_mask[1:3] = True
        return x, y, loss_mask

    def pop_recent_metrics(self, max_items=None):
        return []


class TinyModel(nn.Module):
    def __init__(self, vocab: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros((2, vocab), dtype=torch.float32))

    def forward(self, x: torch.Tensor, segment_ids=None, left_dist_idx=None, right_dist_idx=None) -> torch.Tensor:
        b, k, t = x.shape
        return self.logits.unsqueeze(0).unsqueeze(2).expand(b, -1, t, -1)


def _make_validation_loader(band: str) -> DataLoader:
    y = torch.zeros((2, 6), dtype=torch.long)
    x = y.clone()
    x[:, 2:4] = 9
    loss_mask = torch.zeros(6, dtype=torch.bool)
    loss_mask[2:4] = True
    dataset = FixedMaskedSpanDataset(
        [
            FixedValidationExample(
                x=x,
                y=y,
                loss_mask=loss_mask,
                band=band,
                sample_name=band,
                mask_mean_activity=0.9 if band == "high_activity" else 0.0,
                mask_len=2,
                window_start=0,
                mask_start=2,
            )
        ]
    )
    return DataLoader(dataset, batch_size=1, shuffle=False)


def _build_bare_trainer(tmpdir: str, **cfg_overrides) -> Trainer:
    cfg = TrainConfig(
        output_dir=tmpdir,
        run_name="es_integration",
        warmup_steps=1,
        save_every=1000,
        log_every=1000,
        test_fill_every=0,
        **cfg_overrides,
    )
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")
    trainer.writer = DummyWriter()
    trainer.dataset = TinyTrainDataset()
    trainer.dataloader = DataLoader(trainer.dataset, batch_size=1, shuffle=False)
    trainer.validation_enabled = True
    trainer.validation_dataloaders = {
        "high_activity": _make_validation_loader("high_activity"),
        "low_activity": _make_validation_loader("low_activity"),
    }
    trainer.validation_group_specs = {}
    trainer.validation_inspection_examples = {}
    trainer.model = TinyModel(vocab=5)
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
    trainer.scaler = torch.amp.GradScaler(enabled=False)
    trainer.global_step = 0
    trainer.best_loss = float("inf")
    trainer.best_val_loss = float("inf")
    trainer._val_no_improve_count = 0
    trainer.early_stop_triggered = False
    trainer.diverged = False
    return trainer


class TestTrainLoopStopsOnEarlyStopping(unittest.TestCase):
    def test_train_breaks_as_soon_as_early_stopping_triggers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _build_bare_trainer(
                tmpdir,
                total_steps=20,
                validation_every=2,
                early_stopping_enabled=True,
                early_stopping_patience=1,
            )
            # Force the very first validation check to report "no improvement", which with
            # patience=1 must trigger a stop immediately.
            def fake_run_validation(step):
                trainer.early_stop_triggered = True
                return {"combined_loss": 999.0, "early_stop": True}

            trainer.run_validation = fake_run_validation
            trainer.train()

            # Should have stopped at the first validation check (step 2), nowhere near total_steps=20.
            self.assertEqual(trainer.global_step, 2)
            self.assertTrue(trainer.early_stop_triggered)
            self.assertTrue((trainer.cfg.checkpoint_dir / "latest.pt").exists())
            self.assertTrue((trainer.cfg.checkpoint_dir / "final.pt").exists())


class TestTrainLoopStopsOnNaN(unittest.TestCase):
    def test_train_breaks_immediately_and_skips_the_update_on_nan_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _build_bare_trainer(
                tmpdir,
                total_steps=20,
                validation_every=0,  # isolate the NaN guard from the early-stopping guard
                nan_guard_enabled=True,
            )
            params_before = trainer.model.logits.detach().clone()

            def poisoned_losses(x, logits, y, loss_mask, step):
                zeros = {
                    "decoded_loss_total": 0.0,
                    "decoded_loss_waveform_l1": 0.0,
                    "decoded_loss_stft": 0.0,
                    "decoded_loss_spectral_convergence": 0.0,
                    "decoded_loss_log_magnitude": 0.0,
                    "decoded_loss_items": 0,
                }
                nan = torch.tensor(float("nan"))
                return nan, nan, zeros

            trainer._compute_training_losses = poisoned_losses
            trainer.train()

            self.assertEqual(trainer.global_step, 1)
            self.assertTrue(trainer.diverged)
            self.assertFalse(trainer.early_stop_triggered)
            # The poisoned step's gradient must never have been applied.
            self.assertTrue(torch.equal(trainer.model.logits.detach(), params_before))
            self.assertTrue((trainer.cfg.checkpoint_dir / "latest.pt").exists())


if __name__ == "__main__":
    unittest.main()
