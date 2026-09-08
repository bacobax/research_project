import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import TrainConfig, Trainer


class DummyWriter:
    def __init__(self):
        self.figures = []
        self.flushed = False

    def add_figure(self, name, fig, step):
        self.figures.append((name, step))

    def flush(self):
        self.flushed = True


class _TinyEncoder:
    """EnCodec-free stand-in. `decode` (full-song path) raises by default so tests can assert
    it was never called when save_crop_samples=True."""

    samples_per_frame = 8

    def __init__(self, allow_full_decode: bool = True):
        self.allow_full_decode = allow_full_decode
        self.full_decode_calls = 0

    def codes_to_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.float()

    def decode_embeddings(self, embeddings: torch.Tensor, scale=None) -> torch.Tensor:
        audio = embeddings.mean(dim=1, keepdim=True)
        return audio.repeat_interleave(self.samples_per_frame, dim=-1)

    def decode(self, codes: torch.Tensor, scale) -> np.ndarray:
        self.full_decode_calls += 1
        if not self.allow_full_decode:
            raise AssertionError("full-song decode should not be called when save_crop_samples=True")
        audio = codes.float().mean(dim=0)
        return audio.repeat_interleave(self.samples_per_frame).numpy().astype(np.float32)


class _TinyModel(nn.Module):
    def __init__(self, K: int, vocab: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros((K, vocab), dtype=torch.float32))

    def forward(self, x, segment_ids=None, left_dist_idx=None, right_dist_idx=None):
        b, k, t = x.shape
        return self.logits.unsqueeze(0).unsqueeze(2).expand(b, -1, t, -1)


def _build_bare_inpaint_trainer(tmpdir: str, allow_full_decode: bool = True) -> Trainer:
    K, F, vocab = 2, 40, 5
    cfg = TrainConfig(
        output_dir=tmpdir,
        run_name="inpaint_crop_test",
        boundary_max_distance=8,
        max_len=20,
        ctx_left=None,
        ctx_right=None,
        inpaint_iters=1,
        target_sr=8000,
    )
    cfg.samples_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.device = torch.device("cpu")
    trainer.encoder = _TinyEncoder(allow_full_decode=allow_full_decode)
    trainer.model = _TinyModel(K=K, vocab=vocab)
    trainer.writer = DummyWriter()
    trainer.global_step = 12345
    trainer.mask_token = 0
    trainer.frames = F
    trainer.scale = None
    trainer.codes = torch.randint(1, vocab, (K, F), dtype=torch.long)
    trainer.gaps_f = [(10, 12), (25, 26)]  # two gaps, matching a real 4-gap annotation's shape
    return trainer


class TestInpaintAllGapsCropSamples(unittest.TestCase):
    def test_save_crop_samples_true_writes_per_gap_files_and_skips_full_decode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _build_bare_inpaint_trainer(tmpdir, allow_full_decode=False)
            result = trainer.inpaint_all_gaps(save_crop_samples=True)

            self.assertIsNone(result)
            self.assertEqual(trainer.encoder.full_decode_calls, 0)

            wav_files = sorted(trainer.cfg.samples_dir.glob("gap*_step_12345.wav"))
            self.assertEqual(len(wav_files), len(trainer.gaps_f))
            for f in wav_files:
                self.assertGreater(f.stat().st_size, 0)

            self.assertEqual(len(trainer.writer.figures), len(trainer.gaps_f))
            self.assertTrue(trainer.writer.flushed)

    def test_save_crop_samples_false_preserves_original_full_song_behavior(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _build_bare_inpaint_trainer(tmpdir, allow_full_decode=True)
            result = trainer.inpaint_all_gaps(save_crop_samples=False)

            self.assertIsNotNone(result)
            self.assertEqual(trainer.encoder.full_decode_calls, 1)
            # No per-gap crop files or figures in the default (unchanged) path.
            self.assertEqual(list(trainer.cfg.samples_dir.glob("gap*_step_*.wav")), [])
            self.assertEqual(trainer.writer.figures, [])

    def test_output_path_still_writes_full_song_wav(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = _build_bare_inpaint_trainer(tmpdir, allow_full_decode=True)
            out_path = Path(tmpdir) / "final_infilled.wav"
            result = trainer.inpaint_all_gaps(output_path=str(out_path), save_crop_samples=False)
            self.assertIsNotNone(result)
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
