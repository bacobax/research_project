import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.retrieval import SameSongRetrievalCache, build_retrieval_feature_config, extract_context_feature
from audio_infill.train import (
    EncoderDecoderCodebookInfiller,
    JointCodebookInfiller,
    TrainConfig,
    Trainer,
    build_boundary_condition_tensors,
    resolve_bucket_length,
)


class TestRetrievalConditioning(unittest.TestCase):
    def setUp(self):
        t = np.linspace(0.0, 1.0, 4096, endpoint=False, dtype=np.float32)
        self.waveform = (0.6 * np.sin(2 * np.pi * 3 * t) + 0.3 * np.sin(2 * np.pi * 7 * t)).astype(np.float32)
        self.codes = torch.arange(0, 80, dtype=torch.long).reshape(2, 40)
        self.cfg = TrainConfig(
            retrieval_mel_n_fft=128,
            retrieval_mel_hop_length=32,
            retrieval_mel_n_mels=16,
            retrieval_mel_fmin=20.0,
            retrieval_stft_n_fft=128,
            retrieval_stft_hop_length=32,
        )

    def test_context_feature_is_normalized_for_mel_and_stft(self):
        left = self.waveform[:512]
        right = self.waveform[512:1024]

        cfg = build_retrieval_feature_config(self.cfg)
        mel_vec = extract_context_feature(left, right, 24000, cfg)
        self.assertAlmostEqual(float(np.linalg.norm(mel_vec)), 1.0, places=4)

        self.cfg.retrieval_feature_type = "stft_mag"
        cfg = build_retrieval_feature_config(self.cfg)
        stft_vec = extract_context_feature(left, right, 24000, cfg)
        self.assertAlmostEqual(float(np.linalg.norm(stft_vec)), 1.0, places=4)

    def test_same_song_cache_builds_and_queries_with_exclusion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SameSongRetrievalCache(
                waveform=self.waveform,
                codes=self.codes,
                gaps=[(10, 12)],
                target_sr=24000,
                cache_dir=tmpdir,
                sample_name="demo",
                feature_cfg=build_retrieval_feature_config(self.cfg),
                rebuild_cache=True,
                supported_lengths=(4, 8, 12),
            )
            bank = cache.get_bank(mask_len=4, left_context_frames=3, right_context_frames=3, stride_frames=2)
            self.assertGreater(len(bank.entries), 0)
            self.assertTrue(
                all(
                    not (entry.left_context_start_frame < 12 and entry.right_context_end_frame > 10)
                    for entry in bank.entries
                )
            )

            results = cache.query(
                query_start_frame=14,
                query_end_frame=18,
                left_context_frames=3,
                right_context_frames=3,
                stride_frames=2,
                top_k=3,
                exclusion_margin_frames=1,
            )
            self.assertGreater(len(results), 0)
            self.assertLessEqual(len(results), 3)
            self.assertTrue(all(item.entry.fill_tokens.shape == (2, 4) for item in results))
            self.assertTrue(all(item.similarity <= 1.0001 for item in results))
            self.assertTrue(
                all(
                    not (item.entry.fill_start_frame < 19 and item.entry.fill_end_frame > 13)
                    for item in results
                )
            )

    def test_bucket_resolution_and_prebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SameSongRetrievalCache(
                waveform=self.waveform,
                codes=self.codes,
                gaps=[],
                target_sr=24000,
                cache_dir=tmpdir,
                sample_name="demo",
                feature_cfg=build_retrieval_feature_config(self.cfg),
                rebuild_cache=False,
                supported_lengths=(4, 8, 12),
            )
            self.assertEqual(resolve_bucket_length(7, (4, 8, 12)), 8)
            self.assertEqual(cache.resolve_bucket_length(11), 12)

            summary = cache.prebuild_banks(
                bucket_lengths=(4, 8, 12),
                left_context_resolver=lambda mask_len: (3, 3),
                stride_resolver=lambda mask_len: 2,
            )
            self.assertEqual(summary["bucket_lengths"], (4, 8, 12))
            self.assertEqual(len(summary["banks"]), 3)
            self.assertGreaterEqual(summary["total_entries"], 0)

            summary_2 = cache.prebuild_banks(
                bucket_lengths=(4, 8, 12),
                left_context_resolver=lambda mask_len: (3, 3),
                stride_resolver=lambda mask_len: 2,
            )
            self.assertEqual(summary_2["bucket_lengths"], (4, 8, 12))

    def test_models_accept_retrieval_payload(self):
        payload = {
            "tokens": torch.randint(0, 8, (1, 2, 2, 3), dtype=torch.long),
            "scores": torch.tensor([[0.8, 0.4]], dtype=torch.float32),
            "candidate_mask": torch.tensor([[True, True]], dtype=torch.bool),
            "lengths": torch.tensor([3], dtype=torch.long),
        }
        x = torch.randint(0, 8, (1, 2, 6), dtype=torch.long)
        x[:, :, 2:5] = 8
        loss_mask = torch.tensor([[False, False, True, True, True, False]])
        seg_ids, left_idx, right_idx = build_boundary_condition_tensors(loss_mask, max_distance=4)

        joint = JointCodebookInfiller(
            K=2,
            bins=8,
            mask_token=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_len=16,
            dropout=0.0,
            boundary_max_distance=4,
            retrieval_conditioning=True,
        )
        joint_logits = joint(x, seg_ids, left_idx, right_idx, retrieval_payload=payload)
        self.assertEqual(joint_logits.shape, (1, 2, 6, 8))

        encdec = EncoderDecoderCodebookInfiller(
            K=2,
            bins=8,
            mask_token=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            max_len=16,
            dropout=0.0,
            boundary_max_distance=4,
            retrieval_conditioning=True,
        )
        encdec_logits = encdec(x, seg_ids, left_idx, right_idx, retrieval_payload=payload)
        self.assertEqual(encdec_logits.shape, (1, 2, 6, 8))

    def test_trainer_retrieval_payload_truncates_longer_bucket_candidates(self):
        trainer = Trainer.__new__(Trainer)
        trainer.cfg = TrainConfig(
            retrieval_top_k=2,
            retrieval_candidate_stride_frames=2,
            retrieval_left_context_frames=3,
            retrieval_right_context_frames=3,
            retrieval_exclusion_margin_frames=1,
        )
        trainer.device = torch.device("cpu")
        trainer.retrieval_enabled = True
        trainer.K = 2

        long_tokens = torch.arange(0, 2 * 8, dtype=torch.long).reshape(2, 8)
        exact_tokens = torch.arange(100, 100 + 2 * 6, dtype=torch.long).reshape(2, 6)
        trainer.retrieval_cache = SimpleNamespace(
            query=lambda **kwargs: [
                SimpleNamespace(entry=SimpleNamespace(fill_tokens=long_tokens), similarity=0.9),
                SimpleNamespace(entry=SimpleNamespace(fill_tokens=exact_tokens), similarity=0.5),
            ]
        )

        payload, metrics = trainer._build_retrieval_payload(
            window_starts=[10],
            mask_starts=[4],
            mask_lens=[6],
        )

        self.assertIsNotNone(payload)
        self.assertEqual(tuple(payload["tokens"].shape), (1, 2, 2, 6))
        self.assertTrue(torch.equal(payload["tokens"][0, 0], long_tokens[:, :6]))
        self.assertTrue(torch.equal(payload["tokens"][0, 1], exact_tokens))
        self.assertEqual(int(payload["lengths"][0].item()), 6)
        self.assertTrue(bool(payload["candidate_mask"][0, 0].item()))
        self.assertTrue(bool(payload["candidate_mask"][0, 1].item()))
        self.assertGreater(metrics["retrieval_best_similarity"], 0.0)
        self.assertGreater(metrics["retrieval_used"], 0.0)

    def test_checkpoint_compat_allows_missing_retrieval_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = TrainConfig(output_dir=tmpdir, run_name="retrieval_ckpt", use_retrieval_conditioning=True)
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

            trainer = Trainer.__new__(Trainer)
            trainer.cfg = cfg
            trainer.device = torch.device("cpu")
            trainer.writer = None
            trainer.model = JointCodebookInfiller(
                K=2,
                bins=8,
                mask_token=8,
                d_model=16,
                n_heads=4,
                n_layers=1,
                max_len=16,
                dropout=0.0,
                boundary_max_distance=cfg.boundary_max_distance,
                retrieval_conditioning=True,
            )
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
            trainer.scaler = torch.amp.GradScaler(enabled=False)
            trainer.best_loss = 1.0
            trainer.best_val_loss = 0.5
            trainer.global_step = 7

            legacy_state = {
                key: value
                for key, value in trainer.model.state_dict().items()
                if not key.startswith("retrieval_")
            }
            ckpt_path = cfg.checkpoint_dir / "legacy.pt"
            torch.save(
                {
                    "step": 7,
                    "model": legacy_state,
                    "optimizer": trainer.optimizer.state_dict(),
                    "scaler": trainer.scaler.state_dict(),
                    "best_loss": trainer.best_loss,
                    "best_val_loss": trainer.best_val_loss,
                    "config": {"use_encoder_decoder": False},
                },
                ckpt_path,
            )

            reloaded = Trainer.__new__(Trainer)
            reloaded.cfg = cfg
            reloaded.device = torch.device("cpu")
            reloaded.writer = None
            reloaded.model = JointCodebookInfiller(
                K=2,
                bins=8,
                mask_token=8,
                d_model=16,
                n_heads=4,
                n_layers=1,
                max_len=16,
                dropout=0.0,
                boundary_max_distance=cfg.boundary_max_distance,
                retrieval_conditioning=True,
            )
            reloaded.optimizer = torch.optim.AdamW(reloaded.model.parameters(), lr=1e-3)
            reloaded.scaler = torch.amp.GradScaler(enabled=False)
            reloaded.load_checkpoint(str(ckpt_path))

            self.assertEqual(reloaded.global_step, 7)
            self.assertEqual(reloaded.best_val_loss, 0.5)


if __name__ == "__main__":
    unittest.main()
