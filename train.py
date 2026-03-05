#!/usr/bin/env python3
import os
import sys
import time
import math
import random
import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("infiller")


@dataclass
class TrainConfig:
    wav_path: str = "gapped_audio.wav"
    target_sr: int = 24000
    bandwidth: float = 6.0
    gap_start_s: float = 200.0
    gap_end_s: float = 210.0

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    max_len: int = 2048
    dropout: float = 0.1

    seq_len: int = 1024
    mask_len_min: int = 64
    mask_len_max: int = 256

    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 200
    total_steps: int = 3000
    log_every: int = 100
    save_every: int = 500
    test_fill_every: int = 0
    num_workers: int = 2

    output_dir: str = "runs"
    run_name: str = "infiller"
    seed: int = 42
    device: str = "auto"

    inpaint_iters: int = 10

    ctx_left: Optional[int] = None
    ctx_right: Optional[int] = None

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name / "checkpoints"

    @property
    def tb_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name / "tb"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg_device: str) -> torch.device:
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


class AudioEncoder:
    def __init__(self, bandwidth: float, device: torch.device):
        from encodec import EncodecModel

        self.device = device
        self.model = EncodecModel.encodec_model_24khz().to(device).eval()
        self.model.set_target_bandwidth(bandwidth)
        self.bins = int(self.model.quantizer.bins)
        self.mask_token = self.bins

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> Tuple[torch.Tensor, object]:
        x = wav.unsqueeze(0).to(self.device)
        encoded = self.model.encode(x)
        codes_b, scale = encoded[0]
        codes = codes_b[0].detach().cpu()
        return codes, scale

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, scale) -> np.ndarray:
        codes_b = codes.unsqueeze(0).to(self.device)
        wav_out = self.model.decode([(codes_b, scale)])
        return wav_out.squeeze(0).cpu().squeeze(0).numpy()


class MaskedSpanDataset(Dataset):
    def __init__(
        self,
        codes: torch.Tensor,
        gap_f0: int,
        gap_f1: int,
        seq_len: int = 1024,
        mask_len_range: Tuple[int, int] = (64, 256),
        mask_token: int = 1024,
        virtual_size: int = 50_000,
    ):
        self.codes = codes
        self.K, self.F = codes.shape
        self.g0 = gap_f0
        self.g1 = gap_f1
        self.seq_len = seq_len
        self.mask_len_range = mask_len_range
        self.mask_token = mask_token
        self.virtual_size = virtual_size

        self.starts: List[int] = []
        max_s = self.F - self.seq_len
        for s in range(max_s):
            if (s + self.seq_len) <= self.g0 or s >= self.g1:
                self.starts.append(s)

        if len(self.starts) == 0:
            raise ValueError("No valid windows found. Reduce seq_len or check gap boundaries.")

        logger.info("Dataset: K=%d, F=%d, valid_starts=%d", self.K, self.F, len(self.starts))

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, idx: int):
        s = random.choice(self.starts)
        x = self.codes[:, s : s + self.seq_len].clone()
        y = x.clone()

        mask_len = random.randint(*self.mask_len_range)
        mask_len = min(mask_len, self.seq_len)
        m0 = random.randint(0, self.seq_len - mask_len)
        m1 = m0 + mask_len

        x[:, m0:m1] = self.mask_token

        loss_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        loss_mask[m0:m1] = True

        return x, y, loss_mask


class JointCodebookInfiller(nn.Module):
    def __init__(
        self,
        K: int,
        bins: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        max_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.K = K
        self.bins = bins
        self.vocab = bins + 1

        self.emb = nn.ModuleList([nn.Embedding(self.vocab, d_model) for _ in range(K)])
        self.pos = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.ModuleList([nn.Linear(d_model, bins) for _ in range(K)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, K, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.pos(pos)
        for k in range(K):
            h = h + self.emb[k](x[:, k, :])
        h = self.enc(h)
        logits = torch.stack([self.head[k](h) for k in range(K)], dim=1)
        return logits


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        set_seed(cfg.seed)
        logger.info("Device: %s", self.device)

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg.tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(cfg.tb_dir))

        self._log_hparams()
        self._load_audio()
        self._build_dataset()
        self._build_model()
        self._build_optimizer()

        self.global_step = 0
        self.best_loss = float("inf")

    def _log_hparams(self):
        cfg = self.cfg
        hparams = {k: v for k, v in vars(cfg).items() if not k.startswith("_") and isinstance(v, (int, float, str, bool))}
        hparams_safe = {}
        for k, v in hparams.items():
            if v is None:
                continue
            hparams_safe[k] = v
        logger.info("=== Hyperparameters ===")
        for k, v in sorted(hparams_safe.items()):
            logger.info("  %-20s = %s", k, v)
        logger.info("=======================")
        self.writer.add_text("hparams", "\n".join(f"{k} = {v}" for k, v in sorted(hparams_safe.items())), 0)
        self.writer.add_hparams(
            hparams_safe,
            {"hparam/placeholder": 0.0},
            run_name=".",
        )

    def _load_audio(self):
        import soundfile as sf
        import librosa

        cfg = self.cfg
        logger.info("Loading audio: %s", cfg.wav_path)

        audio, sr = sf.read(cfg.wav_path, always_2d=True)
        audio = audio.astype(np.float32).mean(axis=1)
        if sr != cfg.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.target_sr)
        wav = torch.from_numpy(audio).unsqueeze(0)
        wav = wav / (wav.abs().max() + 1e-9)

        self.wav = wav
        self.duration_s = wav.shape[-1] / cfg.target_sr
        logger.info("Audio: %.1fs @ %dHz, samples=%d", self.duration_s, cfg.target_sr, wav.shape[-1])

        self.encoder = AudioEncoder(cfg.bandwidth, self.device)
        self.codes, self.scale = self.encoder.encode(wav)
        self.K, self.frames = self.codes.shape
        self.bins = self.encoder.bins
        self.mask_token = self.encoder.mask_token

        fps = self.frames / self.duration_s
        self.gap_f0 = max(0, min(self.frames, int(round(cfg.gap_start_s * fps))))
        self.gap_f1 = max(0, min(self.frames, int(round(cfg.gap_end_s * fps))))
        logger.info("Codes: K=%d, F=%d, bins=%d", self.K, self.frames, self.bins)
        logger.info("Gap frames: [%d, %d) = %.1fs", self.gap_f0, self.gap_f1, (self.gap_f1 - self.gap_f0) / fps)

    def _build_dataset(self):
        cfg = self.cfg
        self.dataset = MaskedSpanDataset(
            codes=self.codes,
            gap_f0=self.gap_f0,
            gap_f1=self.gap_f1,
            seq_len=cfg.seq_len,
            mask_len_range=(cfg.mask_len_min, cfg.mask_len_max),
            mask_token=self.mask_token,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )

    def _build_model(self):
        cfg = self.cfg
        self.model = JointCodebookInfiller(
            K=self.K,
            bins=self.bins,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            max_len=cfg.max_len,
            dropout=cfg.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model params: %.2fM", n_params / 1e6)
        self.writer.add_scalar("model/params_M", n_params / 1e6, 0)

    def _build_optimizer(self):
        cfg = self.cfg
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

    def _get_lr(self, step: int) -> float:
        cfg = self.cfg
        if step < cfg.warmup_steps:
            return cfg.lr * step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        B, K, T, V = logits.shape
        logits_flat = logits.permute(0, 2, 1, 3).reshape(B * T * K, V)
        y_flat = y.permute(0, 2, 1).reshape(B * T * K)
        mask_bt = loss_mask.reshape(B * T)
        mask_btk = mask_bt.repeat_interleave(K)
        return F.cross_entropy(logits_flat[mask_btk], y_flat[mask_btk])

    def save_checkpoint(self, tag: str = "latest"):
        path = self.cfg.checkpoint_dir / f"{tag}.pt"
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        torch.save(
            {
                "step": self.global_step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "best_loss": self.best_loss,
                "config": vars(self.cfg),
                "rng_state": rng_state,
            },
            path,
        )
        logger.info("Saved checkpoint: %s (step %d)", path, self.global_step)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["step"]
        self.best_loss = ckpt.get("best_loss", float("inf"))
        if "rng_state" in ckpt:
            rng = ckpt["rng_state"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.random.set_rng_state(rng["torch_cpu"])
            if torch.cuda.is_available() and "torch_cuda" in rng:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])
        logger.info("Loaded checkpoint: %s (step %d)", path, self.global_step)

    def train(self):
        cfg = self.cfg
        self.model.train()
        use_amp = self.device.type == "cuda"

        running_loss = 0.0
        running_acc = 0.0
        running_acc5 = 0.0
        running_ppl = 0.0
        running_nll = 0.0
        running_count = 0
        t0 = time.time()
        data_iter = iter(self.dataloader)

        logger.info("Starting training for %d steps", cfg.total_steps)

        pbar = tqdm(
            range(self.global_step + 1, cfg.total_steps + 1),
            desc="Training",
            initial=self.global_step,
            total=cfg.total_steps,
            unit="step",
        )
        for step in pbar:
            self.global_step = step

            try:
                x, y, loss_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                x, y, loss_mask = next(data_iter)

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            loss_mask = loss_mask.to(self.device, non_blocking=True).bool()

            lr = self._get_lr(step)
            self._set_lr(lr)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                logits = self.model(x)
                loss = self._compute_loss(logits, y, loss_mask)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip).item()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                B, K, T, V = logits.shape
                if loss_mask.dim() == 1:
                    lm = loss_mask.unsqueeze(0).expand(B, -1)
                else:
                    lm = loss_mask
                mask_exp = lm.unsqueeze(1).expand(B, K, T)

                preds = logits.argmax(dim=-1)
                n_masked = mask_exp.sum().item()
                if n_masked > 0:
                    correct = (preds[mask_exp] == y[mask_exp]).float().mean().item()
                    top5 = logits.topk(5, dim=-1).indices
                    hits5 = (top5 == y.unsqueeze(-1)).any(dim=-1)
                    acc5 = hits5[mask_exp].float().mean().item()
                else:
                    correct = 0.0
                    acc5 = 0.0

            loss_val = loss.item()
            nll = loss_val
            ppl = math.exp(min(loss_val, 20.0))

            running_loss += loss_val
            running_acc += correct
            running_acc5 += acc5
            running_ppl += ppl
            running_nll += nll
            running_count += 1

            self.writer.add_scalar("train/loss", loss_val, step)
            self.writer.add_scalar("train/acc_top1", correct, step)
            self.writer.add_scalar("train/acc_top5", acc5, step)
            self.writer.add_scalar("train/ppl", ppl, step)
            self.writer.add_scalar("train/nll", nll, step)
            self.writer.add_scalar("train/lr", lr, step)
            self.writer.add_scalar("train/grad_norm", grad_norm, step)

            pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{correct:.3f}", ppl=f"{ppl:.1f}", lr=f"{lr:.2e}")

            if step % cfg.log_every == 0:
                avg_loss = running_loss / running_count
                avg_acc = running_acc / running_count
                avg_acc5 = running_acc5 / running_count
                avg_ppl = running_ppl / running_count
                avg_nll = running_nll / running_count
                dt = time.time() - t0
                steps_per_sec = running_count / dt
                logger.info(
                    "step %5d | loss %.4f | acc1 %.3f | acc5 %.3f | ppl %.1f | lr %.2e | grad_norm %.2f | %.1f steps/s",
                    step, avg_loss, avg_acc, avg_acc5, avg_ppl, lr, grad_norm, steps_per_sec,
                )
                self.writer.add_scalar("train/avg_loss", avg_loss, step)
                self.writer.add_scalar("train/avg_accuracy", avg_acc, step)
                self.writer.add_scalar("train/avg_acc_top1", avg_acc, step)
                self.writer.add_scalar("train/avg_acc_top5", avg_acc5, step)
                self.writer.add_scalar("train/avg_ppl", avg_ppl, step)
                self.writer.add_scalar("train/avg_nll", avg_nll, step)
                self.writer.add_scalar("train/steps_per_sec", steps_per_sec, step)

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint("best")

                running_loss = 0.0
                running_acc = 0.0
                running_acc5 = 0.0
                running_ppl = 0.0
                running_nll = 0.0
                running_count = 0
                t0 = time.time()

            if step % cfg.save_every == 0:
                self.save_checkpoint("latest")
                self.save_checkpoint(f"step_{step}")

            if cfg.test_fill_every > 0 and step % cfg.test_fill_every == 0:
                logger.info("Running test inpaint at step %d", step)
                wav_filled = self.inpaint()
                self.log_spectrograms(wav_filled)
                self.model.train()

        self.save_checkpoint("final")
        self.writer.close()
        logger.info("Training complete. Best loss: %.4f", self.best_loss)

    @torch.no_grad()
    def inpaint(self, output_path: Optional[str] = None) -> np.ndarray:
        cfg = self.cfg
        self.model.eval()

        gap_len = self.gap_f1 - self.gap_f0
        if cfg.ctx_left is not None and cfg.ctx_right is not None:
            ctx_left = cfg.ctx_left
            ctx_right = cfg.ctx_right
        else:
            budget = max(0, cfg.max_len - gap_len - 16)
            ctx_left = budget // 2
            ctx_right = budget - ctx_left

        L = max(0, self.gap_f0 - ctx_left)
        R = min(self.frames, self.gap_f1 + ctx_right)
        g0 = self.gap_f0 - L
        g1 = self.gap_f1 - L

        x = self.codes[:, L:R].clone()
        x[:, g0:g1] = self.mask_token
        xb = x.unsqueeze(0).to(self.device)

        for i in range(cfg.inpaint_iters):
            logits = self.model(xb)[0]
            pred = logits.argmax(dim=-1)
            xb[0, :, g0:g1] = pred[:, g0:g1]

        codes_filled = self.codes.clone()
        codes_filled[:, L:R] = xb[0].cpu()

        wav_filled = self.encoder.decode(codes_filled, self.scale)

        if output_path:
            import soundfile as sf

            sf.write(output_path, wav_filled, cfg.target_sr)
            logger.info("Saved inpainted audio: %s", output_path)

        return wav_filled

    def _make_spectrogram_figure(self, audio: np.ndarray, title: str, sr: int, n_fft: int = 2048, hop: int = 512):
        y = torch.tensor(audio, dtype=torch.float32)
        S = torch.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                        window=torch.hann_window(n_fft), return_complex=True)
        S_db = 20.0 * np.log10(np.abs(S.numpy()) + 1e-7)
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        im = ax.imshow(S_db, origin="lower", aspect="auto",
                        extent=[0, audio.shape[-1] / sr, 0, sr / 2])
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="dB")
        fig.tight_layout()
        return fig

    def log_spectrograms(self, wav_filled: np.ndarray):
        sr = self.cfg.target_sr
        wav_orig = self.wav.squeeze(0).numpy()

        fig_orig = self._make_spectrogram_figure(wav_orig, "Original (gapped)", sr)
        self.writer.add_figure("audio/spectrogram_original_gapped", fig_orig, self.global_step)
        plt.close(fig_orig)

        fig_recon = self._make_spectrogram_figure(wav_filled, "Reconstructed (infilled)", sr)
        self.writer.add_figure("audio/spectrogram_reconstructed", fig_recon, self.global_step)
        plt.close(fig_recon)

        self.writer.flush()
        logger.info("Logged spectrograms to TensorBoard")


def load_annotation(ds_dir: str, sample: str) -> dict:
    json_path = Path(ds_dir) / f"{sample}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def apply_auto_hparams(cfg: TrainConfig, ann: dict):
    gap = ann["gap"]
    cfg.gap_start_s = gap["gap_start_s"]
    cfg.gap_end_s = gap["gap_end_s"]
    cfg.target_sr = ann["sr"]

    rec = ann["recommendations"]["token_based"]
    cfg.seq_len = rec["seq_len_frames"]
    cfg.mask_len_min = rec["mask_len_min_frames"]
    cfg.mask_len_max = rec["mask_len_max_frames"]
    cfg.max_len = max(cfg.max_len, rec["max_len_frames_required"])
    cfg.ctx_left = rec["ctx_left_frames"]
    cfg.ctx_right = rec["ctx_right_frames"]

    if "encodec_stats_full_audio" in ann:
        stats = ann["encodec_stats_full_audio"]
        cfg.bandwidth = stats["bandwidth_kbps"]

    logger.info("Auto-hparams applied from annotation: seq_len=%d, mask=[%d,%d], max_len=%d, ctx=%d/%d",
                cfg.seq_len, cfg.mask_len_min, cfg.mask_len_max, cfg.max_len, cfg.ctx_left, cfg.ctx_right)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Audio Infiller Training")
    cfg = TrainConfig()

    parser.add_argument("--ds-dir", type=str, default="ds", help="Dataset directory containing .wav and .json pairs")
    parser.add_argument("--sample", type=str, default=None, help="Sample name (stem) within ds-dir, e.g. wav_test_gap_5p000s")
    parser.add_argument("--auto-hparam", action="store_true", help="Use recommended hparams from the annotation JSON")

    parser.add_argument("--wav-path", type=str, default=None, help="Direct wav path (overrides --ds-dir/--sample)")
    parser.add_argument("--target-sr", type=int, default=None)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--gap-start-s", type=float, default=None)
    parser.add_argument("--gap-end-s", type=float, default=None)

    parser.add_argument("--d-model", type=int, default=cfg.d_model)
    parser.add_argument("--n-heads", type=int, default=cfg.n_heads)
    parser.add_argument("--n-layers", type=int, default=cfg.n_layers)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)

    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--mask-len-min", type=int, default=None)
    parser.add_argument("--mask-len-max", type=int, default=None)
    parser.add_argument("--ctx-left", type=int, default=None)
    parser.add_argument("--ctx-right", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=cfg.grad_clip)
    parser.add_argument("--warmup-steps", type=int, default=cfg.warmup_steps)
    parser.add_argument("--total-steps", type=int, default=cfg.total_steps)
    parser.add_argument("--log-every", type=int, default=cfg.log_every)
    parser.add_argument("--save-every", type=int, default=cfg.save_every)
    parser.add_argument("--test-fill-every", type=int, default=cfg.test_fill_every, help="Run inpaint + log spectrogram every N steps (0=disabled)")
    parser.add_argument("--num-workers", type=int, default=cfg.num_workers)

    parser.add_argument("--output-dir", type=str, default=cfg.output_dir)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--device", type=str, default=cfg.device)

    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--inpaint-only", action="store_true", help="Skip training, only run inpainting")
    parser.add_argument("--inpaint-iters", type=int, default=cfg.inpaint_iters)
    parser.add_argument("--inpaint-output", type=str, default=None)

    args = parser.parse_args()

    ann = None
    if args.sample:
        wav_file = Path(args.ds_dir) / f"{args.sample}.wav"
        if not wav_file.exists():
            parser.error(f"Sample wav not found: {wav_file}")
        cfg.wav_path = str(wav_file)
        ann = load_annotation(args.ds_dir, args.sample)
        cfg.gap_start_s = ann["gap"]["gap_start_s"]
        cfg.gap_end_s = ann["gap"]["gap_end_s"]
        cfg.target_sr = ann["sr"]
        if args.auto_hparam:
            apply_auto_hparams(cfg, ann)
    elif args.wav_path:
        cfg.wav_path = args.wav_path

    if args.target_sr is not None:
        cfg.target_sr = args.target_sr
    if args.bandwidth is not None:
        cfg.bandwidth = args.bandwidth
    if args.gap_start_s is not None:
        cfg.gap_start_s = args.gap_start_s
    if args.gap_end_s is not None:
        cfg.gap_end_s = args.gap_end_s
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.mask_len_min is not None:
        cfg.mask_len_min = args.mask_len_min
    if args.mask_len_max is not None:
        cfg.mask_len_max = args.mask_len_max
    if args.max_len is not None:
        cfg.max_len = args.max_len
    if args.ctx_left is not None:
        cfg.ctx_left = args.ctx_left
    if args.ctx_right is not None:
        cfg.ctx_right = args.ctx_right

    cfg.d_model = args.d_model
    cfg.n_heads = args.n_heads
    cfg.n_layers = args.n_layers
    cfg.dropout = args.dropout
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr
    cfg.weight_decay = args.weight_decay
    cfg.grad_clip = args.grad_clip
    cfg.warmup_steps = args.warmup_steps
    cfg.total_steps = args.total_steps
    cfg.log_every = args.log_every
    cfg.save_every = args.save_every
    cfg.test_fill_every = args.test_fill_every
    cfg.num_workers = args.num_workers
    cfg.output_dir = args.output_dir
    cfg.run_name = args.run_name or (args.sample if args.sample else cfg.run_name)
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.inpaint_iters = args.inpaint_iters

    return cfg, args


def main():
    cfg, args = parse_args()
    trainer = Trainer(cfg)

    if args.resume:
        trainer.load_checkpoint(args.resume)

    if not args.inpaint_only:
        trainer.train()

    out = args.inpaint_output or os.path.splitext(cfg.wav_path)[0] + "_infilled.wav"
    wav_filled = trainer.inpaint(output_path=out)
    trainer.log_spectrograms(wav_filled)


if __name__ == "__main__":
    main()
