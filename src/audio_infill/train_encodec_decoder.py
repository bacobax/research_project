#!/usr/bin/env python3
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audio_infill.encodec_decoder_config import EncodecDecoderTrainConfig, parse_args
from audio_infill.encodec_utils import build_encodec_model, codes_to_embeddings, decode_embeddings, export_decoder_artifact
from audio_infill.train import MultiResolutionSTFTLoss, save_waveform
from audio_infill.training_common import (
    log_hparams,
    load_training_checkpoint,
    resolve_device,
    restore_rng_state,
    save_training_checkpoint,
    set_seed,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("encodec_decoder")


@dataclass(frozen=True)
class DecoderWindowExample:
    embeddings: torch.Tensor
    target_audio: torch.Tensor
    scale: torch.Tensor
    start_sample: int


class DecoderWindowDataset(Dataset):
    def __init__(self, examples: Sequence[DecoderWindowExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        return example.embeddings, example.target_audio, example.scale, example.start_sample


def load_audio_mono(wav_path: str, target_sr: int) -> torch.Tensor:
    import librosa
    import soundfile as sf

    audio, sr = sf.read(wav_path, always_2d=True)
    audio = audio.astype(np.float32).mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    wav = torch.from_numpy(audio).unsqueeze(0)
    wav = wav / (wav.abs().max() + 1e-9)
    return wav


def build_window_starts(total_samples: int, window_samples: int, hop_samples: int) -> List[int]:
    if total_samples <= window_samples:
        return [0]
    starts = list(range(0, total_samples - window_samples + 1, hop_samples))
    final_start = total_samples - window_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def resolve_annotation_path(wav_path: str) -> Optional[Path]:
    path = Path(wav_path)
    json_path = path.with_suffix(".json")
    if json_path.exists():
        return json_path
    return None


def load_gap_sample_ranges(wav_path: str) -> List[Tuple[int, int]]:
    json_path = resolve_annotation_path(wav_path)
    if json_path is None:
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    ranges: List[Tuple[int, int]] = []
    if "gaps" in ann:
        for gap in ann["gaps"]:
            start = int(gap["gap_start_sample"])
            end = int(gap["gap_end_sample"])
            ranges.append((start, end))
    elif "gap" in ann:
        gap = ann["gap"]
        start = int(gap["gap_start_sample"])
        end = int(gap["gap_end_sample"])
        ranges.append((start, end))
    return ranges


def window_overlaps_ranges(start: int, length: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    end = start + length
    for range_start, range_end in ranges:
        if start < range_end and end > range_start:
            return True
    return False


def make_examples(
    wav: torch.Tensor,
    *,
    model: torch.nn.Module,
    device: torch.device,
    window_samples: int,
    hop_samples: int,
    blocked_sample_ranges: Optional[Sequence[Tuple[int, int]]] = None,
) -> List[DecoderWindowExample]:
    starts = build_window_starts(int(wav.shape[-1]), window_samples, hop_samples)
    blocked_sample_ranges = list(blocked_sample_ranges or [])
    examples: List[DecoderWindowExample] = []
    for start in starts:
        if blocked_sample_ranges and window_overlaps_ranges(start, window_samples, blocked_sample_ranges):
            continue
        window = wav[:, start : start + window_samples]
        if window.shape[-1] < window_samples:
            pad = window_samples - window.shape[-1]
            window = F.pad(window, (0, pad))
        with torch.no_grad():
            encoded = model.encode(window.unsqueeze(0).to(device))
            codes_b, scale = encoded[0]
            codes = codes_b[0].detach().cpu()
            embeddings = codes_to_embeddings(model, codes, device=device).squeeze(0).detach().cpu()
        if scale is None:
            scale_tensor = torch.ones(1, dtype=torch.float32)
        else:
            scale_tensor = torch.as_tensor(scale, dtype=torch.float32).detach().cpu().reshape(1)
        examples.append(
            DecoderWindowExample(
                embeddings=embeddings,
                target_audio=window.squeeze(0).detach().cpu(),
                scale=scale_tensor,
                start_sample=int(start),
            )
        )
    return examples


class DecoderFinetuneTrainer:
    def __init__(self, cfg: EncodecDecoderTrainConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        set_seed(cfg.seed)
        logger.info("Device: %s", self.device)

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg.tb_dir.mkdir(parents=True, exist_ok=True)
        cfg.samples_dir.mkdir(parents=True, exist_ok=True)
        cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(cfg.tb_dir))
        log_hparams(logger, self.writer, cfg, prefix="=== Decoder Fine-Tune Hyperparameters ===")

        self.window_samples = int(round(cfg.window_seconds * cfg.target_sr))
        self.hop_samples = int(round(cfg.hop_seconds * cfg.target_sr))
        if self.window_samples <= 0:
            raise ValueError("window_seconds produced a non-positive window size")
        if self.hop_samples <= 0:
            raise ValueError("hop_seconds produced a non-positive hop size")

        self._build_model_and_data()
        self._build_optimizer()

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_train_loss = float("inf")
        self.validation_checks_since_improvement = 0
        self._logged_length_mismatch = False

    def _build_model_and_data(self) -> None:
        cfg = self.cfg
        self.encodec = build_encodec_model(cfg.encodec_model).to(self.device)
        self.encodec.set_target_bandwidth(cfg.bandwidth)
        self.encodec.eval()
        self.encodec.requires_grad_(False)
        self.encodec.decoder.requires_grad_(True)

        self.wav = load_audio_mono(cfg.wav_path, cfg.target_sr)
        logger.info("Loaded audio: %s | duration=%.2fs", cfg.wav_path, self.wav.shape[-1] / cfg.target_sr)
        self.blocked_sample_ranges = load_gap_sample_ranges(cfg.wav_path)
        if self.blocked_sample_ranges:
            blocked_total = sum(end - start for start, end in self.blocked_sample_ranges)
            logger.info(
                "Excluding decoder windows overlapping %d gap region(s), blocked_samples=%d",
                len(self.blocked_sample_ranges),
                blocked_total,
            )
        else:
            logger.info("No gap annotations found for decoder fine-tuning wav; using all windows")

        all_examples = make_examples(
            self.wav,
            model=self.encodec,
            device=self.device,
            window_samples=self.window_samples,
            hop_samples=self.hop_samples,
            blocked_sample_ranges=self.blocked_sample_ranges,
        )
        if not all_examples:
            raise ValueError(
                "No decoder fine-tuning windows were generated after excluding gap-overlapping regions"
            )

        holdout_count = len(all_examples) // 10
        if cfg.validation_every > 0 and holdout_count > 0 and len(all_examples) - holdout_count > 0:
            self.validation_examples = all_examples[-holdout_count:]
            self.train_examples = all_examples[:-holdout_count]
            self.validation_enabled = True
            logger.info(
                "Decoder windows: total=%d train=%d val=%d",
                len(all_examples),
                len(self.train_examples),
                len(self.validation_examples),
            )
        else:
            self.train_examples = all_examples
            self.validation_examples = []
            self.validation_enabled = False
            if cfg.validation_every > 0:
                logger.info(
                    "Validation requested but skipped: total_windows=%d does not leave a usable 10%% holdout split",
                    len(all_examples),
                )

        self.train_loader = DataLoader(
            DecoderWindowDataset(self.train_examples),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )
        self.val_loader = None
        if self.validation_enabled:
            self.val_loader = DataLoader(
                DecoderWindowDataset(self.validation_examples),
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
                drop_last=False,
            )

        self.stft_loss = MultiResolutionSTFTLoss(
            n_ffts=cfg.stft_n_ffts,
            hop_lengths=cfg.stft_hop_lengths,
            win_lengths=cfg.stft_win_lengths,
            spectral_convergence_weight=1.0,
            log_magnitude_weight=1.0,
        )

    def _build_optimizer(self) -> None:
        cfg = self.cfg
        self.optimizer = torch.optim.AdamW(
            self.encodec.decoder.parameters(),
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

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _compute_loss(
        self,
        embeddings: torch.Tensor,
        target_audio: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        pred_audio = decode_embeddings(self.encodec.decoder, embeddings, device=self.device, scale=scale).squeeze(1)
        pred_audio, target_audio = self._align_audio_pair(pred_audio, target_audio)

        total = pred_audio.new_zeros(())
        waveform_l1 = pred_audio.new_zeros(())
        waveform_l2 = pred_audio.new_zeros(())
        stft_total = pred_audio.new_zeros(())
        stft_sc = pred_audio.new_zeros(())
        stft_log_mag = pred_audio.new_zeros(())

        if self.cfg.waveform_l1_weight > 0:
            waveform_l1 = F.l1_loss(pred_audio, target_audio)
            total = total + self.cfg.waveform_l1_weight * waveform_l1
        if self.cfg.waveform_l2_weight > 0:
            waveform_l2 = F.mse_loss(pred_audio, target_audio)
            total = total + self.cfg.waveform_l2_weight * waveform_l2
        if self.cfg.stft_weight > 0:
            stft_terms = self.stft_loss(pred_audio, target_audio)
            stft_total = stft_terms["total"]
            stft_sc = stft_terms["spectral_convergence"]
            stft_log_mag = stft_terms["log_magnitude"]
            total = total + self.cfg.stft_weight * stft_total

        metrics = {
            "loss": float(total.detach().item()),
            "waveform_l1": float(waveform_l1.detach().item()),
            "waveform_l2": float(waveform_l2.detach().item()),
            "stft": float(stft_total.detach().item()),
            "stft_spectral_convergence": float(stft_sc.detach().item()),
            "stft_log_magnitude": float(stft_log_mag.detach().item()),
        }
        return total, metrics, pred_audio

    def _decoder_export_path(self) -> Path:
        if self.cfg.decoder_export_path:
            return Path(self.cfg.decoder_export_path)
        return self.cfg.artifacts_dir / "decoder.pt"

    def export_decoder(self, tag: str) -> Path:
        base_path = self._decoder_export_path()
        if tag == "latest":
            export_path = base_path
        else:
            export_path = base_path.with_name(f"{base_path.stem}_{tag}{base_path.suffix}")
        export_decoder_artifact(
            str(export_path),
            decoder_state_dict=self.encodec.decoder.state_dict(),
            encodec_model=self.cfg.encodec_model,
            bandwidth=self.cfg.bandwidth,
            target_sr=self.cfg.target_sr,
            step=self.global_step,
            run_name=self.cfg.run_name,
            config_path=self.cfg.config,
        )
        logger.info("Exported decoder artifact: %s", export_path)
        return export_path

    def save_checkpoint(self, tag: str = "latest") -> None:
        path = self.cfg.checkpoint_dir / f"{tag}.pt"
        save_training_checkpoint(
            path,
            step=self.global_step,
            model_key="decoder",
            model_state=self.encodec.decoder.state_dict(),
            optimizer=self.optimizer,
            scaler=self.scaler,
            cfg=self.cfg,
            extra={
                "best_val_loss": self.best_val_loss,
                "best_train_loss": self.best_train_loss,
                "validation_checks_since_improvement": self.validation_checks_since_improvement,
            },
        )
        logger.info("Saved decoder checkpoint: %s (step %d)", path, self.global_step)
        self.export_decoder("latest")

    def load_checkpoint(self, path: str) -> None:
        ckpt = load_training_checkpoint(path, self.device)
        self.encodec.decoder.load_state_dict(ckpt["decoder"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = int(ckpt["step"])
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        self.best_train_loss = float(ckpt.get("best_train_loss", float("inf")))
        self.validation_checks_since_improvement = int(ckpt.get("validation_checks_since_improvement", 0))
        restore_rng_state(ckpt.get("rng_state"))
        logger.info("Loaded decoder checkpoint: %s (step %d)", path, self.global_step)

    def _align_audio_pair(
        self,
        pred_audio: torch.Tensor,
        target_audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred_audio.shape[-1] == target_audio.shape[-1]:
            return pred_audio, target_audio
        common = min(int(pred_audio.shape[-1]), int(target_audio.shape[-1]))
        if common <= 0:
            raise ValueError(
                f"Cannot align decoder audio with non-positive shared length: "
                f"pred={pred_audio.shape[-1]}, target={target_audio.shape[-1]}"
            )
        if not self._logged_length_mismatch:
            logger.warning(
                "Aligning decoder audio lengths before loss: pred_samples=%d target_samples=%d common=%d",
                pred_audio.shape[-1],
                target_audio.shape[-1],
                common,
            )
            self._logged_length_mismatch = True
        return pred_audio[..., :common], target_audio[..., :common]

    def _should_save_validation_audio(self, step: int) -> bool:
        freq = int(self.cfg.validation_save_audio_every)
        return freq > 0 and step % freq == 0

    @torch.no_grad()
    def run_validation(self, step: int) -> Optional[Tuple[Dict[str, float], bool]]:
        if not self.validation_enabled or self.val_loader is None:
            return None

        self.encodec.decoder.eval()
        use_amp = self.device.type == "cuda"
        totals = {
            "loss": 0.0,
            "waveform_l1": 0.0,
            "waveform_l2": 0.0,
            "stft": 0.0,
            "stft_spectral_convergence": 0.0,
            "stft_log_magnitude": 0.0,
        }
        count = 0
        saved_artifact = False
        save_audio = self._should_save_validation_audio(step)
        step_dir = None
        if save_audio:
            step_dir = self.cfg.samples_dir / "validation" / f"step_{step}"
            step_dir.mkdir(parents=True, exist_ok=True)

        for embeddings, target_audio, scale, start_sample in self.val_loader:
            embeddings = embeddings.to(self.device, non_blocking=True)
            target_audio = target_audio.to(self.device, non_blocking=True)
            scale = scale.to(self.device, non_blocking=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                _, metrics, pred_audio = self._compute_loss(embeddings, target_audio, scale)
            batch_size = int(embeddings.shape[0])
            for key in totals:
                totals[key] += metrics[key] * batch_size
            count += batch_size

            if save_audio and not saved_artifact and step_dir is not None:
                target_np = target_audio[0].detach().cpu().numpy().astype(np.float32, copy=False)
                pred_np = pred_audio[0].detach().cpu().numpy().astype(np.float32, copy=False)
                save_waveform(step_dir / "target.wav", target_np, self.cfg.target_sr)
                save_waveform(step_dir / "pred.wav", pred_np, self.cfg.target_sr)
                with open(step_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "step": int(step),
                            "start_sample": int(start_sample[0]),
                            "target_sr": int(self.cfg.target_sr),
                        },
                        f,
                        indent=2,
                    )
                saved_artifact = True

        avg = {key: value / max(1, count) for key, value in totals.items()}
        for key, value in avg.items():
            self.writer.add_scalar(f"val/{key}", value, step)

        improved = avg["loss"] < self.best_val_loss
        if improved:
            self.best_val_loss = avg["loss"]
            self.validation_checks_since_improvement = 0
            self.save_checkpoint("best_val")
            self.export_decoder("best")
        else:
            self.validation_checks_since_improvement += 1

        self.encodec.decoder.train()
        return avg, improved

    def train(self) -> None:
        cfg = self.cfg
        self.encodec.decoder.train()
        use_amp = self.device.type == "cuda"
        data_iter = iter(self.train_loader)
        running = {"loss": 0.0, "waveform_l1": 0.0, "waveform_l2": 0.0, "stft": 0.0}
        running_count = 0
        t0 = time.time()

        logger.info("Starting EnCodec decoder fine-tuning for %d steps", cfg.total_steps)
        pbar = tqdm(
            range(self.global_step + 1, cfg.total_steps + 1),
            desc="Decoder FT",
            initial=self.global_step,
            total=cfg.total_steps,
            unit="step",
        )
        for step in pbar:
            self.global_step = step
            try:
                embeddings, target_audio, scale, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                embeddings, target_audio, scale, _ = next(data_iter)

            embeddings = embeddings.to(self.device, non_blocking=True)
            target_audio = target_audio.to(self.device, non_blocking=True)
            scale = scale.to(self.device, non_blocking=True)

            lr = self._get_lr(step)
            self._set_lr(lr)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                total_loss, metrics, _ = self._compute_loss(embeddings, target_audio, scale)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.encodec.decoder.parameters(), cfg.grad_clip).item()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key in running:
                running[key] += metrics[key]
            running_count += 1

            self.writer.add_scalar("train/loss", metrics["loss"], step)
            self.writer.add_scalar("train/waveform_l1", metrics["waveform_l1"], step)
            self.writer.add_scalar("train/waveform_l2", metrics["waveform_l2"], step)
            self.writer.add_scalar("train/stft", metrics["stft"], step)
            self.writer.add_scalar("train/stft_spectral_convergence", metrics["stft_spectral_convergence"], step)
            self.writer.add_scalar("train/stft_log_magnitude", metrics["stft_log_magnitude"], step)
            self.writer.add_scalar("train/lr", lr, step)
            self.writer.add_scalar("train/grad_norm", grad_norm, step)

            pbar.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{lr:.2e}")

            if metrics["loss"] < self.best_train_loss:
                self.best_train_loss = metrics["loss"]

            if step % cfg.log_every == 0:
                dt = time.time() - t0
                steps_per_sec = running_count / max(dt, 1e-9)
                self.writer.add_scalar("train/avg_loss", running["loss"] / max(1, running_count), step)
                self.writer.add_scalar("train/avg_waveform_l1", running["waveform_l1"] / max(1, running_count), step)
                self.writer.add_scalar("train/avg_waveform_l2", running["waveform_l2"] / max(1, running_count), step)
                self.writer.add_scalar("train/avg_stft", running["stft"] / max(1, running_count), step)
                self.writer.add_scalar("train/steps_per_sec", steps_per_sec, step)
                running = {key: 0.0 for key in running}
                running_count = 0
                t0 = time.time()

            if self.validation_enabled and cfg.validation_every > 0 and step % cfg.validation_every == 0:
                validation_result = self.run_validation(step)
                if validation_result is not None:
                    val_metrics, improved = validation_result
                    if cfg.patience > 0 and not improved and self.validation_checks_since_improvement >= cfg.patience:
                        logger.info(
                            "Early stopping decoder fine-tuning at step %d after %d validation(s) without total loss improvement; best_val_loss=%.6f current_val_loss=%.6f",
                            step,
                            self.validation_checks_since_improvement,
                            self.best_val_loss,
                            val_metrics["loss"],
                        )
                        break

            if step % cfg.save_every == 0:
                self.save_checkpoint("latest")
                self.save_checkpoint(f"step_{step}")

        self.save_checkpoint("final")
        self.export_decoder("final")
        self.writer.flush()


def main() -> None:
    cfg, _ = parse_args()
    trainer = DecoderFinetuneTrainer(cfg)

    if cfg.resume:
        trainer.load_checkpoint(cfg.resume)

    trainer.train()


if __name__ == "__main__":
    main()
