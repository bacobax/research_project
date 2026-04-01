#!/usr/bin/env python3
import os
import time
import math
import random
import json
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Sequence

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

from audio_infill.config import TrainConfig, parse_args
from audio_infill.encodec_utils import (
    build_encodec_model,
    codes_to_embeddings,
    decode_embeddings,
    load_exported_decoder,
    logits_to_embeddings,
)
from audio_infill.retrieval import SameSongRetrievalCache, build_retrieval_feature_config
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
logger = logging.getLogger("infiller")


def normalize_robust(values: np.ndarray, low_q: float = 0.05, high_q: float = 0.95) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    q0 = float(np.quantile(arr, low_q))
    q1 = float(np.quantile(arr, high_q))
    if q1 <= q0 + 1e-8:
        # Fallback to min/max if quantiles collapse on near-constant content.
        a0 = float(arr.min())
        a1 = float(arr.max())
        if a1 <= a0 + 1e-8:
            return np.zeros_like(arr)
        return np.clip((arr - a0) / (a1 - a0), 0.0, 1.0)
    return np.clip((arr - q0) / (q1 - q0), 0.0, 1.0)


def smooth_1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    k = int(max(1, kernel_size))
    if k <= 1 or arr.size <= 1:
        return arr.copy()
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / float(k)
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def compute_rms_per_frame(wav: torch.Tensor, frames: int) -> np.ndarray:
    samples = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if frames <= 1 or samples.size <= 1:
        return np.zeros(max(1, frames), dtype=np.float32)
    samples_per_frame = max(1, samples.size / float(frames))
    rms = np.zeros(frames, dtype=np.float32)
    for i in range(frames):
        s0 = int(round(i * samples_per_frame))
        s1 = int(round((i + 1) * samples_per_frame))
        if s1 <= s0:
            s1 = min(samples.size, s0 + 1)
        segment = samples[s0:s1]
        if segment.size == 0:
            rms[i] = 0.0
        else:
            rms[i] = float(np.sqrt(np.mean(segment * segment) + 1e-12))
    return rms


def compute_token_change_per_frame(codes: torch.Tensor) -> np.ndarray:
    c = codes.detach().cpu().numpy()
    if c.shape[1] == 0:
        return np.zeros(0, dtype=np.float32)
    out = np.zeros(c.shape[1], dtype=np.float32)
    if c.shape[1] > 1:
        changed = (c[:, 1:] != c[:, :-1]).astype(np.float32)
        out[1:] = changed.mean(axis=0)
        out[0] = out[1]
    return out


def _build_cumsum(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    csum = np.zeros(arr.size + 1, dtype=np.float64)
    csum[1:] = np.cumsum(arr, dtype=np.float64)
    return csum


def span_mean_from_cumsum(csum: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    length = np.maximum(1, end - start)
    return ((csum[end] - csum[start]) / length).astype(np.float32)


def choose_mask_regime(probs: Dict[str, float]) -> str:
    names = ["active", "transition", "low_activity", "uniform"]
    p = np.array([max(0.0, float(probs.get(n, 0.0))) for n in names], dtype=np.float64)
    if p.sum() <= 0:
        p = np.array([0.45, 0.30, 0.15, 0.10], dtype=np.float64)
    p = p / p.sum()
    return names[int(np.random.choice(len(names), p=p))]


def normalize_bucket_lengths(lengths: Sequence[int]) -> Tuple[int, ...]:
    return tuple(sorted({int(v) for v in lengths if int(v) > 0}))


def resolve_bucket_length(mask_len: int, bucket_lengths: Sequence[int]) -> int:
    target = int(mask_len)
    if target <= 0:
        raise ValueError("mask_len must be > 0")
    normalized = normalize_bucket_lengths(bucket_lengths)
    if not normalized:
        return target
    return min(normalized, key=lambda item: (abs(item - target), item))


def compute_mask_candidate_weights(
    regime: str,
    mean_activity: np.ndarray,
    active_ratio: np.ndarray,
    std_activity: np.ndarray,
    variation: np.ndarray,
    proximity: np.ndarray,
    token_change_mean: np.ndarray,
    dead_window_min_mean: float,
    dead_window_min_ratio: float,
) -> np.ndarray:
    if regime == "active":
        w = 0.10 + 0.60 * mean_activity + 0.30 * active_ratio
    elif regime == "transition":
        mid_ratio = 1.0 - np.abs(active_ratio - 0.5) / 0.5
        mid_ratio = np.clip(mid_ratio, 0.0, 1.0)
        w = 0.05 + 0.35 * std_activity + 0.35 * variation + 0.30 * mid_ratio
    elif regime == "low_activity":
        low_pref = np.clip(1.0 - mean_activity, 0.0, 1.0)
        w = 0.05 + 0.45 * low_pref + 0.30 * proximity + 0.25 * token_change_mean
    else:
        w = np.ones_like(mean_activity, dtype=np.float32)

    # Only heavily down-weight truly dead candidate spans.
    dead = (
        (mean_activity < dead_window_min_mean)
        & (active_ratio < dead_window_min_ratio)
        & (token_change_mean < 0.01)
        & (proximity < 0.05)
    )
    w = np.where(dead, 1e-5, w)
    w = np.clip(w, 1e-6, None)
    return w.astype(np.float32)


def window_overlaps_ranges(start: int, seq_len: int, ranges: List[Tuple[int, int]]) -> bool:
    window_end = start + seq_len
    for g0, g1 in ranges:
        if start < g1 and window_end > g0:
            return True
    return False


def is_non_gap_window(
    start: int,
    seq_len: int,
    gaps: List[Tuple[int, int]],
    blocked_ranges: Optional[List[Tuple[int, int]]] = None,
) -> bool:
    if window_overlaps_ranges(start, seq_len, gaps):
        return False
    if blocked_ranges and window_overlaps_ranges(start, seq_len, blocked_ranges):
        return False
    return True


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def validation_holdout_summary(ranges: List[Tuple[int, int]]) -> Dict[str, float]:
    merged = merge_ranges(ranges)
    return {
        "holdout_ranges": float(len(merged)),
        "holdout_frames": float(sum(end - start for start, end in merged)),
    }


def candidate_mask_offsets(seq_len: int, mask_len: int, mask_stride: int) -> np.ndarray:
    max_m0 = seq_len - mask_len
    if max_m0 <= 0:
        return np.array([0], dtype=np.int64)
    stride = int(max(1, mask_stride))
    candidates = np.arange(0, max_m0 + 1, stride, dtype=np.int64)
    if candidates[-1] != max_m0:
        candidates = np.concatenate([candidates, np.array([max_m0], dtype=np.int64)], axis=0)
    return candidates


def compute_activity_features(
    wav: torch.Tensor,
    codes: torch.Tensor,
    activity_smooth_kernel: int,
    activity_low_quantile: float,
    activity_high_quantile: float,
) -> Dict[str, Any]:
    frames = int(codes.shape[1])
    rms = compute_rms_per_frame(wav, frames)
    token_change = compute_token_change_per_frame(codes)
    rms_norm = normalize_robust(rms)
    token_change_norm = normalize_robust(token_change)
    activity_raw = 0.65 * rms_norm + 0.35 * token_change_norm
    activity = smooth_1d(activity_raw, activity_smooth_kernel)
    activity_per_frame = np.clip(activity.astype(np.float32), 0.0, 1.0)
    return {
        "rms_per_frame": rms_norm.astype(np.float32),
        "token_change_per_frame": token_change_norm.astype(np.float32),
        "activity_per_frame": activity_per_frame,
        "activity_low_thr": float(np.quantile(activity_per_frame, activity_low_quantile)),
        "activity_high_thr": float(np.quantile(activity_per_frame, activity_high_quantile)),
    }


@dataclass
class FixedValidationExample:
    x: torch.Tensor
    y: torch.Tensor
    loss_mask: torch.Tensor
    band: str
    sample_name: str
    mask_mean_activity: float
    mask_len: int
    window_start: int
    mask_start: int


@dataclass
class ValidationRegion:
    band: str
    start: int
    end: int
    mean_activity: float
    active_ratio: float


@dataclass
class ValidationGroupSpec:
    band: str
    mask_len: Optional[int] = None


class FixedMaskedSpanDataset(Dataset):
    def __init__(self, examples: List[FixedValidationExample], return_metadata: bool = False):
        self.examples = list(examples)
        self.return_metadata = bool(return_metadata)
        mask_means = np.array([ex.mask_mean_activity for ex in self.examples], dtype=np.float32)
        mask_lengths = np.array([ex.mask_len for ex in self.examples], dtype=np.float32)
        self.summary = {
            "count": len(self.examples),
            "mean_mask_activity": float(mask_means.mean()) if mask_means.size else 0.0,
            "mean_mask_len": float(mask_lengths.mean()) if mask_lengths.size else 0.0,
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        if not self.return_metadata:
            return ex.x.clone(), ex.y.clone(), ex.loss_mask.clone()
        metadata = {
            "window_start": torch.tensor(int(ex.window_start), dtype=torch.long),
            "mask_start": torch.tensor(int(ex.mask_start), dtype=torch.long),
            "mask_len": torch.tensor(int(ex.mask_len), dtype=torch.long),
            "window_len": torch.tensor(int(ex.y.shape[1]), dtype=torch.long),
        }
        return ex.x.clone(), ex.y.clone(), ex.loss_mask.clone(), metadata


def slugify_component(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    safe = safe.strip("_")
    return safe or "item"


def make_validation_example_id(example: FixedValidationExample, group_name: str, index: int) -> str:
    sample_slug = slugify_component(example.sample_name)
    group_slug = slugify_component(group_name)
    return (
        f"{sample_slug}__{group_slug}__idx{index:02d}"
        f"__ws{int(example.window_start):06d}__ms{int(example.mask_start):04d}__ml{int(example.mask_len):04d}"
    )


def derive_validation_crop_context_frames(mask_len: int, crop_context_frames: Optional[int] = None) -> int:
    if crop_context_frames is not None:
        return max(0, int(crop_context_frames))
    return max(96, int(math.ceil(float(mask_len) * 0.5)))


def compute_validation_crop_bounds(
    seq_len: int,
    mask_start: int,
    mask_len: int,
    crop_context_frames: Optional[int] = None,
) -> Tuple[int, int]:
    context = derive_validation_crop_context_frames(mask_len, crop_context_frames)
    crop_start = max(0, int(mask_start) - context)
    crop_end = min(int(seq_len), int(mask_start) + int(mask_len) + context)
    return crop_start, max(crop_start + 1, crop_end)


def frame_bounds_to_sample_bounds(
    total_samples: int,
    total_frames: int,
    start_frame: int,
    end_frame: int,
) -> Tuple[int, int]:
    if total_samples <= 0:
        return 0, 0
    samples_per_frame = float(total_samples) / max(1, int(total_frames))
    start_sample = int(round(max(0, start_frame) * samples_per_frame))
    end_sample = int(round(max(start_frame + 1, end_frame) * samples_per_frame))
    start_sample = min(max(0, start_sample), total_samples - 1)
    end_sample = min(max(start_sample + 1, end_sample), total_samples)
    return start_sample, end_sample


def extract_contiguous_mask_bounds(loss_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if loss_mask.dim() == 1:
        loss_mask = loss_mask.unsqueeze(0)
    if loss_mask.dim() != 2:
        raise ValueError(f"loss_mask must have shape [T] or [B,T], got {tuple(loss_mask.shape)}")

    mask = loss_mask.bool()
    batch, steps = mask.shape
    if steps <= 0:
        raise ValueError("loss_mask must have at least one timestep")
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Each example must contain at least one masked timestep")

    first_mask = mask.float().argmax(dim=1)
    last_mask_from_end = torch.flip(mask, dims=[1]).float().argmax(dim=1)
    gap_end = steps - last_mask_from_end

    positions = torch.arange(steps, device=mask.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    gap_mask = (positions >= first_mask.unsqueeze(1)) & (positions < gap_end.unsqueeze(1))
    if not torch.equal(gap_mask, mask):
        raise ValueError("Expected exactly one contiguous masked span per example")
    return first_mask.long(), gap_end.long()


def extract_contiguous_mask_bounds_from_segment_ids(segment_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if segment_ids.dim() == 1:
        segment_ids = segment_ids.unsqueeze(0)
    if segment_ids.dim() != 2:
        raise ValueError(f"segment_ids must have shape [T] or [B,T], got {tuple(segment_ids.shape)}")
    return extract_contiguous_mask_bounds(segment_ids == 1)


def build_boundary_condition_tensors(
    loss_mask: torch.Tensor,
    max_distance: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if max_distance <= 0:
        raise ValueError("max_distance must be > 0")
    if loss_mask.dim() == 1:
        loss_mask = loss_mask.unsqueeze(0)
    mask = loss_mask.bool()
    batch, steps = mask.shape
    first_mask, gap_end = extract_contiguous_mask_bounds(mask)

    positions = torch.arange(steps, device=mask.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
    gap_mask = (positions >= first_mask.unsqueeze(1)) & (positions < gap_end.unsqueeze(1))

    segment_ids = torch.zeros((batch, steps), device=mask.device, dtype=torch.long)
    segment_ids = torch.where(positions >= gap_end.unsqueeze(1), torch.full_like(segment_ids, 2), segment_ids)
    segment_ids = torch.where(gap_mask, torch.full_like(segment_ids, 1), segment_ids)

    max_d = int(max_distance)
    left_distance = (positions - first_mask.unsqueeze(1)).clamp(min=-max_d, max=max_d) + max_d
    right_distance = (gap_end.unsqueeze(1) - positions).clamp(min=-max_d, max=max_d) + max_d
    return segment_ids, left_distance.long(), right_distance.long()


def build_boundary_condition_tensors_from_mask_token(
    x: torch.Tensor,
    mask_token: int,
    max_distance: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 3:
        raise ValueError(f"x must have shape [B,K,T], got {tuple(x.shape)}")
    mask = (x == int(mask_token)).all(dim=1)
    return build_boundary_condition_tensors(mask, max_distance=max_distance)


def compute_log_spectrogram_data(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop: int = 512,
    eps: float = 1e-7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    waveform = torch.as_tensor(np.asarray(audio, dtype=np.float32)).flatten()
    if waveform.numel() == 0:
        raise ValueError("compute_log_spectrogram_data requires a non-empty waveform")
    if waveform.numel() < 2:
        waveform = torch.cat([waveform, waveform.new_zeros(2 - waveform.numel())], dim=0)
    n_fft_eff = max(2, min(int(n_fft), int(waveform.numel())))
    hop_eff = max(1, min(int(hop), max(1, n_fft_eff // 2)))
    window = torch.hann_window(n_fft_eff)
    spec = torch.stft(
        waveform,
        n_fft=n_fft_eff,
        hop_length=hop_eff,
        win_length=n_fft_eff,
        window=window,
        return_complex=True,
    )
    spec_db = (20.0 * torch.log10(spec.abs().clamp_min(eps))).cpu().numpy()
    freqs = np.fft.rfftfreq(n_fft_eff, d=1.0 / float(sr)).astype(np.float32)
    times = (np.arange(spec_db.shape[1], dtype=np.float32) * float(hop_eff) / float(sr)).astype(np.float32)
    if freqs.size > 1 and freqs[0] <= 0.0:
        freqs = freqs[1:]
        spec_db = spec_db[1:, :]
    return times, freqs, spec_db


def _plot_log_spectrogram(
    ax,
    times: np.ndarray,
    freqs: np.ndarray,
    spec_db: np.ndarray,
    title: str,
    time_offset_s: float = 0.0,
    min_freq: float = 30.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    freq_mask = freqs >= max(float(min_freq), float(freqs[0]) if freqs.size else float(min_freq))
    freqs_plot = freqs[freq_mask]
    spec_plot = spec_db[freq_mask, :]
    if freqs_plot.size == 0 or spec_plot.size == 0:
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        return None
    local_vmax = float(np.max(spec_plot)) if vmax is None else float(vmax)
    local_vmin = local_vmax - 80.0 if vmin is None else float(vmin)
    mesh = ax.pcolormesh(
        times + float(time_offset_s),
        freqs_plot,
        spec_plot,
        shading="auto",
        cmap="magma",
        vmin=local_vmin,
        vmax=local_vmax,
    )
    ax.set_yscale("log")
    ax.set_ylim(max(float(min_freq), float(freqs_plot[0])), float(freqs_plot[-1]))
    if times.size <= 1:
        start_t = float(time_offset_s if times.size == 0 else times[0] + time_offset_s)
        end_t = start_t + (1.0 / max(1.0, float(freqs_plot[-1])))
    else:
        start_t = float(times[0] + time_offset_s)
        end_t = float(times[-1] + time_offset_s)
    ax.set_xlim(start_t, end_t)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return mesh


def make_log_spectrogram_comparison_figure(
    target_audio: np.ndarray,
    pred_audio: np.ndarray,
    sr: int,
    title_prefix: str,
    time_offset_s: float = 0.0,
    n_fft: int = 2048,
    hop: int = 512,
    min_freq: float = 30.0,
):
    target_times, target_freqs, target_db = compute_log_spectrogram_data(target_audio, sr, n_fft=n_fft, hop=hop)
    pred_times, pred_freqs, pred_db = compute_log_spectrogram_data(pred_audio, sr, n_fft=n_fft, hop=hop)
    vmax = max(float(np.max(target_db)), float(np.max(pred_db)))
    vmin = vmax - 80.0
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharex=True, sharey=True, constrained_layout=True)
    mesh = _plot_log_spectrogram(
        axes[0],
        target_times,
        target_freqs,
        target_db,
        title=f"{title_prefix} | Ground Truth",
        time_offset_s=time_offset_s,
        min_freq=min_freq,
        vmin=vmin,
        vmax=vmax,
    )
    _plot_log_spectrogram(
        axes[1],
        pred_times,
        pred_freqs,
        pred_db,
        title=f"{title_prefix} | Prediction",
        time_offset_s=time_offset_s,
        min_freq=min_freq,
        vmin=vmin,
        vmax=vmax,
    )
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.tolist(), label="dB")
    return fig


def _plot_waveform_panel(
    ax,
    target_audio: np.ndarray,
    pred_audio: np.ndarray,
    sr: int,
    title: str,
    mask_sample_bounds: Tuple[int, int],
    time_offset_s: float = 0.0,
):
    target = np.asarray(target_audio, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred_audio, dtype=np.float32).reshape(-1)
    n_samples = min(target.shape[0], pred.shape[0])
    if n_samples == 0:
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        return
    target = target[:n_samples]
    pred = pred[:n_samples]
    times = (np.arange(n_samples, dtype=np.float32) / float(sr)) + float(time_offset_s)
    ax.plot(times, target, label="Ground Truth", linewidth=1.2, alpha=0.95)
    ax.plot(times, pred, label="Prediction", linewidth=1.1, alpha=0.85)
    mask_start, mask_end = mask_sample_bounds
    mask_start = max(0, min(mask_start, n_samples - 1))
    mask_end = max(mask_start + 1, min(mask_end, n_samples))
    ax.axvspan(
        (mask_start / float(sr)) + float(time_offset_s),
        (mask_end / float(sr)) + float(time_offset_s),
        color="tab:red",
        alpha=0.12,
        label="Masked Region",
    )
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")


def make_waveform_comparison_figure(
    target_audio: np.ndarray,
    pred_audio: np.ndarray,
    sr: int,
    title_prefix: str,
    mask_sample_bounds: Tuple[int, int],
    crop_sample_bounds: Optional[Tuple[int, int]] = None,
    time_offset_s: float = 0.0,
):
    target = np.asarray(target_audio, dtype=np.float32).reshape(-1)
    pred = np.asarray(pred_audio, dtype=np.float32).reshape(-1)
    n_samples = min(target.shape[0], pred.shape[0])
    target = target[:n_samples]
    pred = pred[:n_samples]
    full_mask = (
        max(0, min(int(mask_sample_bounds[0]), max(0, n_samples - 1))),
        max(1, min(int(mask_sample_bounds[1]), n_samples)),
    )
    if crop_sample_bounds is None:
        crop_sample_bounds = (0, n_samples)
    crop_start = max(0, min(int(crop_sample_bounds[0]), max(0, n_samples - 1)))
    crop_end = max(crop_start + 1, min(int(crop_sample_bounds[1]), n_samples))
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharey=False)
    _plot_waveform_panel(
        axes[0],
        target,
        pred,
        sr,
        title=f"{title_prefix} | Full Window",
        mask_sample_bounds=full_mask,
        time_offset_s=time_offset_s,
    )
    _plot_waveform_panel(
        axes[1],
        target[crop_start:crop_end],
        pred[crop_start:crop_end],
        sr,
        title=f"{title_prefix} | Gap Crop",
        mask_sample_bounds=(full_mask[0] - crop_start, full_mask[1] - crop_start),
        time_offset_s=time_offset_s + (crop_start / float(sr)),
    )
    fig.tight_layout()
    return fig


def save_waveform(path: Path, audio: np.ndarray, sr: int):
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)


def _valid_non_gap_starts(
    frames: int,
    span_len: int,
    gaps: List[Tuple[int, int]],
    blocked_ranges: Optional[List[Tuple[int, int]]] = None,
) -> List[int]:
    max_start = max(0, frames - span_len)
    return [s for s in range(max_start) if is_non_gap_window(s, span_len, gaps, blocked_ranges)]


def _region_mean_activity(
    activity_cumsum: np.ndarray,
    active_flag_cumsum: np.ndarray,
    start: int,
    end: int,
) -> Tuple[float, float]:
    start_a = np.array([start], dtype=np.int64)
    end_a = np.array([end], dtype=np.int64)
    mean_activity = float(span_mean_from_cumsum(activity_cumsum, start_a, end_a)[0])
    active_ratio = float(span_mean_from_cumsum(active_flag_cumsum, start_a, end_a)[0])
    return mean_activity, active_ratio


def _pick_validation_regions(
    candidates: List[ValidationRegion],
    count: int,
    descending: bool,
    min_separation: int,
    already_selected: Optional[List[ValidationRegion]] = None,
) -> List[ValidationRegion]:
    selected = list(already_selected or [])
    chosen: List[ValidationRegion] = []
    ordered = sorted(
        candidates,
        key=lambda item: ((-item.mean_activity) if descending else item.mean_activity, item.start),
    )
    for candidate in ordered:
        conflict = False
        for existing in selected:
            if candidate.start < existing.end and candidate.end > existing.start:
                conflict = True
                break
            if abs(candidate.start - existing.start) < min_separation:
                conflict = True
                break
        if conflict:
            continue
        chosen.append(candidate)
        selected.append(candidate)
        if len(chosen) >= count:
            break
    return chosen


def build_fixed_validation_examples(
    codes: torch.Tensor,
    gaps: List[Tuple[int, int]],
    seq_len: int,
    mask_len_range: Tuple[int, int],
    mask_token: int,
    activity_per_frame: np.ndarray,
    activity_low_thr: float,
    activity_high_thr: float,
    examples_per_band: int,
    mask_stride: int,
    seed: int,
    sample_name: str,
    bucket_lengths: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, List[FixedValidationExample]], List[Tuple[int, int]]]:
    _, frames = codes.shape
    valid_starts = _valid_non_gap_starts(frames, seq_len, gaps)
    if not valid_starts:
        raise ValueError(f"No valid validation windows found for sample={sample_name}")

    activity_cumsum = _build_cumsum(activity_per_frame)
    rng = random.Random(seed)
    mask_min, mask_max = mask_len_range
    per_band: Dict[str, List[FixedValidationExample]] = {
        "high_activity": [],
        "low_activity": [],
    }
    reserved_ranges: List[Tuple[int, int]] = []
    seen: Dict[str, set] = {
        "high_activity": set(),
        "low_activity": set(),
    }
    max_attempts = max(2000, examples_per_band * 400)

    for band in ["high_activity", "low_activity"]:
        attempts = 0
        while len(per_band[band]) < examples_per_band and attempts < max_attempts:
            attempts += 1
            start = rng.choice(valid_starts)
            mask_len = min(seq_len, rng.randint(mask_min, max(mask_min, mask_max)))
            if bucket_lengths:
                mask_len = resolve_bucket_length(mask_len, bucket_lengths)
            offsets = candidate_mask_offsets(seq_len, mask_len, mask_stride)
            mask_start = int(rng.choice(offsets.tolist()))
            g0 = start + mask_start
            g1 = g0 + mask_len
            mask_mean = float(span_mean_from_cumsum(activity_cumsum, np.array([g0]), np.array([g1]))[0])

            if band == "high_activity" and mask_mean < activity_high_thr:
                continue
            if band == "low_activity" and mask_mean > activity_low_thr:
                continue

            key = (start, mask_start, mask_len)
            if key in seen[band]:
                continue
            seen[band].add(key)
            reserved_ranges.append((start, start + seq_len))

            y = codes[:, start : start + seq_len].clone()
            x = y.clone()
            x[:, mask_start : mask_start + mask_len] = mask_token
            loss_mask = torch.zeros(seq_len, dtype=torch.bool)
            loss_mask[mask_start : mask_start + mask_len] = True
            per_band[band].append(
                FixedValidationExample(
                    x=x,
                    y=y,
                    loss_mask=loss_mask,
                    band=band,
                    sample_name=sample_name,
                    mask_mean_activity=mask_mean,
                    mask_len=mask_len,
                    window_start=start,
                    mask_start=mask_start,
                )
            )

        if len(per_band[band]) == 0:
            raise ValueError(
                f"No validation examples found for band={band} sample={sample_name}. "
                "Adjust validation settings or activity thresholds."
            )
        if len(per_band[band]) < examples_per_band:
            logger.warning(
                "Validation band %s for sample %s requested %d examples but found %d",
                band,
                sample_name,
                examples_per_band,
                len(per_band[band]),
            )

    return per_band, merge_ranges(reserved_ranges)


def build_holdout_region_validation_examples(
    codes: torch.Tensor,
    gaps: List[Tuple[int, int]],
    seq_len: int,
    mask_lengths: Sequence[int],
    mask_token: int,
    activity_per_frame: np.ndarray,
    activity_low_thr: float,
    activity_high_thr: float,
    regions_per_band: int,
    region_len_frames: int,
    region_min_separation_frames: int,
    examples_per_length_band: int,
    mask_stride: int,
    seed: int,
    sample_name: str,
    dead_window_min_mean: float,
    dead_window_min_ratio: float,
    bucket_lengths: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, List[FixedValidationExample]], Dict[str, List[ValidationRegion]], List[Tuple[int, int]], Dict[str, float]]:
    _, frames = codes.shape
    normalized_buckets = normalize_bucket_lengths(bucket_lengths or ())
    if normalized_buckets:
        missing = [int(v) for v in mask_lengths if int(v) not in normalized_buckets]
        if missing:
            raise ValueError(
                f"validation mask lengths {missing} are not present in retrieval bucket lengths {list(normalized_buckets)}"
            )
    if region_len_frames < seq_len:
        raise ValueError(f"validation region length {region_len_frames} must be >= seq_len {seq_len}")

    activity_cumsum = _build_cumsum(activity_per_frame)
    active_flag = (activity_per_frame > activity_low_thr).astype(np.float32)
    active_flag_cumsum = _build_cumsum(active_flag)
    region_starts = _valid_non_gap_starts(frames, region_len_frames, gaps)
    if not region_starts:
        raise ValueError(f"No valid validation holdout regions found for sample={sample_name}")

    candidates_high: List[ValidationRegion] = []
    candidates_low: List[ValidationRegion] = []
    fallback_low: List[ValidationRegion] = []
    for start in region_starts:
        end = start + region_len_frames
        mean_activity, active_ratio = _region_mean_activity(activity_cumsum, active_flag_cumsum, start, end)
        region = ValidationRegion(
            band="high_activity" if mean_activity >= activity_high_thr else "low_activity",
            start=start,
            end=end,
            mean_activity=mean_activity,
            active_ratio=active_ratio,
        )
        if mean_activity >= activity_high_thr:
            candidates_high.append(region)
        if mean_activity <= activity_low_thr:
            fallback_low.append(region)
            if mean_activity >= dead_window_min_mean or active_ratio >= dead_window_min_ratio:
                candidates_low.append(region)

    selected_high = _pick_validation_regions(
        candidates_high if candidates_high else [
            ValidationRegion("high_activity", s, s + region_len_frames, *_region_mean_activity(activity_cumsum, active_flag_cumsum, s, s + region_len_frames))
            for s in region_starts
        ],
        count=regions_per_band,
        descending=True,
        min_separation=region_min_separation_frames,
    )
    if len(selected_high) < regions_per_band:
        raise ValueError(f"Unable to select {regions_per_band} high-activity validation regions for sample={sample_name}")

    low_source = candidates_low if candidates_low else fallback_low
    selected_low = _pick_validation_regions(
        low_source,
        count=regions_per_band,
        descending=False,
        min_separation=region_min_separation_frames,
        already_selected=selected_high,
    )
    if len(selected_low) < regions_per_band:
        raise ValueError(f"Unable to select {regions_per_band} low-activity validation regions for sample={sample_name}")

    regions_by_band = {
        "high_activity": selected_high,
        "low_activity": selected_low,
    }
    grouped_examples: Dict[str, List[FixedValidationExample]] = {}
    rng = random.Random(seed)

    for band, regions in regions_by_band.items():
        aggregated_items: List[FixedValidationExample] = []
        for mask_len in mask_lengths:
            key = f"{band}_len_{int(mask_len)}"
            items: List[FixedValidationExample] = []
            seen = set()
            attempts = 0
            max_attempts = max(4000, examples_per_length_band * 800)
            while len(items) < examples_per_length_band and attempts < max_attempts:
                attempts += 1
                region = rng.choice(regions)
                window_starts = list(range(region.start, region.end - seq_len + 1))
                if not window_starts:
                    continue
                start = rng.choice(window_starts)
                offsets = candidate_mask_offsets(seq_len, int(mask_len), mask_stride)
                mask_start = int(rng.choice(offsets.tolist()))
                g0 = start + mask_start
                g1 = g0 + int(mask_len)
                start_a = np.array([g0], dtype=np.int64)
                end_a = np.array([g1], dtype=np.int64)
                mask_mean = float(span_mean_from_cumsum(activity_cumsum, start_a, end_a)[0])
                mask_ratio = float(span_mean_from_cumsum(active_flag_cumsum, start_a, end_a)[0])
                if band == "high_activity" and mask_mean < activity_high_thr:
                    continue
                if band == "low_activity":
                    if mask_mean > activity_low_thr:
                        continue
                    if mask_mean < dead_window_min_mean and mask_ratio < dead_window_min_ratio:
                        continue
                ex_key = (start, mask_start, int(mask_len))
                if ex_key in seen:
                    continue
                seen.add(ex_key)
                y = codes[:, start : start + seq_len].clone()
                x = y.clone()
                x[:, mask_start : mask_start + int(mask_len)] = mask_token
                loss_mask = torch.zeros(seq_len, dtype=torch.bool)
                loss_mask[mask_start : mask_start + int(mask_len)] = True
                ex = FixedValidationExample(
                    x=x,
                    y=y,
                    loss_mask=loss_mask,
                    band=band,
                    sample_name=sample_name,
                    mask_mean_activity=mask_mean,
                    mask_len=int(mask_len),
                    window_start=start,
                    mask_start=mask_start,
                )
                items.append(ex)
                aggregated_items.append(ex)
            if len(items) != examples_per_length_band:
                raise ValueError(
                    f"Requested {examples_per_length_band} validation examples for band={band} mask_len={mask_len} "
                    f"but built {len(items)} for sample={sample_name}"
                )
            grouped_examples[key] = items
        grouped_examples[band] = aggregated_items

    holdout_ranges = merge_ranges([(region.start, region.end) for band in regions_by_band.values() for region in band])
    metadata: Dict[str, float] = {
        "holdout_regions": float(len(holdout_ranges)),
        "holdout_frames": float(sum(end - start for start, end in holdout_ranges)),
        "high_activity_region_count": float(len(selected_high)),
        "low_activity_region_count": float(len(selected_low)),
        "high_activity_region_mean_activity": float(np.mean([r.mean_activity for r in selected_high])) if selected_high else 0.0,
        "low_activity_region_mean_activity": float(np.mean([r.mean_activity for r in selected_low])) if selected_low else 0.0,
        "high_activity_count": float(len(grouped_examples["high_activity"])),
        "low_activity_count": float(len(grouped_examples["low_activity"])),
        "high_activity_mean_mask_len": float(np.mean([ex.mask_len for ex in grouped_examples["high_activity"]])) if grouped_examples["high_activity"] else 0.0,
        "low_activity_mean_mask_len": float(np.mean([ex.mask_len for ex in grouped_examples["low_activity"]])) if grouped_examples["low_activity"] else 0.0,
    }
    return grouped_examples, regions_by_band, holdout_ranges, metadata


class AudioEncoder:
    def __init__(
        self,
        bandwidth: float,
        device: torch.device,
        *,
        encodec_model: str = "encodec_24khz",
        custom_decoder_checkpoint: Optional[str] = None,
    ):
        self.device = device
        self.encodec_model = encodec_model
        self.model = build_encodec_model(encodec_model).to(device).eval()
        self.model.set_target_bandwidth(bandwidth)
        if custom_decoder_checkpoint:
            load_exported_decoder(
                custom_decoder_checkpoint,
                decoder=self.model.decoder,
                expected_encodec_model=encodec_model,
                expected_bandwidth=bandwidth,
            )
        self.model.requires_grad_(False)
        self.bins = int(self.model.quantizer.bins)
        self.mask_token = self.bins
        self.frame_rate = int(self.model.frame_rate)

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

    def codes_to_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        return codes_to_embeddings(self.model, codes, device=self.device)

    def logits_to_embeddings(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return logits_to_embeddings(self.model, logits, temperature=temperature, device=self.device)

    def decode_embeddings(self, embeddings: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        return decode_embeddings(self.model.decoder, embeddings, device=self.device, scale=scale)


class ActivityAwareMaskedSpanDataset(Dataset):
    """Samples non-gap windows and contiguous masks with optional activity guidance."""

    def __init__(
        self,
        codes: torch.Tensor,
        gaps: List[Tuple[int, int]],
        seq_len: int = 1024,
        mask_len_range: Tuple[int, int] = (64, 256),
        mask_token: int = 1024,
        virtual_size: int = 50_000,
        activity_per_frame: Optional[np.ndarray] = None,
        token_change_per_frame: Optional[np.ndarray] = None,
        activity_low_thr: float = 0.0,
        activity_high_thr: float = 1.0,
        weighted_sampling: bool = True,
        dead_window_min_mean: float = 0.01,
        dead_window_min_ratio: float = 0.03,
        regime_probs: Optional[Dict[str, float]] = None,
        max_resample_tries: int = 12,
        blocked_ranges: Optional[List[Tuple[int, int]]] = None,
        mask_stride: int = 1,
        activity_guided_masking: bool = True,
        return_metadata: bool = False,
        bucket_lengths: Optional[Sequence[int]] = None,
    ):
        self.codes = codes
        self.K, self.F = codes.shape
        self.gaps = sorted(gaps, key=lambda g: g[0])
        self.blocked_ranges = merge_ranges(blocked_ranges or [])
        self.seq_len = seq_len
        self._mask_len_range = list(mask_len_range)
        self.mask_token = mask_token
        self.virtual_size = virtual_size
        self.activity_per_frame = (
            np.asarray(activity_per_frame, dtype=np.float32)
            if activity_per_frame is not None
            else np.zeros(self.F, dtype=np.float32)
        )
        self.token_change_per_frame = (
            np.asarray(token_change_per_frame, dtype=np.float32)
            if token_change_per_frame is not None
            else np.zeros(self.F, dtype=np.float32)
        )
        if self.activity_per_frame.shape[0] != self.F:
            raise ValueError("activity_per_frame length must match frame count")
        if self.token_change_per_frame.shape[0] != self.F:
            raise ValueError("token_change_per_frame length must match frame count")

        self.activity_low_thr = float(activity_low_thr)
        self.activity_high_thr = float(activity_high_thr)
        self.weighted_sampling = bool(weighted_sampling)
        self.dead_window_min_mean = float(dead_window_min_mean)
        self.dead_window_min_ratio = float(dead_window_min_ratio)
        self.max_resample_tries = int(max(1, max_resample_tries))
        self.mask_stride = int(max(1, mask_stride))
        self.activity_guided_masking = bool(activity_guided_masking)
        self.return_metadata = bool(return_metadata)
        self.bucket_lengths = normalize_bucket_lengths(bucket_lengths or ())
        rp = regime_probs or {
            "active": 0.45,
            "transition": 0.30,
            "low_activity": 0.15,
            "uniform": 0.10,
        }
        self.regime_probs = {k: float(v) for k, v in rp.items()}
        p_sum = sum(max(0.0, v) for v in self.regime_probs.values())
        if p_sum <= 0:
            self.regime_probs = {
                "active": 0.45,
                "transition": 0.30,
                "low_activity": 0.15,
                "uniform": 0.10,
            }
        else:
            self.regime_probs = {k: max(0.0, v) / p_sum for k, v in self.regime_probs.items()}

        self.activity_cumsum = _build_cumsum(self.activity_per_frame)
        self.activity_sq_cumsum = _build_cumsum(self.activity_per_frame * self.activity_per_frame)
        active_flag = (self.activity_per_frame > self.activity_low_thr).astype(np.float32)
        self.active_flag_cumsum = _build_cumsum(active_flag)
        self.activity_diff = np.abs(np.diff(self.activity_per_frame, prepend=self.activity_per_frame[0])).astype(np.float32)
        self.activity_diff_cumsum = _build_cumsum(self.activity_diff)
        self.token_change_cumsum = _build_cumsum(self.token_change_per_frame)
        self.active_indices = np.nonzero(active_flag > 0.5)[0]
        self.recent_metrics: deque = deque(maxlen=4096)

        self._build_valid_starts_and_weights()

    def _is_non_gap_window(self, s: int) -> bool:
        return is_non_gap_window(s, self.seq_len, self.gaps, self.blocked_ranges)

    def _build_valid_starts_and_weights(self):
        self.starts: List[int] = []
        window_mean = []
        window_ratio = []
        rejected_dead = 0
        non_gap_candidates = 0
        blocked_by_holdout = 0
        max_s = self.F - self.seq_len
        for s in range(max(0, max_s)):
            if window_overlaps_ranges(s, self.seq_len, self.gaps):
                continue
            non_gap_candidates += 1
            if self.blocked_ranges and window_overlaps_ranges(s, self.seq_len, self.blocked_ranges):
                blocked_by_holdout += 1
                continue
            mean_act = float(span_mean_from_cumsum(self.activity_cumsum, np.array([s]), np.array([s + self.seq_len]))[0])
            active_ratio = float(span_mean_from_cumsum(self.active_flag_cumsum, np.array([s]), np.array([s + self.seq_len]))[0])
            if self.activity_guided_masking and mean_act < self.dead_window_min_mean and active_ratio < self.dead_window_min_ratio:
                rejected_dead += 1
                continue
            self.starts.append(s)
            window_mean.append(mean_act)
            window_ratio.append(active_ratio)

        if len(self.starts) == 0:
            raise ValueError(
                f"No valid windows found. seq_len={self.seq_len}, "
                f"F={self.F}, gaps={self.gaps}. "
                "Reduce seq_len or check gap boundaries."
            )

        self.window_mean = np.asarray(window_mean, dtype=np.float32)
        self.window_active_ratio = np.asarray(window_ratio, dtype=np.float32)
        self.window_weights = (0.05 + 0.60 * self.window_mean + 0.35 * self.window_active_ratio).astype(np.float32)
        self.window_weights = np.clip(self.window_weights, 1e-6, None)
        self.window_weights = self.window_weights / self.window_weights.sum()
        self.window_weights_t = torch.from_numpy(self.window_weights)
        self.start_to_index = {s: i for i, s in enumerate(self.starts)}

        self.summary = {
            "total_valid_starts": len(self.starts),
            "pre_holdout_valid_starts": int(non_gap_candidates),
            "starts_after_filter": len(self.starts),
            "holdout_blocked_starts": int(blocked_by_holdout),
            "holdout_blocked_fraction": float(blocked_by_holdout / max(1, non_gap_candidates)),
            "rejected_dead": int(rejected_dead),
            "blocked_ranges": int(len(self.blocked_ranges)),
            "avg_window_activity": float(self.window_mean.mean()) if self.window_mean.size else 0.0,
            "min_window_weight": float(self.window_weights.min()) if self.window_weights.size else 0.0,
            "max_window_weight": float(self.window_weights.max()) if self.window_weights.size else 0.0,
            "regime_active_prob": self.regime_probs.get("active", 0.0),
            "regime_transition_prob": self.regime_probs.get("transition", 0.0),
            "regime_low_prob": self.regime_probs.get("low_activity", 0.0),
            "regime_uniform_prob": self.regime_probs.get("uniform", 0.0),
        }

        logger.info(
            "Dataset: K=%d, F=%d, gaps=%d, starts=%d (rejected_dead=%d), avg_window_act=%.4f, w[min,max]=[%.6f, %.6f]",
            self.K,
            self.F,
            len(self.gaps),
            len(self.starts),
            rejected_dead,
            self.summary["avg_window_activity"],
            self.summary["min_window_weight"],
            self.summary["max_window_weight"],
        )

    @property
    def mask_len_range(self) -> Tuple[int, int]:
        return (self._mask_len_range[0], self._mask_len_range[1])

    def update_mask_range(self, mask_min: int, mask_max: int):
        """Dynamically update mask range for curriculum learning."""
        self._mask_len_range[0] = mask_min
        self._mask_len_range[1] = mask_max

    def __len__(self) -> int:
        return self.virtual_size

    def _sample_window_start(self) -> int:
        if self.weighted_sampling and self.window_weights_t.numel() > 1:
            idx = int(torch.multinomial(self.window_weights_t, 1, replacement=True).item())
            return self.starts[idx]
        return random.choice(self.starts)

    def _span_mean(self, csum: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        return span_mean_from_cumsum(csum, start, end)

    def _proximity_to_active(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        if self.active_indices.size == 0:
            return np.zeros(start.shape[0], dtype=np.float32)
        centers = ((start + end) // 2).astype(np.int64)
        pos = np.searchsorted(self.active_indices, centers)
        left_idx = np.clip(pos - 1, 0, self.active_indices.size - 1)
        right_idx = np.clip(pos, 0, self.active_indices.size - 1)
        left_dist = np.abs(centers - self.active_indices[left_idx])
        right_dist = np.abs(centers - self.active_indices[right_idx])
        min_dist = np.minimum(left_dist, right_dist).astype(np.float32)
        norm = np.maximum(1.0, self.seq_len / 2.0)
        return np.clip(1.0 - (min_dist / norm), 0.0, 1.0)

    def _choose_mask_span(self, s: int, mask_len: int) -> Tuple[int, int, Dict[str, float]]:
        max_m0 = self.seq_len - mask_len
        if max_m0 <= 0:
            return 0, mask_len, {
                "window_mean_activity": float(self.window_mean[0] if self.window_mean.size else 0.0),
                "window_active_ratio": float(self.window_active_ratio[0] if self.window_active_ratio.size else 0.0),
                "mask_mean_activity": 0.0,
                "mask_active_ratio": 0.0,
                "mask_regime": 3,
            }

        candidates = candidate_mask_offsets(self.seq_len, mask_len, self.mask_stride)

        if not self.activity_guided_masking:
            m0 = int(random.choice(candidates.tolist()))
            m1 = m0 + mask_len
            g0 = s + m0
            g1 = s + m1
            mask_mean = float(self._span_mean(self.activity_cumsum, np.array([g0]), np.array([g1]))[0])
            mask_ratio = float(self._span_mean(self.active_flag_cumsum, np.array([g0]), np.array([g1]))[0])
            return m0, m1, {
                "mask_mean_activity": mask_mean,
                "mask_active_ratio": mask_ratio,
                "mask_regime": 3,
            }

        regime = choose_mask_regime(self.regime_probs)
        g0 = s + candidates
        g1 = g0 + mask_len
        mean_act = self._span_mean(self.activity_cumsum, g0, g1)
        ratio = self._span_mean(self.active_flag_cumsum, g0, g1)
        sq_mean = self._span_mean(self.activity_sq_cumsum, g0, g1)
        std_act = np.sqrt(np.clip(sq_mean - mean_act * mean_act, 0.0, None)).astype(np.float32)
        variation = self._span_mean(self.activity_diff_cumsum, g0, g1)
        token_change = self._span_mean(self.token_change_cumsum, g0, g1)
        proximity = self._proximity_to_active(g0, g1)

        weights = compute_mask_candidate_weights(
            regime=regime,
            mean_activity=mean_act,
            active_ratio=ratio,
            std_activity=std_act,
            variation=variation,
            proximity=proximity,
            token_change_mean=token_change,
            dead_window_min_mean=self.dead_window_min_mean,
            dead_window_min_ratio=self.dead_window_min_ratio,
        )
        w_t = torch.from_numpy(weights)
        c_idx = int(torch.multinomial(w_t / w_t.sum(), 1, replacement=True).item())
        m0 = int(candidates[c_idx])
        m1 = m0 + mask_len
        regime_code = {
            "active": 0,
            "transition": 1,
            "low_activity": 2,
            "uniform": 3,
        }[regime]
        return m0, m1, {
            "mask_mean_activity": float(mean_act[c_idx]),
            "mask_active_ratio": float(ratio[c_idx]),
            "mask_regime": float(regime_code),
        }

    def pop_recent_metrics(self, max_items: Optional[int] = None) -> List[Dict[str, float]]:
        if max_items is None:
            max_items = len(self.recent_metrics)
        items = []
        for _ in range(min(max_items, len(self.recent_metrics))):
            items.append(self.recent_metrics.popleft())
        return items

    def __getitem__(self, idx: int):
        s = self._sample_window_start()
        x = self.codes[:, s : s + self.seq_len].clone()
        y = x.clone()

        mask_min, mask_max = self._mask_len_range
        mask_len = random.randint(mask_min, max(mask_min, mask_max))
        mask_len = min(mask_len, self.seq_len)
        if self.bucket_lengths:
            mask_len = resolve_bucket_length(mask_len, self.bucket_lengths)
        m0, m1, mask_meta = self._choose_mask_span(s, mask_len)

        x[:, m0:m1] = self.mask_token

        loss_mask = torch.zeros(self.seq_len, dtype=torch.bool)
        loss_mask[m0:m1] = True

        w_idx = self.start_to_index.get(s, 0)
        self.recent_metrics.append(
            {
                "window_mean_activity": float(self.window_mean[w_idx]) if self.window_mean.size else 0.0,
                "window_active_ratio": float(self.window_active_ratio[w_idx]) if self.window_active_ratio.size else 0.0,
                "mask_mean_activity": float(mask_meta["mask_mean_activity"]),
                "mask_active_ratio": float(mask_meta["mask_active_ratio"]),
                "mask_regime": float(mask_meta["mask_regime"]),
            }
        )

        if not self.return_metadata:
            return x, y, loss_mask
        metadata = {
            "window_start": torch.tensor(int(s), dtype=torch.long),
            "mask_start": torch.tensor(int(m0), dtype=torch.long),
            "mask_len": torch.tensor(int(mask_len), dtype=torch.long),
            "window_len": torch.tensor(int(self.seq_len), dtype=torch.long),
        }
        return x, y, loss_mask, metadata


# Backward-compatible alias for any existing imports.
MaskedSpanDataset = ActivityAwareMaskedSpanDataset


class _CodebookInfillerBase(nn.Module):
    def __init__(
        self,
        K: int,
        bins: int,
        mask_token: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        max_len: int = 2048,
        dropout: float = 0.1,
        boundary_max_distance: int = 128,
        retrieval_conditioning: bool = False,
    ):
        super().__init__()
        self.K = K
        self.bins = bins
        self.vocab = bins + 1
        self.mask_token = int(mask_token)
        self.boundary_max_distance = int(boundary_max_distance)

        self.emb = nn.ModuleList([nn.Embedding(self.vocab, d_model) for _ in range(K)])
        self.pos = nn.Embedding(max_len, d_model)
        self.segment_emb = nn.Embedding(3, d_model)
        self.left_distance_emb = nn.Embedding(2 * self.boundary_max_distance + 1, d_model)
        self.right_distance_emb = nn.Embedding(2 * self.boundary_max_distance + 1, d_model)
        self.head = nn.ModuleList([nn.Linear(d_model, bins) for _ in range(K)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = int(d_model)
        self.retrieval_conditioning = bool(retrieval_conditioning)
        if self.retrieval_conditioning:
            self.retrieval_query_norm = nn.LayerNorm(d_model)
            self.retrieval_memory_norm = nn.LayerNorm(d_model)
            self.retrieval_cross_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.retrieval_output_norm = nn.LayerNorm(d_model)
            self.retrieval_dropout = nn.Dropout(dropout)

    def _embed_tokens(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        segment_ids: torch.Tensor,
        left_dist_idx: torch.Tensor,
        right_dist_idx: torch.Tensor,
    ) -> torch.Tensor:
        h = self.pos(positions)
        for k in range(self.K):
            h = h + self.emb[k](x[:, k, :])
        h = h + self.segment_emb(segment_ids)
        h = h + self.left_distance_emb(left_dist_idx)
        h = h + self.right_distance_emb(right_dist_idx)
        return self.dropout(h)

    def _project_logits(self, h: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.head[k](h) for k in range(self.K)], dim=1)

    def _fuse_retrieval_tokens(
        self,
        retrieval_payload: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.retrieval_conditioning or retrieval_payload is None:
            return None, None, None

        retrieval_tokens = retrieval_payload["tokens"]
        retrieval_scores = retrieval_payload["scores"]
        candidate_mask = retrieval_payload["candidate_mask"].bool()
        retrieval_lengths = retrieval_payload["lengths"].long()
        if retrieval_tokens.dim() != 4:
            raise ValueError(
                f"retrieval tokens must have shape [B,R,K,T], got {tuple(retrieval_tokens.shape)}"
            )
        bsz, top_k, codebooks, steps = retrieval_tokens.shape
        if codebooks != self.K:
            raise ValueError(f"retrieval codebook mismatch: expected K={self.K}, got {codebooks}")

        flat_tokens = retrieval_tokens.reshape(bsz * top_k, codebooks, steps)
        positions = torch.arange(steps, device=retrieval_tokens.device, dtype=torch.long).unsqueeze(0).expand(bsz * top_k, steps)
        fused = self.pos(positions)
        for k in range(self.K):
            fused = fused + self.emb[k](flat_tokens[:, k, :])
        fused = fused.view(bsz, top_k, steps, self.d_model)

        valid_steps = (
            torch.arange(steps, device=retrieval_tokens.device, dtype=torch.long).unsqueeze(0)
            < retrieval_lengths.unsqueeze(1)
        )
        fused = fused * valid_steps.unsqueeze(1).unsqueeze(-1).to(dtype=fused.dtype)

        score_weights = torch.softmax(
            retrieval_scores.masked_fill(~candidate_mask, -1e9),
            dim=1,
        )
        score_weights = torch.where(candidate_mask, score_weights, torch.zeros_like(score_weights))
        score_weights = score_weights / score_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        fused = torch.sum(fused * score_weights.unsqueeze(-1).unsqueeze(-1), dim=1)
        key_padding_mask = ~valid_steps
        retrieval_present = candidate_mask.any(dim=1) & retrieval_lengths.gt(0)
        return fused, key_padding_mask, retrieval_present

    def _apply_retrieval_cross_attention(
        self,
        hidden: torch.Tensor,
        *,
        retrieval_payload: Optional[Dict[str, torch.Tensor]] = None,
        query_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.retrieval_conditioning or retrieval_payload is None:
            return hidden

        retrieval_memory, key_padding_mask, retrieval_present = self._fuse_retrieval_tokens(retrieval_payload)
        if retrieval_memory is None or key_padding_mask is None or retrieval_present is None:
            return hidden
        if not bool(retrieval_present.any().item()):
            return hidden

        selected = retrieval_present.nonzero(as_tuple=False).flatten()
        query = self.retrieval_query_norm(hidden[selected])
        memory = self.retrieval_memory_norm(retrieval_memory[selected])
        attn_out, _ = self.retrieval_cross_attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=key_padding_mask[selected],
            need_weights=False,
        )
        if query_padding_mask is not None:
            attn_out = attn_out.masked_fill(query_padding_mask[selected].unsqueeze(-1), 0.0)

        updated = hidden.clone()
        updated[selected] = self.retrieval_output_norm(
            hidden[selected] + self.retrieval_dropout(attn_out)
        )
        return updated


class JointCodebookInfiller(_CodebookInfillerBase):
    def __init__(
        self,
        K: int,
        bins: int,
        mask_token: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        max_len: int = 2048,
        dropout: float = 0.1,
        boundary_max_distance: int = 128,
        retrieval_conditioning: bool = False,
    ):
        super().__init__(
            K=K,
            bins=bins,
            mask_token=mask_token,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout,
            boundary_max_distance=boundary_max_distance,
            retrieval_conditioning=retrieval_conditioning,
        )

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

    def forward(
        self,
        x: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        left_dist_idx: Optional[torch.Tensor] = None,
        right_dist_idx: Optional[torch.Tensor] = None,
        retrieval_payload: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, K, T = x.shape
        if segment_ids is None or left_dist_idx is None or right_dist_idx is None:
            segment_ids, left_dist_idx, right_dist_idx = build_boundary_condition_tensors_from_mask_token(
                x,
                mask_token=self.mask_token,
                max_distance=self.boundary_max_distance,
            )
        segment_ids = segment_ids.to(device=x.device, dtype=torch.long)
        left_dist_idx = left_dist_idx.to(device=x.device, dtype=torch.long)
        right_dist_idx = right_dist_idx.to(device=x.device, dtype=torch.long)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self._embed_tokens(x, pos, segment_ids, left_dist_idx, right_dist_idx)
        h = self.enc(h)
        h = self._apply_retrieval_cross_attention(h, retrieval_payload=retrieval_payload)
        return self._project_logits(h)


class EncoderDecoderCodebookInfiller(_CodebookInfillerBase):
    def __init__(
        self,
        K: int,
        bins: int,
        mask_token: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        max_len: int = 2048,
        dropout: float = 0.1,
        boundary_max_distance: int = 128,
        retrieval_conditioning: bool = False,
    ):
        super().__init__(
            K=K,
            bins=bins,
            mask_token=mask_token,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
            dropout=dropout,
            boundary_max_distance=boundary_max_distance,
            retrieval_conditioning=retrieval_conditioning,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dec = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        left_dist_idx: Optional[torch.Tensor] = None,
        right_dist_idx: Optional[torch.Tensor] = None,
        retrieval_payload: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, K, T = x.shape
        if segment_ids is None or left_dist_idx is None or right_dist_idx is None:
            segment_ids, left_dist_idx, right_dist_idx = build_boundary_condition_tensors_from_mask_token(
                x,
                mask_token=self.mask_token,
                max_distance=self.boundary_max_distance,
            )
        segment_ids = segment_ids.to(device=x.device, dtype=torch.long)
        left_dist_idx = left_dist_idx.to(device=x.device, dtype=torch.long)
        right_dist_idx = right_dist_idx.to(device=x.device, dtype=torch.long)
        gap_start, gap_end = extract_contiguous_mask_bounds_from_segment_ids(segment_ids)
        full_positions = torch.arange(T, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, T)

        context_lengths = T - (gap_end - gap_start)
        gap_lengths = gap_end - gap_start
        max_ctx = int(context_lengths.max().item())
        max_gap = int(gap_lengths.max().item())
        if max_ctx <= 0 or max_gap <= 0:
            raise ValueError("Encoder-decoder model requires at least one context token and one gap token")

        ctx_tokens = x.new_zeros((B, K, max_ctx))
        ctx_positions = full_positions.new_zeros((B, max_ctx))
        ctx_segment_ids = segment_ids.new_zeros((B, max_ctx))
        ctx_left_idx = left_dist_idx.new_zeros((B, max_ctx))
        ctx_right_idx = right_dist_idx.new_zeros((B, max_ctx))
        ctx_padding_mask = torch.ones((B, max_ctx), device=x.device, dtype=torch.bool)

        gap_tokens = x.new_full((B, K, max_gap), self.mask_token)
        gap_positions = full_positions.new_zeros((B, max_gap))
        gap_segment_ids = segment_ids.new_ones((B, max_gap))
        gap_left_idx = left_dist_idx.new_zeros((B, max_gap))
        gap_right_idx = right_dist_idx.new_zeros((B, max_gap))
        gap_padding_mask = torch.ones((B, max_gap), device=x.device, dtype=torch.bool)
        gap_indices = full_positions.new_zeros((B, max_gap))

        for b in range(B):
            s = int(gap_start[b].item())
            e = int(gap_end[b].item())
            left_len = s
            right_len = T - e
            ctx_len = left_len + right_len
            gap_len = e - s

            if left_len > 0:
                ctx_tokens[b, :, :left_len] = x[b, :, :s]
                ctx_positions[b, :left_len] = full_positions[b, :s]
                ctx_segment_ids[b, :left_len] = segment_ids[b, :s]
                ctx_left_idx[b, :left_len] = left_dist_idx[b, :s]
                ctx_right_idx[b, :left_len] = right_dist_idx[b, :s]
            if right_len > 0:
                ctx_tokens[b, :, left_len:ctx_len] = x[b, :, e:]
                ctx_positions[b, left_len:ctx_len] = full_positions[b, e:]
                ctx_segment_ids[b, left_len:ctx_len] = segment_ids[b, e:]
                ctx_left_idx[b, left_len:ctx_len] = left_dist_idx[b, e:]
                ctx_right_idx[b, left_len:ctx_len] = right_dist_idx[b, e:]
            ctx_padding_mask[b, :ctx_len] = False

            gap_positions[b, :gap_len] = full_positions[b, s:e]
            gap_segment_ids[b, :gap_len] = segment_ids[b, s:e]
            gap_left_idx[b, :gap_len] = left_dist_idx[b, s:e]
            gap_right_idx[b, :gap_len] = right_dist_idx[b, s:e]
            gap_padding_mask[b, :gap_len] = False
            gap_indices[b, :gap_len] = torch.arange(s, e, device=x.device, dtype=torch.long)

        memory = self._embed_tokens(
            ctx_tokens,
            ctx_positions,
            ctx_segment_ids,
            ctx_left_idx,
            ctx_right_idx,
        )
        memory = self.enc(memory, src_key_padding_mask=ctx_padding_mask)

        tgt = self._embed_tokens(
            gap_tokens,
            gap_positions,
            gap_segment_ids,
            gap_left_idx,
            gap_right_idx,
        )
        decoded = self.dec(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=gap_padding_mask,
            memory_key_padding_mask=ctx_padding_mask,
        )
        decoded = self._apply_retrieval_cross_attention(
            decoded,
            retrieval_payload=retrieval_payload,
            query_padding_mask=gap_padding_mask,
        )
        gap_logits = self._project_logits(decoded)

        full_logits = gap_logits.new_zeros((B, K, T, self.bins))
        for b in range(B):
            gap_len = int(gap_lengths[b].item())
            full_logits[b, :, gap_indices[b, :gap_len], :] = gap_logits[b, :, :gap_len, :]
        return full_logits


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        n_ffts: Sequence[int],
        hop_lengths: Sequence[int],
        win_lengths: Sequence[int],
        spectral_convergence_weight: float = 1.0,
        log_magnitude_weight: float = 1.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        if not (len(n_ffts) == len(hop_lengths) == len(win_lengths)):
            raise ValueError("STFT parameter lists must have the same length")
        if len(n_ffts) == 0:
            raise ValueError("STFT parameter lists must be non-empty")
        self.n_ffts = tuple(int(v) for v in n_ffts)
        self.hop_lengths = tuple(int(v) for v in hop_lengths)
        self.win_lengths = tuple(int(v) for v in win_lengths)
        self.spectral_convergence_weight = float(spectral_convergence_weight)
        self.log_magnitude_weight = float(log_magnitude_weight)
        self.eps = float(eps)

    def _stft_mag(self, audio: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
        window = torch.hann_window(win_length, device=audio.device, dtype=audio.dtype)
        spec = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
        return spec.abs().clamp_min(self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
        if pred.dim() != 2:
            raise ValueError(f"MultiResolutionSTFTLoss expects [B,T], got shape={tuple(pred.shape)}")
        total_sc = pred.new_zeros(())
        total_log_mag = pred.new_zeros(())
        for n_fft, hop_length, win_length in zip(self.n_ffts, self.hop_lengths, self.win_lengths):
            pred_mag = self._stft_mag(pred, n_fft, hop_length, win_length)
            target_mag = self._stft_mag(target, n_fft, hop_length, win_length)
            diff = pred_mag - target_mag
            sc_num = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)
            sc_den = torch.linalg.vector_norm(target_mag.reshape(target_mag.shape[0], -1), dim=1).clamp_min(self.eps)
            spectral_convergence = (sc_num / sc_den).mean()
            log_magnitude = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
            total_sc = total_sc + spectral_convergence
            total_log_mag = total_log_mag + log_magnitude

        n_res = float(len(self.n_ffts))
        total_sc = total_sc / n_res
        total_log_mag = total_log_mag / n_res
        total = (
            self.spectral_convergence_weight * total_sc
            + self.log_magnitude_weight * total_log_mag
        )
        return {
            "total": total,
            "spectral_convergence": total_sc,
            "log_magnitude": total_log_mag,
        }


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        set_seed(cfg.seed)
        logger.info("Device: %s", self.device)

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg.tb_dir.mkdir(parents=True, exist_ok=True)
        cfg.samples_dir.mkdir(parents=True, exist_ok=True)
        cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(cfg.tb_dir))

        self._log_hparams()
        self._load_audio()
        self._build_retrieval()
        self._build_validation()
        self._build_dataset()
        self._build_model()
        self._build_optimizer()
        self._build_decoded_loss()

        self.global_step = 0
        self.best_loss = float("inf")
        self.best_val_loss = float("inf")

        # Curriculum state
        if cfg.curriculum:
            self._init_curriculum()

    def _log_hparams(self):
        log_hparams(logger, self.writer, self.cfg)

    def _resolve_gap_frames(
        self,
        frames: int,
        duration_s: float,
        gap_start_s: float,
        gap_end_s: float,
        ann: Optional[dict],
    ) -> List[Tuple[int, int]]:
        fps = frames / max(duration_s, 1e-9)
        gaps_f: List[Tuple[int, int]] = []
        if ann is not None:
            if "gaps" in ann:
                for gap in ann["gaps"]:
                    g0 = max(0, min(frames, int(round(gap["gap_start_s"] * fps))))
                    g1 = max(0, min(frames, int(round(gap["gap_end_s"] * fps))))
                    gaps_f.append((g0, g1))
            elif "gap" in ann:
                gap = ann["gap"]
                g0 = max(0, min(frames, int(round(gap["gap_start_s"] * fps))))
                g1 = max(0, min(frames, int(round(gap["gap_end_s"] * fps))))
                gaps_f.append((g0, g1))
        else:
            g0 = max(0, min(frames, int(round(gap_start_s * fps))))
            g1 = max(0, min(frames, int(round(gap_end_s * fps))))
            gaps_f.append((g0, g1))
        return gaps_f

    def _load_audio_sample(
        self,
        wav_path: str,
        target_sr: int,
        gap_start_s: float,
        gap_end_s: float,
        ann: Optional[dict] = None,
    ) -> Dict[str, Any]:
        import soundfile as sf
        import librosa

        logger.info("Loading audio: %s", wav_path)

        audio, sr = sf.read(wav_path, always_2d=True)
        audio = audio.astype(np.float32).mean(axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        wav = torch.from_numpy(audio).unsqueeze(0)
        wav = wav / (wav.abs().max() + 1e-9)

        duration_s = wav.shape[-1] / target_sr
        logger.info("Audio: %.1fs @ %dHz, samples=%d", duration_s, target_sr, wav.shape[-1])

        codes, scale = self.encoder.encode(wav)
        features = compute_activity_features(
            wav=wav,
            codes=codes,
            activity_smooth_kernel=self.cfg.activity_smooth_kernel,
            activity_low_quantile=self.cfg.activity_low_quantile,
            activity_high_quantile=self.cfg.activity_high_quantile,
        )
        gaps_f = self._resolve_gap_frames(
            frames=int(codes.shape[1]),
            duration_s=duration_s,
            gap_start_s=gap_start_s,
            gap_end_s=gap_end_s,
            ann=ann,
        )
        fps = codes.shape[1] / max(duration_s, 1e-9)
        for i, (g0, g1) in enumerate(gaps_f):
            gap_dur = (g1 - g0) / fps
            logger.info("  Gap %d: frames [%d, %d) = %.2fs", i, g0, g1, gap_dur)

        return {
            "wav": wav,
            "duration_s": duration_s,
            "codes": codes,
            "scale": scale,
            "frames": int(codes.shape[1]),
            "gaps_f": gaps_f,
            **features,
        }

    def _load_audio(self):
        cfg = self.cfg
        self.encoder = AudioEncoder(
            cfg.bandwidth,
            self.device,
            encodec_model=cfg.encodec_model,
            custom_decoder_checkpoint=cfg.custom_decoder_checkpoint,
        )
        sample_data = self._load_audio_sample(
            wav_path=cfg.wav_path,
            target_sr=cfg.target_sr,
            gap_start_s=cfg.gap_start_s,
            gap_end_s=cfg.gap_end_s,
            ann=getattr(cfg, "_annotation", None),
        )

        self.wav = sample_data["wav"]
        self.duration_s = sample_data["duration_s"]
        self.codes = sample_data["codes"]
        self.scale = sample_data["scale"]
        self.K, self.frames = self.codes.shape
        self.bins = self.encoder.bins
        self.mask_token = self.encoder.mask_token
        self.gaps_f = sample_data["gaps_f"]
        self.rms_per_frame = sample_data["rms_per_frame"]
        self.token_change_per_frame = sample_data["token_change_per_frame"]
        self.activity_per_frame = sample_data["activity_per_frame"]
        self.activity_low_thr = sample_data["activity_low_thr"]
        self.activity_high_thr = sample_data["activity_high_thr"]
        self.activity_cumsum = _build_cumsum(self.activity_per_frame)
        self.activity_active_cumsum = _build_cumsum((self.activity_per_frame > self.activity_low_thr).astype(np.float32))
        self.largest_gap_frames = max(g1 - g0 for g0, g1 in self.gaps_f)
        logger.info(
            "Codes: K=%d, F=%d, bins=%d, largest_gap_frames=%d",
            self.K, self.frames, self.bins, self.largest_gap_frames,
        )
        logger.info(
            "Activity: low_thr=%.4f (q=%.2f), high_thr=%.4f (q=%.2f)",
            self.activity_low_thr,
            self.cfg.activity_low_quantile,
            self.activity_high_thr,
            self.cfg.activity_high_quantile,
        )

        # For backward compat, keep gap_f0/gap_f1 pointing to first gap
        self.gap_f0 = self.gaps_f[0][0]
        self.gap_f1 = self.gaps_f[0][1]

    def _build_retrieval(self):
        cfg = self.cfg
        self.retrieval_enabled = bool(cfg.use_retrieval_conditioning)
        self.retrieval_cache: Optional[SameSongRetrievalCache] = None
        self.retrieval_bucket_lengths: Tuple[int, ...] = ()
        self.writer.add_scalar("model/use_retrieval_conditioning", float(self.retrieval_enabled), 0)
        if not self.retrieval_enabled:
            logger.info("Retrieval conditioning disabled")
            return

        if cfg.retrieval_bucket_lengths:
            bucket_lengths = normalize_bucket_lengths(cfg.retrieval_bucket_lengths)
        elif cfg.validation_mask_lengths:
            bucket_lengths = normalize_bucket_lengths(cfg.validation_mask_lengths)
        else:
            midpoint = int((int(cfg.mask_len_min) + int(cfg.mask_len_max)) // 2)
            bucket_lengths = normalize_bucket_lengths([cfg.mask_len_min, midpoint, cfg.mask_len_max])
        if not bucket_lengths:
            raise ValueError("retrieval-enabled runs require at least one retrieval bucket length")
        self.retrieval_bucket_lengths = bucket_lengths

        sample_name = cfg.sample or Path(cfg.wav_path).stem
        self.retrieval_cache = SameSongRetrievalCache(
            waveform=self.wav.squeeze(0),
            codes=self.codes,
            gaps=self.gaps_f,
            target_sr=cfg.target_sr,
            cache_dir=cfg.retrieval_cache_dir,
            sample_name=sample_name,
            feature_cfg=build_retrieval_feature_config(cfg),
            rebuild_cache=cfg.retrieval_rebuild_cache,
            supported_lengths=bucket_lengths,
        )
        logger.info(
            "Retrieval conditioning enabled: sample=%s feature=%s pool=%s top_k=%d exclusion_margin=%d cache_dir=%s bucket_lengths=%s",
            sample_name,
            cfg.retrieval_feature_type,
            cfg.retrieval_pool_mode,
            cfg.retrieval_top_k,
            cfg.retrieval_exclusion_margin_frames,
            cfg.retrieval_cache_dir,
            list(bucket_lengths),
        )
        if cfg.retrieval_prebuild_banks:
            summary = self.retrieval_cache.prebuild_banks(
                bucket_lengths=bucket_lengths,
                left_context_resolver=self._resolve_retrieval_context_frames,
                stride_resolver=self._resolve_retrieval_stride_frames,
                logger=logger,
            )
            logger.info(
                "Retrieval prebuild complete: banks=%d total_entries=%d elapsed=%.2fs",
                len(summary["banks"]),
                summary["total_entries"],
                summary["elapsed_sec"],
            )
        self.writer.add_scalar("retrieval/cache_size", float(self.retrieval_cache.cache_size()), 0)
        self.writer.add_scalar("retrieval/top_k", float(cfg.retrieval_top_k), 0)
        self.writer.add_scalar(
            "retrieval/exclusion_margin_frames",
            float(cfg.retrieval_exclusion_margin_frames),
            0,
        )
        for idx, length in enumerate(bucket_lengths):
            self.writer.add_scalar(f"retrieval/bucket_len_{idx}", float(length), 0)

    def _resolve_retrieval_context_frames(self, mask_len: int) -> Tuple[int, int]:
        cfg = self.cfg
        if cfg.retrieval_left_context_frames is not None:
            left = int(cfg.retrieval_left_context_frames)
        elif cfg.ctx_left is not None:
            left = int(cfg.ctx_left)
        else:
            left = int(mask_len)

        if cfg.retrieval_right_context_frames is not None:
            right = int(cfg.retrieval_right_context_frames)
        elif cfg.ctx_right is not None:
            right = int(cfg.ctx_right)
        else:
            right = int(mask_len)
        return left, right

    def _resolve_retrieval_stride_frames(self, mask_len: int) -> int:
        cfg = self.cfg
        if cfg.retrieval_candidate_stride_frames is not None:
            return int(cfg.retrieval_candidate_stride_frames)
        return max(1, int(mask_len) // 2)

    def _unpack_batch(
        self,
        batch: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            x, y, loss_mask, metadata = batch
            return x, y, loss_mask, metadata
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            x, y, loss_mask = batch
            return x, y, loss_mask, None
        raise ValueError("Expected batch to contain (x, y, loss_mask) or (x, y, loss_mask, metadata)")

    def _build_retrieval_payload(
        self,
        *,
        window_starts: Sequence[int],
        mask_starts: Sequence[int],
        mask_lens: Sequence[int],
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, float]]:
        zero_metrics = {
            "retrieval_best_similarity": 0.0,
            "retrieval_mean_similarity_topk": 0.0,
            "retrieval_topk": float(self.cfg.retrieval_top_k),
            "retrieval_used": 0.0,
        }
        if not getattr(self, "retrieval_enabled", False) or getattr(self, "retrieval_cache", None) is None:
            return None, zero_metrics

        batch_size = len(window_starts)
        if batch_size == 0:
            return None, zero_metrics

        top_k = int(self.cfg.retrieval_top_k)
        max_mask_len = max(int(v) for v in mask_lens)
        tokens = torch.zeros((batch_size, top_k, self.K, max_mask_len), dtype=torch.long)
        scores = torch.zeros((batch_size, top_k), dtype=torch.float32)
        candidate_mask = torch.zeros((batch_size, top_k), dtype=torch.bool)
        lengths = torch.zeros((batch_size,), dtype=torch.long)

        best_scores: List[float] = []
        mean_topk_scores: List[float] = []

        for b_idx, (window_start, mask_start, mask_len) in enumerate(zip(window_starts, mask_starts, mask_lens)):
            mask_len = int(mask_len)
            gap_start = int(window_start) + int(mask_start)
            gap_end = gap_start + mask_len
            left_ctx, right_ctx = self._resolve_retrieval_context_frames(mask_len)
            stride = self._resolve_retrieval_stride_frames(mask_len)
            retrieved = self.retrieval_cache.query(
                query_start_frame=gap_start,
                query_end_frame=gap_end,
                left_context_frames=left_ctx,
                right_context_frames=right_ctx,
                stride_frames=stride,
                top_k=top_k,
                exclusion_margin_frames=int(self.cfg.retrieval_exclusion_margin_frames),
            )
            if not retrieved:
                continue

            lengths[b_idx] = mask_len
            best_scores.append(float(retrieved[0].similarity))
            mean_topk_scores.append(float(np.mean([item.similarity for item in retrieved])))
            for cand_idx, item in enumerate(retrieved[:top_k]):
                fill_tokens = item.entry.fill_tokens
                tokens[b_idx, cand_idx, :, : fill_tokens.shape[1]] = fill_tokens
                scores[b_idx, cand_idx] = float(item.similarity)
                candidate_mask[b_idx, cand_idx] = True

        used = candidate_mask.any(dim=1)
        if not bool(used.any().item()):
            return None, zero_metrics

        payload = {
            "tokens": tokens.to(self.device, non_blocking=True),
            "scores": scores.to(self.device, non_blocking=True),
            "candidate_mask": candidate_mask.to(self.device, non_blocking=True),
            "lengths": lengths.to(self.device, non_blocking=True),
        }
        metrics = {
            "retrieval_best_similarity": float(np.mean(best_scores)) if best_scores else 0.0,
            "retrieval_mean_similarity_topk": float(np.mean(mean_topk_scores)) if mean_topk_scores else 0.0,
            "retrieval_topk": float(top_k),
            "retrieval_used": float(used.float().mean().item()),
        }
        return payload, metrics

    def _build_retrieval_payload_from_metadata(
        self,
        metadata: Optional[Dict[str, torch.Tensor]],
        loss_mask: torch.Tensor,
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Dict[str, float]]:
        if metadata is None:
            return None, {
                "retrieval_best_similarity": 0.0,
                "retrieval_mean_similarity_topk": 0.0,
                "retrieval_topk": float(self.cfg.retrieval_top_k),
                "retrieval_used": 0.0,
            }

        def _to_int_list(value: Any) -> List[int]:
            if isinstance(value, torch.Tensor):
                items = value.detach().cpu().tolist()
            elif isinstance(value, np.ndarray):
                items = value.tolist()
            elif isinstance(value, (list, tuple)):
                items = list(value)
            else:
                items = [value]
            return [int(v) for v in items]

        mask_start_t, gap_end_t = extract_contiguous_mask_bounds(loss_mask)
        inferred_mask_lens = _to_int_list(gap_end_t - mask_start_t)
        window_starts = _to_int_list(metadata["window_start"])
        mask_starts = _to_int_list(metadata.get("mask_start", mask_start_t))
        mask_lens = _to_int_list(metadata.get("mask_len", inferred_mask_lens))
        return self._build_retrieval_payload(
            window_starts=window_starts,
            mask_starts=mask_starts,
            mask_lens=mask_lens,
        )

    def _build_dataset(self):
        cfg = self.cfg
        self.dataset = ActivityAwareMaskedSpanDataset(
            codes=self.codes,
            gaps=self.gaps_f,
            seq_len=cfg.seq_len,
            mask_len_range=(cfg.mask_len_min, cfg.mask_len_max),
            mask_token=self.mask_token,
            activity_per_frame=self.activity_per_frame,
            token_change_per_frame=self.token_change_per_frame,
            activity_low_thr=self.activity_low_thr,
            activity_high_thr=self.activity_high_thr,
            weighted_sampling=cfg.weighted_sampling,
            dead_window_min_mean=cfg.dead_window_min_mean,
            dead_window_min_ratio=cfg.dead_window_min_ratio,
            regime_probs={
                "active": cfg.regime_active_prob,
                "transition": cfg.regime_transition_prob,
                "low_activity": cfg.regime_low_prob,
                "uniform": cfg.regime_uniform_prob,
            },
            blocked_ranges=getattr(self, "validation_holdout_ranges", []),
            mask_stride=cfg.mask_stride,
            activity_guided_masking=cfg.activity_guided_masking,
            return_metadata=cfg.use_retrieval_conditioning,
            bucket_lengths=self.retrieval_bucket_lengths if cfg.use_retrieval_conditioning else (),
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )
        if hasattr(self.dataset, "summary"):
            logger.info("Dataset sampling summary: %s", self.dataset.summary)
            for key, value in self.dataset.summary.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"dataset/{key}", float(value), 0)
        self.writer.add_scalar("train/activity_low_thr", self.activity_low_thr, 0)
        self.writer.add_scalar("train/activity_high_thr", self.activity_high_thr, 0)

    def _build_validation(self):
        cfg = self.cfg
        self.validation_dataloaders: Dict[str, DataLoader] = {}
        self.validation_group_specs: Dict[str, ValidationGroupSpec] = {}
        self.validation_inspection_examples: Dict[str, List[FixedValidationExample]] = {}
        self.validation_holdout_ranges: List[Tuple[int, int]] = []
        self.validation_enabled = cfg.validation_every > 0
        if not self.validation_enabled:
            return

        val_batch_size = cfg.validation_batch_size or cfg.batch_size
        sample_examples: Dict[str, List[FixedValidationExample]]
        validation_metadata: Dict[str, float]
        loader_examples: Dict[str, List[FixedValidationExample]]
        if cfg.validation_strategy == "holdout_regions":
            mask_lengths = tuple(int(v) for v in (cfg.validation_mask_lengths or (cfg.mask_len_min, cfg.mask_len_max)))
            max_ctx = max(int(cfg.ctx_left or 0), int(cfg.ctx_right or 0))
            region_len_frames = cfg.validation_region_len_frames
            if region_len_frames is None:
                region_len_frames = max(cfg.seq_len, max(mask_lengths) + 2 * max_ctx)
            region_min_separation = cfg.validation_region_min_separation_frames
            if region_min_separation is None:
                region_min_separation = cfg.seq_len
            sample_examples, _, self.validation_holdout_ranges, validation_metadata = build_holdout_region_validation_examples(
                codes=self.codes,
                gaps=self.gaps_f,
                seq_len=cfg.seq_len,
                mask_lengths=mask_lengths,
                mask_token=self.mask_token,
                activity_per_frame=self.activity_per_frame,
                activity_low_thr=self.activity_low_thr,
                activity_high_thr=self.activity_high_thr,
                regions_per_band=cfg.validation_regions_per_band,
                region_len_frames=int(region_len_frames),
                region_min_separation_frames=int(region_min_separation),
                examples_per_length_band=cfg.validation_examples_per_length_band,
                mask_stride=cfg.mask_stride,
                seed=cfg.seed + 1009,
                sample_name=cfg.sample or Path(cfg.wav_path).stem,
                dead_window_min_mean=cfg.dead_window_min_mean,
                dead_window_min_ratio=cfg.dead_window_min_ratio,
                bucket_lengths=self.retrieval_bucket_lengths if cfg.use_retrieval_conditioning else (),
            )
            self.validation_group_specs = {
                key: ValidationGroupSpec(
                    band="high_activity" if key.startswith("high_activity") else "low_activity",
                    mask_len=(int(key.rsplit("_", 1)[-1]) if "_len_" in key else None),
                )
                for key in sample_examples
            }
            loader_examples = {
                key: items
                for key, items in sample_examples.items()
                if "_len_" in key
            }
        else:
            sample_examples, self.validation_holdout_ranges = build_fixed_validation_examples(
                codes=self.codes,
                gaps=self.gaps_f,
                seq_len=cfg.seq_len,
                mask_len_range=(cfg.mask_len_min, cfg.mask_len_max),
                mask_token=self.mask_token,
                activity_per_frame=self.activity_per_frame,
                activity_low_thr=self.activity_low_thr,
                activity_high_thr=self.activity_high_thr,
                examples_per_band=cfg.validation_examples_per_band,
                mask_stride=cfg.mask_stride,
                seed=cfg.seed + 1009,
                sample_name=cfg.sample or Path(cfg.wav_path).stem,
                bucket_lengths=self.retrieval_bucket_lengths if cfg.use_retrieval_conditioning else (),
            )
            validation_metadata = {}
            self.validation_group_specs = {
                key: ValidationGroupSpec(band=key, mask_len=None)
                for key in sample_examples
            }
            loader_examples = sample_examples

        if cfg.validation_inspection_enabled:
            limit = int(cfg.validation_inspection_examples_per_group)
            self.validation_inspection_examples = {
                group_name: list(items[:limit])
                for group_name, items in loader_examples.items()
                if len(items) > 0
            }

        holdout_summary = validation_holdout_summary(self.validation_holdout_ranges)
        holdout_summary["holdout_fraction"] = holdout_summary["holdout_frames"] / max(1, self.frames)
        for key, value in {**holdout_summary, **validation_metadata}.items():
            self.writer.add_scalar(f"validation/{key}", float(value), 0)

        for group_name, items in loader_examples.items():
            dataset = FixedMaskedSpanDataset(items, return_metadata=cfg.use_retrieval_conditioning)
            self.validation_dataloaders[group_name] = DataLoader(
                dataset,
                batch_size=val_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
                drop_last=False,
            )
            logger.info("Validation %s summary: %s", group_name, dataset.summary)
            for key, value in dataset.summary.items():
                self.writer.add_scalar(f"validation/{group_name}_{key}", float(value), 0)

    def _build_model(self):
        cfg = self.cfg
        model_cls = EncoderDecoderCodebookInfiller if cfg.use_encoder_decoder else JointCodebookInfiller
        self.model = model_cls(
            K=self.K,
            bins=self.bins,
            mask_token=self.mask_token,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            max_len=cfg.max_len,
            dropout=cfg.dropout,
            boundary_max_distance=cfg.boundary_max_distance,
            retrieval_conditioning=cfg.use_retrieval_conditioning,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model params: %.2fM", n_params / 1e6)
        self.writer.add_scalar("model/params_M", n_params / 1e6, 0)
        self.writer.add_scalar("model/use_encoder_decoder", float(cfg.use_encoder_decoder), 0)

    def _build_model_boundary_tensors(
        self,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return build_boundary_condition_tensors(
            loss_mask,
            max_distance=self.cfg.boundary_max_distance,
        )

    def _build_optimizer(self):
        cfg = self.cfg
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

    def _model_forward(
        self,
        x: torch.Tensor,
        segment_ids: torch.Tensor,
        left_dist_idx: torch.Tensor,
        right_dist_idx: torch.Tensor,
        retrieval_payload: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if retrieval_payload is None:
            return self.model(x, segment_ids, left_dist_idx, right_dist_idx)
        return self.model(
            x,
            segment_ids,
            left_dist_idx,
            right_dist_idx,
            retrieval_payload=retrieval_payload,
        )

    def _build_decoded_loss(self):
        cfg = self.cfg
        self.decoded_loss_enabled = bool(cfg.decoded_loss_enabled and cfg.decoded_loss_weight > 0)
        self.decoded_stft_loss: Optional[MultiResolutionSTFTLoss] = None
        if not self.decoded_loss_enabled:
            return
        if getattr(self.encoder.model, "normalize", False):
            raise NotImplementedError(
                "decoded-domain loss currently assumes normalize=False for the EnCodec decoder path"
            )
        self.decoded_stft_loss = MultiResolutionSTFTLoss(
            n_ffts=cfg.decoded_loss_n_ffts,
            hop_lengths=cfg.decoded_loss_hop_lengths,
            win_lengths=cfg.decoded_loss_win_lengths,
            spectral_convergence_weight=cfg.decoded_loss_spectral_convergence_weight,
            log_magnitude_weight=cfg.decoded_loss_log_magnitude_weight,
        )
        logger.info(
            "Decoded-domain loss enabled (hard argmax path): weight=%.4f every=%d start=%d max_items=%d margin_frames=%d temperature=%.3f (parsed, unused)",
            cfg.decoded_loss_weight,
            cfg.decoded_loss_every,
            cfg.decoded_loss_start_step,
            cfg.decoded_loss_max_items,
            cfg.decoded_loss_margin_frames,
            cfg.decoded_loss_temperature,
        )

    # --- Curriculum learning ---

    def _init_curriculum(self):
        cfg = self.cfg
        self.curriculum_start = cfg.curriculum_start_mask if cfg.curriculum_start_mask is not None \
            else min(cfg.mask_len_max, 128)
        self.curriculum_end = cfg.curriculum_end_mask if cfg.curriculum_end_mask is not None \
            else self.largest_gap_frames
        self.curriculum_warmup_steps = int(cfg.curriculum_warmup_frac * cfg.total_steps)

        logger.info(
            "Curriculum enabled: start_mask=%d, end_mask=%d, warmup_steps=%d, schedule=%s",
            self.curriculum_start, self.curriculum_end,
            self.curriculum_warmup_steps, cfg.curriculum_schedule,
        )

    def _curriculum_update(self, step: int):
        """Update mask range based on curriculum progress."""
        cfg = self.cfg
        warmup = self.curriculum_warmup_steps
        total = cfg.total_steps

        if step <= warmup:
            progress = 0.0
        else:
            progress = min(1.0, (step - warmup) / max(1, total - warmup))

        if cfg.curriculum_schedule == "cosine":
            # Cosine: slow start, accelerate, slow finish
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        # else: linear (progress stays as-is)

        mask_len_max_current = int(
            self.curriculum_start + (self.curriculum_end - self.curriculum_start) * progress
        )
        mask_len_max_current = max(self.curriculum_start, min(self.curriculum_end, mask_len_max_current))

        mask_len_min_current = max(32, mask_len_max_current // 4)

        # Update dataset mask range
        self.dataset.update_mask_range(mask_len_min_current, mask_len_max_current)

        # Log to TensorBoard
        self.writer.add_scalar("curriculum/progress", progress, step)
        self.writer.add_scalar("curriculum/mask_len_min", mask_len_min_current, step)
        self.writer.add_scalar("curriculum/mask_len_max", mask_len_max_current, step)
        self.writer.add_scalar("curriculum/largest_gap_frames", self.largest_gap_frames, step)

        return mask_len_min_current, mask_len_max_current, progress

    # --- LR scheduling ---

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

    def _should_apply_decoded_loss(self, step: int) -> bool:
        if not getattr(self, "decoded_loss_enabled", False):
            return False
        cfg = self.cfg
        if step < cfg.decoded_loss_start_step:
            return False
        return (step - cfg.decoded_loss_start_step) % cfg.decoded_loss_every == 0

    def _compute_decoded_domain_loss(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not getattr(self, "decoded_loss_enabled", False) or self.decoded_stft_loss is None:
            zero = logits.new_zeros(())
            return zero, {
                "decoded_loss_total": 0.0,
                "decoded_loss_waveform_l1": 0.0,
                "decoded_loss_stft": 0.0,
                "decoded_loss_spectral_convergence": 0.0,
                "decoded_loss_log_magnitude": 0.0,
                "decoded_loss_items": 0.0,
            }

        cfg = self.cfg
        logits_f = logits.float()
        pred_tokens = logits.argmax(dim=-1)
        if loss_mask.dim() == 1:
            mask_bt = loss_mask.unsqueeze(0).expand(logits.shape[0], -1)
        else:
            mask_bt = loss_mask
        mask_bt = mask_bt.bool()
        mask_lengths = mask_bt.sum(dim=1)
        valid = torch.nonzero(mask_lengths > 0, as_tuple=False).flatten()
        if valid.numel() == 0:
            zero = logits.new_zeros(())
            return zero, {
                "decoded_loss_total": 0.0,
                "decoded_loss_waveform_l1": 0.0,
                "decoded_loss_stft": 0.0,
                "decoded_loss_spectral_convergence": 0.0,
                "decoded_loss_log_magnitude": 0.0,
                "decoded_loss_items": 0.0,
            }

        ranked = valid[torch.argsort(mask_lengths[valid], descending=True)]
        selected = ranked[: cfg.decoded_loss_max_items]

        total_wave_l1 = logits.new_zeros((), dtype=torch.float32)
        total_stft = logits.new_zeros((), dtype=torch.float32)
        total_sc = logits.new_zeros((), dtype=torch.float32)
        total_log_mag = logits.new_zeros((), dtype=torch.float32)

        with torch.autocast(device_type=self.device.type, enabled=False):
            for b_idx in selected.tolist():
                sample_mask = mask_bt[b_idx]
                gap_start, gap_end = extract_contiguous_mask_bounds(sample_mask)
                gap_start_t = int(gap_start[0].item())
                gap_end_t = int(gap_end[0].item())
                start_t = max(0, gap_start_t - cfg.decoded_loss_margin_frames)
                end_t = min(int(sample_mask.shape[0]), gap_end_t + cfg.decoded_loss_margin_frames)

                target_codes = y[b_idx : b_idx + 1, :, start_t:end_t]
                pred_codes = x[b_idx : b_idx + 1, :, start_t:end_t].clone()
                local_gap_start = gap_start_t - start_t
                local_gap_end = gap_end_t - start_t
                pred_codes[:, :, local_gap_start:local_gap_end] = pred_tokens[
                    b_idx : b_idx + 1,
                    :,
                    gap_start_t:gap_end_t,
                ]
                logits_slice = logits_f[b_idx : b_idx + 1, :, start_t:end_t, :]
                logits_gap = logits_f[b_idx : b_idx + 1, :, gap_start_t:gap_end_t, :]

                with torch.no_grad():
                    target_embeddings = self.encoder.codes_to_embeddings(target_codes).detach()
                    hard_embeddings = self.encoder.codes_to_embeddings(pred_codes).detach()
                soft_gap_embeddings = self.encoder.logits_to_embeddings(logits_gap)
                hard_gap_embeddings = hard_embeddings[:, :, local_gap_start:local_gap_end]
                st_gap_embeddings = soft_gap_embeddings + (hard_gap_embeddings - soft_gap_embeddings).detach()
                pred_embeddings = target_embeddings.to(dtype=logits_f.dtype).clone()
                pred_embeddings[:, :, local_gap_start:local_gap_end] = st_gap_embeddings

                pred_audio = self.encoder.decode_embeddings(pred_embeddings).squeeze(1)
                with torch.no_grad():
                    target_audio = self.encoder.decode_embeddings(target_embeddings).squeeze(1).detach()

                wave_l1 = F.l1_loss(pred_audio, target_audio)
                stft_terms = self.decoded_stft_loss(pred_audio, target_audio)

                total_wave_l1 = total_wave_l1 + wave_l1
                total_stft = total_stft + stft_terms["total"]
                total_sc = total_sc + stft_terms["spectral_convergence"]
                total_log_mag = total_log_mag + stft_terms["log_magnitude"]

        denom = float(len(selected))
        avg_wave_l1 = total_wave_l1 / denom
        avg_stft = total_stft / denom
        avg_sc = total_sc / denom
        avg_log_mag = total_log_mag / denom
        total = cfg.decoded_loss_weight * (
            cfg.decoded_loss_waveform_l1_weight * avg_wave_l1
            + cfg.decoded_loss_stft_weight * avg_stft
        )
        return total, {
            "decoded_loss_total": float(total.detach().item()),
            "decoded_loss_waveform_l1": float(avg_wave_l1.detach().item()),
            "decoded_loss_stft": float(avg_stft.detach().item()),
            "decoded_loss_spectral_convergence": float(avg_sc.detach().item()),
            "decoded_loss_log_magnitude": float(avg_log_mag.detach().item()),
            "decoded_loss_items": float(len(selected)),
        }

    def _compute_training_losses(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        y: torch.Tensor,
        loss_mask: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        token_loss = self._compute_loss(logits, y, loss_mask)
        decoded_loss = logits.new_zeros(())
        decoded_metrics = {
            "decoded_loss_total": 0.0,
            "decoded_loss_waveform_l1": 0.0,
            "decoded_loss_stft": 0.0,
            "decoded_loss_spectral_convergence": 0.0,
            "decoded_loss_log_magnitude": 0.0,
            "decoded_loss_items": 0.0,
        }
        if self._should_apply_decoded_loss(step):
            decoded_loss, decoded_metrics = self._compute_decoded_domain_loss(x, logits, y, loss_mask)
        total_loss = token_loss + decoded_loss
        return total_loss, token_loss, decoded_metrics

    def _compute_masked_metrics(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Dict[str, float]:
        B, K, T, _ = logits.shape
        if loss_mask.dim() == 1:
            lm = loss_mask.unsqueeze(0).expand(B, -1)
        else:
            lm = loss_mask
        mask_exp = lm.unsqueeze(1).expand(B, K, T)

        preds = logits.argmax(dim=-1)
        n_masked = int(mask_exp.sum().item())
        if n_masked > 0:
            acc_top1 = float((preds[mask_exp] == y[mask_exp]).float().mean().item())
            top5 = logits.topk(5, dim=-1).indices
            hits5 = (top5 == y.unsqueeze(-1)).any(dim=-1)
            acc_top5 = float(hits5[mask_exp].float().mean().item())
        else:
            acc_top1 = 0.0
            acc_top5 = 0.0
        return {
            "acc_top1": acc_top1,
            "acc_top5": acc_top5,
            "masked_tokens": float(n_masked),
        }

    @torch.no_grad()
    def _decode_validation_window_audio(self, codes: torch.Tensor) -> np.ndarray:
        embeddings = self.encoder.codes_to_embeddings(codes.unsqueeze(0))
        audio = self.encoder.decode_embeddings(embeddings, scale=None)
        return audio.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def _predict_validation_window(self, example: FixedValidationExample) -> torch.Tensor:
        xb = example.x.unsqueeze(0).to(self.device, non_blocking=True)
        seg_ids, left_idx, right_idx = self._build_model_boundary_tensors(example.loss_mask.to(self.device))
        retrieval_payload, _ = self._build_retrieval_payload(
            window_starts=[int(example.window_start)],
            mask_starts=[int(example.mask_start)],
            mask_lens=[int(example.mask_len)],
        )
        logits = self._model_forward(xb, seg_ids, left_idx, right_idx, retrieval_payload=retrieval_payload)
        pred = logits.argmax(dim=-1).squeeze(0).detach().cpu()
        filled = example.y.clone()
        filled[:, example.loss_mask] = pred[:, example.loss_mask]
        return filled

    def _save_validation_artifacts(
        self,
        step: int,
        group_name: str,
        example_id: str,
        example: FixedValidationExample,
        pred_codes: torch.Tensor,
        target_audio: np.ndarray,
        pred_audio: np.ndarray,
        crop_frame_bounds: Tuple[int, int],
        crop_sample_bounds: Tuple[int, int],
    ) -> Dict[str, Any]:
        step_dir = self.cfg.samples_dir / "validation" / f"step_{step}"
        example_dir = step_dir / group_name / example_id
        example_dir.mkdir(parents=True, exist_ok=True)
        mask_sample_bounds = frame_bounds_to_sample_bounds(
            total_samples=min(target_audio.shape[0], pred_audio.shape[0]),
            total_frames=int(example.y.shape[1]),
            start_frame=int(example.mask_start),
            end_frame=int(example.mask_start + example.mask_len),
        )
        target_crop = target_audio[crop_sample_bounds[0] : crop_sample_bounds[1]]
        pred_crop = pred_audio[crop_sample_bounds[0] : crop_sample_bounds[1]]
        files = {
            "bundle": "bundle.pt",
            "metadata": "metadata.json",
            "pred_window_wav": "pred_window.wav",
            "target_window_wav": "target_window.wav",
            "pred_gap_crop_wav": "pred_gap_crop.wav",
            "target_gap_crop_wav": "target_gap_crop.wav",
        }
        bundle = {
            "x_masked_codes": example.x.clone(),
            "y_target_codes": example.y.clone(),
            "pred_filled_codes": pred_codes.clone(),
            "loss_mask": example.loss_mask.clone(),
            "mask_start": int(example.mask_start),
            "mask_len": int(example.mask_len),
            "window_start": int(example.window_start),
            "window_end": int(example.window_start + example.y.shape[1]),
            "group_name": group_name,
            "band": example.band,
            "sample_name": example.sample_name,
            "step": int(step),
            "target_window_audio": target_audio.astype(np.float32, copy=False),
            "pred_window_audio": pred_audio.astype(np.float32, copy=False),
            "crop_frame_bounds": tuple(int(v) for v in crop_frame_bounds),
            "crop_sample_bounds": tuple(int(v) for v in crop_sample_bounds),
            "mask_sample_bounds": tuple(int(v) for v in mask_sample_bounds),
        }
        torch.save(bundle, example_dir / files["bundle"])
        save_waveform(example_dir / files["target_window_wav"], target_audio, self.cfg.target_sr)
        save_waveform(example_dir / files["pred_window_wav"], pred_audio, self.cfg.target_sr)
        save_waveform(example_dir / files["target_gap_crop_wav"], target_crop, self.cfg.target_sr)
        save_waveform(example_dir / files["pred_gap_crop_wav"], pred_crop, self.cfg.target_sr)

        metadata = {
            "example_id": example_id,
            "group_name": group_name,
            "band": example.band,
            "sample_name": example.sample_name,
            "run_name": self.cfg.run_name,
            "step": int(step),
            "target_sr": int(self.cfg.target_sr),
            "window_start": int(example.window_start),
            "window_end": int(example.window_start + example.y.shape[1]),
            "mask_start": int(example.mask_start),
            "mask_len": int(example.mask_len),
            "crop_frame_bounds": [int(v) for v in crop_frame_bounds],
            "crop_sample_bounds": [int(v) for v in crop_sample_bounds],
            "files": files,
        }
        with open(example_dir / files["metadata"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return {
            **metadata,
            "artifact_dir": str(example_dir),
        }

    @torch.no_grad()
    def _log_validation_inspection(self, step: int) -> None:
        if not self.cfg.validation_inspection_enabled:
            return
        inspection_examples = getattr(self, "validation_inspection_examples", {})
        if not inspection_examples:
            return
        manifest_examples: List[Dict[str, Any]] = []
        step_dir = self.cfg.samples_dir / "validation" / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        token_fps = float(getattr(self.encoder, "frame_rate", 1))

        for group_name, examples in inspection_examples.items():
            for idx, example in enumerate(examples):
                example_id = make_validation_example_id(example, group_name, idx)
                pred_codes = self._predict_validation_window(example)
                target_audio = self._decode_validation_window_audio(example.y)
                pred_audio = self._decode_validation_window_audio(pred_codes)

                crop_frame_bounds = compute_validation_crop_bounds(
                    seq_len=int(example.y.shape[1]),
                    mask_start=int(example.mask_start),
                    mask_len=int(example.mask_len),
                    crop_context_frames=self.cfg.validation_crop_context_frames,
                )
                total_samples = min(target_audio.shape[0], pred_audio.shape[0])
                crop_sample_bounds = frame_bounds_to_sample_bounds(
                    total_samples=total_samples,
                    total_frames=int(example.y.shape[1]),
                    start_frame=int(crop_frame_bounds[0]),
                    end_frame=int(crop_frame_bounds[1]),
                )
                mask_sample_bounds = frame_bounds_to_sample_bounds(
                    total_samples=total_samples,
                    total_frames=int(example.y.shape[1]),
                    start_frame=int(example.mask_start),
                    end_frame=int(example.mask_start + example.mask_len),
                )

                title_prefix = (
                    f"{example.sample_name} | {group_name} | step={step} | "
                    f"window={example.window_start}:{example.window_start + example.y.shape[1]} | "
                    f"mask={example.mask_start}:{example.mask_start + example.mask_len}"
                )
                full_time_offset_s = float(example.window_start) / token_fps
                crop_time_offset_s = float(example.window_start + crop_frame_bounds[0]) / token_fps
                target_crop_audio = target_audio[crop_sample_bounds[0] : crop_sample_bounds[1]]
                pred_crop_audio = pred_audio[crop_sample_bounds[0] : crop_sample_bounds[1]]

                full_fig = make_log_spectrogram_comparison_figure(
                    target_audio=target_audio,
                    pred_audio=pred_audio,
                    sr=self.cfg.target_sr,
                    title_prefix=title_prefix,
                    time_offset_s=full_time_offset_s,
                )
                crop_fig = make_log_spectrogram_comparison_figure(
                    target_audio=target_crop_audio,
                    pred_audio=pred_crop_audio,
                    sr=self.cfg.target_sr,
                    title_prefix=f"{title_prefix} | crop",
                    time_offset_s=crop_time_offset_s,
                )
                wave_fig = make_waveform_comparison_figure(
                    target_audio=target_audio,
                    pred_audio=pred_audio,
                    sr=self.cfg.target_sr,
                    title_prefix=title_prefix,
                    mask_sample_bounds=mask_sample_bounds,
                    crop_sample_bounds=crop_sample_bounds,
                    time_offset_s=full_time_offset_s,
                )

                figure_prefix = f"validation_inspect/{group_name}/{example_id}"
                if hasattr(self.writer, "add_figure"):
                    self.writer.add_figure(f"{figure_prefix}/spectrogram_full", full_fig, step)
                    self.writer.add_figure(f"{figure_prefix}/spectrogram_gap_crop", crop_fig, step)
                    self.writer.add_figure(f"{figure_prefix}/waveform", wave_fig, step)
                plt.close(full_fig)
                plt.close(crop_fig)
                plt.close(wave_fig)

                if self.cfg.validation_save_artifacts:
                    manifest_examples.append(
                        self._save_validation_artifacts(
                            step=step,
                            group_name=group_name,
                            example_id=example_id,
                            example=example,
                            pred_codes=pred_codes,
                            target_audio=target_audio,
                            pred_audio=pred_audio,
                            crop_frame_bounds=crop_frame_bounds,
                            crop_sample_bounds=crop_sample_bounds,
                        )
                    )
                else:
                    manifest_examples.append(
                        {
                            "example_id": example_id,
                            "group_name": group_name,
                            "band": example.band,
                            "sample_name": example.sample_name,
                            "run_name": self.cfg.run_name,
                            "step": int(step),
                            "window_start": int(example.window_start),
                            "window_end": int(example.window_start + example.y.shape[1]),
                            "mask_start": int(example.mask_start),
                            "mask_len": int(example.mask_len),
                            "crop_frame_bounds": [int(v) for v in crop_frame_bounds],
                            "crop_sample_bounds": [int(v) for v in crop_sample_bounds],
                        }
                    )

        with open(step_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "step": int(step),
                    "run_name": self.cfg.run_name,
                    "examples": manifest_examples,
                },
                f,
                indent=2,
            )
        if hasattr(self.writer, "flush"):
            self.writer.flush()

    def run_validation(self, step: int) -> Optional[Dict[str, float]]:
        if not getattr(self, "validation_enabled", False):
            return None

        self.model.eval()
        use_amp = self.device.type == "cuda"
        band_totals: Dict[str, Dict[str, float]] = {
            "high_activity": {"loss": 0.0, "acc_top1": 0.0, "acc_top5": 0.0, "count": 0.0},
            "low_activity": {"loss": 0.0, "acc_top1": 0.0, "acc_top5": 0.0, "count": 0.0},
        }
        with torch.no_grad():
            group_specs = getattr(self, "validation_group_specs", {})
            for group_name, loader in self.validation_dataloaders.items():
                spec = group_specs.get(group_name, ValidationGroupSpec(band=group_name, mask_len=None))
                total_loss = 0.0
                total_acc1 = 0.0
                total_acc5 = 0.0
                count = 0
                for batch in loader:
                    x, y, loss_mask, metadata = self._unpack_batch(batch)
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    loss_mask = loss_mask.to(self.device, non_blocking=True).bool()
                    seg_ids, left_idx, right_idx = self._build_model_boundary_tensors(loss_mask)
                    retrieval_payload, _ = self._build_retrieval_payload_from_metadata(metadata, loss_mask)
                    with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                        logits = self._model_forward(x, seg_ids, left_idx, right_idx, retrieval_payload=retrieval_payload)
                        loss = self._compute_loss(logits, y, loss_mask)
                    metrics = self._compute_masked_metrics(logits, y, loss_mask)
                    batch_size = int(x.shape[0])
                    total_loss += float(loss.item()) * batch_size
                    total_acc1 += metrics["acc_top1"] * batch_size
                    total_acc5 += metrics["acc_top5"] * batch_size
                    count += batch_size

                avg_loss = total_loss / max(1, count)
                avg_acc1 = total_acc1 / max(1, count)
                avg_acc5 = total_acc5 / max(1, count)
                avg_ppl = math.exp(min(avg_loss, 20.0))
                prefix_root = "high" if spec.band == "high_activity" else "low"
                if spec.mask_len is None:
                    prefix = f"val/{prefix_root}"
                else:
                    prefix = f"val/{prefix_root}_len_{spec.mask_len}"
                self.writer.add_scalar(f"{prefix}_loss", avg_loss, step)
                self.writer.add_scalar(f"{prefix}_nll", avg_loss, step)
                self.writer.add_scalar(f"{prefix}_ppl", avg_ppl, step)
                self.writer.add_scalar(f"{prefix}_acc_top1", avg_acc1, step)
                self.writer.add_scalar(f"{prefix}_acc_top5", avg_acc5, step)

                band_totals[spec.band]["loss"] += avg_loss * count
                band_totals[spec.band]["acc_top1"] += avg_acc1 * count
                band_totals[spec.band]["acc_top5"] += avg_acc5 * count
                band_totals[spec.band]["count"] += count

        band_results: Dict[str, Dict[str, float]] = {}
        for band, totals in band_totals.items():
            count = max(1.0, totals["count"])
            avg_loss = totals["loss"] / count
            avg_acc1 = totals["acc_top1"] / count
            avg_acc5 = totals["acc_top5"] / count
            avg_ppl = math.exp(min(avg_loss, 20.0))
            band_results[band] = {
                "loss": avg_loss,
                "nll": avg_loss,
                "ppl": avg_ppl,
                "acc_top1": avg_acc1,
                "acc_top5": avg_acc5,
            }
            prefix = "val/high" if band == "high_activity" else "val/low"
            self.writer.add_scalar(f"{prefix}_loss", avg_loss, step)
            self.writer.add_scalar(f"{prefix}_nll", avg_loss, step)
            self.writer.add_scalar(f"{prefix}_ppl", avg_ppl, step)
            self.writer.add_scalar(f"{prefix}_acc_top1", avg_acc1, step)
            self.writer.add_scalar(f"{prefix}_acc_top5", avg_acc5, step)

        combined_loss = float(np.mean([band_results["high_activity"]["loss"], band_results["low_activity"]["loss"]]))
        self.writer.add_scalar("val/combined_loss", combined_loss, step)
        self._log_validation_inspection(step)
        self.model.train()

        if combined_loss < self.best_val_loss:
            self.best_val_loss = combined_loss
            self.save_checkpoint("best_val")

        result = {
            "combined_loss": combined_loss,
            "high_loss": band_results["high_activity"]["loss"],
            "low_loss": band_results["low_activity"]["loss"],
        }
        return result

    def save_checkpoint(self, tag: str = "latest"):
        path = self.cfg.checkpoint_dir / f"{tag}.pt"
        save_training_checkpoint(
            path,
            step=self.global_step,
            model_key="model",
            model_state=self.model.state_dict(),
            optimizer=self.optimizer,
            scaler=self.scaler,
            cfg=self.cfg,
            extra={
                "best_loss": self.best_loss,
                "best_val_loss": self.best_val_loss,
            },
        )
        logger.info("Saved checkpoint: %s (step %d)", path, self.global_step)

    def load_checkpoint(self, path: str):
        ckpt = load_training_checkpoint(path, self.device)
        ckpt_cfg = ckpt.get("config") or {}
        ckpt_use_encoder_decoder = bool(ckpt_cfg.get("use_encoder_decoder", False))
        if ckpt_use_encoder_decoder != bool(self.cfg.use_encoder_decoder):
            raise ValueError(
                "Checkpoint architecture mismatch: checkpoint use_encoder_decoder="
                f"{ckpt_use_encoder_decoder}, current config use_encoder_decoder={self.cfg.use_encoder_decoder}"
            )
        model_state = ckpt["model"]
        compat_mode = False
        try:
            self.model.load_state_dict(model_state)
        except RuntimeError as exc:
            incompatible = self.model.load_state_dict(model_state, strict=False)
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            allowed_missing = {
                "segment_emb.weight",
                "left_distance_emb.weight",
                "right_distance_emb.weight",
            }
            allowed_prefixes = (
                "retrieval_query_norm.",
                "retrieval_memory_norm.",
                "retrieval_cross_attn.",
                "retrieval_output_norm.",
            )
            allowed_missing_ok = all(
                key in allowed_missing or any(key.startswith(prefix) for prefix in allowed_prefixes)
                for key in missing
            )
            if unexpected or not allowed_missing_ok:
                raise exc
            compat_mode = True
            logger.warning(
                "Loaded checkpoint with newly initialized compatibility keys=%s",
                sorted(missing),
            )
        if compat_mode:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                logger.warning(
                    "Checkpoint optimizer state is incompatible with the current parameter set; using fresh optimizer state"
                )
        else:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["step"]
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        restore_rng_state(ckpt.get("rng_state"))
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

            # Curriculum update (before sampling)
            if cfg.curriculum:
                cur_min, cur_max, cur_progress = self._curriculum_update(step)

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            x, y, loss_mask, metadata = self._unpack_batch(batch)

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            loss_mask = loss_mask.to(self.device, non_blocking=True).bool()
            seg_ids, left_idx, right_idx = self._build_model_boundary_tensors(loss_mask)
            retrieval_payload, retrieval_metrics = self._build_retrieval_payload_from_metadata(metadata, loss_mask)

            lr = self._get_lr(step)
            self._set_lr(lr)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                logits = self._model_forward(x, seg_ids, left_idx, right_idx, retrieval_payload=retrieval_payload)
                total_loss, token_loss, decoded_metrics = self._compute_training_losses(
                    x,
                    logits,
                    y,
                    loss_mask,
                    step=step,
                )

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip).item()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                train_metrics = self._compute_masked_metrics(logits, y, loss_mask)
                correct = train_metrics["acc_top1"]
                acc5 = train_metrics["acc_top5"]

            loss_val = total_loss.item()
            token_loss_val = token_loss.item()
            nll = token_loss_val
            ppl = math.exp(min(token_loss_val, 20.0))

            running_loss += loss_val
            running_acc += correct
            running_acc5 += acc5
            running_ppl += ppl
            running_nll += nll
            running_count += 1

            self.writer.add_scalar("train/loss", loss_val, step)
            self.writer.add_scalar("train/token_loss", token_loss_val, step)
            self.writer.add_scalar("train/acc_top1", correct, step)
            self.writer.add_scalar("train/acc_top5", acc5, step)
            self.writer.add_scalar("train/ppl", ppl, step)
            self.writer.add_scalar("train/nll", nll, step)
            self.writer.add_scalar("train/lr", lr, step)
            self.writer.add_scalar("train/grad_norm", grad_norm, step)
            self.writer.add_scalar("train/decoded_loss", decoded_metrics["decoded_loss_total"], step)
            self.writer.add_scalar("train/decoded_loss_waveform_l1", decoded_metrics["decoded_loss_waveform_l1"], step)
            self.writer.add_scalar("train/decoded_loss_stft", decoded_metrics["decoded_loss_stft"], step)
            self.writer.add_scalar(
                "train/decoded_loss_spectral_convergence",
                decoded_metrics["decoded_loss_spectral_convergence"],
                step,
            )
            self.writer.add_scalar(
                "train/decoded_loss_log_magnitude",
                decoded_metrics["decoded_loss_log_magnitude"],
                step,
            )
            self.writer.add_scalar("train/decoded_loss_items", decoded_metrics["decoded_loss_items"], step)
            self.writer.add_scalar("train/retrieval_best_similarity", retrieval_metrics["retrieval_best_similarity"], step)
            self.writer.add_scalar(
                "train/retrieval_mean_similarity_topk",
                retrieval_metrics["retrieval_mean_similarity_topk"],
                step,
            )
            self.writer.add_scalar("train/retrieval_topk", retrieval_metrics["retrieval_topk"], step)
            self.writer.add_scalar("train/retrieval_used", retrieval_metrics["retrieval_used"], step)

            if hasattr(self.dataset, "pop_recent_metrics"):
                sample_metrics = self.dataset.pop_recent_metrics(cfg.batch_size)
            else:
                sample_metrics = []
            if sample_metrics:
                w_mean = float(np.mean([m["window_mean_activity"] for m in sample_metrics]))
                w_ratio = float(np.mean([m["window_active_ratio"] for m in sample_metrics]))
                m_mean = float(np.mean([m["mask_mean_activity"] for m in sample_metrics]))
                m_ratio = float(np.mean([m["mask_active_ratio"] for m in sample_metrics]))
                m_regime = float(np.mean([m["mask_regime"] for m in sample_metrics]))
                self.writer.add_scalar("train/sample_window_mean_activity", w_mean, step)
                self.writer.add_scalar("train/sample_window_active_ratio", w_ratio, step)
                self.writer.add_scalar("train/sample_mask_mean_activity", m_mean, step)
                self.writer.add_scalar("train/sample_mask_active_ratio", m_ratio, step)
                self.writer.add_scalar("train/sample_mask_regime", m_regime, step)

                reg_hist = defaultdict(int)
                for m in sample_metrics:
                    reg_hist[int(m["mask_regime"])] += 1
                total_reg = max(1, len(sample_metrics))
                for reg_id, reg_name in [(0, "active"), (1, "transition"), (2, "low_activity"), (3, "uniform")]:
                    self.writer.add_scalar(f"train/sample_mask_regime_frac_{reg_name}", reg_hist[reg_id] / total_reg, step)

            postfix = dict(loss=f"{loss_val:.4f}", acc=f"{correct:.3f}", ppl=f"{ppl:.1f}", lr=f"{lr:.2e}")
            if cfg.curriculum:
                postfix["mask"] = f"{cur_min}-{cur_max}"
            pbar.set_postfix(**postfix)

            if step % cfg.log_every == 0:
                avg_loss = running_loss / running_count
                avg_acc = running_acc / running_count
                avg_acc5 = running_acc5 / running_count
                avg_ppl = running_ppl / running_count
                avg_nll = running_nll / running_count
                dt = time.time() - t0
                steps_per_sec = running_count / dt

                cur_info = ""
                if cfg.curriculum:
                    cur_info = f" | mask [{cur_min},{cur_max}]"

                # logger.info(
                #     "step %5d | loss %.4f | acc1 %.3f | acc5 %.3f | ppl %.1f | lr %.2e | grad_norm %.2f | %.1f steps/s%s",
                #     step, avg_loss, avg_acc, avg_acc5, avg_ppl, lr, grad_norm, steps_per_sec, cur_info,
                # )
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

            if self.validation_enabled and cfg.validation_every > 0 and step % cfg.validation_every == 0:
                self.run_validation(step)

            if cfg.test_fill_every > 0 and step % cfg.test_fill_every == 0:
                logger.info("Running test inpaint (all gaps) at step %d", step)
                wav_filled = self.inpaint_all_gaps()
                self.log_spectrograms(wav_filled)
                self.model.train()

        self.save_checkpoint("final")
        self.writer.close()
        logger.info("Training complete. Best loss: %.4f", self.best_loss)

    # --- Inpainting ---

    @torch.no_grad()
    def inpaint_all_gaps(self, output_path: Optional[str] = None) -> np.ndarray:
        """Inpaint ALL gaps sequentially.

        For each gap:
        1) Compute maximum possible context within max_len
        2) Extract window [L, R]
        3) Mask the gap region
        4) Run iterative refinement
        5) Write predicted tokens back into codes

        Returns the reconstructed wav as a numpy array.
        """
        cfg = self.cfg
        self.model.eval()

        codes_filled = self.codes.clone()

        for i, (g0, g1) in enumerate(self.gaps_f):
            gap_len = g1 - g0
            logger.info(
                "Inpainting gap %d/%d: frames [%d, %d), len=%d",
                i + 1, len(self.gaps_f), g0, g1, gap_len,
            )

            # Compute maximum context within max_len
            if cfg.ctx_left is not None and cfg.ctx_right is not None:
                ctx_left = cfg.ctx_left
                ctx_right = cfg.ctx_right
            else:
                budget = max(0, cfg.max_len - gap_len - 16)
                ctx_left = budget // 2
                ctx_right = budget - ctx_left

            L = max(0, g0 - ctx_left)
            R = min(self.frames, g1 + ctx_right)
            local_g0 = g0 - L
            local_g1 = g1 - L

            logger.info(
                "  Context window: [%d, %d), local gap [%d, %d), window_len=%d",
                L, R, local_g0, local_g1, R - L,
            )

            x = codes_filled[:, L:R].clone()
            x[:, local_g0:local_g1] = self.mask_token
            xb = x.unsqueeze(0).to(self.device)
            local_loss_mask = torch.zeros((1, R - L), device=self.device, dtype=torch.bool)
            local_loss_mask[:, local_g0:local_g1] = True
            seg_ids, left_idx, right_idx = self._build_model_boundary_tensors(local_loss_mask)
            retrieval_payload, _ = self._build_retrieval_payload(
                window_starts=[int(L)],
                mask_starts=[int(local_g0)],
                mask_lens=[int(local_g1 - local_g0)],
            )

            for it in range(cfg.inpaint_iters):
                logits = self._model_forward(
                    xb,
                    seg_ids,
                    left_idx,
                    right_idx,
                    retrieval_payload=retrieval_payload,
                )[0]
                pred = logits.argmax(dim=-1)
                xb[0, :, local_g0:local_g1] = pred[:, local_g0:local_g1]

            # Write back
            codes_filled[:, L:R] = xb[0].cpu()

        wav_filled = self.encoder.decode(codes_filled, self.scale)

        if output_path:
            import soundfile as sf
            sf.write(output_path, wav_filled, cfg.target_sr)
            logger.info("Saved inpainted audio: %s", output_path)

        return wav_filled

    @torch.no_grad()
    def inpaint(self, output_path: Optional[str] = None) -> np.ndarray:
        """Backward-compatible single-call: delegates to inpaint_all_gaps."""
        return self.inpaint_all_gaps(output_path=output_path)

    def _make_spectrogram_figure(self, audio: np.ndarray, title: str, sr: int, n_fft: int = 2048, hop: int = 512):
        times, freqs, spec_db = compute_log_spectrogram_data(audio, sr, n_fft=n_fft, hop=hop)
        vmax = float(np.max(spec_db))
        vmin = vmax - 80.0
        fig, ax = plt.subplots(1, 1, figsize=(12, 4.5))
        mesh = _plot_log_spectrogram(
            ax,
            times,
            freqs,
            spec_db,
            title=title,
            time_offset_s=0.0,
            min_freq=30.0,
            vmin=vmin,
            vmax=vmax,
        )
        if mesh is not None:
            fig.colorbar(mesh, ax=ax, label="dB")
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

        # Goal E: save reconstructed wav
        import soundfile as sf
        wav_path = self.cfg.samples_dir / f"infilled_step_{self.global_step}.wav"
        sf.write(str(wav_path), wav_filled, sr)
        logger.info("Saved reconstructed wav: %s", wav_path)


def main():
    cfg, args = parse_args()
    trainer = Trainer(cfg)

    if cfg.resume:
        trainer.load_checkpoint(cfg.resume)

    if not cfg.inpaint_only:
        trainer.train()

    out = cfg.inpaint_output or os.path.splitext(cfg.wav_path)[0] + "_infilled.wav"
    wav_filled = trainer.inpaint_all_gaps(output_path=out)
    trainer.log_spectrograms(wav_filled)


if __name__ == "__main__":
    main()
