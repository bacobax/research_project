#!/usr/bin/env python3
"""Training, validation and inference for the codebook-token audio infiller.

Audio is encoded to EnCodec RVQ tokens [K, T]; a transformer (joint or encoder-decoder) predicts the
tokens of masked spans (gaps), trained with cross-entropy plus an optional decoded-audio (STFT) loss.
Contents: activity/mask-sampling helpers, fixed validation-set builders and plotting, the AudioEncoder
codec wrapper, the model classes, the multi-resolution STFT loss and the Trainer, then the CLI (main).
Settings come from audio_infill.config / YAML (see configs/train/).
"""

import os
import sys
import time
import math
import random
import argparse
import json
import logging
import fnmatch
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

from audio_infill.config import validate_train_config as validate_shared_train_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("infiller")

# Pre-clip grad-norm threshold above which a training step is counted as a "spike" in the
# aggregated train/grad_norm_spike_count metric (see Trainer.train). Matches the threshold
# used to audit the decoded-loss STFT epsilon bug (near-silent bins -> log() gradient blowup).
GRAD_NORM_SPIKE_THRESHOLD = 100.0


@dataclass
class TrainConfig:
    """Training/inference settings (YAML keys / CLI flags). Mirrors audio_infill.config.TrainConfig, which
    is what the CLI and tests use; keep both in sync when adding fields.
    """
    config: Optional[str] = None
    ds_dir: str = "data/processed/single_gap"
    sample: Optional[str] = None
    auto_hparam: bool = False

    wav_path: str = "data/interim/gapped_audio.wav"
    target_sr: int = 24000
    bandwidth: float = 6.0
    gap_start_s: float = 200.0
    gap_end_s: float = 210.0

    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    max_len: int = 2048
    dropout: float = 0.1
    boundary_max_distance: int = 128

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
    validation_every: int = 0
    validation_examples_per_band: int = 64
    validation_batch_size: Optional[int] = None
    validation_strategy: str = "random_windows"
    validation_regions_per_band: int = 1
    validation_region_len_frames: Optional[int] = None
    validation_region_min_separation_frames: Optional[int] = None
    validation_examples_per_length_band: int = 8
    validation_mask_lengths: Tuple[int, ...] = ()
    validation_inspection_enabled: bool = False
    validation_inspection_examples_per_group: int = 1
    validation_crop_context_frames: Optional[int] = None
    validation_save_artifacts: bool = True
    num_workers: int = 2

    output_dir: str = "outputs/runs"
    run_name: str = "infiller"
    seed: int = 42
    device: str = "auto"

    resume: Optional[str] = None
    inpaint_only: bool = False
    inpaint_iters: int = 10
    inpaint_output: Optional[str] = None

    ctx_left: Optional[int] = None
    ctx_right: Optional[int] = None

    # Curriculum learning
    curriculum: bool = False
    curriculum_start_mask: Optional[int] = None  # default: min(mask_len_max, 128)
    curriculum_end_mask: Optional[int] = None      # default: largest_gap_frames
    curriculum_warmup_frac: float = 0.1
    curriculum_schedule: str = "linear"  # "linear" or "cosine"

    activity_smooth_kernel: int = 9
    activity_low_quantile: float = 0.30
    activity_high_quantile: float = 0.70
    weighted_sampling: bool = True
    dead_window_min_mean: float = 0.01
    dead_window_min_ratio: float = 0.03
    regime_active_prob: float = 0.45
    regime_transition_prob: float = 0.30
    regime_low_prob: float = 0.15
    regime_uniform_prob: float = 0.10
    mask_stride: int = 1
    activity_guided_masking: bool = True

    use_encoder_decoder: bool = False
    decoded_loss_enabled: bool = False
    decoded_loss_weight: float = 0.0
    decoded_loss_start_step: int = 0
    decoded_loss_every: int = 1
    decoded_loss_max_items: int = 1
    decoded_loss_margin_frames: int = 8
    decoded_loss_temperature: float = 1.0
    decoded_loss_waveform_l1_weight: float = 0.0
    decoded_loss_stft_weight: float = 1.0
    decoded_loss_spectral_convergence_weight: float = 1.0
    decoded_loss_log_magnitude_weight: float = 1.0
    decoded_loss_n_ffts: Tuple[int, ...] = (512, 1024, 2048)
    decoded_loss_hop_lengths: Tuple[int, ...] = (128, 256, 512)
    decoded_loss_win_lengths: Tuple[int, ...] = (512, 1024, 2048)
    decoded_loss_log_eps: float = 1e-7  # magnitude floor for the log-magnitude STFT term;
                                         # d/dx log(x) = 1/x, so a small eps lets near-silent
                                         # audio (common in low-activity windows) produce
                                         # gradients as large as 1/eps -- raise this to bound the
                                         # worst-case gradient from this term if that's a factor
                                         # in an observed gradient-explosion incident.

    # Multi-song training: extra clean (gap-free) songs sampled alongside the pivot.
    extra_wavs_dir: Optional[str] = None
    extra_wavs_glob: str = "*.wav"
    extra_wavs_exclude: Tuple[str, ...] = ()
    extra_wavs_limit: Optional[int] = None
    extra_wav_paths: Tuple[str, ...] = ()
    song_sampling: str = "duration"

    # Automated early stopping / divergence guard.
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 6
    early_stopping_min_delta: float = 0.0
    nan_guard_enabled: bool = True

    # Validation-time decoded-audio-quality metrics (SI-SDR, spectral convergence).
    validation_audio_metrics_enabled: bool = False
    validation_audio_metrics_max_examples: Optional[int] = None

    # Phase 6: train/inference-consistency options (defaults reproduce the earlier behaviour).
    zeroed_context_prob: float = 0.0   # share of training windows whose context is re-encoded from audio with the gap zeroed
    zeroed_context_warmup_frames: int = 300  # extra frames encoded before the window so the causal encoder state settles
    codebook_conditioning: str = "none"  # "none" (independent heads) or "depth_ar" (head k sees tokens of codebooks < k)
    depth_layers: int = 2
    validation_autocast: bool = True   # False evaluates validation in fp32
    audio_selection_metric: Optional[str] = None  # e.g. "si_sdr_db": also keep best_audio.pt selected on it
    inpaint_output_mode: str = "decode_all"  # "decode_all" or "splice" (only the gap's samples replace the original)
    tb_log_figures: bool = True  # False: no matplotlib figures in TensorBoard (they were ~97% of event-file size)

    @property
    def checkpoint_dir(self) -> Path:
        """Directory for checkpoints of this run."""
        return Path(self.output_dir) / self.run_name / "checkpoints"

    @property
    def tb_dir(self) -> Path:
        """Directory for TensorBoard event files of this run."""
        return Path(self.output_dir) / self.run_name / "tb"

    @property
    def samples_dir(self) -> Path:
        """Directory for audio/figure samples of this run."""
        return Path(self.output_dir) / self.run_name / "samples"


def set_seed(seed: int):
    """Seed python, numpy and torch (CPU and all CUDA devices) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg_device: str) -> torch.device:
    """Map 'auto' to cuda, else mps, else cpu; any other string is passed to torch.device."""
    if cfg_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(cfg_device)


def normalize_robust(values: np.ndarray, low_q: float = 0.05, high_q: float = 0.95) -> np.ndarray:
    """Rescale to ~[0, 1] using the low/high quantiles (robust to outliers); falls back to min/max when the quantiles collapse."""
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
    """Moving-average smoothing with an odd kernel (even sizes are bumped by one) and edge padding."""
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
    """RMS of the waveform within each codec frame (samples are split evenly across `frames`)."""
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
    """Fraction of codebooks whose token differs from the previous frame, per frame (frame 0 copies frame 1)."""
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
    """Zero-prefixed float64 cumulative sum so span sums are csum[end] - csum[start]."""
    arr = np.asarray(values, dtype=np.float32)
    csum = np.zeros(arr.size + 1, dtype=np.float64)
    csum[1:] = np.cumsum(arr, dtype=np.float64)
    return csum


def span_mean_from_cumsum(csum: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Mean of the underlying values over [start, end) spans, vectorised via a prefix sum."""
    length = np.maximum(1, end - start)
    return ((csum[end] - csum[start]) / length).astype(np.float32)


def choose_mask_regime(probs: Dict[str, float]) -> str:
    """Sample a masking regime (active/transition/low_activity/uniform) from the given probabilities (normalised; defaults used if all zero)."""
    names = ["active", "transition", "low_activity", "uniform"]
    p = np.array([max(0.0, float(probs.get(n, 0.0))) for n in names], dtype=np.float64)
    if p.sum() <= 0:
        p = np.array([0.45, 0.30, 0.15, 0.10], dtype=np.float64)
    p = p / p.sum()
    return names[int(np.random.choice(len(names), p=p))]


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
    """Per-candidate sampling weights for placing a mask under `regime`, from span activity statistics; near-dead spans are heavily down-weighted."""
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
    """True if [start, start+seq_len) overlaps any (begin, end) range."""
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
    """True if the window overlaps neither the real gaps nor any blocked (e.g. validation holdout) ranges."""
    if window_overlaps_ranges(start, seq_len, gaps):
        return False
    if blocked_ranges and window_overlaps_ranges(start, seq_len, blocked_ranges):
        return False
    return True


def merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or touching (start, end) ranges into a sorted disjoint list."""
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
    """Count and total frames of the merged holdout ranges, for logging."""
    merged = merge_ranges(ranges)
    return {
        "holdout_ranges": float(len(merged)),
        "holdout_frames": float(sum(end - start for start, end in merged)),
    }


def candidate_mask_offsets(seq_len: int, mask_len: int, mask_stride: int) -> np.ndarray:
    """Candidate mask start offsets within a window at the given stride, always including the last valid offset."""
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
    """Per-frame activity = smoothed 0.65*normalised RMS + 0.35*normalised token change, plus low/high quantile thresholds."""
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
    """One fixed validation item: masked input x, target y, loss mask and its provenance (band, window/mask position)."""
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
    """A held-out contiguous frame region [start, end) with its activity statistics, used by the holdout_regions strategy."""
    band: str
    start: int
    end: int
    mean_activity: float
    active_ratio: float


@dataclass
class ValidationGroupSpec:
    """Identifies a validation group: activity band, optionally restricted to one mask length."""
    band: str
    mask_len: Optional[int] = None


class FixedMaskedSpanDataset(Dataset):
    """Dataset over a precomputed list of FixedValidationExample (identical examples at every validation)."""
    def __init__(self, examples: List[FixedValidationExample]):
        """Store the examples and precompute their mean activity / mask length summary."""
        self.examples = list(examples)
        mask_means = np.array([ex.mask_mean_activity for ex in self.examples], dtype=np.float32)
        mask_lengths = np.array([ex.mask_len for ex in self.examples], dtype=np.float32)
        self.summary = {
            "count": len(self.examples),
            "mean_mask_activity": float(mask_means.mean()) if mask_means.size else 0.0,
            "mean_mask_len": float(mask_lengths.mean()) if mask_lengths.size else 0.0,
        }

    def __len__(self) -> int:
        """Number of items."""
        return len(self.examples)

    def __getitem__(self, idx: int):
        """Return the idx-th item."""
        ex = self.examples[idx]
        return ex.x.clone(), ex.y.clone(), ex.loss_mask.clone()


def slugify_component(text: str) -> str:
    """Make a string filesystem-safe (alnum, '-' and '_' only)."""
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    safe = safe.strip("_")
    return safe or "item"


def make_validation_example_id(example: FixedValidationExample, group_name: str, index: int) -> str:
    """Stable id for a validation example, encoding sample, group, index and window/mask position."""
    sample_slug = slugify_component(example.sample_name)
    group_slug = slugify_component(group_name)
    return (
        f"{sample_slug}__{group_slug}__idx{index:02d}"
        f"__ws{int(example.window_start):06d}__ms{int(example.mask_start):04d}__ml{int(example.mask_len):04d}"
    )


def derive_validation_crop_context_frames(mask_len: int, crop_context_frames: Optional[int] = None) -> int:
    """Context frames to show around a mask in inspection crops: explicit value, else max(96, ceil(mask_len / 2))."""
    if crop_context_frames is not None:
        return max(0, int(crop_context_frames))
    return max(96, int(math.ceil(float(mask_len) * 0.5)))


def compute_validation_crop_bounds(
    seq_len: int,
    mask_start: int,
    mask_len: int,
    crop_context_frames: Optional[int] = None,
) -> Tuple[int, int]:
    """Frame bounds [start, end) of the inspection crop: the mask plus context on both sides, clipped to the window."""
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
    """Convert codec-frame bounds to audio sample bounds (uniform samples-per-frame), clipped to the signal."""
    if total_samples <= 0:
        return 0, 0
    samples_per_frame = float(total_samples) / max(1, int(total_frames))
    start_sample = int(round(max(0, start_frame) * samples_per_frame))
    end_sample = int(round(max(start_frame + 1, end_frame) * samples_per_frame))
    start_sample = min(max(0, start_sample), total_samples - 1)
    end_sample = min(max(start_sample + 1, end_sample), total_samples)
    return start_sample, end_sample


def audio_metric_crop_bounds(seq_len: int, mask_start: int, mask_len: int, margin_frames: int) -> Tuple[int, int]:
    """Frame bounds of a gap+margin crop, clamped to the window. `margin_frames=0` gives the
    gap span alone; `margin_frames=cfg.decoded_loss_margin_frames` matches the crop used by
    `_compute_decoded_domain_loss` and Phase 0's `decode_gap_crop_audio`."""
    lo = max(0, int(mask_start) - int(margin_frames))
    hi = min(int(seq_len), int(mask_start) + int(mask_len) + int(margin_frames))
    return lo, hi


def si_sdr_db(pred: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> float:
    """Scale-invariant SDR in dB. Ported verbatim from `phase_0/common.py:si_sdr_db` — kept as a
    plain function here (not imported) since `phase_0` imports *from* this module."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = min(len(pred), len(target))
    pred, target = pred[:n], target[:n]
    alpha = float(np.dot(pred, target)) / (float(np.dot(target, target)) + eps)
    e = pred - alpha * target
    return float(10.0 * np.log10((np.mean((alpha * target) ** 2) + eps) / (np.mean(e**2) + eps)))


def extract_contiguous_mask_bounds(loss_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-row (first_masked_index, end_exclusive) of the single contiguous masked span in a [T] or [B, T] boolean mask."""
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
    """Same as extract_contiguous_mask_bounds, taking the gap as segment id 1 of a segment-id tensor."""
    if segment_ids.dim() == 1:
        segment_ids = segment_ids.unsqueeze(0)
    if segment_ids.dim() != 2:
        raise ValueError(f"segment_ids must have shape [T] or [B,T], got {tuple(segment_ids.shape)}")
    return extract_contiguous_mask_bounds(segment_ids == 1)


def build_boundary_condition_tensors(
    loss_mask: torch.Tensor,
    max_distance: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Boundary-conditioning inputs from a loss mask: segment ids (0 left ctx, 1 gap, 2 right ctx) and clamped signed distances
    to the gap start / end, offset by max_distance to be valid embedding indices.
    """
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
    """Boundary tensors for x [B, K, T], treating positions where all codebooks equal mask_token as the gap."""
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
    """STFT log-magnitude (dB) of a mono waveform. Returns (times, freqs, spec_db)."""
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
    """Draw one log-frequency spectrogram on `ax` (shared vmin/vmax allow side-by-side comparison)."""
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
    """Figure with target and prediction log-spectrograms on a common colour scale."""
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
    """Overlay target and predicted waveforms on `ax` with the masked sample range shaded."""
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
    """Figure comparing target vs predicted waveforms for the full window and the inspection crop around the mask."""
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
    """Write float32 audio to a wav file, creating parent directories."""
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(audio, dtype=np.float32), sr)


def _valid_non_gap_starts(
    frames: int,
    span_len: int,
    gaps: List[Tuple[int, int]],
    blocked_ranges: Optional[List[Tuple[int, int]]] = None,
) -> List[int]:
    """All window starts whose span avoids the gaps and blocked ranges."""
    max_start = max(0, frames - span_len)
    return [s for s in range(max_start) if is_non_gap_window(s, span_len, gaps, blocked_ranges)]


def _region_mean_activity(
    activity_cumsum: np.ndarray,
    active_flag_cumsum: np.ndarray,
    start: int,
    end: int,
) -> Tuple[float, float]:
    """Mean activity and active-frame ratio over [start, end) from prefix sums."""
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
    """Greedily choose `count` regions ordered by activity (descending or ascending) that stay min_separation frames from each other and from already-selected ones."""
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
) -> Tuple[Dict[str, List[FixedValidationExample]], List[Tuple[int, int]]]:
    """Build the random-window validation pool: per activity band (high/low), fixed masked windows avoiding the real gaps.
    Returns (examples per band, ranges used).
    """
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
) -> Tuple[Dict[str, List[FixedValidationExample]], Dict[str, List[ValidationRegion]], List[Tuple[int, int]], Dict[str, float]]:
    """Build the holdout-region validation pool: pick high/low-activity regions, exclude them from training, and draw fixed
    masked examples per mask length inside them. Returns (examples per group, regions per band, holdout ranges, summary).
    """
    _, frames = codes.shape
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


def touched_frames(s: int, e: int, hop: int) -> Tuple[int, int]:
    """Frames [floor(s/hop), ceil(e/hop)) that a zeroed sample span [s, e) touches."""
    return s // hop, -(-e // hop)


def reallike_span(g0: int, g1: int, hop: int, rng: Optional[np.random.Generator] = None) -> Tuple[int, int]:
    """A zeroed sample span whose rounded frame bounds are exactly [g0, g1), with random
    sub-frame offsets (start and end each within +-half a frame of the mask edge), like the real
    gaps. Deterministic per (g0, g1) when no generator is given (used by validation and Phase 5)."""
    if rng is None:
        rng = np.random.default_rng(1_000_003 * g0 + 17 * (g1 - g0))
    u0, u1 = rng.uniform(0.02, 0.98, size=2)
    return int(round((g0 - 0.5 + u0) * hop)), int(round((g1 - 0.5 + u1) * hop))


def splice_gap_audio(
    base: np.ndarray,
    decoded: np.ndarray,
    decoded_offset: int,
    start: int,
    end: int,
    fade: int,
) -> np.ndarray:
    """Copy of `base` (the gapped input audio) with its samples [start, end) taken from `decoded`.

    `decoded[i]` corresponds to sample `decoded_offset + i` of `base`. A raised-cosine cross-fade
    of `fade` samples is applied just outside the gap, so the context stays the input audio apart
    from those few samples and no re-encoded context reaches the output."""
    out = np.array(base, dtype=np.float32, copy=True)
    lo = max(0, start - fade)
    hi = min(len(out), end + fade)
    a = max(lo, decoded_offset)
    b = min(hi, decoded_offset + len(decoded))
    if b <= a:
        return out
    weight = np.ones(b - a, dtype=np.float32)
    idx = np.arange(a, b)
    if fade > 0:
        left = idx < start
        right = idx >= end
        weight[left] = 0.5 - 0.5 * np.cos(np.pi * (idx[left] - (start - fade) + 1) / (fade + 1))
        weight[right] = 0.5 + 0.5 * np.cos(np.pi * (idx[right] - end + 1) / (fade + 1))
    else:
        weight[(idx < start) | (idx >= end)] = 0.0
    out[a:b] = weight * decoded[a - decoded_offset:b - decoded_offset] + (1.0 - weight) * out[a:b]
    return out


class AudioEncoder:
    """Frozen EnCodec 24 kHz wrapper: waveform -> RVQ codes [K, T] and codes -> waveform; mask_token == codebook size."""
    def __init__(self, bandwidth: float, device: torch.device):
        """Load the frozen EnCodec 24 kHz model at `bandwidth` kbps; exposes bins, mask_token (= bins) and frame_rate."""
        from encodec import EncodecModel

        self.device = device
        self.model = EncodecModel.encodec_model_24khz().to(device).eval()
        self.model.set_target_bandwidth(bandwidth)
        self.model.requires_grad_(False)
        self.bins = int(self.model.quantizer.bins)
        self.mask_token = self.bins
        self.frame_rate = int(self.model.frame_rate)

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> Tuple[torch.Tensor, object]:
        """Encode a mono waveform [1, T] to RVQ codes [K, frames] (on CPU) plus the codec's scale."""
        x = wav.unsqueeze(0).to(self.device)
        encoded = self.model.encode(x)
        codes_b, scale = encoded[0]
        codes = codes_b[0].detach().cpu()
        return codes, scale

    @torch.no_grad()
    def decode(self, codes: torch.Tensor, scale) -> np.ndarray:
        """Decode codes [K, frames] (with the scale from encode) back to a numpy waveform."""
        codes_b = codes.unsqueeze(0).to(self.device)
        wav_out = self.model.decode([(codes_b, scale)])
        return wav_out.squeeze(0).cpu().squeeze(0).numpy()

    def codes_to_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum the quantizer-layer embeddings of hard codes [B, K, T] (or [K, T]) into decoder input embeddings [B, D, T]."""
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        if codes.dim() != 3:
            raise ValueError(f"codes_to_embeddings expects [B,K,T] or [K,T], got shape={tuple(codes.shape)}")
        codes = codes.to(self.device, dtype=torch.long)
        quantized_out = None
        layers = self.model.quantizer.vq.layers
        if codes.shape[1] > len(layers):
            raise ValueError(f"codes K={codes.shape[1]} exceeds quantizer layers={len(layers)}")
        for q in range(codes.shape[1]):
            quantized = layers[q].decode(codes[:, q, :])
            quantized_out = quantized if quantized_out is None else quantized_out + quantized
        assert quantized_out is not None
        return quantized_out

    def logits_to_embeddings(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Differentiable counterpart of codes_to_embeddings: softmax(logits / temperature) mixes each codebook's vectors, so audio losses can backprop to logits."""
        if logits.dim() != 4:
            raise ValueError(f"logits_to_embeddings expects [B,K,T,V], got shape={tuple(logits.shape)}")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits = logits.to(self.device, dtype=torch.float32)
        probs = torch.softmax(logits / temperature, dim=-1)
        quantized_out = None
        layers = self.model.quantizer.vq.layers
        if logits.shape[1] > len(layers):
            raise ValueError(f"logits K={logits.shape[1]} exceeds quantizer layers={len(layers)}")
        for q in range(logits.shape[1]):
            layer = layers[q]
            codebook = layer.codebook.to(device=logits.device, dtype=logits.dtype)
            soft_quantized = torch.matmul(probs[:, q, :, :], codebook)
            soft_quantized = layer.project_out(soft_quantized)
            soft_quantized = soft_quantized.transpose(1, 2)
            quantized_out = soft_quantized if quantized_out is None else quantized_out + soft_quantized
        assert quantized_out is not None
        return quantized_out

    def decode_embeddings(self, embeddings: torch.Tensor, scale: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run the frozen EnCodec decoder on embeddings [B, D, T] (differentiable); optional per-item scale is applied to the output waveform."""
        if embeddings.dim() != 3:
            raise ValueError(f"decode_embeddings expects [B,D,T], got shape={tuple(embeddings.shape)}")
        embeddings = embeddings.to(self.device, dtype=torch.float32)
        decoder = self.model.decoder
        decoder_was_training = decoder.training
        if torch.is_grad_enabled():
            # cuDNN LSTM backward requires the decoder forward to run in training mode,
            # even though the EnCodec weights themselves stay frozen.
            decoder.train(True)
        try:
            wav_out = decoder(embeddings)
        finally:
            decoder.train(decoder_was_training)
        if scale is not None:
            wav_out = wav_out * scale.view(-1, 1, 1)
        return wav_out


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
    ):
        """Index the valid (non-gap, non-holdout, non-dead) window starts, their sampling weights, and the activity statistics used for mask placement."""
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
        # When True, __getitem__ also returns [song_idx, window_start] so the trainer can
        # re-encode the window from audio with the gap zeroed.
        self.return_meta = False
        self.song_idx = 0

        self._build_valid_starts_and_weights()

    def _is_non_gap_window(self, s: int) -> bool:
        """Whether a window starting at `s` avoids gaps and blocked (validation holdout) ranges."""
        return is_non_gap_window(s, self.seq_len, self.gaps, self.blocked_ranges)

    def _build_valid_starts_and_weights(self):
        """Enumerate usable window starts and compute their sampling weights from window activity; dead windows are rejected or down-weighted. Logs the filtering counts."""
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
        """Current (min, max) mask length; updated by the curriculum."""
        return (self._mask_len_range[0], self._mask_len_range[1])

    def update_mask_range(self, mask_min: int, mask_max: int):
        """Dynamically update mask range for curriculum learning."""
        self._mask_len_range[0] = mask_min
        self._mask_len_range[1] = mask_max

    def __len__(self) -> int:
        """Number of items."""
        return self.virtual_size

    def _sample_window_start(self) -> int:
        """Draw a window start, weighted by activity when weighted_sampling is on, else uniformly."""
        if self.weighted_sampling and self.window_weights_t.numel() > 1:
            idx = int(torch.multinomial(self.window_weights_t, 1, replacement=True).item())
            return self.starts[idx]
        return random.choice(self.starts)

    def _span_mean(self, csum: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Mean over [start, end) spans via a prefix sum."""
        return span_mean_from_cumsum(csum, start, end)

    def _proximity_to_active(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Per-span closeness (1 = at an active frame, 0 = half a window or more away) from the span centre to the nearest active frame."""
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
        """Pick where the mask of `mask_len` goes inside the window: uniformly, or by activity regime-weighted candidate offsets.
        Returns (mask_start, mask_end, stats) with the stats recorded for logging.
        """
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
        """Drain and return up to max_items recent sampling metrics (collected only when meta is enabled)."""
        if max_items is None:
            max_items = len(self.recent_metrics)
        items = []
        for _ in range(min(max_items, len(self.recent_metrics))):
            items.append(self.recent_metrics.popleft())
        return items

    def __getitem__(self, idx: int):
        """Return the idx-th item."""
        s = self._sample_window_start()
        x = self.codes[:, s : s + self.seq_len].clone()
        y = x.clone()

        mask_min, mask_max = self._mask_len_range
        mask_len = random.randint(mask_min, max(mask_min, mask_max))
        mask_len = min(mask_len, self.seq_len)
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

        if self.return_meta:
            return x, y, loss_mask, torch.tensor([self.song_idx, s], dtype=torch.long)
        return x, y, loss_mask


# Backward-compatible alias for any existing imports.
MaskedSpanDataset = ActivityAwareMaskedSpanDataset


class MultiSongMaskedSpanDataset(Dataset):
    """Samples training windows across several per-song `ActivityAwareMaskedSpanDataset`s.

    Each sub-dataset keeps its own per-song activity thresholds, valid window starts, and
    mask-span placement logic untouched (see that class for the actual sampling policy) — this
    wrapper only decides which song a given draw comes from, then delegates. `__getitem__`
    ignores `idx` on the sub-datasets exactly as the single-song dataset does, so plain
    delegation is safe.
    """

    def __init__(
        self,
        datasets: List[ActivityAwareMaskedSpanDataset],
        song_weights: np.ndarray,
        virtual_size: int = 50_000,
    ):
        """Combine per-song datasets; each sample first picks a song by song_weights, then delegates to that song's dataset."""
        if not datasets:
            raise ValueError("MultiSongMaskedSpanDataset requires at least one sub-dataset")
        if len(datasets) != len(song_weights):
            raise ValueError("datasets and song_weights must have the same length")
        self.datasets = datasets
        self.virtual_size = int(virtual_size)

        weights = np.clip(np.asarray(song_weights, dtype=np.float32), 1e-9, None)
        weights = weights / weights.sum()
        self.song_weights = weights
        self.song_weights_t = torch.from_numpy(weights)

        starts_per_song = [len(ds.starts) for ds in datasets]
        count_weights = [max(1, n) for n in starts_per_song]
        self.summary: Dict[str, Any] = {
            "num_songs": len(datasets),
            "total_valid_starts": int(sum(starts_per_song)),
            "avg_window_activity": float(
                np.average(
                    [ds.summary.get("avg_window_activity", 0.0) for ds in datasets],
                    weights=count_weights,
                )
            ),
            "min_window_weight": float(min(ds.summary.get("min_window_weight", 0.0) for ds in datasets)),
            "max_window_weight": float(max(ds.summary.get("max_window_weight", 0.0) for ds in datasets)),
        }
        for i, n in enumerate(starts_per_song):
            self.summary[f"song_{i}_starts"] = int(n)
            self.summary[f"song_{i}_weight"] = float(weights[i])

    def __len__(self) -> int:
        """Number of items."""
        return self.virtual_size

    def enable_meta(self) -> None:
        """Make every sample also carry [song_idx, window_start] (see ActivityAwareMaskedSpanDataset)."""
        for i, ds in enumerate(self.datasets):
            ds.song_idx = i
            ds.return_meta = True

    def _sample_song_index(self) -> int:
        """Draw a song index according to song_weights."""
        return int(torch.multinomial(self.song_weights_t, 1, replacement=True).item())

    def __getitem__(self, idx: int):
        """Return the idx-th item."""
        i = self._sample_song_index()
        return self.datasets[i][idx]

    @property
    def mask_len_range(self) -> Tuple[int, int]:
        """Current (min, max) mask length (shared across the sub-datasets)."""
        return self.datasets[0].mask_len_range

    def update_mask_range(self, mask_min: int, mask_max: int):
        """Fan out curriculum-learning mask-range updates to every sub-dataset."""
        for ds in self.datasets:
            ds.update_mask_range(mask_min, mask_max)

    def pop_recent_metrics(self, max_items: Optional[int] = None) -> List[Dict[str, float]]:
        """Merge recently-sampled window/mask metrics across sub-datasets.

        Note: like the single-song dataset, each sub-dataset's metrics deque lives in whichever
        process filled it, so under `num_workers > 0` this mostly returns empty in the main
        process — pre-existing behavior, unrelated to multi-song support.
        """
        items: List[Dict[str, float]] = []
        remaining = max_items
        for ds in self.datasets:
            take = None if remaining is None else max(0, remaining)
            batch = ds.pop_recent_metrics(take)
            items.extend(batch)
            if remaining is not None:
                remaining -= len(batch)
                if remaining <= 0:
                    break
        return items


class _CodebookInfillerBase(nn.Module):
    """Shared pieces of both infillers: per-codebook token embeddings (vocab = bins + mask token), positional,
    segment (left ctx / gap / right ctx) and clamped left/right gap-distance embeddings, and one linear head per codebook."""
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
    ):
        """Create the embeddings, positional/segment/distance tables and per-codebook heads."""
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

    def _embed_tokens(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        segment_ids: torch.Tensor,
        left_dist_idx: torch.Tensor,
        right_dist_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Sum token (per codebook), positional, segment and left/right distance embeddings, then apply dropout. Returns [B, T, d_model]."""
        h = self.pos(positions)
        for k in range(self.K):
            h = h + self.emb[k](x[:, k, :])
        h = h + self.segment_emb(segment_ids)
        h = h + self.left_distance_emb(left_dist_idx)
        h = h + self.right_distance_emb(right_dist_idx)
        return self.dropout(h)

    def _project_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the per-codebook heads to hidden states [B, T, d] -> logits [B, K, T, bins]."""
        return torch.stack([self.head[k](h) for k in range(self.K)], dim=1)


class JointCodebookInfiller(_CodebookInfillerBase):
    """Bidirectional transformer encoder over the whole window (context and masked gap together); the K codebook
    embeddings are summed per frame and each codebook has its own output head."""
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
    ):
        """Build the shared embeddings plus a pre-norm transformer encoder."""
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
        gap_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """x: [B, K, T] tokens with the gap set to mask_token. Returns logits [B, K, T, bins]. Boundary tensors are
        derived from the mask token when not given; gap_targets is unused by this model."""
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
        return self._project_logits(h)


class EncoderDecoderCodebookInfiller(_CodebookInfillerBase):
    """Encoder over the context frames (gap removed) and decoder over the gap frames, which cross-attend to the encoded
    context. codebook_conditioning="depth_ar" adds a small causal transformer across the K codebooks of each gap frame
    (codebook k sees codebooks < k); "none" predicts all codebooks independently from the decoder state.
    """
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
        codebook_conditioning: str = "none",
        depth_layers: int = 2,
    ):
        """Build the shared embeddings, the context encoder, the gap decoder and, for depth_ar, the codebook-depth transformer."""
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

        if codebook_conditioning not in {"none", "depth_ar"}:
            raise ValueError("codebook_conditioning must be 'none' or 'depth_ar'")
        self.codebook_conditioning = codebook_conditioning
        if codebook_conditioning == "depth_ar":
            # Small causal transformer over the K codebook positions of each gap frame
            # (RQ-Transformer style): position k sees the decoder state and the tokens of
            # codebooks 0..k-1 of the same frame. Created only when requested, so the
            # state dict of an unconditioned model is unchanged.
            self.depth_tok_emb = nn.ModuleList([nn.Embedding(bins, d_model) for _ in range(K - 1)])
            self.depth_start = nn.Parameter(torch.zeros(d_model))
            self.depth_pos = nn.Embedding(K, d_model)
            depth_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.depth = nn.TransformerEncoder(depth_layer, num_layers=depth_layers)

    def _depth_inputs(self, h: torch.Tensor, prev_tokens: torch.Tensor, upto: int) -> torch.Tensor:
        """h: (N, d) decoder states; prev_tokens: (N, >=upto) tokens of codebooks 0..upto-1.
        Returns (N, upto + 1, d) inputs for depth positions 0..upto."""
        N = h.shape[0]
        pos = self.depth_pos(torch.arange(upto + 1, device=h.device)).unsqueeze(0)
        parts = [self.depth_start.view(1, 1, -1).expand(N, 1, -1)]
        for k in range(upto):
            parts.append(self.depth_tok_emb[k](prev_tokens[:, k]).unsqueeze(1))
        return self.dropout(h.unsqueeze(1) + torch.cat(parts, dim=1) + pos)

    def _depth_logits(self, decoded: torch.Tensor, gap_tgt: Optional[torch.Tensor]) -> torch.Tensor:
        """decoded: (B, G, d). gap_tgt: (B, K, G) teacher-forcing tokens, or None for greedy
        decoding over codebooks. Returns logits (B, K, G, bins)."""
        B, G, d = decoded.shape
        K = self.K
        h = decoded.reshape(B * G, d)
        if gap_tgt is not None:
            prev = gap_tgt.permute(0, 2, 1).reshape(B * G, K)
            inp = self._depth_inputs(h, prev, K - 1)
            causal = torch.triu(torch.ones(K, K, device=h.device, dtype=torch.bool), diagonal=1)
            out = self.depth(inp, mask=causal)
            logits = torch.stack([self.head[k](out[:, k]) for k in range(K)], dim=1)  # (N, K, bins)
        else:
            prev = h.new_zeros((B * G, K), dtype=torch.long)
            step_logits = []
            for k in range(K):
                inp = self._depth_inputs(h, prev, k)
                causal = torch.triu(torch.ones(k + 1, k + 1, device=h.device, dtype=torch.bool), diagonal=1)
                out = self.depth(inp, mask=causal)
                lk = self.head[k](out[:, k])
                step_logits.append(lk)
                prev[:, k] = lk.argmax(dim=-1)
            logits = torch.stack(step_logits, dim=1)
        return logits.reshape(B, G, K, self.bins).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        left_dist_idx: Optional[torch.Tensor] = None,
        right_dist_idx: Optional[torch.Tensor] = None,
        gap_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """`gap_targets` (B, K, T) are the clean tokens; they are used only when
        codebook_conditioning == "depth_ar", for teacher forcing. Without them the codebooks
        of each gap frame are decoded greedily, which is what inference does."""
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

        # Split each row into context tokens (left + right of the gap, original positions kept) and gap tokens,
        # right-padded to the batch maximum; padding masks keep attention off the pad slots.
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
        if self.codebook_conditioning == "depth_ar":
            gap_tgt = None
            if gap_targets is not None:
                gap_tgt = gap_targets.new_zeros((B, K, max_gap))
                for b in range(B):
                    gl = int(gap_lengths[b].item())
                    gap_tgt[b, :, :gl] = gap_targets[b, :, gap_indices[b, :gl]]
            gap_logits = self._depth_logits(decoded, gap_tgt)
        else:
            gap_logits = self._project_logits(decoded)

        # Scatter gap logits back to their frame positions; context frames stay zero (never used in the loss).
        full_logits = gap_logits.new_zeros((B, K, T, self.bins))
        for b in range(B):
            gap_len = int(gap_lengths[b].item())
            full_logits[b, :, gap_indices[b, :gap_len], :] = gap_logits[b, :, :gap_len, :]
        return full_logits


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (spectral convergence + log-magnitude) between waveforms, used as the decoded-audio loss."""
    def __init__(
        self,
        n_ffts: Sequence[int],
        hop_lengths: Sequence[int],
        win_lengths: Sequence[int],
        spectral_convergence_weight: float = 1.0,
        log_magnitude_weight: float = 1.0,
        eps: float = 1e-7,
    ):
        """Store the FFT / hop / window lists (equal length) and the term weights; eps floors magnitudes before log."""
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
        """Hann-windowed STFT magnitude, floored at eps (bounds the log gradient)."""
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
        """Average over resolutions of spectral convergence and log-magnitude L1 between pred and target [B, T]. Returns the weighted total and its parts."""
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


def resolve_extra_wav_paths(cfg: TrainConfig) -> List[Path]:
    """Resolve the list of extra (clean, gap-free) training wavs from `cfg`.

    Deterministic and sort-stable so that regimes built with different `extra_wavs_limit`
    values are nested subsets of each other (e.g. limit=3 is a prefix of the full list) — this
    is what makes a data-scaling sweep comparable across regimes.
    """
    if cfg.extra_wav_paths:
        paths = [Path(p) for p in cfg.extra_wav_paths]
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"extra_wav_paths entry not found: {p}")
        return paths

    if not cfg.extra_wavs_dir:
        return []

    wavs_dir = Path(cfg.extra_wavs_dir)
    if not wavs_dir.is_dir():
        raise FileNotFoundError(f"extra_wavs_dir not found or not a directory: {wavs_dir}")

    candidates = sorted(wavs_dir.glob(cfg.extra_wavs_glob))
    exclude_patterns = list(cfg.extra_wavs_exclude or ())
    if exclude_patterns:
        candidates = [
            p for p in candidates
            if not any(fnmatch.fnmatch(p.name, pat) for pat in exclude_patterns)
        ]
    if cfg.extra_wavs_limit is not None:
        candidates = candidates[: cfg.extra_wavs_limit]
    return candidates


class Trainer:
    """Owns the codec, model, data sampling, validation pools and the train / validate / checkpoint / inpaint loops."""
    def __init__(self, cfg: TrainConfig):
        """Set up the run: seed, output dirs, TensorBoard, audio/codes, validation pool, dataset, model, optimizer and decoded loss."""
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        set_seed(cfg.seed)
        logger.info("Device: %s", self.device)

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cfg.tb_dir.mkdir(parents=True, exist_ok=True)
        cfg.samples_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(cfg.tb_dir))

        self._log_hparams()
        self._load_audio()
        self._build_validation()
        self._build_dataset()
        self._build_model()
        self._build_optimizer()
        self._build_decoded_loss()

        self.global_step = 0
        self.best_loss = float("inf")
        self.best_val_loss = float("inf")

        # Early stopping / divergence guard state.
        self._val_no_improve_count = 0
        self.early_stop_triggered = False
        self.diverged = False

        # Curriculum state
        if cfg.curriculum:
            self._init_curriculum()

    def _log_hparams(self):
        """Log the config values (and write them to TensorBoard) at startup."""
        cfg = self.cfg
        hparams = {
            k: v
            for k, v in vars(cfg).items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool, list, tuple))
        }
        hparams_safe = {}
        for k, v in hparams.items():
            if v is None:
                continue
            hparams_safe[k] = json.dumps(v) if isinstance(v, (list, tuple)) else v
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

    def _resolve_gap_frames(
        self,
        frames: int,
        duration_s: float,
        gap_start_s: float,
        gap_end_s: float,
        ann: Optional[dict],
    ) -> List[Tuple[int, int]]:
        """Convert gap times (from the annotation, else gap_start_s/gap_end_s) to clamped codec-frame ranges."""
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
        has_gaps: bool = True,
    ) -> Dict[str, Any]:
        """Load, mono-mix, resample and encode one wav; returns the waveform, codes, scale, gap frames and activity features."""
        import soundfile as sf
        import librosa

        logger.info("Loading audio: %s", wav_path)
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(
                f"input audio not found: {wav_path} (see SETUP.md 'Reproduce From Scratch'; the legacy "
                "data/processed/{single_gap,multigap} and data/interim inputs cannot be regenerated)"
            )

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
        if has_gaps:
            gaps_f = self._resolve_gap_frames(
                frames=int(codes.shape[1]),
                duration_s=duration_s,
                gap_start_s=gap_start_s,
                gap_end_s=gap_end_s,
                ann=ann,
            )
        else:
            gaps_f = []
        fps = codes.shape[1] / max(duration_s, 1e-9)
        for i, (g0, g1) in enumerate(gaps_f):
            gap_dur = (g1 - g0) / fps
            logger.info("  Gap %d: frames [%d, %d) = %.2fs", i, g0, g1, gap_dur)

        return {
            "wav_path": wav_path,
            "wav": wav,
            "duration_s": duration_s,
            "codes": codes,
            "scale": scale,
            "frames": int(codes.shape[1]),
            "gaps_f": gaps_f,
            **features,
        }

    def _load_audio(self):
        """Load the pivot song and any extra_wavs (extra songs, no gaps) and derive shared sizes (K, bins, largest gap)."""
        cfg = self.cfg
        self.encoder = AudioEncoder(cfg.bandwidth, self.device)
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

        # Extra clean (gap-free) songs trained on alongside the pivot sample above. These never
        # touch self.wav/self.codes/self.gaps_f/... (the pivot's attributes, used by validation,
        # inpainting, and phase_0's Trainer reconstruction) — they only feed _build_dataset.
        self.pivot_wav_path = cfg.wav_path
        extra_paths = resolve_extra_wav_paths(cfg)
        self.extra_samples: List[Dict[str, Any]] = []
        for p in extra_paths:
            self.extra_samples.append(
                self._load_audio_sample(
                    wav_path=str(p),
                    target_sr=cfg.target_sr,
                    gap_start_s=0.0,
                    gap_end_s=0.0,
                    ann=None,
                    has_gaps=False,
                )
            )
        if self.extra_samples:
            total_extra_s = sum(s["duration_s"] for s in self.extra_samples)
            logger.info(
                "Loaded %d extra training song(s), %.1fs (%.1f min) total, alongside pivot %.1fs",
                len(self.extra_samples), total_extra_s, total_extra_s / 60.0, self.duration_s,
            )
        self._write_songs_manifest(extra_paths)

    def _write_songs_manifest(self, extra_paths: List[Path]):
        """Record which songs this run trained on, for provenance (mirrors the annotation JSON
        already saved by the dataset-generation step)."""
        manifest = {
            "pivot": {
                "wav_path": self.pivot_wav_path,
                "duration_s": self.duration_s,
                "frames": self.frames,
                "gaps_f": self.gaps_f,
            },
            "extra_songs": [
                {
                    "wav_path": s["wav_path"],
                    "duration_s": s["duration_s"],
                    "frames": s["frames"],
                }
                for s in self.extra_samples
            ],
            "song_sampling": self.cfg.song_sampling,
        }
        try:
            with open(Path(self.cfg.output_dir) / self.cfg.run_name / "songs.json", "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except OSError as e:
            logger.warning("Could not write songs.json manifest: %s", e)

    def _build_dataset(self):
        """Build the training dataset(s): per-song activity-aware masked-span datasets, combined by song_sampling when there are several."""
        cfg = self.cfg
        pivot_dataset = ActivityAwareMaskedSpanDataset(
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
        )

        if not self.extra_samples:
            # No extra songs: keep the exact single-song construction, byte-identical to every
            # run before multi-song support existed.
            self.dataset = pivot_dataset
        else:
            datasets = [pivot_dataset]
            starts_counts = [len(pivot_dataset.starts)]
            for sample in self.extra_samples:
                ds = ActivityAwareMaskedSpanDataset(
                    codes=sample["codes"],
                    gaps=[],  # clean songs: no absolute gaps to exclude
                    seq_len=cfg.seq_len,
                    mask_len_range=(cfg.mask_len_min, cfg.mask_len_max),
                    mask_token=self.mask_token,
                    activity_per_frame=sample["activity_per_frame"],
                    token_change_per_frame=sample["token_change_per_frame"],
                    activity_low_thr=sample["activity_low_thr"],
                    activity_high_thr=sample["activity_high_thr"],
                    weighted_sampling=cfg.weighted_sampling,
                    dead_window_min_mean=cfg.dead_window_min_mean,
                    dead_window_min_ratio=cfg.dead_window_min_ratio,
                    regime_probs={
                        "active": cfg.regime_active_prob,
                        "transition": cfg.regime_transition_prob,
                        "low_activity": cfg.regime_low_prob,
                        "uniform": cfg.regime_uniform_prob,
                    },
                    blocked_ranges=[],  # only the pivot has a validation holdout to protect
                    mask_stride=cfg.mask_stride,
                    activity_guided_masking=cfg.activity_guided_masking,
                )
                datasets.append(ds)
                starts_counts.append(len(ds.starts))

            if cfg.song_sampling == "uniform":
                song_weights = np.ones(len(datasets), dtype=np.float32)
            else:  # "duration" (default): proportional to valid window starts, i.e. usable
                   # training material per song, so short tracks aren't over-weighted and this
                   # behaves like sampling from one long concatenated song.
                song_weights = np.asarray(starts_counts, dtype=np.float32)

            self.dataset = MultiSongMaskedSpanDataset(
                datasets=datasets,
                song_weights=song_weights,
                virtual_size=pivot_dataset.virtual_size,
            )

        self.song_wavs = [self.wav] + [smp["wav"] for smp in self.extra_samples]
        if cfg.zeroed_context_prob > 0:
            if hasattr(self.dataset, "enable_meta"):
                self.dataset.enable_meta()
            else:
                self.dataset.song_idx = 0
                self.dataset.return_meta = True
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
            self.writer.add_scalar("dataset/num_songs", float(1 + len(self.extra_samples)), 0)
            total_frames = self.frames + sum(s["frames"] for s in self.extra_samples)
            total_minutes = (self.duration_s + sum(s["duration_s"] for s in self.extra_samples)) / 60.0
            self.writer.add_scalar("dataset/total_frames", float(total_frames), 0)
            self.writer.add_scalar("dataset/total_minutes", total_minutes, 0)
            for key, value in self.dataset.summary.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"dataset/{key}", float(value), 0)
        self.writer.add_scalar("train/activity_low_thr", self.activity_low_thr, 0)
        self.writer.add_scalar("train/activity_high_thr", self.activity_high_thr, 0)

    def _build_validation(self):
        """Build the fixed validation pool (random windows or holdout regions) and its dataloaders; also marks holdout ranges to exclude from training."""
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

        self.validation_dataloaders_clean: Dict[str, DataLoader] = {}
        if cfg.zeroed_context_prob > 0:
            # Primary validation uses inference-style (zeroed-gap) context; the clean-context
            # copy of the same examples is kept as a reference ("val_clean/*").
            for group_name, items in loader_examples.items():
                self.validation_dataloaders_clean[group_name] = DataLoader(
                    FixedMaskedSpanDataset(items), batch_size=val_batch_size, shuffle=False, num_workers=0,
                    pin_memory=(self.device.type == "cuda"), drop_last=False,
                )
            loader_examples = {
                group_name: [self._zero_context_example(ex) for ex in items]
                for group_name, items in loader_examples.items()
            }
            logger.info("Validation context re-encoded with the gap zeroed (%d groups).", len(loader_examples))

        for group_name, items in loader_examples.items():
            dataset = FixedMaskedSpanDataset(items)
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

        self.validation_audio_targets: Dict[str, List[Dict[str, np.ndarray]]] = {}
        if cfg.validation_audio_metrics_enabled:
            margin = int(cfg.decoded_loss_margin_frames)
            for group_name, items in loader_examples.items():
                targets: List[Dict[str, np.ndarray]] = []
                for ex in items:
                    seq_len = int(ex.y.shape[1])
                    lo_m, hi_m = audio_metric_crop_bounds(seq_len, ex.mask_start, ex.mask_len, margin)
                    lo_g, hi_g = audio_metric_crop_bounds(seq_len, ex.mask_start, ex.mask_len, 0)
                    targets.append({
                        "margin": self._decode_validation_window_audio(ex.y[:, lo_m:hi_m]),
                        "gap_only": self._decode_validation_window_audio(ex.y[:, lo_g:hi_g]),
                    })
                self.validation_audio_targets[group_name] = targets
            logger.info(
                "Cached target audio crops for validation audio-quality metrics (%d groups).",
                len(self.validation_audio_targets),
            )

    def _build_model(self):
        """Instantiate the joint or encoder-decoder infiller from the config and move it to the device."""
        cfg = self.cfg
        model_cls = EncoderDecoderCodebookInfiller if cfg.use_encoder_decoder else JointCodebookInfiller
        extra_kwargs: Dict[str, Any] = {}
        if cfg.codebook_conditioning != "none":
            if not cfg.use_encoder_decoder:
                raise ValueError("codebook_conditioning requires use_encoder_decoder: true")
            extra_kwargs = dict(codebook_conditioning=cfg.codebook_conditioning, depth_layers=cfg.depth_layers)
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
            **extra_kwargs,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model params: %.2fM", n_params / 1e6)
        self.writer.add_scalar("model/params_M", n_params / 1e6, 0)
        self.writer.add_scalar("model/use_encoder_decoder", float(cfg.use_encoder_decoder), 0)

    def _build_model_boundary_tensors(
        self,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Boundary (segment/distance) tensors for a loss mask using cfg.boundary_max_distance."""
        return build_boundary_condition_tensors(
            loss_mask,
            max_distance=self.cfg.boundary_max_distance,
        )

    def _build_optimizer(self):
        """AdamW (+ AMP GradScaler on CUDA) from the config."""
        cfg = self.cfg
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
        )
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

    def _build_decoded_loss(self):
        """Create the shared MultiResolutionSTFTLoss, needed by the decoded training loss and/or validation audio metrics."""
        cfg = self.cfg
        self.decoded_loss_enabled = bool(cfg.decoded_loss_enabled and cfg.decoded_loss_weight > 0)
        self.decoded_stft_loss: Optional[MultiResolutionSTFTLoss] = None
        # The STFT-loss module is shared between the training-time decoded loss and the
        # validation-time audio-quality metrics (Section 2) — either flag can request it, but
        # only cfg.decoded_loss_enabled gates whether it's actually added to the training loss.
        needs_stft_module = self.decoded_loss_enabled or cfg.validation_audio_metrics_enabled
        if not needs_stft_module:
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
            eps=cfg.decoded_loss_log_eps,
        )
        if self.decoded_loss_enabled:
            logger.info(
                "Decoded-domain loss enabled (hard argmax path): weight=%.4f every=%d start=%d max_items=%d margin_frames=%d temperature=%.3f (parsed, unused)",
                cfg.decoded_loss_weight,
                cfg.decoded_loss_every,
                cfg.decoded_loss_start_step,
                cfg.decoded_loss_max_items,
                cfg.decoded_loss_margin_frames,
                cfg.decoded_loss_temperature,
            )
        if cfg.validation_audio_metrics_enabled:
            logger.info(
                "Validation audio-quality metrics enabled: SI-SDR + spectral convergence, "
                "margin-included (margin_frames=%d) and gap-only variants.",
                cfg.decoded_loss_margin_frames,
            )

    # --- Curriculum learning ---

    def _init_curriculum(self):
        """Resolve curriculum start/end mask lengths and warmup steps from the config."""
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
        """Linear warmup to cfg.lr, then cosine decay to 0 at total_steps."""
        cfg = self.cfg
        if step < cfg.warmup_steps:
            return cfg.lr * step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float):
        """Set the learning rate on every optimizer param group."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    @torch.no_grad()
    def _encode_zeroed_windows(
        self,
        items: List[Tuple[torch.Tensor, int, int, int, int, Optional[np.random.Generator]]],
    ) -> Tuple[torch.Tensor, List[int]]:
        """Tokens of windows encoded the way inference encodes a gapped file.

        Each item is (wav [1, N], window_start_frame, window_len_frames, gap_local_start,
        gap_local_end, rng). The audio from `zeroed_context_warmup_frames` before the window to
        its end is cut out, a real-like span rounding to the gap is set to zero, everything is
        encoded (EnCodec's encoder is causal, so the warm-up only settles its state) and the
        warm-up frames are dropped. Returns ([n, K, window_len] tokens on the model device, local
        index of the first frame the zeroing touches for each item). Frames before that index equal
        the clean encoding (the encoder is causal) and the windowed encoding approximates them worse
        than the clean whole-song tokens do, so callers keep the clean tokens there
        (phase_6/results/preflight_zeroed_context.json)."""
        hop = self.cfg.target_sr // self.encoder.frame_rate
        warm = int(self.cfg.zeroed_context_warmup_frames)
        segs: List[Tuple[int, torch.Tensor, int, int]] = []
        first_touched: List[int] = []
        for wav, s, T, m0, m1, rng in items:
            w0 = max(0, s - warm)
            lo, hi = w0 * hop, (s + T) * hop
            seg = wav[:, lo:min(hi, wav.shape[-1])]
            if seg.shape[-1] < hi - lo:
                seg = F.pad(seg, (0, hi - lo - seg.shape[-1]))
            seg = seg.clone()
            zs, ze = reallike_span(s + m0, s + m1, hop, rng)
            seg[:, max(0, zs - lo):max(0, ze - lo)] = 0.0
            first_touched.append(max(0, min(m0, touched_frames(zs, ze, hop)[0] - s)))
            segs.append((s - w0, seg, T, len(segs)))
        out: List[Optional[torch.Tensor]] = [None] * len(items)
        by_len: Dict[Tuple[int, int], List[Tuple[int, torch.Tensor, int, int]]] = defaultdict(list)
        for entry in segs:
            by_len[(entry[0], entry[1].shape[-1])].append(entry)
        for (drop, _), group in by_len.items():
            batch = torch.stack([g[1] for g in group], dim=0).to(self.device)
            codes = self.encoder.model.encode(batch)[0][0]  # [B, K, frames]
            for j, g in enumerate(group):
                out[g[3]] = codes[j, :, drop:drop + g[2]].long()
        return torch.stack(out, dim=0), first_touched

    def _apply_zeroed_context(self, x: torch.Tensor, loss_mask: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Replace the context tokens of a random subset of the batch (probability
        `zeroed_context_prob` per window) by the zeroed-gap encoding; `y` stays clean."""
        B, K, T = x.shape
        use = [b for b in range(B) if random.random() < self.cfg.zeroed_context_prob]
        if not use:
            return x
        rng = np.random.default_rng(random.getrandbits(63))
        lm = loss_mask.detach().cpu()
        items = []
        for b in use:
            idx = lm[b].nonzero().flatten()
            items.append((self.song_wavs[int(meta[b, 0])], int(meta[b, 1]), T,
                          int(idx[0]), int(idx[-1]) + 1, rng))
        enc, first = self._encode_zeroed_windows(items)
        x = x.clone()
        for j, b in enumerate(use):
            xb = x[b].clone()  # clean tokens (gap masked): kept before the first touched frame
            xb[:, first[j]:] = enc[j][:, first[j]:].to(x.device)
            xb[:, loss_mask[b]] = self.mask_token
            x[b] = xb
        return x

    def _zero_context_example(self, ex: "FixedValidationExample") -> "FixedValidationExample":
        """A validation example whose context comes from the zeroed-gap encoding (deterministic)."""
        T = int(ex.y.shape[1])
        enc, first = self._encode_zeroed_windows(
            [(self.wav, int(ex.window_start), T, int(ex.mask_start), int(ex.mask_start + ex.mask_len), None)]
        )
        x = ex.x.clone()
        x[:, first[0]:] = enc[0][:, first[0]:].cpu()
        x[:, ex.loss_mask] = self.mask_token
        return FixedValidationExample(
            x=x, y=ex.y, loss_mask=ex.loss_mask, band=ex.band, sample_name=ex.sample_name,
            mask_mean_activity=ex.mask_mean_activity, mask_len=ex.mask_len,
            window_start=ex.window_start, mask_start=ex.mask_start,
        )

    def _teacher_forcing_kwargs(self, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Clean targets for the conditioned codebook heads (teacher forcing); nothing otherwise."""
        if self.cfg.codebook_conditioning == "depth_ar":
            return {"gap_targets": y}
        return {}

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """Token cross-entropy over masked frames only (all K codebooks of each masked frame)."""
        B, K, T, V = logits.shape
        logits_flat = logits.permute(0, 2, 1, 3).reshape(B * T * K, V)
        y_flat = y.permute(0, 2, 1).reshape(B * T * K)
        mask_bt = loss_mask.reshape(B * T)
        mask_btk = mask_bt.repeat_interleave(K)
        return F.cross_entropy(logits_flat[mask_btk], y_flat[mask_btk])

    def _should_apply_decoded_loss(self, step: int) -> bool:
        """Whether the decoded-audio loss is active at `step` (enabled, past decoded_loss_start_step, every decoded_loss_every steps)."""
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
        """Decoded-audio loss on the gap: soft-decode predicted logits and the targets with EnCodec and compare
        (waveform L1 + multi-resolution STFT). Returns (loss, metrics); zero when disabled.
        """
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
        """Token loss plus (when scheduled) the weighted decoded loss. Returns (total, token_loss, decoded metrics)."""
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
        """Masked-frame metrics: NLL, top-1/top-5 accuracy and perplexity."""
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
        """Decode a window's codes straight to audio (no codec scale) for validation audio metrics."""
        embeddings = self.encoder.codes_to_embeddings(codes.unsqueeze(0))
        audio = self.encoder.decode_embeddings(embeddings, scale=None)
        return audio.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def _predict_validation_window(self, example: FixedValidationExample) -> torch.Tensor:
        """Greedy fill: argmax tokens substituted into the masked frames of a validation example."""
        xb = example.x.unsqueeze(0).to(self.device, non_blocking=True)
        seg_ids, left_idx, right_idx = self._build_model_boundary_tensors(example.loss_mask.to(self.device))
        logits = self.model(xb, seg_ids, left_idx, right_idx)
        pred = logits.argmax(dim=-1).squeeze(0).detach().cpu()
        filled = example.y.clone()
        filled[:, example.loss_mask] = pred[:, example.loss_mask]
        return filled

    def _spectral_convergence(self, pred_audio: np.ndarray, target_audio: np.ndarray) -> float:
        """Spectral convergence via self.decoded_stft_loss, zero-padded up to the largest STFT
        window if the crop is shorter (the gap-only, margin=0 crop can be as short as one
        EnCodec frame = 320 samples, well under a 2048-sample win_length) — matches
        phase_0/common.py's stft_metrics padding safeguard."""
        if self.decoded_stft_loss is None:
            return float("nan")
        pred_t = torch.from_numpy(np.asarray(pred_audio, dtype=np.float32)).unsqueeze(0)
        tgt_t = torch.from_numpy(np.asarray(target_audio, dtype=np.float32)).unsqueeze(0)
        n = min(pred_t.shape[1], tgt_t.shape[1])
        pred_t, tgt_t = pred_t[:, :n], tgt_t[:, :n]
        min_len = max(self.decoded_stft_loss.win_lengths)
        if n < min_len:
            pad = min_len - n
            pred_t = F.pad(pred_t, (0, pad))
            tgt_t = F.pad(tgt_t, (0, pad))
        with torch.no_grad():
            out = self.decoded_stft_loss(pred_t, tgt_t)
        return float(out["spectral_convergence"].item())

    @torch.no_grad()
    def _compute_validation_audio_metrics(
        self, group_name: str, examples: List[FixedValidationExample]
    ) -> Dict[str, float]:
        """Decoded-audio SI-SDR / spectral convergence for one validation group, both
        margin-included (comparable to Phase 0's exp03/exp04) and gap-only (margin=0, more
        directly sensitive to fill quality — see the plan's note on margin dilution). Reuses
        `_predict_validation_window` per example (simpler and already tested, versus threading
        batched logits through from the token-CE loop — the added cost is negligible, ~1-5s per
        validation call total)."""
        max_examples = self.cfg.validation_audio_metrics_max_examples
        if max_examples is not None:
            examples = examples[: int(max_examples)]
        targets = self.validation_audio_targets.get(group_name, [])
        margin = int(self.cfg.decoded_loss_margin_frames)

        si_sdr_margin: List[float] = []
        si_sdr_gap: List[float] = []
        sc_margin: List[float] = []
        sc_gap: List[float] = []
        for idx, ex in enumerate(examples):
            if idx >= len(targets):
                break
            filled = self._predict_validation_window(ex)
            seq_len = int(ex.y.shape[1])
            lo_m, hi_m = audio_metric_crop_bounds(seq_len, ex.mask_start, ex.mask_len, margin)
            lo_g, hi_g = audio_metric_crop_bounds(seq_len, ex.mask_start, ex.mask_len, 0)
            pred_margin = self._decode_validation_window_audio(filled[:, lo_m:hi_m])
            pred_gap = self._decode_validation_window_audio(filled[:, lo_g:hi_g])
            target_margin = targets[idx]["margin"]
            target_gap = targets[idx]["gap_only"]

            si_sdr_margin.append(si_sdr_db(pred_margin, target_margin))
            si_sdr_gap.append(si_sdr_db(pred_gap, target_gap))
            sc_margin.append(self._spectral_convergence(pred_margin, target_margin))
            sc_gap.append(self._spectral_convergence(pred_gap, target_gap))

        def _mean(values: List[float]) -> float:
            """NaN-safe mean of a list (nan if empty)."""
            return float(np.mean(values)) if values else float("nan")

        return {
            "si_sdr_db": _mean(si_sdr_margin),
            "si_sdr_db_gap_only": _mean(si_sdr_gap),
            "spectral_convergence": _mean(sc_margin),
            "spectral_convergence_gap_only": _mean(sc_gap),
        }

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
        """Write the target/predicted audio and comparison figures for one validation example under samples_dir."""
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
        """Save audio/figures for the fixed inspection examples at `step` and write a manifest (no-op unless validation_inspection_enabled)."""
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
                if hasattr(self.writer, "add_figure") and self.cfg.tb_log_figures:
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
        """Evaluate all validation groups: token metrics and, if enabled, decoded-audio metrics; logs to TensorBoard
        and returns aggregated metrics (None if validation is off).
        """
        if not getattr(self, "validation_enabled", False):
            return None

        self.model.eval()
        t_val0 = time.time()
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
                for x, y, loss_mask in loader:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    loss_mask = loss_mask.to(self.device, non_blocking=True).bool()
                    seg_ids, left_idx, right_idx = self._build_model_boundary_tensors(loss_mask)
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=use_amp and self.cfg.validation_autocast,
                    ):
                        if self.cfg.codebook_conditioning == "depth_ar":
                            # NLL is the teacher-forced (exact joint) one; accuracy comes from
                            # the greedy pass, which is what inference produces.
                            logits_tf = self.model(x, seg_ids, left_idx, right_idx, gap_targets=y)
                            loss = self._compute_loss(logits_tf, y, loss_mask)
                            logits = self.model(x, seg_ids, left_idx, right_idx)
                        else:
                            logits = self.model(x, seg_ids, left_idx, right_idx)
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

        t_tok = time.time()
        audio_combined: Optional[Dict[str, float]] = None
        if self.cfg.validation_audio_metrics_enabled:
            audio_combined = self._run_validation_audio_metrics(step, group_specs)
        t_audio = time.time()
        if getattr(self, "validation_dataloaders_clean", None):
            self._run_clean_reference_validation(step, use_amp)
        t_clean = time.time()

        self._log_validation_inspection(step)
        logger.info(
            "Validation timing at step %d: token pass %.1fs, audio metrics %.1fs, clean reference %.1fs, inspection %.1fs",
            step, t_tok - t_val0, t_audio - t_tok, t_clean - t_audio, time.time() - t_clean,
        )
        self.model.train()

        improved = combined_loss < self.best_val_loss
        if improved:
            self.best_val_loss = combined_loss
            self.save_checkpoint("best_val")
        if self.cfg.audio_selection_metric and audio_combined is not None:
            value = audio_combined.get(self.cfg.audio_selection_metric, float("nan"))
            if math.isfinite(value) and value > getattr(self, "best_val_audio", float("-inf")):
                self.best_val_audio = value
                self.save_checkpoint("best_audio")
                logger.info("New best %s=%.3f at step %d (best_audio.pt)", self.cfg.audio_selection_metric, value, step)

        self.early_stop_triggered = self._update_early_stopping(improved, combined_loss, step)

        result = {
            "combined_loss": combined_loss,
            "high_loss": band_results["high_activity"]["loss"],
            "low_loss": band_results["low_activity"]["loss"],
            "early_stop": self.early_stop_triggered,
        }
        return result

    _AUDIO_METRIC_KEYS = ("si_sdr_db", "si_sdr_db_gap_only", "spectral_convergence", "spectral_convergence_gap_only")

    def _run_validation_audio_metrics(self, step: int, group_specs: Dict[str, "ValidationGroupSpec"]) -> Dict[str, float]:
        """Decoded-audio SI-SDR / spectral convergence, logged per group/band/combined with the
        same aggregation convention as the token-loss metrics above: count-weighted mean across
        groups within a band, then an unweighted mean of the two band means for the combined
        `val/*` scalar. Reporting-only — does not affect checkpoint selection or early stopping."""
        band_audio_totals: Dict[str, Dict[str, float]] = {
            "high_activity": {**{k: 0.0 for k in self._AUDIO_METRIC_KEYS}, "count": 0.0},
            "low_activity": {**{k: 0.0 for k in self._AUDIO_METRIC_KEYS}, "count": 0.0},
        }
        max_examples = self.cfg.validation_audio_metrics_max_examples
        for group_name, loader in self.validation_dataloaders.items():
            spec = group_specs.get(group_name, ValidationGroupSpec(band=group_name, mask_len=None))
            examples = loader.dataset.examples
            metrics = self._compute_validation_audio_metrics(group_name, examples)
            count = len(examples) if max_examples is None else min(len(examples), int(max_examples))

            prefix_root = "high" if spec.band == "high_activity" else "low"
            prefix = f"val/{prefix_root}" if spec.mask_len is None else f"val/{prefix_root}_len_{spec.mask_len}"
            for key, value in metrics.items():
                self.writer.add_scalar(f"{prefix}_{key}", value, step)
                if math.isfinite(value):
                    band_audio_totals[spec.band][key] += value * count
            band_audio_totals[spec.band]["count"] += count

        band_audio_results: Dict[str, Dict[str, float]] = {}
        for band, totals in band_audio_totals.items():
            count = max(1.0, totals["count"])
            band_audio_results[band] = {key: totals[key] / count for key in self._AUDIO_METRIC_KEYS}
            prefix = "val/high" if band == "high_activity" else "val/low"
            for key, value in band_audio_results[band].items():
                self.writer.add_scalar(f"{prefix}_{key}", value, step)

        combined_audio: Dict[str, float] = {}
        for key in self._AUDIO_METRIC_KEYS:
            combined_value = float(np.mean([band_audio_results["high_activity"][key], band_audio_results["low_activity"][key]]))
            self.writer.add_scalar(f"val/{key}", combined_value, step)
            combined_audio[key] = combined_value
        return combined_audio

    @torch.no_grad()
    def _run_clean_reference_validation(self, step: int, use_amp: bool) -> None:
        """Token loss/accuracy of the same validation examples with clean (training-style) context,
        logged as val_clean/*; only used when the primary validation set has zeroed-gap context."""
        totals: Dict[str, List[float]] = {"high": [0.0, 0.0, 0.0], "low": [0.0, 0.0, 0.0]}  # loss, acc1, count
        for group_name, loader in self.validation_dataloaders_clean.items():
            spec = self.validation_group_specs.get(group_name)
            root = "high" if spec is not None and spec.band == "high_activity" else "low"
            for x, y, loss_mask in loader:
                x = x.to(self.device); y = y.to(self.device); loss_mask = loss_mask.to(self.device).bool()
                seg_ids, left_idx, right_idx = self._build_model_boundary_tensors(loss_mask)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                    enabled=use_amp and self.cfg.validation_autocast):
                    if self.cfg.codebook_conditioning == "depth_ar":
                        logits_tf = self.model(x, seg_ids, left_idx, right_idx, gap_targets=y)
                        loss = self._compute_loss(logits_tf, y, loss_mask)
                        logits = self.model(x, seg_ids, left_idx, right_idx)
                    else:
                        logits = self.model(x, seg_ids, left_idx, right_idx)
                        loss = self._compute_loss(logits, y, loss_mask)
                n = int(x.shape[0])
                totals[root][0] += float(loss.item()) * n
                totals[root][1] += self._compute_masked_metrics(logits, y, loss_mask)["acc_top1"] * n
                totals[root][2] += n
        losses = []
        for root, (l, a, n) in totals.items():
            if n > 0:
                self.writer.add_scalar(f"val_clean/{root}_loss", l / n, step)
                self.writer.add_scalar(f"val_clean/{root}_acc_top1", a / n, step)
                losses.append(l / n)
        if len(losses) == 2:
            self.writer.add_scalar("val_clean/combined_loss", float(np.mean(losses)), step)

    def _update_early_stopping(self, improved: bool, combined_loss: float, step: int) -> bool:
        """Track consecutive meaningfully-non-improving validation checks; return True once
        training should stop. Base criterion matches what was used by hand to kill the two
        completed single-song runs ("zero sign of improvement across 6 consecutive eval
        checkpoints"), with one refinement: `early_stopping_min_delta` gives a tolerance band
        around the best loss so far. A check that lands within that band of the best (but isn't
        itself a new best) is treated as noise, not degradation — it neither resets nor advances
        the patience counter. Only a check that's *meaningfully* worse than the best counts
        against patience. This matters because validation runs on only 64 examples, so small
        fluctuations around a genuinely flat loss are expected and shouldn't be indistinguishable
        from real, sustained degradation."""
        if not self.cfg.early_stopping_enabled:
            return False
        if improved:
            self._val_no_improve_count = 0
            return False
        min_delta = self.cfg.early_stopping_min_delta
        if combined_loss <= self.best_val_loss + min_delta:
            logger.info(
                "Validation at step %d within tolerance of best (combined_loss=%.4f, best=%.4f, "
                "min_delta=%.4f) - not counted against patience (%d/%d)",
                step, combined_loss, self.best_val_loss, min_delta,
                self._val_no_improve_count, self.cfg.early_stopping_patience,
            )
            return False
        self._val_no_improve_count += 1
        logger.info(
            "Validation did not improve at step %d: combined_loss=%.4f vs best=%.4f "
            "(%d/%d consecutive non-improving checks)",
            step, combined_loss, self.best_val_loss,
            self._val_no_improve_count, self.cfg.early_stopping_patience,
        )
        if self._val_no_improve_count >= self.cfg.early_stopping_patience:
            logger.warning(
                "Early stopping at step %d: val/combined_loss has not improved for %d "
                "consecutive validation checks (every %d steps).",
                step, self._val_no_improve_count, self.cfg.validation_every,
            )
            return True
        return False

    def save_checkpoint(self, tag: str = "latest"):
        """Save model/optimizer/scaler/RNG state, step and config to <checkpoint_dir>/<tag>.pt."""
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
                "best_val_loss": self.best_val_loss,
                "best_val_audio": getattr(self, "best_val_audio", float("-inf")),
                "config": vars(self.cfg),
                "rng_state": rng_state,
            },
            path,
        )
        logger.info("Saved checkpoint: %s (step %d)", path, self.global_step)

    def load_checkpoint(self, path: str):
        """Restore a checkpoint (after checking architecture compatibility with the config) and its training/RNG state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_cfg = ckpt.get("config") or {}
        ckpt_use_encoder_decoder = bool(ckpt_cfg.get("use_encoder_decoder", False))
        if ckpt_use_encoder_decoder != bool(self.cfg.use_encoder_decoder):
            raise ValueError(
                "Checkpoint architecture mismatch: checkpoint use_encoder_decoder="
                f"{ckpt_use_encoder_decoder}, current config use_encoder_decoder={self.cfg.use_encoder_decoder}"
            )
        ckpt_conditioning = str(ckpt_cfg.get("codebook_conditioning", "none"))
        if ckpt_conditioning != self.cfg.codebook_conditioning:
            raise ValueError(
                f"Checkpoint codebook_conditioning={ckpt_conditioning} does not match "
                f"config codebook_conditioning={self.cfg.codebook_conditioning}"
            )
        model_state = ckpt["model"]
        boundary_compat_mode = False
        try:
            self.model.load_state_dict(model_state)
        except RuntimeError as exc:
            incompatible = self.model.load_state_dict(model_state, strict=False)
            allowed_missing = {
                "segment_emb.weight",
                "left_distance_emb.weight",
                "right_distance_emb.weight",
            }
            missing = set(incompatible.missing_keys)
            unexpected = set(incompatible.unexpected_keys)
            if unexpected or not missing.issubset(allowed_missing):
                raise exc
            boundary_compat_mode = True
            logger.warning(
                "Loaded checkpoint without boundary-aware embeddings; newly initialized keys=%s",
                sorted(missing),
            )
        if boundary_compat_mode:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                logger.warning(
                    "Checkpoint optimizer state is incompatible with boundary-aware parameters; using fresh optimizer state"
                )
        else:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.global_step = ckpt["step"]
        self.best_loss = ckpt.get("best_loss", float("inf"))
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_val_audio = ckpt.get("best_val_audio", float("-inf"))
        if "rng_state" in ckpt:
            rng = ckpt["rng_state"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            cpu_rng = rng["torch_cpu"]
            if not isinstance(cpu_rng, torch.ByteTensor):
                cpu_rng = cpu_rng.cpu().byte() if hasattr(cpu_rng, 'cpu') else torch.ByteTensor(cpu_rng)
            torch.random.set_rng_state(cpu_rng)
            if torch.cuda.is_available() and "torch_cuda" in rng:
                cuda_states = rng["torch_cuda"]
                cuda_states = [s.cpu() if s.device.type != 'cpu' else s for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)
        logger.info("Loaded checkpoint: %s (step %d)", path, self.global_step)

    def train(self):
        """Main loop: sample batches, schedule LR (and curriculum), compute losses, clip/step, aggregate metrics, and periodically
        validate, checkpoint and run test-gap fills. Honours early stopping and the NaN guard when enabled.
        """
        cfg = self.cfg
        self.model.train()
        use_amp = self.device.type == "cuda"

        running_loss = 0.0
        running_acc = 0.0
        running_acc5 = 0.0
        running_ppl = 0.0
        running_nll = 0.0
        running_count = 0
        running_grad_norm_sum = 0.0
        running_grad_norm_max = 0.0
        running_grad_norm_spike_count = 0
        running_grad_norm_nonfinite = False
        running_decoded_loss_sum = 0.0
        running_decoded_loss_waveform_l1_sum = 0.0
        running_decoded_loss_stft_sum = 0.0
        running_decoded_loss_spectral_convergence_sum = 0.0
        running_decoded_loss_log_magnitude_sum = 0.0
        running_decoded_loss_items_sum = 0.0
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
            x, y, loss_mask = batch[:3]
            batch_meta = batch[3] if len(batch) > 3 else None

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            loss_mask = loss_mask.to(self.device, non_blocking=True).bool()
            if batch_meta is not None:
                x = self._apply_zeroed_context(x, loss_mask, batch_meta)
            seg_ids, left_idx, right_idx = self._build_model_boundary_tensors(loss_mask)

            lr = self._get_lr(step)
            self._set_lr(lr)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                logits = self.model(x, seg_ids, left_idx, right_idx, **self._teacher_forcing_kwargs(y))
                total_loss, token_loss, decoded_metrics = self._compute_training_losses(
                    x,
                    logits,
                    y,
                    loss_mask,
                    step=step,
                )

            if cfg.nan_guard_enabled and not torch.isfinite(total_loss):
                logger.error(
                    "Training loss became non-finite at step %d (total_loss=%s) - stopping "
                    "immediately without applying this update, to avoid corrupting the model "
                    "with a NaN/Inf gradient step.",
                    step, total_loss.detach().item() if total_loss.numel() == 1 else "non-scalar",
                )
                self.diverged = True
                self.save_checkpoint("latest")
                break

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

            # Per-step scalars are aggregated here and flushed every cfg.log_every steps
            # instead of written every step - at long run lengths (hundreds of thousands to
            # millions of steps) per-step TB scalars are the dominant contributor to on-disk
            # log size. grad_norm keeps max/spike-count aggregates (not just an average) so
            # rare single-step gradient spikes are still visible after aggregation.
            is_grad_norm_finite = math.isfinite(grad_norm)
            if is_grad_norm_finite:
                running_grad_norm_sum += grad_norm
                running_grad_norm_max = max(running_grad_norm_max, grad_norm)
            else:
                running_grad_norm_nonfinite = True
            if (not is_grad_norm_finite) or grad_norm > GRAD_NORM_SPIKE_THRESHOLD:
                running_grad_norm_spike_count += 1

            running_decoded_loss_sum += decoded_metrics["decoded_loss_total"]
            running_decoded_loss_waveform_l1_sum += decoded_metrics["decoded_loss_waveform_l1"]
            running_decoded_loss_stft_sum += decoded_metrics["decoded_loss_stft"]
            running_decoded_loss_spectral_convergence_sum += decoded_metrics["decoded_loss_spectral_convergence"]
            running_decoded_loss_log_magnitude_sum += decoded_metrics["decoded_loss_log_magnitude"]
            running_decoded_loss_items_sum += decoded_metrics["decoded_loss_items"]

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
                self.writer.add_scalar("train/lr", lr, step)

                finite_count = running_count - (1 if running_grad_norm_nonfinite else 0)
                avg_grad_norm = running_grad_norm_sum / max(1, finite_count)
                self.writer.add_scalar("train/grad_norm_avg", avg_grad_norm, step)
                self.writer.add_scalar("train/grad_norm_max", running_grad_norm_max, step)
                self.writer.add_scalar("train/grad_norm_spike_count", running_grad_norm_spike_count, step)
                self.writer.add_scalar(
                    "train/grad_norm_nonfinite", 1.0 if running_grad_norm_nonfinite else 0.0, step
                )

                self.writer.add_scalar("train/avg_decoded_loss", running_decoded_loss_sum / running_count, step)
                self.writer.add_scalar(
                    "train/avg_decoded_loss_waveform_l1", running_decoded_loss_waveform_l1_sum / running_count, step
                )
                self.writer.add_scalar(
                    "train/avg_decoded_loss_stft", running_decoded_loss_stft_sum / running_count, step
                )
                self.writer.add_scalar(
                    "train/avg_decoded_loss_spectral_convergence",
                    running_decoded_loss_spectral_convergence_sum / running_count,
                    step,
                )
                self.writer.add_scalar(
                    "train/avg_decoded_loss_log_magnitude",
                    running_decoded_loss_log_magnitude_sum / running_count,
                    step,
                )
                self.writer.add_scalar(
                    "train/avg_decoded_loss_items", running_decoded_loss_items_sum / running_count, step
                )

                if hasattr(self.dataset, "pop_recent_metrics"):
                    sample_metrics = self.dataset.pop_recent_metrics()
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
                        self.writer.add_scalar(
                            f"train/sample_mask_regime_frac_{reg_name}", reg_hist[reg_id] / total_reg, step
                        )

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self.save_checkpoint("best")

                running_loss = 0.0
                running_acc = 0.0
                running_acc5 = 0.0
                running_ppl = 0.0
                running_nll = 0.0
                running_count = 0
                running_grad_norm_sum = 0.0
                running_grad_norm_max = 0.0
                running_grad_norm_spike_count = 0
                running_grad_norm_nonfinite = False
                running_decoded_loss_sum = 0.0
                running_decoded_loss_waveform_l1_sum = 0.0
                running_decoded_loss_stft_sum = 0.0
                running_decoded_loss_spectral_convergence_sum = 0.0
                running_decoded_loss_log_magnitude_sum = 0.0
                running_decoded_loss_items_sum = 0.0
                t0 = time.time()

            if step % cfg.save_every == 0:
                self.save_checkpoint("latest")
                self.save_checkpoint(f"step_{step}")

            if self.validation_enabled and cfg.validation_every > 0 and step % cfg.validation_every == 0:
                self.run_validation(step)
                if self.early_stop_triggered:
                    self.save_checkpoint("latest")
                    break

            if cfg.test_fill_every > 0 and step % cfg.test_fill_every == 0:
                logger.info("Running test inpaint (gap crops only) at step %d", step)
                self.inpaint_all_gaps(save_crop_samples=True)
                self.model.train()

        self.save_checkpoint("final")
        self.writer.close()
        if getattr(self, "diverged", False):
            logger.warning("Training stopped early at step %d: loss diverged (non-finite).", self.global_step)
        elif getattr(self, "early_stop_triggered", False):
            logger.warning("Training stopped early at step %d: early stopping triggered.", self.global_step)
        logger.info("Training complete. Best loss: %.4f", self.best_loss)

    # --- Inpainting ---

    @torch.no_grad()
    def inpaint_all_gaps(
        self, output_path: Optional[str] = None, save_crop_samples: bool = False
    ) -> Optional[np.ndarray]:
        """Inpaint ALL gaps sequentially.

        For each gap:
        1) Compute maximum possible context within max_len
        2) Extract window [L, R]
        3) Mask the gap region
        4) Run iterative refinement
        5) Write predicted tokens back into codes

        Returns the reconstructed wav as a numpy array — unless `save_crop_samples` is True, in
        which case the (expensive, ~400s) whole-song decode is skipped entirely; instead, right
        after each gap's context window is filled, just that window is decoded and saved (a
        small wav + a comparison figure), and this returns None. Used for periodic in-training
        snapshots (`test_fill_every`) so they cost a handful of ~6s crop decodes instead of one
        full-song decode+save every call.
        """
        cfg = self.cfg
        self.model.eval()

        codes_filled = self.codes.clone()

        splice = cfg.inpaint_output_mode == "splice"
        hop = 0
        wav_out: Optional[np.ndarray] = None
        sample_spans: List[Tuple[int, int]] = []
        if splice:
            hop = cfg.target_sr // self.encoder.frame_rate
            ann = getattr(cfg, "_annotation", None)
            if not ann or "gaps" not in ann:
                raise ValueError("inpaint_output_mode='splice' needs the annotation's exact gap sample spans")
            sample_spans = [(int(g["gap_start_sample"]), int(g["gap_end_sample"])) for g in ann["gaps"]]
            wav_out = self.wav.squeeze(0).numpy().astype(np.float32, copy=True)
        fade = int(round(0.0025 * cfg.target_sr))  # ~2.5 ms

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

            for it in range(cfg.inpaint_iters):
                logits = self.model(xb, seg_ids, left_idx, right_idx)[0]
                pred = logits.argmax(dim=-1)
                xb[0, :, local_g0:local_g1] = pred[:, local_g0:local_g1]

            # Write back
            codes_filled[:, L:R] = xb[0].cpu()

            if splice:
                decoded_win = self._decode_validation_window_audio(codes_filled[:, L:R])
                wav_out = splice_gap_audio(wav_out, decoded_win, L * hop, sample_spans[i][0], sample_spans[i][1], fade)

            if save_crop_samples:
                orig_crop = self._decode_validation_window_audio(self.codes[:, L:R])
                if splice:
                    filled_crop = wav_out[L * hop:L * hop + len(orig_crop)]
                else:
                    filled_crop = self._decode_validation_window_audio(codes_filled[:, L:R])
                wav_path = cfg.samples_dir / f"gap{i}_step_{self.global_step}.wav"
                save_waveform(wav_path, filled_crop, cfg.target_sr)
                fig = make_log_spectrogram_comparison_figure(
                    target_audio=orig_crop,
                    pred_audio=filled_crop,
                    sr=cfg.target_sr,
                    title_prefix=f"gap{i} frames[{g0},{g1}) step={self.global_step}",
                )
                if cfg.tb_log_figures:
                    self.writer.add_figure(f"audio/gap{i}_comparison", fig, self.global_step)
                plt.close(fig)

        if save_crop_samples:
            self.writer.flush()
            logger.info(
                "Saved %d gap-crop sample(s) at step %d (samples_dir=%s)",
                len(self.gaps_f), self.global_step, cfg.samples_dir,
            )
            return None

        wav_filled = wav_out if splice else self.encoder.decode(codes_filled, self.scale)

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
        """Log-spectrogram figure of one audio signal with an 80 dB dynamic range."""
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
        """Log spectrogram figures of the original gapped audio and the infilled output (to TensorBoard when tb_log_figures)."""
        sr = self.cfg.target_sr
        wav_orig = self.wav.squeeze(0).numpy()

        fig_orig = self._make_spectrogram_figure(wav_orig, "Original (gapped)", sr)
        if self.cfg.tb_log_figures:
            self.writer.add_figure("audio/spectrogram_original_gapped", fig_orig, self.global_step)
        plt.close(fig_orig)

        fig_recon = self._make_spectrogram_figure(wav_filled, "Reconstructed (infilled)", sr)
        if self.cfg.tb_log_figures:
            self.writer.add_figure("audio/spectrogram_reconstructed", fig_recon, self.global_step)
        plt.close(fig_recon)

        self.writer.flush()
        logger.info("Logged spectrograms to TensorBoard")

        # Goal E: save reconstructed wav
        import soundfile as sf
        wav_path = self.cfg.samples_dir / f"infilled_step_{self.global_step}.wav"
        sf.write(str(wav_path), wav_filled, sr)
        logger.info("Saved reconstructed wav: %s", wav_path)


def load_annotation(ds_dir: str, sample: str) -> dict:
    """Load the JSON annotation written by make_gapped_dataset for <ds_dir>/<sample>.json."""
    json_path = Path(ds_dir) / f"{sample}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML mapping (PyYAML if available, else a minimal flat parser)."""
    try:
        import yaml
    except Exception as e:
        raise RuntimeError(
            "YAML config requested but PyYAML is not installed. Install with: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping at top-level: {path}")
    return data


def _set_cfg_field(cfg: TrainConfig, key: str, value: Any, source: str):
    """Set one config field from a (possibly kebab-case) key; unknown keys are logged and ignored."""
    name = key.replace("-", "_")
    if not hasattr(cfg, name):
        logger.warning("Ignoring unknown config key from %s: %s", source, key)
        return
    if name in {"betas", "validation_mask_lengths", "extra_wavs_exclude", "extra_wav_paths"} and isinstance(value, list):
        value = tuple(value)
    setattr(cfg, name, value)


def apply_mapping_to_cfg(cfg: TrainConfig, mapping: Dict[str, Any], source: str):
    """Apply a mapping of YAML values onto the TrainConfig (lists become tuples where the field expects them)."""
    for key, value in mapping.items():
        _set_cfg_field(cfg, key, value, source)


def apply_auto_hparams(cfg: TrainConfig, ann: dict):
    """Apply auto hyperparameters from annotation JSON.

    Supports both old single-gap and new multi-gap annotation formats.
    """
    # Load gap timing info
    if "gaps" in ann:
        # New multi-gap format: use first gap for gap_start/end_s (backward compat)
        first_gap = ann["gaps"][0]
        cfg.gap_start_s = first_gap["gap_start_s"]
        cfg.gap_end_s = first_gap["gap_end_s"]
    elif "gap" in ann:
        # Old format
        gap = ann["gap"]
        cfg.gap_start_s = gap["gap_start_s"]
        cfg.gap_end_s = gap["gap_end_s"]

    cfg.target_sr = ann["sr"]

    rec = ann["recommendations"]["token_based"]
    if rec is not None:
        cfg.seq_len = rec["seq_len_frames"]
        cfg.mask_len_min = rec["mask_len_min_frames"]
        cfg.mask_len_max = rec["mask_len_max_frames"]
        cfg.max_len = max(cfg.max_len, rec["max_len_frames_required"])

        if "ctx_left_frames" in rec:
            cfg.ctx_left = rec["ctx_left_frames"]
        if "ctx_right_frames" in rec:
            cfg.ctx_right = rec["ctx_right_frames"]

        # If curriculum and we have largest_gap_frames, set end mask
        if rec.get("largest_gap_frames") is not None:
            cfg.curriculum_end_mask = rec["largest_gap_frames"]

    if "encodec_stats_full_audio" in ann:
        stats = ann["encodec_stats_full_audio"]
        if stats is not None:
            cfg.bandwidth = stats["bandwidth_kbps"]

    logger.info(
        "Auto-hparams applied from annotation: seq_len=%d, mask=[%d,%d], max_len=%d, ctx=%s/%s",
        cfg.seq_len, cfg.mask_len_min, cfg.mask_len_max, cfg.max_len,
        cfg.ctx_left, cfg.ctx_right,
    )


def parse_args(argv: Optional[List[str]] = None):
    """Parse CLI flags; precedence is defaults < YAML --config < explicit flags. Returns (cfg, args)."""
    parser = argparse.ArgumentParser(description="Audio Infiller Training")
    cfg = TrainConfig()

    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")

    parser.add_argument("--ds-dir", type=str, default=None, help="Dataset directory containing .wav and .json pairs")
    parser.add_argument("--sample", type=str, default=None, help="Sample name (stem) within ds-dir, e.g. wav_test_gap_5p000s")
    parser.add_argument("--auto-hparam", action="store_true", help="Use recommended hparams from the annotation JSON")

    parser.add_argument("--wav-path", type=str, default=None, help="Direct wav path (overrides --ds-dir/--sample)")
    parser.add_argument("--target-sr", type=int, default=None)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--gap-start-s", type=float, default=None)
    parser.add_argument("--gap-end-s", type=float, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--boundary-max-distance", type=int, default=None)

    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--mask-len-min", type=int, default=None)
    parser.add_argument("--mask-len-max", type=int, default=None)
    parser.add_argument("--ctx-left", type=int, default=None)
    parser.add_argument("--ctx-right", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--betas", nargs=2, type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--test-fill-every", type=int, default=None, help="Run inpaint + log spectrogram every N steps (0=disabled)")
    parser.add_argument("--validation-every", type=int, default=None, help="Run dual-band validation on held-out windows from the training sample every N steps (0=disabled)")
    parser.add_argument("--validation-examples-per-band", type=int, default=None, help="Fixed same-sample validation examples per activity band")
    parser.add_argument("--validation-batch-size", type=int, default=None, help="Validation batch size (default: batch-size)")
    parser.add_argument("--validation-strategy", choices=["random_windows", "holdout_regions"], default=None)
    parser.add_argument("--validation-regions-per-band", type=int, default=None)
    parser.add_argument("--validation-region-len-frames", type=int, default=None)
    parser.add_argument("--validation-region-min-separation-frames", type=int, default=None)
    parser.add_argument("--validation-examples-per-length-band", type=int, default=None)
    parser.add_argument("--validation-mask-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--validation-inspection-enabled", dest="validation_inspection_enabled", action="store_true")
    parser.add_argument("--no-validation-inspection-enabled", dest="validation_inspection_enabled", action="store_false")
    parser.set_defaults(validation_inspection_enabled=None)
    parser.add_argument("--validation-inspection-examples-per-group", type=int, default=None)
    parser.add_argument("--validation-crop-context-frames", type=int, default=None)
    parser.add_argument("--validation-save-artifacts", dest="validation_save_artifacts", action="store_true")
    parser.add_argument("--no-validation-save-artifacts", dest="validation_save_artifacts", action="store_false")
    parser.set_defaults(validation_save_artifacts=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--inpaint-only", action="store_true", help="Skip training, only run inpainting")
    parser.add_argument("--inpaint-iters", type=int, default=None)
    parser.add_argument("--inpaint-output", type=str, default=None)

    # Curriculum learning args
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning (grow mask span)")
    parser.add_argument("--curriculum-start-mask", type=int, default=None,
                        help="Initial max mask length (default: min(mask_len_max, 128))")
    parser.add_argument("--curriculum-end-mask", type=int, default=None,
                        help="Final max mask length (default: largest_gap_frames)")
    parser.add_argument("--curriculum-warmup-frac", type=float, default=None,
                        help="Fraction of total steps for curriculum warmup (default 0.1)")
    parser.add_argument("--curriculum-schedule", choices=["linear", "cosine"],
                        default=None,
                        help="Curriculum interpolation schedule (default: linear)")

    parser.add_argument("--activity-smooth-kernel", type=int, default=None)
    parser.add_argument("--activity-low-quantile", type=float, default=None)
    parser.add_argument("--activity-high-quantile", type=float, default=None)
    parser.add_argument("--weighted-sampling", dest="weighted_sampling", action="store_true")
    parser.add_argument("--no-weighted-sampling", dest="weighted_sampling", action="store_false")
    parser.set_defaults(weighted_sampling=None)
    parser.add_argument("--dead-window-min-mean", type=float, default=None)
    parser.add_argument("--dead-window-min-ratio", type=float, default=None)
    parser.add_argument("--regime-active-prob", type=float, default=None)
    parser.add_argument("--regime-transition-prob", type=float, default=None)
    parser.add_argument("--regime-low-prob", type=float, default=None)
    parser.add_argument("--regime-uniform-prob", type=float, default=None)
    parser.add_argument("--mask-stride", type=int, default=None)
    parser.add_argument("--activity-guided-masking", dest="activity_guided_masking", action="store_true")
    parser.add_argument("--no-activity-guided-masking", dest="activity_guided_masking", action="store_false")
    parser.set_defaults(activity_guided_masking=None)

    parser.add_argument("--use-encoder-decoder", dest="use_encoder_decoder", action="store_true")
    parser.add_argument("--no-use-encoder-decoder", dest="use_encoder_decoder", action="store_false")
    parser.set_defaults(use_encoder_decoder=None)
    parser.add_argument("--decoded-loss-enabled", dest="decoded_loss_enabled", action="store_true")
    parser.add_argument("--no-decoded-loss-enabled", dest="decoded_loss_enabled", action="store_false")
    parser.set_defaults(decoded_loss_enabled=None)
    parser.add_argument("--decoded-loss-weight", type=float, default=None)
    parser.add_argument("--decoded-loss-start-step", type=int, default=None)
    parser.add_argument("--decoded-loss-every", type=int, default=None)
    parser.add_argument("--decoded-loss-max-items", type=int, default=None)
    parser.add_argument("--decoded-loss-margin-frames", type=int, default=None)
    parser.add_argument("--decoded-loss-temperature", type=float, default=None)
    parser.add_argument("--decoded-loss-waveform-l1-weight", type=float, default=None)
    parser.add_argument("--decoded-loss-stft-weight", type=float, default=None)
    parser.add_argument("--decoded-loss-spectral-convergence-weight", type=float, default=None)
    parser.add_argument("--decoded-loss-log-magnitude-weight", type=float, default=None)
    parser.add_argument("--decoded-loss-log-eps", type=float, default=None,
                        help="Magnitude floor for the log-magnitude STFT term (raise to bound its gradient near silence)")
    parser.add_argument("--decoded-loss-n-ffts", nargs="+", type=int, default=None)
    parser.add_argument("--decoded-loss-hop-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--decoded-loss-win-lengths", nargs="+", type=int, default=None)

    parser.add_argument("--extra-wavs-dir", type=str, default=None,
                        help="Directory of additional clean (gap-free) wavs to train on alongside the pivot sample")
    parser.add_argument("--extra-wavs-glob", type=str, default=None)
    parser.add_argument("--extra-wavs-exclude", nargs="+", type=str, default=None,
                        help="fnmatch patterns (against filename) to exclude from --extra-wavs-dir")
    parser.add_argument("--extra-wavs-limit", type=int, default=None,
                        help="Take only the first N extra wavs (sorted); combine with a fixed sort order for nested subsets")
    parser.add_argument("--extra-wav-paths", nargs="+", type=str, default=None,
                        help="Explicit list of extra wav paths; overrides --extra-wavs-dir when set")
    parser.add_argument("--song-sampling", choices=["duration", "uniform"], default=None)

    parser.add_argument("--early-stopping-enabled", dest="early_stopping_enabled", action="store_true",
                        help="Stop training once val/combined_loss hasn't improved for --early-stopping-patience consecutive validation checks")
    parser.add_argument("--no-early-stopping-enabled", dest="early_stopping_enabled", action="store_false")
    parser.set_defaults(early_stopping_enabled=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Consecutive non-improving validation checks before stopping")
    parser.add_argument("--early-stopping-min-delta", type=float, default=None,
                        help="Validation checks within this margin of the best combined_loss don't count against patience")
    parser.add_argument("--nan-guard-enabled", dest="nan_guard_enabled", action="store_true",
                        help="Stop immediately (without applying the update) if the training loss becomes non-finite")
    parser.add_argument("--no-nan-guard-enabled", dest="nan_guard_enabled", action="store_false")
    parser.set_defaults(nan_guard_enabled=None)

    parser.add_argument("--validation-audio-metrics-enabled", dest="validation_audio_metrics_enabled", action="store_true",
                        help="Compute decoded-audio SI-SDR/spectral-convergence (margin-included and gap-only) during validation")
    parser.add_argument("--no-validation-audio-metrics-enabled", dest="validation_audio_metrics_enabled", action="store_false")
    parser.set_defaults(validation_audio_metrics_enabled=None)
    parser.add_argument("--validation-audio-metrics-max-examples", type=int, default=None,
                        help="Cap the number of validation examples scored per call (default: all)")

    args = parser.parse_args(argv)

    if args.config:
        cfg.config = args.config
        cfg_map = load_yaml_config(args.config)
        apply_mapping_to_cfg(cfg, cfg_map, args.config)

    cli_overrides = {
        "ds_dir": args.ds_dir,
        "sample": args.sample,
        "wav_path": args.wav_path,
        "target_sr": args.target_sr,
        "bandwidth": args.bandwidth,
        "gap_start_s": args.gap_start_s,
        "gap_end_s": args.gap_end_s,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_len": args.max_len,
        "dropout": args.dropout,
        "boundary_max_distance": args.boundary_max_distance,
        "seq_len": args.seq_len,
        "mask_len_min": args.mask_len_min,
        "mask_len_max": args.mask_len_max,
        "ctx_left": args.ctx_left,
        "ctx_right": args.ctx_right,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "betas": tuple(args.betas) if args.betas is not None else None,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "warmup_steps": args.warmup_steps,
        "total_steps": args.total_steps,
        "log_every": args.log_every,
        "save_every": args.save_every,
        "test_fill_every": args.test_fill_every,
        "validation_every": args.validation_every,
        "validation_examples_per_band": args.validation_examples_per_band,
        "validation_batch_size": args.validation_batch_size,
        "validation_strategy": args.validation_strategy,
        "validation_regions_per_band": args.validation_regions_per_band,
        "validation_region_len_frames": args.validation_region_len_frames,
        "validation_region_min_separation_frames": args.validation_region_min_separation_frames,
        "validation_examples_per_length_band": args.validation_examples_per_length_band,
        "validation_mask_lengths": tuple(args.validation_mask_lengths) if args.validation_mask_lengths is not None else None,
        "validation_inspection_enabled": args.validation_inspection_enabled,
        "validation_inspection_examples_per_group": args.validation_inspection_examples_per_group,
        "validation_crop_context_frames": args.validation_crop_context_frames,
        "validation_save_artifacts": args.validation_save_artifacts,
        "num_workers": args.num_workers,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        "device": args.device,
        "resume": args.resume,
        "inpaint_iters": args.inpaint_iters,
        "inpaint_output": args.inpaint_output,
        "curriculum_start_mask": args.curriculum_start_mask,
        "curriculum_end_mask": args.curriculum_end_mask,
        "curriculum_warmup_frac": args.curriculum_warmup_frac,
        "curriculum_schedule": args.curriculum_schedule,
        "activity_smooth_kernel": args.activity_smooth_kernel,
        "activity_low_quantile": args.activity_low_quantile,
        "activity_high_quantile": args.activity_high_quantile,
        "weighted_sampling": args.weighted_sampling,
        "dead_window_min_mean": args.dead_window_min_mean,
        "dead_window_min_ratio": args.dead_window_min_ratio,
        "regime_active_prob": args.regime_active_prob,
        "regime_transition_prob": args.regime_transition_prob,
        "regime_low_prob": args.regime_low_prob,
        "regime_uniform_prob": args.regime_uniform_prob,
        "mask_stride": args.mask_stride,
        "activity_guided_masking": args.activity_guided_masking,
        "use_encoder_decoder": args.use_encoder_decoder,
        "decoded_loss_enabled": args.decoded_loss_enabled,
        "decoded_loss_weight": args.decoded_loss_weight,
        "decoded_loss_start_step": args.decoded_loss_start_step,
        "decoded_loss_every": args.decoded_loss_every,
        "decoded_loss_max_items": args.decoded_loss_max_items,
        "decoded_loss_margin_frames": args.decoded_loss_margin_frames,
        "decoded_loss_temperature": args.decoded_loss_temperature,
        "decoded_loss_waveform_l1_weight": args.decoded_loss_waveform_l1_weight,
        "decoded_loss_stft_weight": args.decoded_loss_stft_weight,
        "decoded_loss_spectral_convergence_weight": args.decoded_loss_spectral_convergence_weight,
        "decoded_loss_log_magnitude_weight": args.decoded_loss_log_magnitude_weight,
        "decoded_loss_log_eps": args.decoded_loss_log_eps,
        "decoded_loss_n_ffts": tuple(args.decoded_loss_n_ffts) if args.decoded_loss_n_ffts is not None else None,
        "decoded_loss_hop_lengths": tuple(args.decoded_loss_hop_lengths) if args.decoded_loss_hop_lengths is not None else None,
        "decoded_loss_win_lengths": tuple(args.decoded_loss_win_lengths) if args.decoded_loss_win_lengths is not None else None,
        "extra_wavs_dir": args.extra_wavs_dir,
        "extra_wavs_glob": args.extra_wavs_glob,
        "extra_wavs_exclude": tuple(args.extra_wavs_exclude) if args.extra_wavs_exclude is not None else None,
        "extra_wavs_limit": args.extra_wavs_limit,
        "extra_wav_paths": tuple(args.extra_wav_paths) if args.extra_wav_paths is not None else None,
        "song_sampling": args.song_sampling,
        "early_stopping_enabled": args.early_stopping_enabled,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "nan_guard_enabled": args.nan_guard_enabled,
        "validation_audio_metrics_enabled": args.validation_audio_metrics_enabled,
        "validation_audio_metrics_max_examples": args.validation_audio_metrics_max_examples,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    if args.auto_hparam:
        cfg.auto_hparam = True
    if args.curriculum:
        cfg.curriculum = True
    if args.inpaint_only:
        cfg.inpaint_only = True

    ann = None
    if cfg.sample:
        wav_file = Path(cfg.ds_dir) / f"{cfg.sample}.wav"
        if not wav_file.exists():
            parser.error(f"Sample wav not found: {wav_file}")
        cfg.wav_path = str(wav_file)
        ann = load_annotation(cfg.ds_dir, cfg.sample)

        # Load gap info (support both old and new format)
        if "gaps" in ann:
            first_gap = ann["gaps"][0]
            cfg.gap_start_s = first_gap["gap_start_s"]
            cfg.gap_end_s = first_gap["gap_end_s"]
        elif "gap" in ann:
            cfg.gap_start_s = ann["gap"]["gap_start_s"]
            cfg.gap_end_s = ann["gap"]["gap_end_s"]

        cfg.target_sr = ann["sr"]
        if cfg.auto_hparam:
            apply_auto_hparams(cfg, ann)

    if cfg.run_name is None:
        cfg.run_name = cfg.sample if cfg.sample else "infiller"

    # Store annotation for multi-gap loading in Trainer._load_audio
    cfg._annotation = ann  # type: ignore[attr-defined]

    validate_shared_train_config(cfg)
    return cfg, args


def main():
    """CLI entry point: build the Trainer, optionally resume, train (unless inpaint_only), then inpaint all gaps and log spectrograms."""
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
