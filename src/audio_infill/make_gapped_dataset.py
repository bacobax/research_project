#!/usr/bin/env python3
"""
make_gapped_dataset.py

Create gapped .wav samples from a source .wav.
Supports single-gap (one file per gap duration) and multi-gap (one file with
multiple gaps) modes.

For each gap length (seconds) provided, produces:
- a gapped wav (silence inserted by zeroing samples)
- a JSON annotation file

Usage examples:
  # Single-gap mode (default): one wav per gap duration
  python make_gapped_dataset.py --wav input.wav --gap-seconds 0.5 1 2 5 10 --outdir out

  # Multi-gap mode: one wav with 3 gaps of specified durations
  python make_gapped_dataset.py --wav input.wav --num-gaps 3 --gap-seconds 1 2 5 --outdir out

  # Multi-gap mode: one wav with 3 gaps all of the same duration
  python make_gapped_dataset.py --wav input.wav --num-gaps 3 --gap-seconds 2 --outdir out

  # Random placement with margin:
  python make_gapped_dataset.py --wav input.wav --gap-seconds 1 2 5 --center-mode random --seed 0 --outdir out

Notes:
- Uses soundfile to read/write wav.
- If input sample-rate != --target-sr, resamples (needs librosa, otherwise raises).
- If encodec is installed, computes actual token fps and recommends seq_len/mask_len in frames.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from audio_infill.config import load_yaml_config


@dataclass
class DataConfig:
    config: Optional[str] = None
    wav: str = "data/raw/wav_test.wav"
    outdir: str = "data/processed/single_gap"
    gap_seconds: List[float] = None  # type: ignore[assignment]
    num_gaps: int = 1
    min_gap_separation_seconds: float = 1.0
    target_sr: int = 24000
    bandwidth: float = 6.0
    center_mode: str = "random"
    margin_seconds: float = 2.0
    seed: int = 42
    max_len_cap: int = 2048
    prefer_pow2: bool = False
    extra_ctx_s: float = 2.0

    def __post_init__(self):
        if self.gap_seconds is None:
            self.gap_seconds = [1.0]


def validate_data_config(cfg: DataConfig):
    if not cfg.gap_seconds:
        raise ValueError("gap_seconds must contain at least one value")
    if any(float(g) <= 0 for g in cfg.gap_seconds):
        raise ValueError("gap_seconds values must all be > 0")
    if cfg.num_gaps <= 0:
        raise ValueError("num_gaps must be > 0")
    if cfg.min_gap_separation_seconds < 0:
        raise ValueError("min_gap_separation_seconds must be >= 0")
    if cfg.target_sr <= 0:
        raise ValueError("target_sr must be > 0")
    if cfg.bandwidth <= 0:
        raise ValueError("bandwidth must be > 0")
    if cfg.center_mode not in {"middle", "random"}:
        raise ValueError("center_mode must be 'middle' or 'random'")
    if cfg.margin_seconds < 0:
        raise ValueError("margin_seconds must be >= 0")
    if cfg.max_len_cap <= 0:
        raise ValueError("max_len_cap must be > 0")
    if cfg.extra_ctx_s < 0:
        raise ValueError("extra_ctx_s must be >= 0")


def parse_args(argv: Optional[List[str]] = None) -> DataConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="Path to YAML dataset config")
    ap.add_argument("--wav", type=str, default=None, help="Input .wav path")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory")
    ap.add_argument("--gap-seconds", nargs="+", type=float, default=None, help="Gap lengths in seconds (list)")
    ap.add_argument("--num-gaps", type=int, default=None,
                    help="Number of gaps to insert in a single wav (default 1). "
                         "If >1, creates multi-gap output.")
    ap.add_argument("--min-gap-separation-seconds", type=float, default=None,
                    help="Minimum time separation between gaps (seconds, default 1.0)")
    ap.add_argument("--target-sr", type=int, default=None, help="Target sample rate for output (default 24000)")
    ap.add_argument("--bandwidth", type=float, default=None, help="EnCodec target bandwidth kbps (for token stats)")
    ap.add_argument("--center-mode", choices=["middle", "random"], default=None,
                    help="How to place the gap center")
    ap.add_argument("--margin-seconds", type=float, default=None,
                    help="Minimum distance of gap boundaries from edges (seconds)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-len-cap", type=int, default=None,
                    help="Cap for recommended max_len/seq_len in token frames (for learned pos embeddings)")
    ap.add_argument("--prefer-pow2", dest="prefer_pow2", action="store_true")
    ap.add_argument("--no-prefer-pow2", dest="prefer_pow2", action="store_false")
    ap.set_defaults(prefer_pow2=None)
    ap.add_argument("--extra-ctx-s", type=float, default=None,
                    help="Extra context (seconds) to include on both sides in recommended seq_len")
    args = ap.parse_args(argv)

    cfg = DataConfig()
    if args.config:
        cfg.config = args.config
        data = load_yaml_config(args.config)
        for key, value in data.items():
            name = key.replace("-", "_")
            if hasattr(cfg, name):
                setattr(cfg, name, value)

    overrides = {
        "wav": args.wav,
        "outdir": args.outdir,
        "gap_seconds": args.gap_seconds,
        "num_gaps": args.num_gaps,
        "min_gap_separation_seconds": args.min_gap_separation_seconds,
        "target_sr": args.target_sr,
        "bandwidth": args.bandwidth,
        "center_mode": args.center_mode,
        "margin_seconds": args.margin_seconds,
        "seed": args.seed,
        "max_len_cap": args.max_len_cap,
        "prefer_pow2": args.prefer_pow2,
        "extra_ctx_s": args.extra_ctx_s,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    validate_data_config(cfg)
    return cfg


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def load_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)  # [T, C]
    audio = audio.astype(np.float32)
    audio = audio.mean(axis=1)  # mono [T]
    # Normalize (avoid clipping, keep relative dynamics)
    mx = float(np.max(np.abs(audio))) + 1e-12
    audio = audio / mx
    return audio, sr


def resample_if_needed(audio: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    if sr == target_sr:
        return audio, sr
    try:
        import librosa  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Resampling required but librosa is not installed. "
            "Install with: pip install librosa"
        ) from e

    audio_rs = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio_rs, target_sr


def choose_gap_start_end(
    duration_s: float,
    gap_len_s: float,
    center_mode: str,
    margin_s: float,
    rng: random.Random,
) -> Tuple[float, float]:
    if gap_len_s <= 0:
        raise ValueError("gap length must be > 0")
    if gap_len_s >= duration_s - 2 * margin_s:
        raise ValueError(
            f"gap_len_s={gap_len_s} too large for duration={duration_s:.3f}s "
            f"with margin_s={margin_s:.3f}s"
        )

    half = gap_len_s / 2.0
    min_center = margin_s + half
    max_center = duration_s - margin_s - half

    if center_mode == "middle":
        center = duration_s / 2.0
        # If middle violates margins (rare), clamp
        center = max(min_center, min(max_center, center))
    elif center_mode == "random":
        center = rng.uniform(min_center, max_center)
    else:
        raise ValueError("center_mode must be 'middle' or 'random'")

    start = center - half
    end = center + half
    return start, end


def choose_multiple_gap_positions(
    duration_s: float,
    gap_durations: List[float],
    center_mode: str,
    margin_s: float,
    min_gap_separation_s: float,
    rng: random.Random,
    max_attempts: int = 10000,
) -> List[Tuple[float, float]]:
    """Place multiple non-overlapping gaps in the audio.

    Returns list of (start_s, end_s) tuples sorted by start time.
    """
    n = len(gap_durations)

    if center_mode == "middle":
        # Distribute gaps evenly across the audio
        total_gap = sum(gap_durations)
        total_sep = (n - 1) * min_gap_separation_s
        available = duration_s - 2 * margin_s
        if total_gap + total_sep > available:
            raise ValueError(
                f"Cannot fit {n} gaps (total {total_gap:.2f}s + separations "
                f"{total_sep:.2f}s) in {available:.2f}s of usable audio"
            )
        # Place gaps evenly: divide usable space into n equal bands
        usable_start = margin_s
        usable_end = duration_s - margin_s
        usable_len = usable_end - usable_start
        # Each gap gets a proportional zone
        band_width = usable_len / n
        result: List[Tuple[float, float]] = []
        for i, g_dur in enumerate(gap_durations):
            band_center = usable_start + band_width * (i + 0.5)
            g_start = band_center - g_dur / 2.0
            g_end = band_center + g_dur / 2.0
            # Clamp to margins
            g_start = max(margin_s, g_start)
            g_end = min(duration_s - margin_s, g_end)
            result.append((g_start, g_end))
        # Verify no overlaps/spacing violations
        result.sort(key=lambda x: x[0])
        for i in range(1, len(result)):
            if result[i][0] - result[i - 1][1] < min_gap_separation_s - 1e-9:
                raise ValueError(
                    f"Middle-mode placement failed: gap {i-1} ends at "
                    f"{result[i-1][1]:.3f}s, gap {i} starts at {result[i][0]:.3f}s, "
                    f"separation {result[i][0] - result[i-1][1]:.3f}s < "
                    f"min {min_gap_separation_s:.3f}s"
                )
        return result

    elif center_mode == "random":
        # Random placement with rejection sampling
        for attempt in range(max_attempts):
            candidates: List[Tuple[float, float]] = []
            success = True
            # Shuffle order for variety, then place greedily
            indices = list(range(n))
            rng.shuffle(indices)
            for idx in indices:
                g_dur = gap_durations[idx]
                half = g_dur / 2.0
                min_center = margin_s + half
                max_center = duration_s - margin_s - half

                # Also respect separation from already-placed gaps
                for placed_start, placed_end in candidates:
                    # Forbidden zone: [placed_start - sep - g_dur/2, placed_end + sep + g_dur/2]
                    forbidden_min = placed_start - min_gap_separation_s - half
                    forbidden_max = placed_end + min_gap_separation_s + half
                    # We need center not in [forbidden_min, forbidden_max]
                    # This is handled via clipping available intervals
                    pass

                # Build list of available intervals
                available: List[Tuple[float, float]] = [(min_center, max_center)]
                for placed_start, placed_end in sorted(candidates, key=lambda x: x[0]):
                    new_available: List[Tuple[float, float]] = []
                    forbidden_lo = placed_start - min_gap_separation_s - half
                    forbidden_hi = placed_end + min_gap_separation_s + half
                    for (a, b) in available:
                        if b <= forbidden_lo or a >= forbidden_hi:
                            new_available.append((a, b))
                        else:
                            if a < forbidden_lo:
                                new_available.append((a, forbidden_lo))
                            if b > forbidden_hi:
                                new_available.append((forbidden_hi, b))
                    available = [(a, b) for (a, b) in new_available if b > a + 1e-9]

                if not available:
                    success = False
                    break

                # Pick uniformly from available intervals (weighted by length)
                total_len = sum(b - a for a, b in available)
                pick = rng.uniform(0, total_len)
                cumul = 0.0
                chosen_center = None
                for (a, b) in available:
                    seg_len = b - a
                    if cumul + seg_len >= pick:
                        chosen_center = a + (pick - cumul)
                        break
                    cumul += seg_len
                if chosen_center is None:
                    chosen_center = available[-1][1]

                g_start = chosen_center - half
                g_end = chosen_center + half
                candidates.append((g_start, g_end))

            if success and len(candidates) == n:
                # Re-order to match original gap_durations order
                reordered = [None] * n
                for i, idx in enumerate(indices):
                    reordered[idx] = candidates[i]
                # Sort by start time for output
                reordered.sort(key=lambda x: x[0])
                return reordered

        raise RuntimeError(
            f"Failed to place {n} gaps after {max_attempts} attempts. "
            f"Try reducing gap durations, min separation, or margins."
        )
    else:
        raise ValueError("center_mode must be 'middle' or 'random'")


def apply_gap(audio: np.ndarray, sr: int, start_s: float, end_s: float) -> Tuple[np.ndarray, int, int]:
    start_i = int(round(start_s * sr))
    end_i = int(round(end_s * sr))
    start_i = max(0, min(len(audio), start_i))
    end_i = max(0, min(len(audio), end_i))
    if end_i <= start_i:
        raise ValueError("gap end <= gap start after rounding")
    out = audio.copy()
    out[start_i:end_i] = 0.0
    return out, start_i, end_i


def apply_multiple_gaps(
    audio: np.ndarray, sr: int, gap_positions: List[Tuple[float, float]]
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Apply multiple gaps to audio. Returns (gapped_audio, list of (start_i, end_i))."""
    out = audio.copy()
    sample_ranges: List[Tuple[int, int]] = []
    for start_s, end_s in gap_positions:
        start_i = int(round(start_s * sr))
        end_i = int(round(end_s * sr))
        start_i = max(0, min(len(audio), start_i))
        end_i = max(0, min(len(audio), end_i))
        if end_i <= start_i:
            raise ValueError(f"gap end <= gap start after rounding: ({start_s}, {end_s})")
        out[start_i:end_i] = 0.0
        sample_ranges.append((start_i, end_i))
    return out, sample_ranges


def try_encodec_token_stats(audio: np.ndarray, sr: int, bandwidth: float) -> Optional[Dict[str, Any]]:
    """
    If encodec is available, compute:
      - codes shape [K, F]
      - token fps = F / duration_s
      - samples per token = sr / fps
    """
    try:
        import torch  # type: ignore
        from encodec import EncodecModel  # type: ignore
    except Exception:
        return None

    if sr != 24000:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncodecModel.encodec_model_24khz().to(device).eval()
    model.set_target_bandwidth(float(bandwidth))
    bins = int(model.quantizer.bins)

    wav = torch.from_numpy(audio).float().unsqueeze(0)  # [1, T]
    x = wav.unsqueeze(0).to(device)  # [1, 1, T]
    with torch.no_grad():
        encoded = model.encode(x)
    codes_b, _scale = encoded[0]  # [B, K, F]
    K = int(codes_b.shape[1])
    F = int(codes_b.shape[2])

    duration_s = len(audio) / sr
    fps = F / max(1e-9, duration_s)
    samples_per_token = sr / fps

    return {
        "encodec_available": True,
        "device": device,
        "bandwidth_kbps": float(bandwidth),
        "bins": bins,
        "K_codebooks": K,
        "F_frames": F,
        "token_fps": float(fps),
        "samples_per_token": float(samples_per_token),
    }


def recommend_training_lengths(
    token_fps: float,
    gap_len_s: float,
    max_len_cap: int,
    prefer_pow2: bool = True,
    extra_ctx_s: float = 2.0,
) -> Dict[str, Any]:
    """
    Recommend (in codec frames):
      - gap_len_frames
      - seq_len_frames
      - mask_len_min/max
      - ctx_left/right (frames)
    The goal is: seq_len >= gap + context.
    """
    gap_frames = int(round(gap_len_s * token_fps))

    # context: at least extra_ctx_s on both sides, and also at least gap/4
    ctx_frames_min = int(round(extra_ctx_s * token_fps))
    ctx_frames = max(ctx_frames_min, gap_frames // 4)

    seq_len = gap_frames + 2 * ctx_frames

    # don't go below a reasonable floor
    seq_len = max(seq_len, 512)

    if prefer_pow2:
        seq_len = next_pow2(seq_len)

    if seq_len > max_len_cap:
        seq_len = max_len_cap
        ctx_frames = max(0, (seq_len - gap_frames) // 2)

    mask_min = max(32, gap_frames // 4)
    mask_max = min(seq_len, max(mask_min, gap_frames))

    return {
        "gap_len_frames": int(gap_frames),
        "ctx_left_frames": int(ctx_frames),
        "ctx_right_frames": int(ctx_frames),
        "seq_len_frames": int(seq_len),
        "mask_len_min_frames": int(mask_min),
        "mask_len_max_frames": int(mask_max),
        "max_len_frames_required": int(seq_len),
        "notes": (
            "These recommendations assume a bidirectional (encoder/BERT-style) masked-span model "
            "over EnCodec tokens. If you want larger context, increase max_len and retrain."
        ),
    }


def recommend_training_lengths_multi(
    token_fps: float,
    gap_durations_s: List[float],
    max_len_cap: int,
    prefer_pow2: bool = True,
    extra_ctx_s: float = 2.0,
) -> Dict[str, Any]:
    """Recommend training hyperparams for multi-gap scenarios.

    The key insight: seq_len must accommodate the LARGEST gap plus context.
    The curriculum will train up to largest_gap_frames mask length.
    """
    gap_frames_per_gap = [int(round(g * token_fps)) for g in gap_durations_s]
    largest_gap_frames = max(gap_frames_per_gap)

    ctx_frames_min = int(round(extra_ctx_s * token_fps))
    ctx_frames = max(ctx_frames_min, largest_gap_frames // 4)

    seq_len = largest_gap_frames + 2 * ctx_frames
    seq_len = max(seq_len, 512)

    if prefer_pow2:
        seq_len = next_pow2(seq_len)

    if seq_len > max_len_cap:
        seq_len = max_len_cap
        ctx_frames = max(0, (seq_len - largest_gap_frames) // 2)

    mask_min = max(32, largest_gap_frames // 4)
    mask_max = min(seq_len, max(mask_min, largest_gap_frames))

    return {
        "seq_len_frames": int(seq_len),
        "mask_len_min_frames": int(mask_min),
        "mask_len_max_frames": int(mask_max),
        "largest_gap_frames": int(largest_gap_frames),
        "gap_frames_per_gap": gap_frames_per_gap,
        "ctx_left_frames": int(ctx_frames),
        "ctx_right_frames": int(ctx_frames),
        "max_len_frames_required": int(seq_len),
        "notes": (
            "Multi-gap recommendations. seq_len accommodates the largest gap. "
            "Use curriculum learning to train up to largest_gap_frames mask length."
        ),
    }


def main():
    cfg = parse_args()

    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(cfg.seed)

    audio, sr = load_wav_mono(cfg.wav)
    audio, sr = resample_if_needed(audio, sr, cfg.target_sr)
    duration_s = len(audio) / sr

    base_name = Path(cfg.wav).stem

    # Optional: compute token fps once (on the *full* audio)
    encodec_stats = try_encodec_token_stats(audio, sr, cfg.bandwidth)

    num_gaps = cfg.num_gaps

    if num_gaps > 1:
        # --- MULTI-GAP MODE ---
        # Resolve gap durations for each gap
        if len(cfg.gap_seconds) == 1:
            gap_durations = [cfg.gap_seconds[0]] * num_gaps
        elif len(cfg.gap_seconds) == num_gaps:
            gap_durations = list(cfg.gap_seconds)
        else:
            raise ValueError(
                f"In multi-gap mode (--num-gaps {num_gaps}), --gap-seconds must "
                f"have exactly 1 or {num_gaps} values, got {len(cfg.gap_seconds)}"
            )

        # Place all gaps
        gap_positions = choose_multiple_gap_positions(
            duration_s=duration_s,
            gap_durations=gap_durations,
            center_mode=cfg.center_mode,
            margin_s=cfg.margin_seconds,
            min_gap_separation_s=cfg.min_gap_separation_seconds,
            rng=rng,
        )

        # Apply all gaps
        gapped, sample_ranges = apply_multiple_gaps(audio, sr, gap_positions)

        # Build gap info list
        gaps_info: List[Dict[str, Any]] = []
        for i, ((start_s, end_s), (start_i, end_i)) in enumerate(zip(gap_positions, sample_ranges)):
            g_len_s = end_s - start_s
            gaps_info.append({
                "gap_len_s": float(g_len_s),
                "gap_start_s": float(start_s),
                "gap_end_s": float(end_s),
                "gap_start_sample": int(start_i),
                "gap_end_sample": int(end_i),
                "gap_len_samples": int(end_i - start_i),
            })

        # Filenames
        durations_tag = "_".join(f"{d:.1f}s" for d in gap_durations).replace(".", "p")
        tag = f"multigap_{num_gaps}x_{durations_tag}"
        wav_out = outdir / f"{base_name}_{tag}.wav"
        ann_out = outdir / f"{base_name}_{tag}.json"

        sf.write(str(wav_out), gapped, sr)

        ann: Dict[str, Any] = {
            "source_wav": os.path.abspath(cfg.wav),
            "config": cfg.config,
            "generated_wav": os.path.abspath(str(wav_out)),
            "sr": int(sr),
            "duration_s": float(duration_s),
            "num_gaps": num_gaps,
            "min_gap_separation_seconds": float(cfg.min_gap_separation_seconds),
            "center_mode": cfg.center_mode,
            "margin_seconds": float(cfg.margin_seconds),
            "gaps": gaps_info,
            # Backward compat: also include "gap" if single gap
            "recommendations": {
                "token_based": None,
                "seconds_based": {
                    "suggested_context_each_side_s": float(cfg.extra_ctx_s),
                    "notes": (
                        "If EnCodec token stats are unavailable, use seconds-based windowing. "
                        "After encoding, convert seconds to frames using token_fps."
                    ),
                },
            },
            "encodec_stats_full_audio": encodec_stats,
        }

        # Backward compat: if single gap, also include old-style "gap" key
        if num_gaps == 1:
            ann["gap"] = gaps_info[0].copy()
            ann["gap"]["center_mode"] = cfg.center_mode
            ann["gap"]["margin_seconds"] = float(cfg.margin_seconds)

        # Token-based recommendations
        if encodec_stats is not None and encodec_stats.get("token_fps") is not None:
            token_fps = float(encodec_stats["token_fps"])
            ann["recommendations"]["token_based"] = recommend_training_lengths_multi(
                token_fps=token_fps,
                gap_durations_s=[g["gap_len_s"] for g in gaps_info],
                max_len_cap=int(cfg.max_len_cap),
                prefer_pow2=bool(cfg.prefer_pow2),
                extra_ctx_s=float(cfg.extra_ctx_s),
            )

        with open(ann_out, "w", encoding="utf-8") as f:
            json.dump(ann, f, indent=2)

        print(f"[OK] wrote {wav_out.name} + {ann_out.name}  ({num_gaps} gaps)")

    else:
        # --- SINGLE-GAP MODE (backward compatible) ---
        summary_rows: List[Dict[str, Any]] = []

        for gap_len_s in cfg.gap_seconds:
            gap_len_s = float(gap_len_s)

            start_s, end_s = choose_gap_start_end(
                duration_s=duration_s,
                gap_len_s=gap_len_s,
                center_mode=cfg.center_mode,
                margin_s=cfg.margin_seconds,
                rng=rng,
            )

            gapped, start_i, end_i = apply_gap(audio, sr, start_s, end_s)

            # filenames
            tag = f"gap_{gap_len_s:.3f}s".replace(".", "p")
            wav_out = outdir / f"{base_name}_{tag}.wav"
            ann_out = outdir / f"{base_name}_{tag}.json"

            sf.write(str(wav_out), gapped, sr)

            gap_info = {
                "gap_len_s": float(gap_len_s),
                "gap_start_s": float(start_s),
                "gap_end_s": float(end_s),
                "gap_start_sample": int(start_i),
                "gap_end_sample": int(end_i),
                "gap_len_samples": int(end_i - start_i),
            }

            ann: Dict[str, Any] = {
                "source_wav": os.path.abspath(cfg.wav),
                "config": cfg.config,
                "generated_wav": os.path.abspath(str(wav_out)),
                "sr": int(sr),
                "duration_s": float(duration_s),
                "num_gaps": 1,
                "min_gap_separation_seconds": float(cfg.min_gap_separation_seconds),
                "center_mode": cfg.center_mode,
                "margin_seconds": float(cfg.margin_seconds),
                # New format
                "gaps": [gap_info],
                # Old format (backward compat)
                "gap": {
                    **gap_info,
                    "center_mode": cfg.center_mode,
                    "margin_seconds": float(cfg.margin_seconds),
                },
                "recommendations": {
                    "token_based": None,
                    "seconds_based": {
                        "suggested_context_each_side_s": float(cfg.extra_ctx_s),
                        "suggested_total_window_s": float(gap_len_s + 2 * cfg.extra_ctx_s),
                        "notes": (
                            "If EnCodec token stats are unavailable, use seconds-based windowing. "
                            "After encoding, convert seconds to frames using token_fps."
                        ),
                    },
                },
                "encodec_stats_full_audio": encodec_stats,
            }

            if encodec_stats is not None and encodec_stats.get("token_fps") is not None:
                token_fps = float(encodec_stats["token_fps"])
                # Single-gap: include both old-style and new multi-gap recs
                ann["recommendations"]["token_based"] = recommend_training_lengths(
                    token_fps=token_fps,
                    gap_len_s=gap_len_s,
                    max_len_cap=int(cfg.max_len_cap),
                    prefer_pow2=bool(cfg.prefer_pow2),
                    extra_ctx_s=float(cfg.extra_ctx_s),
                )
                # Also add multi-gap fields for consistency
                ann["recommendations"]["token_based"]["largest_gap_frames"] = \
                    ann["recommendations"]["token_based"]["gap_len_frames"]
                ann["recommendations"]["token_based"]["gap_frames_per_gap"] = \
                    [ann["recommendations"]["token_based"]["gap_len_frames"]]

            with open(ann_out, "w", encoding="utf-8") as f:
                json.dump(ann, f, indent=2)

            summary_rows.append({
                "wav_out": str(wav_out),
                "ann_out": str(ann_out),
                "gap_len_s": gap_len_s,
                "gap_start_s": start_s,
                "gap_end_s": end_s,
                "sr": sr,
            })

            print(f"[OK] wrote {wav_out.name} + {ann_out.name}")

        # Write a summary JSON
        summary_path = outdir / f"{base_name}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_wav": os.path.abspath(cfg.wav),
                    "config": cfg.config,
                    "outdir": os.path.abspath(str(outdir)),
                    "n_samples": len(summary_rows),
                    "encodec_stats_full_audio": encodec_stats,
                    "samples": summary_rows,
                },
                f,
                indent=2,
            )
        print(f"[OK] wrote {summary_path.name}")


if __name__ == "__main__":
    main()
