#!/usr/bin/env python3
"""
make_gapped_dataset.py

Create multiple gapped .wav samples from a source .wav.
For each gap length (seconds) provided, produces:
- a gapped wav (silence inserted by zeroing samples)
- a JSON annotation file with:
  - gap start/end (seconds + samples)
  - suggested EnCodec-token training hyperparams (seq_len / mask_len) if EnCodec is available
  - otherwise, a seconds-based recommendation

Usage examples:
  python make_gapped_dataset.py --wav input.wav --gap-seconds 0.5 1 2 5 10 --outdir out

  # deterministic center gap at exact middle:
  python make_gapped_dataset.py --wav input.wav --gap-seconds 1 2 5 --center-mode middle --outdir out

  # random center but not near edges:
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
        # EnCodec model used here is 24kHz; caller should resample beforehand.
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
        # clamp (and shrink context accordingly)
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
        "max_len_frames_required": int(seq_len),  # for learned pos-emb
        "notes": (
            "These recommendations assume a bidirectional (encoder/BERT-style) masked-span model "
            "over EnCodec tokens. If you want larger context, increase max_len and retrain."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="Input .wav path")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--gap-seconds", nargs="+", type=float, required=True, help="Gap lengths in seconds (list)")
    ap.add_argument("--target-sr", type=int, default=24000, help="Target sample rate for output (default 24000)")
    ap.add_argument("--bandwidth", type=float, default=6.0, help="EnCodec target bandwidth kbps (for token stats)")
    ap.add_argument("--center-mode", choices=["middle", "random"], default="random",
                    help="How to place the gap center")
    ap.add_argument("--margin-seconds", type=float, default=2.0,
                    help="Minimum distance of gap boundaries from edges (seconds)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len-cap", type=int, default=2048,
                    help="Cap for recommended max_len/seq_len in token frames (for learned pos embeddings)")
    ap.add_argument("--prefer-pow2", action="store_true", help="Round recommended seq_len to next power of 2")
    ap.add_argument("--extra-ctx-s", type=float, default=2.0,
                    help="Extra context (seconds) to include on both sides in recommended seq_len")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    audio, sr = load_wav_mono(args.wav)
    audio, sr = resample_if_needed(audio, sr, args.target_sr)
    duration_s = len(audio) / sr

    base_name = Path(args.wav).stem

    # Optional: compute token fps once (on the *full* audio)
    encodec_stats = try_encodec_token_stats(audio, sr, args.bandwidth)

    summary_rows: List[Dict[str, Any]] = []

    for gap_len_s in args.gap_seconds:
        gap_len_s = float(gap_len_s)

        start_s, end_s = choose_gap_start_end(
            duration_s=duration_s,
            gap_len_s=gap_len_s,
            center_mode=args.center_mode,
            margin_s=args.margin_seconds,
            rng=rng,
        )

        gapped, start_i, end_i = apply_gap(audio, sr, start_s, end_s)

        # filenames
        tag = f"gap_{gap_len_s:.3f}s".replace(".", "p")
        wav_out = outdir / f"{base_name}_{tag}.wav"
        ann_out = outdir / f"{base_name}_{tag}.json"

        sf.write(str(wav_out), gapped, sr)

        ann: Dict[str, Any] = {
            "source_wav": os.path.abspath(args.wav),
            "generated_wav": os.path.abspath(str(wav_out)),
            "sr": int(sr),
            "duration_s": float(duration_s),
            "gap": {
                "gap_len_s": float(gap_len_s),
                "gap_start_s": float(start_s),
                "gap_end_s": float(end_s),
                "gap_start_sample": int(start_i),
                "gap_end_sample": int(end_i),
                "gap_len_samples": int(end_i - start_i),
                "center_mode": args.center_mode,
                "margin_seconds": float(args.margin_seconds),
            },
            "recommendations": {
                "token_based": None,
                "seconds_based": {
                    "suggested_context_each_side_s": float(args.extra_ctx_s),
                    "suggested_total_window_s": float(gap_len_s + 2 * args.extra_ctx_s),
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
            ann["recommendations"]["token_based"] = recommend_training_lengths(
                token_fps=token_fps,
                gap_len_s=gap_len_s,
                max_len_cap=int(args.max_len_cap),
                prefer_pow2=bool(args.prefer_pow2),
                extra_ctx_s=float(args.extra_ctx_s),
            )

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

    # Write a summary CSV-ish JSON (simple)
    summary_path = outdir / f"{base_name}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_wav": os.path.abspath(args.wav),
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