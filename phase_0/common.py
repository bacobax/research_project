"""
Phase 0 shared infrastructure.

Reconstructs Trainer objects from the two completed runs' real training configs plus a saved
checkpoint's weights -- WITHOUT ever touching outputs/runs/**/{checkpoints,tb,samples}/. All
Phase 0 output (TensorBoard writer, any inspection artifacts) is redirected to
phase_0/results/scratch/ before the Trainer is constructed.

Also provides: a train-region control example builder (same construction as the real
holdout_regions validation, but sampling regions that are additionally required to avoid the
run's real validation holdout ranges -- i.e. drawn from training territory), non-learned
token-space baseline fillers, and audio-domain metrics (multi-res STFT, SI-SDR, log-mel L1)
for scoring decoded gap fills.

See phase_0/README.md for how each exp0N_*.py script uses these.
"""
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio_infill.train import (  # noqa: E402
    FixedMaskedSpanDataset,
    FixedValidationExample,
    MultiResolutionSTFTLoss,
    Trainer,
    ValidationGroupSpec,
    ValidationRegion,
    _pick_validation_regions,
    _region_mean_activity,
    _build_cumsum,
    _valid_non_gap_starts,
    build_holdout_region_validation_examples,
    candidate_mask_offsets,
    frame_bounds_to_sample_bounds,
    merge_ranges,
    parse_args,
    set_seed,
    span_mean_from_cumsum,
)

RESULTS_DIR = REPO_ROOT / "phase_0" / "results"
SCRATCH_OUTPUT_DIR = str(RESULTS_DIR / "scratch")
SUMMARY_CSV = RESULTS_DIR / "summary.csv"
SUMMARY_COLUMNS = ["experiment", "run", "checkpoint", "step", "band", "mask_len", "method", "metric", "value"]

class RecordingWriter:
    """Drop-in replacement for SummaryWriter that records every add_scalar call instead of
    writing TensorBoard event files, so scripts can read back e.g. val/high_acc_top1
    programmatically. Trainer.run_validation() only calls add_scalar/flush on its writer."""

    def __init__(self) -> None:
        self.scalars: List[Tuple[str, float, int]] = []

    def add_scalar(self, name: str, value: float, step: int) -> None:
        self.scalars.append((name, float(value), int(step)))

    def add_figure(self, *args, **kwargs) -> None:
        pass  # inspection is disabled on cfg, but be defensive

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def latest(self, tag: str) -> Optional[float]:
        for name, value, _ in reversed(self.scalars):
            if name == tag:
                return value
        return None

    def as_dict(self) -> Dict[str, float]:
        return {name: value for name, value, _ in self.scalars}


RUNS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "config": REPO_ROOT / "configs/train/nuvole_bianche_short_gaps_encoder_decoder.yaml",
        "checkpoints": {
            "best_val": REPO_ROOT
            / "outputs/runs/nuvole_bianche_short_gaps/nuvole_bianche_short_gaps_encoder_decoder/checkpoints/best_val.pt",
            "best": REPO_ROOT
            / "outputs/runs/nuvole_bianche_short_gaps/nuvole_bianche_short_gaps_encoder_decoder/checkpoints/best.pt",
        },
    },
    "bigmodel": {
        "config": REPO_ROOT / "configs/train/nuvole_bianche_short_gaps_encoder_decoder_bigmodel.yaml",
        "checkpoints": {
            "best_val": REPO_ROOT
            / "outputs/runs/nuvole_bianche_short_gaps/nuvole_bianche_short_gaps_encoder_decoder_bigmodel/checkpoints/best_val.pt",
            "best": REPO_ROOT
            / "outputs/runs/nuvole_bianche_short_gaps/nuvole_bianche_short_gaps_encoder_decoder_bigmodel/checkpoints/best.pt",
        },
    },
}


# ---------------------------------------------------------------------------
# Trainer reconstruction (safe: never writes into outputs/runs/)
# ---------------------------------------------------------------------------


def build_trainer(run_name: str, checkpoint_tag: str) -> Trainer:
    """Reconstruct a Trainer exactly as the real training entry point would (same
    parse_args() path: yaml -> TrainConfig -> annotation loading), then load one
    checkpoint's weights. output_dir/run_name are forced to scratch space *before*
    Trainer.__init__ so its checkpoints/tb/samples directories are never the real ones.
    """
    run = RUNS[run_name]
    checkpoint_path = run["checkpoints"][checkpoint_tag]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    cfg, _ = parse_args(["--config", str(run["config"])])
    cfg.output_dir = SCRATCH_OUTPUT_DIR
    cfg.run_name = f"{run_name}_{checkpoint_tag}_phase0"
    cfg.validation_inspection_enabled = False
    cfg.validation_save_artifacts = False

    trainer = Trainer(cfg)
    trainer.load_checkpoint(str(checkpoint_path))
    trainer.model.eval()
    # load_checkpoint restores the training run's python/numpy/torch RNG state; re-seed so
    # any sampling this script does (e.g. random_tokens baseline) is reproducible on its own.
    set_seed(cfg.seed)
    # Replace the real SummaryWriter (already flushed __init__-time hparams/holdout-summary
    # scalars into scratch space, harmless) with a recorder so run_validation_safe's acc/ppl
    # scalars can be read back programmatically instead of only its {combined,high,low}_loss
    # return dict.
    trainer.writer.close()
    trainer.writer = RecordingWriter()
    return trainer


def run_validation_safe(
    trainer: Trainer,
    dataloaders: Dict[str, DataLoader],
    group_specs: Dict[str, ValidationGroupSpec],
    step: int = 0,
) -> Optional[Dict[str, float]]:
    """Call Trainer.run_validation() with a substituted example set, guaranteed not to write
    a best_val.pt (best_val_loss forced to -inf) or any inspection artifacts (already disabled
    on cfg by build_trainer)."""
    trainer.validation_dataloaders = dataloaders
    trainer.validation_group_specs = group_specs
    trainer.validation_inspection_examples = {}
    trainer.best_val_loss = float("-inf")
    return trainer.run_validation(step=step)


def build_validation_dataloaders(
    trainer: Trainer,
    grouped_examples: Dict[str, List[FixedValidationExample]],
) -> Tuple[Dict[str, DataLoader], Dict[str, ValidationGroupSpec]]:
    """Mirror Trainer._build_validation's loader construction (train.py ~2022-2046): only
    '*_len_*' keyed groups become dataloaders; band-only aggregate keys stay in group_specs
    for bookkeeping but are not loaded (run_validation recomputes band aggregates itself)."""
    cfg = trainer.cfg
    val_batch_size = cfg.validation_batch_size or cfg.batch_size
    group_specs = {
        key: ValidationGroupSpec(
            band="high_activity" if key.startswith("high_activity") else "low_activity",
            mask_len=(int(key.rsplit("_", 1)[-1]) if "_len_" in key else None),
        )
        for key in grouped_examples
    }
    loader_examples = {k: v for k, v in grouped_examples.items() if "_len_" in k}
    dataloaders = {}
    for group_name, items in loader_examples.items():
        dataset = FixedMaskedSpanDataset(items)
        dataloaders[group_name] = DataLoader(
            dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(trainer.device.type == "cuda"),
            drop_last=False,
        )
    return dataloaders, group_specs


# ---------------------------------------------------------------------------
# Validation-region construction (real validation, and its train-region control variant)
# ---------------------------------------------------------------------------


def _validation_build_params(trainer: Trainer) -> Tuple[Tuple[int, ...], int, int, str]:
    """The exact derivation Trainer._build_validation uses for region sizing (train.py
    ~1965-1975), factored out so both the real-validation rebuild and the train-region
    control use identical parameters."""
    cfg = trainer.cfg
    mask_lengths = tuple(int(v) for v in (cfg.validation_mask_lengths or (cfg.mask_len_min, cfg.mask_len_max)))
    max_ctx = max(int(cfg.ctx_left or 0), int(cfg.ctx_right or 0))
    region_len_frames = cfg.validation_region_len_frames
    if region_len_frames is None:
        region_len_frames = max(cfg.seq_len, max(mask_lengths) + 2 * max_ctx)
    region_min_separation = cfg.validation_region_min_separation_frames
    if region_min_separation is None:
        region_min_separation = cfg.seq_len
    sample_name = cfg.sample or Path(cfg.wav_path).stem
    return mask_lengths, int(region_len_frames), int(region_min_separation), sample_name


def build_real_validation_examples(
    trainer: Trainer,
) -> Tuple[Dict[str, List[FixedValidationExample]], List[Tuple[int, int]]]:
    """Rebuild the exact same validation example set the real training run used (same seed,
    cfg.seed + 1009, per Trainer._build_validation) -- deterministic, so this reproduces what
    training actually evaluated on without reading any on-disk artifacts."""
    cfg = trainer.cfg
    mask_lengths, region_len_frames, region_min_separation, sample_name = _validation_build_params(trainer)
    grouped, _, holdout_ranges, _ = build_holdout_region_validation_examples(
        codes=trainer.codes,
        gaps=trainer.gaps_f,
        seq_len=cfg.seq_len,
        mask_lengths=mask_lengths,
        mask_token=trainer.mask_token,
        activity_per_frame=trainer.activity_per_frame,
        activity_low_thr=trainer.activity_low_thr,
        activity_high_thr=trainer.activity_high_thr,
        regions_per_band=cfg.validation_regions_per_band,
        region_len_frames=region_len_frames,
        region_min_separation_frames=region_min_separation,
        examples_per_length_band=cfg.validation_examples_per_length_band,
        mask_stride=cfg.mask_stride,
        seed=cfg.seed + 1009,
        sample_name=sample_name,
        dead_window_min_mean=cfg.dead_window_min_mean,
        dead_window_min_ratio=cfg.dead_window_min_ratio,
    )
    return grouped, holdout_ranges


def build_train_region_control_examples(
    trainer: Trainer,
    seed_offset: int = 5051,
) -> Dict[str, List[FixedValidationExample]]:
    """Same construction as audio_infill.train.build_holdout_region_validation_examples
    (copied here, not monkeypatched, to avoid mutating shared module state), except candidate
    regions are additionally required to avoid trainer.validation_holdout_ranges -- i.e. every
    example is drawn from territory the training dataset was allowed to sample from. Uses a
    seed distinct from the real validation's (cfg.seed + 1009) so masks aren't correlated with
    the real val set.

    _valid_non_gap_starts already accepts a blocked_ranges kwarg
    (train.py:730) that the real pipeline's call never passes (train.py:902) -- that's the one
    line changed below.
    """
    cfg = trainer.cfg
    codes = trainer.codes
    gaps = trainer.gaps_f
    real_holdout = getattr(trainer, "validation_holdout_ranges", [])
    activity_per_frame = trainer.activity_per_frame
    activity_low_thr = trainer.activity_low_thr
    activity_high_thr = trainer.activity_high_thr
    mask_token = trainer.mask_token
    mask_lengths, region_len_frames, region_min_separation_frames, sample_name = _validation_build_params(trainer)
    regions_per_band = cfg.validation_regions_per_band
    examples_per_length_band = cfg.validation_examples_per_length_band
    mask_stride = cfg.mask_stride
    dead_window_min_mean = cfg.dead_window_min_mean
    dead_window_min_ratio = cfg.dead_window_min_ratio
    seed = cfg.seed + seed_offset

    _, frames = codes.shape
    if region_len_frames < cfg.seq_len:
        raise ValueError(f"control region length {region_len_frames} must be >= seq_len {cfg.seq_len}")

    activity_cumsum = _build_cumsum(activity_per_frame)
    active_flag = (activity_per_frame > activity_low_thr).astype(np.float32)
    active_flag_cumsum = _build_cumsum(active_flag)
    # The one change vs. build_holdout_region_validation_examples: also block the real
    # validation's holdout ranges, so candidate regions are training territory.
    region_starts = _valid_non_gap_starts(frames, region_len_frames, gaps, blocked_ranges=real_holdout)
    if not region_starts:
        raise ValueError(f"No valid train-region control regions found for sample={sample_name}")

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
        candidates_high
        if candidates_high
        else [
            ValidationRegion(
                "high_activity", s, s + region_len_frames, *_region_mean_activity(activity_cumsum, active_flag_cumsum, s, s + region_len_frames)
            )
            for s in region_starts
        ],
        count=regions_per_band,
        descending=True,
        min_separation=region_min_separation_frames,
    )
    if len(selected_high) < regions_per_band:
        raise ValueError(f"Unable to select {regions_per_band} high-activity control regions for sample={sample_name}")

    low_source = candidates_low if candidates_low else fallback_low
    selected_low = _pick_validation_regions(
        low_source,
        count=regions_per_band,
        descending=False,
        min_separation=region_min_separation_frames,
        already_selected=selected_high,
    )
    if len(selected_low) < regions_per_band:
        raise ValueError(f"Unable to select {regions_per_band} low-activity control regions for sample={sample_name}")

    regions_by_band = {"high_activity": selected_high, "low_activity": selected_low}
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
                window_starts = list(range(region.start, region.end - cfg.seq_len + 1))
                if not window_starts:
                    continue
                start = rng.choice(window_starts)
                offsets = candidate_mask_offsets(cfg.seq_len, int(mask_len), mask_stride)
                mask_start = int(rng.choice(offsets.tolist()))
                g0 = start + mask_start
                g1 = g0 + int(mask_len)
                mask_mean = float(span_mean_from_cumsum(activity_cumsum, np.array([g0]), np.array([g1]))[0])
                mask_ratio = float(span_mean_from_cumsum(active_flag_cumsum, np.array([g0]), np.array([g1]))[0])
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
                y = codes[:, start : start + cfg.seq_len].clone()
                x = y.clone()
                x[:, mask_start : mask_start + int(mask_len)] = mask_token
                loss_mask = torch.zeros(cfg.seq_len, dtype=torch.bool)
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
                    f"Requested {examples_per_length_band} control examples for band={band} mask_len={mask_len} "
                    f"but built {len(items)} for sample={sample_name}"
                )
            grouped_examples[key] = items
        grouped_examples[band] = aggregated_items

    return grouped_examples


# ---------------------------------------------------------------------------
# Non-learned token-space baselines
# ---------------------------------------------------------------------------

BASELINE_NAMES = ["repeat_left", "repeat_right", "nearest", "random_tokens", "codec_ceiling"]


def fill_baseline(name: str, example: FixedValidationExample, rng: np.random.Generator, vocab: int = 1024) -> torch.Tensor:
    """Return a [K, seq_len] filled-codes tensor for the named non-learned baseline. All
    baselines only ever read example.y outside the gap span (never example.x, which carries
    the mask token) so they're valid zero-training references."""
    y = example.y
    K, seq_len = y.shape
    m0, m1 = int(example.mask_start), int(example.mask_start + example.mask_len)
    filled = y.clone()

    if name == "codec_ceiling":
        return filled  # ground truth, unmasked -- not a "filling" method, a ceiling

    if name == "random_tokens":
        filled[:, m0:m1] = torch.from_numpy(rng.integers(0, vocab, size=(K, m1 - m0))).long()
        return filled

    has_left = m0 > 0
    has_right = m1 < seq_len

    if name == "repeat_left":
        src = m0 - 1 if has_left else m1  # fall back to right if no left context
        filled[:, m0:m1] = y[:, src : src + 1].expand(-1, m1 - m0)
        return filled

    if name == "repeat_right":
        src = m1 if has_right else m0 - 1  # fall back to left if no right context
        filled[:, m0:m1] = y[:, src : src + 1].expand(-1, m1 - m0)
        return filled

    if name == "nearest":
        mid = (m0 + m1) // 2
        left_src = m0 - 1 if has_left else None
        right_src = m1 if has_right else None
        for t in range(m0, m1):
            if t < mid and left_src is not None:
                filled[:, t] = y[:, left_src]
            elif right_src is not None:
                filled[:, t] = y[:, right_src]
            elif left_src is not None:
                filled[:, t] = y[:, left_src]
        return filled

    raise ValueError(f"unknown baseline: {name}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def token_accuracy(filled: torch.Tensor, example: FixedValidationExample) -> Dict[str, float]:
    m0, m1 = int(example.mask_start), int(example.mask_start + example.mask_len)
    pred = filled[:, m0:m1]
    tgt = example.y[:, m0:m1]
    correct = (pred == tgt).float()
    return {"acc_top1": float(correct.mean().item()), "masked_tokens": float(correct.numel())}


def si_sdr_db(pred: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> float:
    alpha = float(np.dot(pred, target)) / (float(np.dot(target, target)) + eps)
    e = pred - alpha * target
    return float(10.0 * np.log10((np.mean((alpha * target) ** 2) + eps) / (np.mean(e**2) + eps)))


def snr_db(pred: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> float:
    e = pred - target
    return float(10.0 * np.log10((np.mean(target**2) + eps) / (np.mean(e**2) + eps)))


def log_mel_l1(pred: np.ndarray, target: np.ndarray, sr: int, n_mels: int = 80, eps: float = 1e-7) -> float:
    import librosa

    n = min(len(pred), len(target))
    if n < 4:
        return float("nan")
    n_fft = min(1024, 1 << max(2, int(np.floor(np.log2(max(4, n))))))
    hop = max(1, n_fft // 4)
    n_mels_eff = max(4, min(n_mels, n_fft // 2))
    pred_mel = librosa.feature.melspectrogram(y=pred[:n].astype(np.float32), sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels_eff)
    tgt_mel = librosa.feature.melspectrogram(y=target[:n].astype(np.float32), sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels_eff)
    return float(np.mean(np.abs(np.log(pred_mel + eps) - np.log(tgt_mel + eps))))


def build_stft_loss(trainer: Trainer) -> MultiResolutionSTFTLoss:
    cfg = trainer.cfg
    return MultiResolutionSTFTLoss(
        n_ffts=cfg.decoded_loss_n_ffts,
        hop_lengths=cfg.decoded_loss_hop_lengths,
        win_lengths=cfg.decoded_loss_win_lengths,
        spectral_convergence_weight=cfg.decoded_loss_spectral_convergence_weight,
        log_magnitude_weight=cfg.decoded_loss_log_magnitude_weight,
    )


def stft_metrics(pred: np.ndarray, target: np.ndarray, stft_loss: MultiResolutionSTFTLoss) -> Dict[str, float]:
    pred_t = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0)
    tgt_t = torch.from_numpy(target.astype(np.float32)).unsqueeze(0)
    n = min(pred_t.shape[1], tgt_t.shape[1])
    min_len = max(stft_loss.win_lengths)
    pred_t, tgt_t = pred_t[:, :n], tgt_t[:, :n]
    if n < min_len:
        pad = min_len - n
        pred_t = F.pad(pred_t, (0, pad))
        tgt_t = F.pad(tgt_t, (0, pad))
    with torch.no_grad():
        out = stft_loss(pred_t, tgt_t)
    return {f"stft_{k}": float(v.item()) for k, v in out.items()}


def audio_metrics(pred_audio: np.ndarray, target_audio: np.ndarray, sr: int, stft_loss: MultiResolutionSTFTLoss) -> Dict[str, float]:
    n = min(len(pred_audio), len(target_audio))
    pred = np.asarray(pred_audio[:n], dtype=np.float64)
    target = np.asarray(target_audio[:n], dtype=np.float64)
    metrics = {
        "wave_l1": float(np.mean(np.abs(pred - target))),
        "wave_mse": float(np.mean((pred - target) ** 2)),
        "si_sdr_db": si_sdr_db(pred, target),
        "snr_db": snr_db(pred, target),
        "log_mel_l1": log_mel_l1(pred, target, sr),
    }
    metrics.update(stft_metrics(pred_audio[:n], target_audio[:n], stft_loss))
    return metrics


def crop_frame_bounds(seq_len: int, mask_start: int, mask_len: int, margin_frames: int) -> Tuple[int, int]:
    lo = max(0, int(mask_start) - int(margin_frames))
    hi = min(int(seq_len), int(mask_start) + int(mask_len) + int(margin_frames))
    return lo, hi


def decode_gap_crop_audio(trainer: Trainer, codes: torch.Tensor, mask_start: int, mask_len: int, margin_frames: int) -> np.ndarray:
    """Decode only the gap+margin span of codes -- NOT the full seq_len window -- matching
    exactly what Trainer._compute_decoded_domain_loss does at training time (crop codes to
    [mask_start-margin, mask_start+mask_len+margin) before decoding). Far cheaper than decoding
    the whole window and slicing samples afterward: for seq_len=512 and a 1-4 frame gap with a
    16-frame margin, this decodes ~33-36 frames instead of 512 (~14x fewer samples)."""
    seq_len = int(codes.shape[1])
    lo, hi = crop_frame_bounds(seq_len, mask_start, mask_len, margin_frames)
    return trainer._decode_validation_window_audio(codes[:, lo:hi])


def gap_crop_bounds(example: FixedValidationExample, total_samples: int, margin_frames: int) -> Tuple[int, int]:
    """Sample-domain bounds of the gap span expanded by margin_frames on each side (clamped
    to the window), matching cfg.decoded_loss_margin_frames semantics."""
    seq_len = int(example.y.shape[1])
    g0 = max(0, int(example.mask_start) - int(margin_frames))
    g1 = min(seq_len, int(example.mask_start) + int(example.mask_len) + int(margin_frames))
    return frame_bounds_to_sample_bounds(total_samples, seq_len, g0, g1)


# ---------------------------------------------------------------------------
# Real-gap ground truth (exp04)
# ---------------------------------------------------------------------------


def load_source_wav_matching_annotation(path: str, target_sr: int) -> np.ndarray:
    """Reproduce audio_infill.make_gapped_dataset.load_wav_mono + resample_if_needed exactly
    (mono mean -> peak normalize -> resample), so gap_start_sample/gap_end_sample from the
    dataset annotation JSON index correctly into the returned array. NOT the same normalization
    order as Trainer._load_audio_sample (which normalizes after resampling) -- order doesn't
    change sample alignment, but this matches the script that produced the annotation."""
    import librosa
    import soundfile as sf

    audio, sr = sf.read(path, always_2d=True)
    audio = audio.astype(np.float32).mean(axis=1)
    mx = float(np.max(np.abs(audio))) + 1e-12
    audio = audio / mx
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return audio


def build_real_gap_example(
    trainer: Trainer,
    true_codes: torch.Tensor,
    gap_frame_bounds: Tuple[int, int],
    gap_index: int,
) -> FixedValidationExample:
    """Build a FixedValidationExample for one of the actual carved-out gaps: a seq_len window
    centered on the gap, with the gap span's target codes taken from true_codes (encoded from
    the original ungapped source) rather than from trainer.codes (which has whatever the
    gapped/silenced audio encoded to at that span)."""
    cfg = trainer.cfg
    seq_len = int(cfg.seq_len)
    frames = trainer.frames
    g0, g1 = gap_frame_bounds
    mask_len = g1 - g0

    window_start = max(0, min(frames - seq_len, g0 - seq_len // 2))
    local_g0, local_g1 = g0 - window_start, g1 - window_start

    y = trainer.codes[:, window_start : window_start + seq_len].clone()
    y[:, local_g0:local_g1] = true_codes[:, g0:g1]
    x = y.clone()
    x[:, local_g0:local_g1] = trainer.mask_token
    loss_mask = torch.zeros(seq_len, dtype=torch.bool)
    loss_mask[local_g0:local_g1] = True

    mask_mean = float(
        span_mean_from_cumsum(_build_cumsum(trainer.activity_per_frame), np.array([g0]), np.array([g1]))[0]
    )
    return FixedValidationExample(
        x=x,
        y=y,
        loss_mask=loss_mask,
        band="real_gap",
        sample_name=f"real_gap_{gap_index}",
        mask_mean_activity=mask_mean,
        mask_len=mask_len,
        window_start=window_start,
        mask_start=local_g0,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def append_summary_rows(rows: List[Dict[str, Any]]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({**{c: "" for c in SUMMARY_COLUMNS}, **row})
