import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TrainConfig:
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

    curriculum: bool = False
    curriculum_start_mask: Optional[int] = None
    curriculum_end_mask: Optional[int] = None
    curriculum_warmup_frac: float = 0.1
    curriculum_schedule: str = "linear"

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


def validate_train_config(cfg: TrainConfig):
    if cfg.seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if cfg.mask_len_min <= 0 or cfg.mask_len_max <= 0:
        raise ValueError("mask_len_min and mask_len_max must be > 0")
    if cfg.mask_len_max < cfg.mask_len_min:
        raise ValueError("mask_len_max must be >= mask_len_min")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg.total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    if cfg.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if cfg.log_every <= 0:
        raise ValueError("log_every must be > 0")
    if cfg.save_every <= 0:
        raise ValueError("save_every must be > 0")
    if cfg.test_fill_every < 0:
        raise ValueError("test_fill_every must be >= 0")
    if cfg.validation_every < 0:
        raise ValueError("validation_every must be >= 0")
    if cfg.validation_examples_per_band <= 0:
        raise ValueError("validation_examples_per_band must be > 0")
    if cfg.validation_batch_size is not None and cfg.validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be > 0 when provided")
    if cfg.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if cfg.mask_stride <= 0:
        raise ValueError("mask_stride must be > 0")
    if not (0.0 <= cfg.activity_low_quantile <= 1.0):
        raise ValueError("activity_low_quantile must be in [0, 1]")
    if not (0.0 <= cfg.activity_high_quantile <= 1.0):
        raise ValueError("activity_high_quantile must be in [0, 1]")
    if cfg.activity_low_quantile > cfg.activity_high_quantile:
        raise ValueError("activity_low_quantile must be <= activity_high_quantile")
    if cfg.curriculum_schedule not in {"linear", "cosine"}:
        raise ValueError("curriculum_schedule must be 'linear' or 'cosine'")
    if (cfg.ctx_left is None) != (cfg.ctx_right is None):
        raise ValueError("ctx_left and ctx_right must both be set or both be null")
    if cfg.ctx_left is not None and cfg.ctx_left < 0:
        raise ValueError("ctx_left must be >= 0")
    if cfg.ctx_right is not None and cfg.ctx_right < 0:
        raise ValueError("ctx_right must be >= 0")
    for name in [
        "regime_active_prob",
        "regime_transition_prob",
        "regime_low_prob",
        "regime_uniform_prob",
    ]:
        if getattr(cfg, name) < 0:
            raise ValueError(f"{name} must be >= 0")


def _parse_simple_yaml_value(raw: str) -> Any:
    text = raw.strip()
    if text == "null":
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    if text.startswith("[") and text.endswith("]"):
        normalized = text.replace("null", "None").replace("true", "True").replace("false", "False")
        return ast.literal_eval(normalized)
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def load_yaml_config(path: str) -> Dict[str, Any]:
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping at top-level: {path}")
        return data
    except ModuleNotFoundError:
        mapping: Dict[str, Any] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                mapping[key.strip()] = _parse_simple_yaml_value(value)
        return mapping


def load_annotation(ds_dir: str, sample: str) -> dict:
    json_path = Path(ds_dir) / f"{sample}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Annotation not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_auto_hparams(cfg: TrainConfig, ann: dict):
    if "gaps" in ann:
        first_gap = ann["gaps"][0]
        cfg.gap_start_s = first_gap["gap_start_s"]
        cfg.gap_end_s = first_gap["gap_end_s"]
    elif "gap" in ann:
        gap = ann["gap"]
        cfg.gap_start_s = gap["gap_start_s"]
        cfg.gap_end_s = gap["gap_end_s"]

    cfg.target_sr = ann["sr"]

    rec = ann.get("recommendations", {}).get("token_based")
    if rec is not None:
        cfg.seq_len = rec["seq_len_frames"]
        cfg.mask_len_min = rec["mask_len_min_frames"]
        cfg.mask_len_max = rec["mask_len_max_frames"]
        cfg.max_len = max(cfg.max_len, rec["max_len_frames_required"])

        if "ctx_left_frames" in rec:
            cfg.ctx_left = rec["ctx_left_frames"]
        if "ctx_right_frames" in rec:
            cfg.ctx_right = rec["ctx_right_frames"]
        if rec.get("largest_gap_frames") is not None:
            cfg.curriculum_end_mask = rec["largest_gap_frames"]

    stats = ann.get("encodec_stats_full_audio")
    if stats is not None:
        cfg.bandwidth = stats["bandwidth_kbps"]


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Audio Infiller Training")
    cfg = TrainConfig()

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ds-dir", type=str, default=None)
    parser.add_argument("--sample", type=str, default=None)
    parser.add_argument("--auto-hparam", action="store_true")

    parser.add_argument("--wav-path", type=str, default=None)
    parser.add_argument("--target-sr", type=int, default=None)
    parser.add_argument("--bandwidth", type=float, default=None)
    parser.add_argument("--gap-start-s", type=float, default=None)
    parser.add_argument("--gap-end-s", type=float, default=None)

    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

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
    parser.add_argument("--test-fill-every", type=int, default=None)
    parser.add_argument("--validation-every", type=int, default=None)
    parser.add_argument("--validation-examples-per-band", type=int, default=None)
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)

    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--inpaint-only", action="store_true")
    parser.add_argument("--inpaint-iters", type=int, default=None)
    parser.add_argument("--inpaint-output", type=str, default=None)

    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum-start-mask", type=int, default=None)
    parser.add_argument("--curriculum-end-mask", type=int, default=None)
    parser.add_argument("--curriculum-warmup-frac", type=float, default=None)
    parser.add_argument("--curriculum-schedule", choices=["linear", "cosine"], default=None)

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

    args = parser.parse_args(argv)

    if args.config:
        cfg.config = args.config
        data = load_yaml_config(args.config)
        for key, value in data.items():
            name = key.replace("-", "_")
            if hasattr(cfg, name):
                setattr(cfg, name, tuple(value) if name == "betas" and isinstance(value, list) else value)

    overrides = {
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
    }
    for key, value in overrides.items():
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
        if cfg.auto_hparam:
            apply_auto_hparams(cfg, ann)

    if cfg.run_name is None:
        cfg.run_name = cfg.sample if cfg.sample else "infiller"

    cfg._annotation = ann  # type: ignore[attr-defined]
    validate_train_config(cfg)
    return cfg, args
