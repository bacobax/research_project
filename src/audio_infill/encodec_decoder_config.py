import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from audio_infill.config import load_yaml_config
from audio_infill.encodec_utils import SUPPORTED_ENCODEC_MODELS
from audio_infill.training_common import build_run_paths


@dataclass
class EncodecDecoderTrainConfig:
    config: Optional[str] = None

    output_dir: str = "outputs/runs/encodec_decoder"
    run_name: str = "decoder_finetune"
    seed: int = 42
    device: str = "auto"

    wav_path: str = "data/processed/multigap/wav_test_multigap_4x_1p0s_2p0s_5p0s_10p0s.wav"
    target_sr: int = 24000

    encodec_model: str = "encodec_24khz"
    bandwidth: float = 6.0

    window_seconds: float = 1.5
    hop_seconds: float = 0.75

    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    warmup_steps: int = 0
    total_steps: int = 2000

    log_every: int = 50
    save_every: int = 250
    validation_every: int = 250
    validation_save_audio_every: int = 0
    patience: int = 0

    resume: Optional[str] = None
    decoder_export_path: Optional[str] = None

    waveform_l1_weight: float = 1.0
    waveform_l2_weight: float = 0.0
    stft_weight: float = 1.0
    stft_n_ffts: Tuple[int, ...] = (512, 1024, 2048)
    stft_hop_lengths: Tuple[int, ...] = (128, 256, 512)
    stft_win_lengths: Tuple[int, ...] = (512, 1024, 2048)

    @property
    def checkpoint_dir(self) -> Path:
        return build_run_paths(self.output_dir, self.run_name).checkpoint_dir

    @property
    def tb_dir(self) -> Path:
        return build_run_paths(self.output_dir, self.run_name).tb_dir

    @property
    def samples_dir(self) -> Path:
        return build_run_paths(self.output_dir, self.run_name).samples_dir

    @property
    def artifacts_dir(self) -> Path:
        return build_run_paths(self.output_dir, self.run_name).artifacts_dir


def validate_encodec_decoder_config(cfg: EncodecDecoderTrainConfig) -> None:
    if cfg.encodec_model not in SUPPORTED_ENCODEC_MODELS:
        raise ValueError(
            f"encodec_model must be one of {sorted(SUPPORTED_ENCODEC_MODELS)}, got {cfg.encodec_model!r}"
        )
    if cfg.target_sr <= 0:
        raise ValueError("target_sr must be > 0")
    if cfg.window_seconds <= 0:
        raise ValueError("window_seconds must be > 0")
    if cfg.hop_seconds <= 0:
        raise ValueError("hop_seconds must be > 0")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg.lr <= 0:
        raise ValueError("lr must be > 0")
    if cfg.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0")
    if cfg.grad_clip <= 0:
        raise ValueError("grad_clip must be > 0")
    if cfg.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if cfg.total_steps <= 0:
        raise ValueError("total_steps must be > 0")
    if cfg.log_every <= 0:
        raise ValueError("log_every must be > 0")
    if cfg.save_every <= 0:
        raise ValueError("save_every must be > 0")
    if cfg.validation_every < 0:
        raise ValueError("validation_every must be >= 0")
    if cfg.validation_save_audio_every < 0:
        raise ValueError("validation_save_audio_every must be >= 0")
    if cfg.patience < 0:
        raise ValueError("patience must be >= 0")
    for name in ["waveform_l1_weight", "waveform_l2_weight", "stft_weight"]:
        if getattr(cfg, name) < 0:
            raise ValueError(f"{name} must be >= 0")
    if cfg.waveform_l1_weight == 0 and cfg.waveform_l2_weight == 0 and cfg.stft_weight == 0:
        raise ValueError("at least one decoder reconstruction loss weight must be > 0")
    if not (len(cfg.stft_n_ffts) == len(cfg.stft_hop_lengths) == len(cfg.stft_win_lengths)):
        raise ValueError("STFT parameter lists must have the same length")
    if len(cfg.stft_n_ffts) == 0:
        raise ValueError("STFT parameter lists must be non-empty")
    for idx, (n_fft, hop, win) in enumerate(zip(cfg.stft_n_ffts, cfg.stft_hop_lengths, cfg.stft_win_lengths)):
        if n_fft <= 0 or hop <= 0 or win <= 0:
            raise ValueError(f"STFT params must be > 0 at index {idx}")
        if win > n_fft:
            raise ValueError(f"stft_win_lengths[{idx}] must be <= stft_n_ffts[{idx}]")


def _set_cfg_field(cfg: EncodecDecoderTrainConfig, key: str, value: Any) -> None:
    name = key.replace("-", "_")
    if not hasattr(cfg, name):
        return
    if name in {"betas", "stft_n_ffts", "stft_hop_lengths", "stft_win_lengths"} and isinstance(value, list):
        value = tuple(value)
    setattr(cfg, name, value)


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="EnCodec decoder fine-tuning")
    cfg = EncodecDecoderTrainConfig()

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--wav-path", type=str, default=None)
    parser.add_argument("--target-sr", type=int, default=None)
    parser.add_argument("--encodec-model", type=str, default=None)
    parser.add_argument("--bandwidth", type=float, default=None)

    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--hop-seconds", type=float, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--betas", nargs=2, type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)

    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--validation-every", type=int, default=None)
    parser.add_argument("--validation-save-audio-every", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--decoder-export-path", type=str, default=None)

    parser.add_argument("--waveform-l1-weight", type=float, default=None)
    parser.add_argument("--waveform-l2-weight", type=float, default=None)
    parser.add_argument("--stft-weight", type=float, default=None)
    parser.add_argument("--stft-n-ffts", nargs="+", type=int, default=None)
    parser.add_argument("--stft-hop-lengths", nargs="+", type=int, default=None)
    parser.add_argument("--stft-win-lengths", nargs="+", type=int, default=None)

    args = parser.parse_args(argv)

    if args.config:
        cfg.config = args.config
        mapping = load_yaml_config(args.config)
        for key, value in mapping.items():
            _set_cfg_field(cfg, key, value)

    overrides: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "seed": args.seed,
        "device": args.device,
        "wav_path": args.wav_path,
        "target_sr": args.target_sr,
        "encodec_model": args.encodec_model,
        "bandwidth": args.bandwidth,
        "window_seconds": args.window_seconds,
        "hop_seconds": args.hop_seconds,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "betas": tuple(args.betas) if args.betas is not None else None,
        "grad_clip": args.grad_clip,
        "warmup_steps": args.warmup_steps,
        "total_steps": args.total_steps,
        "log_every": args.log_every,
        "save_every": args.save_every,
        "validation_every": args.validation_every,
        "validation_save_audio_every": args.validation_save_audio_every,
        "patience": args.patience,
        "resume": args.resume,
        "decoder_export_path": args.decoder_export_path,
        "waveform_l1_weight": args.waveform_l1_weight,
        "waveform_l2_weight": args.waveform_l2_weight,
        "stft_weight": args.stft_weight,
        "stft_n_ffts": tuple(args.stft_n_ffts) if args.stft_n_ffts is not None else None,
        "stft_hop_lengths": tuple(args.stft_hop_lengths) if args.stft_hop_lengths is not None else None,
        "stft_win_lengths": tuple(args.stft_win_lengths) if args.stft_win_lengths is not None else None,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    validate_encodec_decoder_config(cfg)
    return cfg, args
