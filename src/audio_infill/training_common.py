import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoint_dir: Path
    tb_dir: Path
    samples_dir: Path
    artifacts_dir: Path


def build_run_paths(output_dir: str, run_name: str) -> RunPaths:
    root = Path(output_dir) / run_name
    return RunPaths(
        root=root,
        checkpoint_dir=root / "checkpoints",
        tb_dir=root / "tb",
        samples_dir=root / "samples",
        artifacts_dir=root / "artifacts",
    )


def ensure_run_dirs(paths: RunPaths) -> None:
    paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.tb_dir.mkdir(parents=True, exist_ok=True)
    paths.samples_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
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


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    cpu_rng = state["torch_cpu"]
    if not isinstance(cpu_rng, torch.ByteTensor):
        cpu_rng = cpu_rng.cpu().byte() if hasattr(cpu_rng, "cpu") else torch.ByteTensor(cpu_rng)
    torch.random.set_rng_state(cpu_rng)

    if torch.cuda.is_available() and "torch_cuda" in state:
        cuda_states = state["torch_cuda"]
        cuda_states = [s.cpu() if s.device.type != "cpu" else s for s in cuda_states]
        torch.cuda.set_rng_state_all(cuda_states)


def log_hparams(
    logger: logging.Logger,
    writer: Any,
    cfg: Any,
    *,
    prefix: str = "=== Hyperparameters ===",
) -> Dict[str, Any]:
    hparams = {
        k: v
        for k, v in vars(cfg).items()
        if not k.startswith("_") and isinstance(v, (int, float, str, bool, list, tuple))
    }
    hparams_safe: Dict[str, Any] = {}
    for key, value in hparams.items():
        if value is None:
            continue
        hparams_safe[key] = json.dumps(value) if isinstance(value, (list, tuple)) else value

    logger.info(prefix)
    for key, value in sorted(hparams_safe.items()):
        logger.info("  %-24s = %s", key, value)
    logger.info("=" * len(prefix))

    if writer is not None:
        if hasattr(writer, "add_text"):
            writer.add_text("hparams", "\n".join(f"{k} = {v}" for k, v in sorted(hparams_safe.items())), 0)
        if hasattr(writer, "add_hparams"):
            writer.add_hparams(hparams_safe, {"hparam/placeholder": 0.0}, run_name=".")
    return hparams_safe


def save_training_checkpoint(
    path: Path,
    *,
    step: int,
    model_key: str,
    model_state: Dict[str, Any],
    optimizer: Any,
    scaler: Any,
    cfg: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "step": int(step),
        model_key: model_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": vars(cfg),
        "rng_state": capture_rng_state(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_training_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)
