import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torchviz import make_dot

from audio_infill.config import TrainConfig, load_yaml_config
from audio_infill.train import JointCodebookInfiller


@dataclass
class GraphConfig:
    config: Optional[str] = None
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    max_len: int = 2048
    dropout: float = 0.1
    codebooks: int = 8
    bins: int = 1024
    batch_size: int = 2
    time_steps: int = 128
    out_dir: str = "docs/figures"
    dot_name: str = "network.dot"
    png_name: str = "network"


def parse_args(argv=None) -> GraphConfig:
    parser = argparse.ArgumentParser(description="Export a graphviz view of the infiller model")
    parser.add_argument("--config", type=str, default=None, help="Optional train config YAML to source model hyperparameters from")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--codebooks", type=int, default=None)
    parser.add_argument("--bins", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--time-steps", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--dot-name", type=str, default=None)
    parser.add_argument("--png-name", type=str, default=None)
    args = parser.parse_args(argv)

    cfg = GraphConfig()
    if args.config:
        cfg.config = args.config
        train_cfg = TrainConfig()
        for key, value in load_yaml_config(args.config).items():
            name = key.replace("-", "_")
            if hasattr(train_cfg, name):
                setattr(train_cfg, name, value)
        cfg.d_model = train_cfg.d_model
        cfg.n_heads = train_cfg.n_heads
        cfg.n_layers = train_cfg.n_layers
        cfg.max_len = train_cfg.max_len
        cfg.dropout = train_cfg.dropout

    for key, value in {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "max_len": args.max_len,
        "dropout": args.dropout,
        "codebooks": args.codebooks,
        "bins": args.bins,
        "batch_size": args.batch_size,
        "time_steps": args.time_steps,
        "out_dir": args.out_dir,
        "dot_name": args.dot_name,
        "png_name": args.png_name,
    }.items():
        if value is not None:
            setattr(cfg, key, value)

    validate_graph_config(cfg)
    return cfg


def validate_graph_config(cfg: GraphConfig):
    for name in ["d_model", "n_heads", "n_layers", "max_len", "codebooks", "bins", "batch_size", "time_steps"]:
        if getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    if cfg.dropout < 0:
        raise ValueError("dropout must be >= 0")


def main(argv=None):
    cfg = parse_args(argv)

    model = JointCodebookInfiller(
        K=cfg.codebooks,
        bins=cfg.bins,
        mask_token=cfg.bins,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
    )

    dummy_input = torch.randint(0, cfg.bins, (cfg.batch_size, cfg.codebooks, cfg.time_steps))
    gap_start = max(0, cfg.time_steps // 3)
    gap_end = min(cfg.time_steps, gap_start + max(1, cfg.time_steps // 6))
    dummy_input[:, :, gap_start:gap_end] = cfg.bins
    output = model(dummy_input)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dot_path = out_dir / cfg.dot_name
    png_base = out_dir / cfg.png_name

    dot = make_dot(output, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
    dot.save(str(dot_path))
    print(f"Saved: {dot_path}")
    try:
        dot.render(str(png_base), format="png")
        print(f"Saved: {png_base}.png")
    except Exception as e:
        print(f"Could not render PNG (install graphviz system package): {e}")
        print(f"You can render manually: dot -Tpng {dot_path} -o {png_base}.png")


if __name__ == "__main__":
    main()
