"""Shared figure style for Phase 5 (static matplotlib figures for the report).

Colours come from the dataviz skill's validated reference palette: categorical slots in fixed order
(blue, orange, aqua, yellow -- the first three pass the all-pairs colour-vision checks, so charts that
overlap marks use at most three), a one-hue blue ramp for ordinal/sequential encodings, recessive grid
and axes, and text in neutral ink (never in a series colour). Every categorical series also gets a
distinct marker / line style so identity never depends on colour alone.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow (fixed order)
MARKERS = ["o", "s", "^", "D"]
LINESTYLES = ["-", "--", "-.", ":"]
BLUE_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]  # ordinal, 8 steps


def apply_style() -> None:
    """Set matplotlib rcParams (surface/ink colours, grid, fonts) for the Phase 5 report figures."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0,
        "lines.markersize": 6.5,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
    })


def save(fig, out_dir: Path, name: str) -> None:
    """Save a figure as PNG and PDF under out_dir/<name> and close it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=170 if ext == "png" else None, bbox_inches="tight",
                    metadata={"Creator": "phase_5"} if ext == "pdf" else None)
    plt.close(fig)
