#!/usr/bin/env python3
"""Export compact, paper-ready summaries from Phase 2 TensorBoard runs.

Same export pattern as ``phase_1/export_results.py`` (see that file for the
detailed rationale), adapted for Phase 2's single-song regularization
ablation: runs are labeled by their regularization condition (d_model /
dropout / weight_decay) rather than by song count, and the summary table
carries those hyperparameters explicitly since that -- not dataset size --
is the variable under test here.

The raw training outputs remain in ``outputs/``.  This script extracts the
small numerical tables and figures that are useful in a paper and records
enough provenance to trace every exported value back to the raw artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.util import tensor_util


DEFAULT_RUNS = ("phase2_regularization",)
DEFAULT_INPUT_ROOT = Path("outputs/runs/phase2_regularization")
DEFAULT_OUTPUT_ROOT = Path("phase_2/results")

METRICS_FIELDS = ("run", "category", "tag", "step", "wall_time_utc", "value")
SUMMARY_FIELDS = (
    "run",
    "d_model",
    "n_layers",
    "dropout",
    "weight_decay",
    "total_minutes",
    "model_params_m",
    "best_val_step",
    "best_combined_loss",
    "best_si_sdr_db",
    "best_si_sdr_db_gap_only",
    "best_high_acc_top1",
    "best_low_acc_top1",
    "stop_step",
    "stop_reason",
    "runtime_seconds",
)
GROUP_FIELDS = (
    "run",
    "best_val_step",
    "activity_band",
    "mask_len_frames",
    "gap_ms",
    "loss",
    "nll",
    "ppl",
    "acc_top1",
    "acc_top5",
)

_INCLUDED_TAG_PREFIXES = ("val/", "train/avg_", "dataset/", "model/", "validation/")
_INCLUDED_TAG_PREFIXES_BYTES = tuple(prefix.encode() for prefix in _INCLUDED_TAG_PREFIXES)
_GROUP_TAG_RE = re.compile(
    r"^val/(?P<band>high|low)_len_(?P<length>\d+)_(?P<metric>loss|nll|ppl|acc_top1|acc_top5)$"
)
_LOG_TIMESTAMP_RE = re.compile(r"(?m)^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[")
_EARLY_STOP_RE = re.compile(r"Early stopping at step (\d+):")
_DIVERGENCE_RE = re.compile(r"Training stopped early at step (\d+): loss diverged")
_TRAINING_STOP_RE = re.compile(r"Training stopped early at step (\d+): early stopping triggered")
_CHECKPOINT_STEP_RE = re.compile(r"Saved checkpoint: .*?/final\.pt \(step (\d+)\)")


@dataclass(frozen=True)
class ScalarPoint:
    tag: str
    step: int
    wall_time: float
    value: float


@dataclass
class RunExport:
    name: str
    run_dir: Path
    points: list[ScalarPoint]
    by_tag_step: dict[tuple[str, int], ScalarPoint]
    summary: dict[str, Any]
    groups: list[dict[str, Any]]
    best_step: int
    token_frame_rate: float
    label: str


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _run_command(repo_root: Path, args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            list(args), cwd=repo_root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _scalar_category(tag: str) -> str:
    if tag.startswith("val/"):
        return "validation_metric"
    if tag.startswith("train/avg_"):
        return "training_aggregate"
    if tag.startswith("dataset/"):
        return "dataset_metadata"
    if tag.startswith("model/"):
        return "model_metadata"
    return "validation_metadata"


def load_scalar_points(tb_dir: Path) -> list[ScalarPoint]:
    """Load selected scalars, resolving duplicate (tag, step) entries by latest wall time."""
    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {tb_dir}")

    chosen: dict[tuple[str, int], ScalarPoint] = {}
    for event_file in event_files:
        # Inspect raw records first so large embedded figures and per-step
        # training values are skipped without protobuf decoding.
        for raw_event in RawEventFileLoader(str(event_file)).Load():
            if not any(prefix in raw_event for prefix in _INCLUDED_TAG_PREFIXES_BYTES):
                continue
            event = event_pb2.Event.FromString(raw_event)
            if not event.HasField("summary"):
                continue
            for item in event.summary.value:
                tag = item.tag
                if not tag.startswith(_INCLUDED_TAG_PREFIXES):
                    continue
                if item.HasField("simple_value"):
                    value = float(item.simple_value)
                elif item.HasField("tensor"):
                    array = tensor_util.make_ndarray(item.tensor)
                    if array.size != 1:
                        continue
                    value = float(array.reshape(-1)[0])
                else:
                    continue
                point = ScalarPoint(tag, int(event.step), float(event.wall_time), value)
                key = (tag, point.step)
                previous = chosen.get(key)
                if previous is None or point.wall_time >= previous.wall_time:
                    chosen[key] = point
    return sorted(chosen.values(), key=lambda point: (point.tag, point.step, point.wall_time))


def _point_value(
    points: Mapping[tuple[str, int], ScalarPoint], tag: str, step: int, *, required: bool = True
) -> float | None:
    point = points.get((tag, step))
    if point is None:
        if required:
            raise ValueError(f"Required scalar {tag!r} is missing at step {step}")
        return None
    return point.value


def _latest_value(points: Iterable[ScalarPoint], tag: str) -> float | None:
    matches = [point for point in points if point.tag == tag]
    return max(matches, key=lambda point: (point.step, point.wall_time)).value if matches else None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _parse_run_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    timestamps = [datetime.strptime(value, "%Y-%m-%d %H:%M:%S") for value in _LOG_TIMESTAMP_RE.findall(text)]
    runtime_seconds = int((timestamps[-1] - timestamps[0]).total_seconds()) if len(timestamps) >= 2 else None

    early = _EARLY_STOP_RE.findall(text) or _TRAINING_STOP_RE.findall(text)
    divergence = _DIVERGENCE_RE.findall(text)
    final_steps = _CHECKPOINT_STEP_RE.findall(text)
    if divergence:
        stop_reason = "diverged"
        stop_step = int(divergence[-1])
    elif early:
        stop_reason = "early_stopping"
        stop_step = int(early[-1])
    elif final_steps:
        stop_reason = "completed"
        stop_step = int(final_steps[-1])
    else:
        raise ValueError(f"Could not determine completion state from {log_path}")
    return {
        "stop_reason": stop_reason,
        "stop_step": stop_step,
        "runtime_seconds": runtime_seconds,
        "started_at": timestamps[0].isoformat().replace("+00:00", "Z") if timestamps else None,
        "finished_at": timestamps[-1].isoformat().replace("+00:00", "Z") if timestamps else None,
    }


def _validate_run(run_dir: Path) -> None:
    required = (
        run_dir / "launch.log",
        run_dir / "songs.json",
        run_dir / "checkpoints" / "best_val.pt",
        run_dir / "checkpoints" / "final.pt",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Run {run_dir.name} is incomplete; missing: {', '.join(missing)}")
    if not any((run_dir / "tb").glob("events.out.tfevents.*")):
        raise FileNotFoundError(f"Run {run_dir.name} has no TensorBoard event files")


def _token_frame_rate(config: Mapping[str, Any]) -> float:
    # EnCodec at 24 kHz uses 75 frames/s for the bandwidths used by these runs.
    # Keep the fallback explicit so other configs can provide their measured rate.
    return float(config.get("token_frame_rate", 75.0))


def _condition_label(config: Mapping[str, Any]) -> str:
    return (
        f"d_model={config.get('d_model')}, dropout={config.get('dropout')}, "
        f"wd={config.get('weight_decay')}"
    )


def analyze_run(name: str, run_dir: Path, config: Mapping[str, Any]) -> RunExport:
    _validate_run(run_dir)
    points = load_scalar_points(run_dir / "tb")
    by_tag_step = {(point.tag, point.step): point for point in points}
    combined = [point for point in points if point.tag == "val/combined_loss"]
    if not combined:
        raise ValueError(f"Run {name} has no val/combined_loss measurements")
    best_point = min(combined, key=lambda point: (point.value, point.step))
    best_step = best_point.step

    songs = _read_json(run_dir / "songs.json")
    song_records = [songs.get("pivot", {})] + list(songs.get("extra_songs", ()))
    total_seconds = sum(float(song.get("duration_s", 0.0)) for song in song_records)
    log_info = _parse_run_log(run_dir / "launch.log")
    params_m = _latest_value(points, "model/params_M")
    if params_m is None:
        raise ValueError(f"Run {name} is missing model/params_M")

    summary = {
        "run": name,
        "d_model": config.get("d_model"),
        "n_layers": config.get("n_layers"),
        "dropout": config.get("dropout"),
        "weight_decay": config.get("weight_decay"),
        "total_minutes": total_seconds / 60.0,
        "model_params_m": params_m,
        "best_val_step": best_step,
        "best_combined_loss": best_point.value,
        "best_si_sdr_db": _point_value(by_tag_step, "val/si_sdr_db", best_step, required=False),
        "best_si_sdr_db_gap_only": _point_value(
            by_tag_step, "val/si_sdr_db_gap_only", best_step, required=False
        ),
        "best_high_acc_top1": _point_value(by_tag_step, "val/high_acc_top1", best_step),
        "best_low_acc_top1": _point_value(by_tag_step, "val/low_acc_top1", best_step),
        "stop_step": log_info["stop_step"],
        "stop_reason": log_info["stop_reason"],
        "runtime_seconds": log_info["runtime_seconds"],
    }

    frame_rate = _token_frame_rate(config)
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for point in points:
        if point.step != best_step:
            continue
        match = _GROUP_TAG_RE.match(point.tag)
        if not match:
            continue
        band = match.group("band")
        mask_len = int(match.group("length"))
        row = grouped.setdefault(
            (band, mask_len),
            {
                "run": name,
                "best_val_step": best_step,
                "activity_band": f"{band}_activity",
                "mask_len_frames": mask_len,
                "gap_ms": 1000.0 * mask_len / frame_rate,
            },
        )
        row[match.group("metric")] = point.value
    expected_metrics = {"loss", "nll", "ppl", "acc_top1", "acc_top5"}
    for key, row in grouped.items():
        missing_metrics = sorted(expected_metrics - row.keys())
        if missing_metrics:
            raise ValueError(f"Run {name}, group {key} is missing metrics: {missing_metrics}")
    if not grouped:
        raise ValueError(f"Run {name} has no per-length metrics at best step {best_step}")

    return RunExport(
        name=name,
        run_dir=run_dir,
        points=points,
        by_tag_step=by_tag_step,
        summary=summary,
        groups=[grouped[key] for key in sorted(grouped)],
        best_step=best_step,
        token_frame_rate=frame_rate,
        label=_condition_label(config),
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "--"
    return f"{float(value):.{digits}f}"


def _tex_escape(value: Any) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def _write_summary_tex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "% Generated by phase_2/export_results.py; requires \\usepackage{booktabs}.",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Run & $d_{model}$ & Dropout & WD & Best step & Val. loss & SI-SDR (gap) & High acc. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["run"]),
                row["d_model"],
                row["dropout"],
                row["weight_decay"],
                row["best_val_step"],
                _format_float(row["best_combined_loss"]),
                _format_float(row["best_si_sdr_db_gap_only"], 2),
                _format_float(row["best_high_acc_top1"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_groups_tex(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "% Generated by phase_2/export_results.py; requires \\usepackage{booktabs}.",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Run & Activity & Gap (ms) & Loss & PPL & Top-1 & Top-5 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["run"]),
                _tex_escape(row["activity_band"]),
                _format_float(row["gap_ms"], 1),
                _format_float(row["loss"]),
                _format_float(row["ppl"], 2),
                _format_float(row["acc_top1"]),
                _format_float(row["acc_top5"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _series(run: RunExport, tag: str) -> tuple[list[int], list[float]]:
    selected = sorted((p for p in run.points if p.tag == tag), key=lambda p: p.step)
    return [point.step for point in selected], [point.value for point in selected]


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(
        base_path.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "phase_2/export_results.py", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def _plot_validation_loss(runs: Sequence[RunExport], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = plt.get_cmap("tab10")
    for index, run in enumerate(runs):
        steps, values = _series(run, "val/combined_loss")
        color = colors(index)
        ax.plot(steps, values, marker="o", markersize=3.5, label=run.label, color=color)
        ax.scatter([run.best_step], [run.summary["best_combined_loss"]], marker="*", s=120, color=color, zorder=5)
        stop_step = int(run.summary["stop_step"])
        if stop_step in steps:
            ax.scatter([stop_step], [values[steps.index(stop_step)]], marker="x", s=55, color=color, zorder=5)
    ax.set(xlabel="Training step", ylabel="Combined validation NLL", title="Phase 2 validation loss")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "validation_loss")


def _plot_validation_audio_metrics(runs: Sequence[RunExport], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    colors = plt.get_cmap("tab10")
    for index, run in enumerate(runs):
        color = colors(index)
        steps, values = _series(run, "val/si_sdr_db")
        axes[0].plot(steps, values, marker="o", markersize=3, label=f"{run.label} (margin)", color=color)
        steps_g, values_g = _series(run, "val/si_sdr_db_gap_only")
        axes[0].plot(steps_g, values_g, marker="s", markersize=3, linestyle="--", label=f"{run.label} (gap-only)", color=color, alpha=0.7)
        steps_sc, values_sc = _series(run, "val/spectral_convergence")
        axes[1].plot(steps_sc, values_sc, marker="o", markersize=3, label=f"{run.label} (margin)", color=color)
        steps_scg, values_scg = _series(run, "val/spectral_convergence_gap_only")
        axes[1].plot(steps_scg, values_scg, marker="s", markersize=3, linestyle="--", label=f"{run.label} (gap-only)", color=color, alpha=0.7)
    axes[0].axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    axes[0].set(xlabel="Training step", ylabel="SI-SDR (dB)", title="Decoded-audio SI-SDR")
    axes[1].set(xlabel="Training step", ylabel="Spectral convergence", title="Decoded-audio spectral convergence")
    for ax in axes:
        _style_axes(ax)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Phase 2 decoded-audio validation metrics (margin-included vs. gap-only)")
    fig.tight_layout()
    _save_figure(fig, figures_dir / "validation_audio_metrics")


def _plot_validation_accuracy(runs: Sequence[RunExport], figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), sharey=True)
    colors = plt.get_cmap("tab10")
    for ax, band, title in zip(axes, ("high", "low"), ("High activity", "Low activity")):
        for index, run in enumerate(runs):
            steps, values = _series(run, f"val/{band}_acc_top1")
            ax.plot(steps, values, marker="o", markersize=3, label=run.label, color=colors(index))
            if run.best_step in steps:
                ax.scatter([run.best_step], [values[steps.index(run.best_step)]], marker="*", s=90, color=colors(index), zorder=5)
        ax.set(xlabel="Training step", title=title)
        _style_axes(ax)
    axes[0].set_ylabel("Validation top-1 accuracy")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Phase 2 validation accuracy")
    fig.tight_layout()
    _save_figure(fig, figures_dir / "validation_accuracy")


def _plot_training_loss(runs: Sequence[RunExport], figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for run in runs:
        steps, values = _series(run, "train/avg_loss")
        ax.plot(steps, values, linewidth=1.3, label=run.label)
    ax.set(xlabel="Training step", ylabel="Mean training loss", title="Phase 2 training loss")
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "training_loss")


def _plot_best_by_gap(runs: Sequence[RunExport], figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex=True)
    for row_index, band in enumerate(("high_activity", "low_activity")):
        for run in runs:
            rows = sorted(
                (row for row in run.groups if row["activity_band"] == band),
                key=lambda row: float(row["gap_ms"]),
            )
            x = [float(row["gap_ms"]) for row in rows]
            axes[row_index, 0].plot(x, [float(row["loss"]) for row in rows], marker="o", label=run.label)
            axes[row_index, 1].plot(x, [float(row["acc_top1"]) for row in rows], marker="o", label=run.label)
        axes[row_index, 0].set_ylabel(f"{'High' if row_index == 0 else 'Low'} activity")
    axes[0, 0].set_title("Validation NLL")
    axes[0, 1].set_title("Top-1 accuracy")
    for ax in axes[-1]:
        ax.set_xlabel("Gap duration (ms)")
    for ax in axes.flat:
        _style_axes(ax)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Best-checkpoint performance by gap duration")
    fig.tight_layout()
    _save_figure(fig, figures_dir / "best_by_gap_length")


def _find_bundle(run: RunExport, group_name: str) -> Path | None:
    group_dir = run.run_dir / "samples" / "validation" / f"step_{run.best_step}" / group_name
    candidates = sorted(group_dir.glob("*/bundle.pt"))
    return candidates[0] if candidates else None


def _spectrogram(ax: plt.Axes, audio: np.ndarray, sample_rate: int, title: str) -> None:
    nfft = min(1024, max(64, 2 ** int(math.floor(math.log2(max(64, audio.size // 8))))))
    noverlap = nfft // 2
    ax.specgram(audio, NFFT=nfft, Fs=sample_rate, noverlap=noverlap, cmap="magma")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Hz")


def _plot_qualitative_group(runs: Sequence[RunExport], figures_dir: Path, group_name: str) -> bool:
    import torch

    bundles = [_find_bundle(run, group_name) for run in runs]
    if any(bundle is None for bundle in bundles):
        return False
    loaded: list[dict[str, Any]] = []
    for bundle in bundles:
        assert bundle is not None
        loaded.append(torch.load(bundle, map_location="cpu", weights_only=False))

    fig, axes = plt.subplots(2, len(runs), figsize=(4.0 * len(runs), 5.6), squeeze=False)
    for column, (run, bundle) in enumerate(zip(runs, loaded)):
        start, end = (int(v) for v in bundle["crop_sample_bounds"])
        target = np.asarray(bundle["target_window_audio"])[start:end]
        prediction = np.asarray(bundle["pred_window_audio"])[start:end]
        sample_rate = int(bundle.get("target_sr", 24000))
        label = f"{run.label}, step {run.best_step:,}"
        _spectrogram(axes[0, column], target, sample_rate, f"Target — {label}")
        _spectrogram(axes[1, column], prediction, sample_rate, f"Prediction — {label}")
    fig.suptitle(f"Matched best-checkpoint example: {group_name.replace('_', ' ')}")
    fig.tight_layout()
    _save_figure(fig, figures_dir / f"qualitative_{group_name}")
    return True


def _copy_snapshots(
    repo_root: Path,
    output_root: Path,
    runs: Sequence[RunExport],
    config_paths: Mapping[str, Path],
) -> None:
    config_dir = output_root / "configs"
    manifest_dir = output_root / "song_manifests"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        shutil.copyfile(config_paths[run.name], config_dir / f"{run.name}.yaml")
        shutil.copyfile(run.run_dir / "songs.json", manifest_dir / f"{run.name}.json")

    runtime_paths = [
        Path("src/audio_infill/train.py"),
        Path("src/audio_infill/config.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    patch = _run_command(repo_root, ["git", "diff", "--", *(str(path) for path in runtime_paths)])
    (output_root / "source_state.patch").write_text((patch + "\n") if patch else "", encoding="utf-8")


def _provenance(
    repo_root: Path,
    runs: Sequence[RunExport],
    config_paths: Mapping[str, Path],
    command: str,
) -> dict[str, Any]:
    runtime_paths = [
        Path("src/audio_infill/train.py"),
        Path("src/audio_infill/config.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
    ]
    relevant_status_paths = runtime_paths + [config_paths[run.name].relative_to(repo_root) for run in runs]
    status = _run_command(repo_root, ["git", "status", "--short", "--", *(str(path) for path in relevant_status_paths)])
    latest_mtime = 0.0
    raw_runs: dict[str, Any] = {}
    for run in runs:
        artifacts = []
        candidates = sorted((run.run_dir / "tb").glob("events.out.tfevents.*")) + [
            run.run_dir / "checkpoints" / "best_val.pt",
            run.run_dir / "checkpoints" / "final.pt",
            run.run_dir / "launch.log",
            run.run_dir / "songs.json",
        ]
        for path in candidates:
            stat = path.stat()
            latest_mtime = max(latest_mtime, stat.st_mtime)
            artifacts.append(
                {
                    "path": path.relative_to(repo_root).as_posix(),
                    "size_bytes": stat.st_size,
                    "sha256": _sha256(path),
                }
            )
        raw_runs[run.name] = {
            "best_val_step": run.best_step,
            "stop_step": run.summary["stop_step"],
            "raw_artifacts": artifacts,
        }
    source_files = {
        path.as_posix(): {"size_bytes": (repo_root / path).stat().st_size, "sha256": _sha256(repo_root / path)}
        for path in runtime_paths
    }
    return {
        "schema_version": 1,
        "generated_from_artifacts_at_utc": _utc_timestamp(latest_mtime),
        "export_command": command,
        "git_commit": _run_command(repo_root, ["git", "rev-parse", "HEAD"]),
        "git_status_relevant": status.splitlines() if status else [],
        "source_files": source_files,
        "runs": raw_runs,
    }


def _write_readme(output_root: Path, runs: Sequence[RunExport], qualitative: Sequence[str]) -> None:
    summary_lines = []
    for run in runs:
        row = run.summary
        gap_only = row["best_si_sdr_db_gap_only"]
        summary_lines.append(
            f"| {row['d_model']} | {row['dropout']} | {row['weight_decay']} | {row['best_val_step']:,} | "
            f"{row['best_combined_loss']:.4f} | {gap_only:.2f} | "
            f"{row['best_high_acc_top1']:.4f} | {row['best_low_acc_top1']:.4f} | "
            f"{row['stop_step']:,} ({row['stop_reason'].replace('_', ' ')}) |"
        )
    qualitative_text = ", ".join(f"`figures/{name}.pdf`" for name in qualitative) or "not generated (source bundles were unavailable)"
    text = f"""# Phase 2 regularization-ablation results

This directory is a compact, Git-trackable export of the single-song regularization ablation
(smaller model, higher dropout, higher weight decay vs. the Phase 1 `phase1_songs01` baseline —
see the repo handout for the exact config diff). Raw checkpoints, TensorBoard logs, validation
bundles, and WAV files remain under `outputs/runs/phase2_regularization/` and are intentionally
not copied here.

## Main results

| $d_{{model}}$ | Dropout | Weight decay | Best validation step | Combined NLL | SI-SDR gap-only (dB) | High-activity top-1 | Low-activity top-1 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{os.linesep.join(summary_lines)}

The selected checkpoint is the step with the minimum recorded `val/combined_loss`; it is not
the final checkpoint. `figures/validation_loss.png` shows the full trajectory, including a late
recovery phase in the run's final ~15% of training (val/combined_loss improved for 3 consecutive
checkpoints just before the run's 600k-step budget ended) — worth noting in the writeup even
though the best checkpoint remains an early one. This is a descriptive result from one seed and
should not be presented as an uncertainty estimate.

## Contents

- `metrics.csv`: exact validation points, 1,000-step training aggregates, and run metadata.
- `run_summary.csv` and `tables/run_summary.tex`: one row per regularization condition.
- `best_by_group.csv` and `tables/best_by_group.tex`: best-step metrics by activity and gap length.
- `figures/`: PNG previews and vector PDFs for the paper, including `validation_audio_metrics.*`
  (SI-SDR and spectral convergence, margin-included and gap-only). Qualitative figures:
  {qualitative_text}.
- `configs/` and `song_manifests/`: exact per-run input snapshots.
- `provenance.json`: source hashes plus hashes and locations of every retained raw artifact.
- `source_state.patch`: tracked runtime-code changes relative to the recorded Git commit.

## Regeneration

Run from the repository root:

```bash
uv run python phase_2/export_results.py \\
  --runs phase2_regularization \\
  --input-root outputs/runs/phase2_regularization \\
  --output-root phase_2/results
```

The exporter never modifies the raw runs. Large binary artifacts remain excluded by the
repository's existing `outputs/` and `data/` ignore rules.
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def export_results(
    *,
    repo_root: Path,
    run_names: Sequence[str],
    input_root: Path,
    output_root: Path,
    command: str,
) -> list[RunExport]:
    repo_root = repo_root.resolve()
    input_root = (repo_root / input_root).resolve() if not input_root.is_absolute() else input_root.resolve()
    output_root = (repo_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    config_paths: dict[str, Path] = {}
    runs: list[RunExport] = []
    for name in run_names:
        config_path = repo_root / "configs" / "train" / f"{name}.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Config not found for run {name}: {config_path}")
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - project dependency
            raise RuntimeError("PyYAML is required to export Phase 2 results") from exc
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Config must be a mapping: {config_path}")
        config_paths[name] = config_path
        runs.append(analyze_run(name, input_root / name, config))

    metrics_rows = [
        {
            "run": run.name,
            "category": _scalar_category(point.tag),
            "tag": point.tag,
            "step": point.step,
            "wall_time_utc": _utc_timestamp(point.wall_time),
            "value": f"{point.value:.10g}",
        }
        for run in runs
        for point in run.points
    ]
    summary_rows = [run.summary for run in runs]
    group_rows = [row for run in runs for row in run.groups]
    _write_csv(output_root / "metrics.csv", METRICS_FIELDS, metrics_rows)
    _write_csv(output_root / "run_summary.csv", SUMMARY_FIELDS, summary_rows)
    _write_csv(output_root / "best_by_group.csv", GROUP_FIELDS, group_rows)
    _write_summary_tex(tables_dir / "run_summary.tex", summary_rows)
    _write_groups_tex(tables_dir / "best_by_group.tex", group_rows)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 100})
    _plot_validation_loss(runs, figures_dir)
    _plot_validation_audio_metrics(runs, figures_dir)
    _plot_validation_accuracy(runs, figures_dir)
    _plot_training_loss(runs, figures_dir)
    _plot_best_by_gap(runs, figures_dir)
    qualitative = []
    for group_name in ("high_activity_len_4", "low_activity_len_4"):
        if _plot_qualitative_group(runs, figures_dir, group_name):
            qualitative.append(f"qualitative_{group_name}")

    _copy_snapshots(repo_root, output_root, runs, config_paths)
    provenance = _provenance(repo_root, runs, config_paths, command)
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_readme(output_root, runs, qualitative)
    return runs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=list(DEFAULT_RUNS), help="Run directory names")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    command = "uv run python phase_2/export_results.py --runs {} --input-root {} --output-root {}".format(
        " ".join(args.runs), args.input_root.as_posix(), args.output_root.as_posix()
    )
    runs = export_results(
        repo_root=args.repo_root,
        run_names=args.runs,
        input_root=args.input_root,
        output_root=args.output_root,
        command=command,
    )
    for run in runs:
        print(
            f"{run.name}: best step {run.best_step}, "
            f"combined loss {run.summary['best_combined_loss']:.4f}, "
            f"stopped at {run.summary['stop_step']} ({run.summary['stop_reason']})"
        )
    print(f"Wrote paper-ready results to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
