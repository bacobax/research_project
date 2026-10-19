# Repository Guidelines

Audio-infill research project: a transformer inpaints short gaps in EnCodec tokens of solo piano. Phases 0-6 are experiments (`phase_N/`); human-oriented map in `docs/SUPERVISOR_GUIDE.md`. Read docs on demand, not up front.

## Layout
- `src/audio_infill/`: runtime code (`train.py`, `config.py`, `make_gapped_dataset.py`, `download_data.py`, `graph.py`). New modules go here.
- `configs/train|data/*.yaml`: all experiment settings; prefer new YAML over hard-coded values.
- `scripts/`: thin `uv run` wrappers plus `scaffold_phase.py`, `export_gap_audio_samples.py`.
- `phase_N/`: experiment scripts; tracked results in `phase_N/results/`; reading guide `phase_N/INTERPRETING_RESULTS.md`.
- `docs/`: `SUPERVISOR_GUIDE.md`, `interpreting_results.md`, `experiment_log.md` (one row per run; update when launching/finishing a run), `RERUN_PLAN.md`, `archive/` (stale, ignore).
- `tests/test_*.py`, `notebooks/`, `docs/figures/`, `assets/prompts/`.
- Untracked/huge: `outputs/` (~9 GB), `data/`, `dataset/`, checkpoints. Never read or commit them; never glob `outputs/`.

## Commands (repo root; uv only, never pip/conda/activate)
- `uv sync`; `uv run pytest -q`; `scripts/smoke_test.sh`; `scripts/run_train.sh` / `resume.sh`.
- `scripts/check_replicability.sh [setup|full]`: post-setup check that every script runs (`tests/test_replicability.py`).
- `scripts/download_data.sh` (`audio-infill-download`) fetches data/weights; see `SETUP.md` for the full reproduce sequence.
- `uv run audio-infill-dataset --help`; `uv run python scripts/scaffold_phase.py N` starts a new phase.

## Style
4-space indent, snake_case (files, functions, YAML keys), PascalCase classes. Add docstrings to new code; match local logging/config patterns; keep diffs small. Shell scripts: `set -euo pipefail`.

## Testing
Tests are `unittest`, pytest-compatible, in `tests/test_<feature>.py`. Config schema changes: update both config dataclasses (`config.py` and `train.py`'s `TrainConfig` must stay in sync) and add parser tests. Training-loop changes need a smoke-level assertion. New scripts under `scripts/` or `phase_*/` must be registered in `SCRIPTS` in `tests/test_replicability.py` (the static test fails otherwise).
Known failures on a clean checkout (9): 8 need legacy `data/processed` fixtures that can't be rebuilt; `test_zeroed_context::test_context_changes_only_after_gap_and_targets_untouched` fails on HEAD too.

## Gotchas
- Check `nvidia-smi` before launching training; configs pin a `device` (often `cuda:1`).
- Check `df -h /` before large writes; keep phase results small (CSV/JSON/PNG/PDF).
- Report validation with `val/combined_loss`, not the log's `Best loss:` (a training average).
- Every analysis gets charts as PNG+PDF, not just tables.

## Commits / PRs
Short imperative subjects. PRs: purpose, configs/scripts/modules touched, exact test command run. Don't commit `outputs/`, checkpoints, `data/`.
