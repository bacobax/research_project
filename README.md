# Audio Infill Research Project

This repository is now organized as a config-driven Python project. Training parameters are defined in YAML files and shell scripts are thin wrappers that only select a config.

## Setup

Install [uv](https://docs.astral.sh/uv/), then create and synchronize the
project-local `.venv` from the lockfile:

```bash
uv sync
```

The project pins Python 3.11 in `.python-version`; uv downloads it when needed.
Run Python tools through `uv run` so the environment stays synchronized.

## Project Layout

- `src/audio_infill/`: core source code (`train.py`, `make_gapped_dataset.py`, `graph.py`)
- `scripts/`: runnable wrappers (`run_train.sh`, `resume.sh`, `smoke_test.sh`, `check_replicability.sh`; legacy `run_multigap*_train.sh`)
- `configs/train/`: explicit training configs (all parameters listed)
- `configs/data/`: dataset generation config templates
- `data/raw/`: source media
- `data/processed/`: generated dataset samples and annotations
- `data/interim/`: temporary/intermediate audio artifacts
- `outputs/runs/`: training outputs (checkpoints, tensorboard, samples)
- `outputs/runs_test/`: smoke-test outputs
- `phase_N/`: experiment scripts per research phase (0-6); `phase_N/results/` holds compact, Git-trackable results, `phase_N/INTERPRETING_RESULTS.md` explains how to read them
- `docs/`: guides (`SUPERVISOR_GUIDE.md`, `interpreting_results.md`), `experiment_log.md`, `RERUN_PLAN.md`, `archive/`
- `notebooks/`: notebooks
- `docs/figures/`: generated diagrams and figures
- `assets/prompts/`: prompt assets
- `tests/`: smoke tests for config loading and parsing behavior

## Documentation

- Start here (supervisors): `docs/SUPERVISOR_GUIDE.md`
- Reading result files and figures: `docs/interpreting_results.md`
- Reproducing from scratch (data download, seeds): `SETUP.md`
- Run history: `docs/experiment_log.md`

## Training (Config-Driven)

Training now runs through a YAML config:

```bash
uv run audio-infill-train --config configs/train/phase6_final_songs17_seed43.yaml
```

CLI overrides are optional and remain supported:

```bash
uv run audio-infill-train \
  --config configs/train/phase6_final_songs17_seed43.yaml \
  --total-steps 1000 \
  --device cuda:0
```

Shell wrappers (thin):

```bash
scripts/smoke_test.sh   # configs/train/smoke.yaml: 30-step pipeline check on the setup data
scripts/run_train.sh    # configs/train/phase6_final_songs17_seed43.yaml: the final run
scripts/resume.sh       # resume the final run from its latest.pt (RESUME_CKPT=... to choose)
```

Each script accepts optional overrides after the config call, for example:

```bash
scripts/run_train.sh --device cuda:0 --total-steps 10000
```

`scripts/run_multigap_train.sh` and `scripts/run_multigap_boundary_train.sh` are legacy
(pre-Phase 0): their inputs cannot be rebuilt from `SETUP.md` (see there).

## Config Files

All train configs under `configs/train/` contain explicit values for all training/runtime parameters, including values that otherwise have Python defaults.

Key files:

- `configs/train/phase6_final_songs17_seed43.yaml`: the final run
- `configs/train/phase{1..4}_*.yaml`, `nuvole_bianche_short_gaps_encoder_decoder*.yaml`: the Phase 0-5 runs
- `configs/train/smoke.yaml`: tiny smoke run for pipeline checks (`scripts/smoke_test.sh`)
- Legacy, inputs not rebuildable: `base.yaml`, `longrun.yaml`, `multigap*.yaml`, `resume_multigap.yaml`

## Dataset Generation

Dataset generation script remains available:

```bash
uv run audio-infill-dataset --help
```

Data config templates are provided in `configs/data/` for reproducible parameter sets.

## Outputs

Generated training artifacts are written under `outputs/`:

- `outputs/runs/...`
- `outputs/runs_test/...`

Raw outputs are intentionally ignored by Git. Each research phase should export
its portable CSV, JSON, PDF, PNG, and LaTeX assets to `phase_N/results/`. Create
a new phase with both locations and starter documentation using:

```bash
uv run python scripts/scaffold_phase.py 2
```

This creates the local raw directory `outputs/runs/phase_2/` and the tracked
paper-results directory `phase_2/results/`. Checkpoints, TensorBoard event files,
and WAV files remain ignored even if accidentally placed inside a results bundle.

## Tests

Run tests from the repository root:

```bash
uv run pytest -q
```

After the `SETUP.md` steps, check that every script runs on this checkout
(`tests/test_replicability.py`; plain `pytest` runs only its static part):

```bash
scripts/check_replicability.sh          # ~2 min: data, every script starts, results regenerate
scripts/check_replicability.sh full     # + few-step training via each wrapper, ~10 min on a GPU
```

To run the standard-library fallback:

```bash
uv run python -m unittest discover -s tests -p 'test_*.py'
```
