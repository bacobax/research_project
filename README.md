# Audio Infill Research Project

This repository is now organized as a config-driven Python project. Training parameters are defined in YAML files and shell scripts are thin wrappers that only select a config.

## Project Layout

- `src/audio_infill/`: core source code (`train.py`, `make_gapped_dataset.py`, `graph.py`)
- `scripts/`: runnable wrappers (`run_train.sh`, `run_multigap_train.sh`, `resume.sh`, `smoke_test.sh`)
- `configs/train/`: explicit training configs (all parameters listed)
- `configs/data/`: dataset generation config templates
- `data/raw/`: source media
- `data/processed/`: generated dataset samples and annotations
- `data/interim/`: temporary/intermediate audio artifacts
- `outputs/runs/`: training outputs (checkpoints, tensorboard, samples)
- `outputs/runs_test/`: smoke-test outputs
- `notebooks/`: notebooks
- `docs/figures/`: generated diagrams and figures
- `assets/prompts/`: prompt assets
- `tests/`: smoke tests for config loading and parsing behavior

## Training (Config-Driven)

Training now runs through a YAML config:

```bash
PYTHONPATH=src python -m audio_infill.train --config configs/train/longrun.yaml
```

CLI overrides are optional and remain supported:

```bash
PYTHONPATH=src python -m audio_infill.train \
  --config configs/train/longrun.yaml \
  --total-steps 1000 \
  --device cuda:0
```

Shell wrappers (thin):

```bash
scripts/run_train.sh
scripts/run_multigap_train.sh
scripts/resume.sh
scripts/smoke_test.sh
```

Each script accepts optional overrides after the config call, for example:

```bash
scripts/run_multigap_train.sh --device cuda:0 --total-steps 10000
```

## Config Files

All train configs under `configs/train/` contain explicit values for all training/runtime parameters, including values that otherwise have Python defaults.

Key files:

- `configs/train/base.yaml`: baseline explicit config
- `configs/train/longrun.yaml`: single-gap long training
- `configs/train/multigap.yaml`: multi-gap curriculum training
- `configs/train/resume_multigap.yaml`: resume/continue setup
- `configs/train/smoke.yaml`: tiny smoke run for pipeline checks

## Dataset Generation

Dataset generation script remains available:

```bash
PYTHONPATH=src python -m audio_infill.make_gapped_dataset --help
```

Data config templates are provided in `configs/data/` for reproducible parameter sets.

## Outputs

Generated training artifacts are written under `outputs/`:

- `outputs/runs/...`
- `outputs/runs_test/...`

## Tests

Run tests from the repository root:

```bash
PYTHONPATH=src python -m pytest -q
```

If `pytest` is unavailable, install it or run via `python -m unittest discover -s tests -p 'test_*.py'`.
