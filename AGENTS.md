# Repository Guidelines

## Project Structure & Module Organization
`src/audio_infill/` contains runtime code: `train.py`, `config.py`, `make_gapped_dataset.py`, and `graph.py`. Keep new Python modules under `src/audio_infill/`.

`configs/train/*.yaml` and `configs/data/*.yaml` store experiment settings; prefer new YAML over hard-coded values. `scripts/*.sh` are thin wrappers that use `uv run` and pick a config. `tests/test_*.py` covers config parsing, dataset logic, validation, and decoded-loss behavior. Exploratory work belongs in `notebooks/`, figures in `docs/figures/`, and prompt assets in `assets/prompts/`.

## Build, Test, and Development Commands
Run all commands from the repository root. uv uses `.python-version` and manages the project-local `.venv`; do not activate or modify a shared environment.

- `uv sync`: create or refresh `.venv` from `uv.lock`.
- `uv run pytest -q`: run the full test suite.
- `uv run python -m unittest discover -s tests -p 'test_*.py'`: standard-library test fallback.
- `scripts/smoke_test.sh`: run the smallest training config for a fast pipeline check.
- `scripts/run_train.sh` or `scripts/resume.sh`: start the default long run or resume flow.
- `uv run audio-infill-dataset --help`: inspect dataset-generation options.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions, variables, files, and YAML keys, and PascalCase for dataclasses and `unittest.TestCase` classes. Keep imports readable, use type hints where surrounding code does, and match local logging/config patterns.

No formatter or linter config is checked in, so keep changes small, consistent, and easy to diff. Shell scripts should stay minimal and use `set -euo pipefail`.

## Testing Guidelines
Add tests as `tests/test_<feature>.py`. The suite uses `unittest` and should stay `pytest`-compatible. When changing config schema, update both config dataclasses and add parser coverage. For training-loop changes, include at least one smoke-level assertion around validation, checkpoints, or scheduling.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `validation set` and `big refactor + activity sampling`. Keep commit titles brief and specific. Pull requests should summarize the purpose, list the configs/scripts/modules touched, and include the exact test command run. Attach plots or screenshots only when training behavior or generated figures changed. Do not commit generated `outputs/`, checkpoints, or `data/` artifacts.
