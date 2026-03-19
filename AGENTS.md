# Repository Guidelines

## Project Structure & Module Organization
`src/audio_infill/` contains runtime code: `train.py`, `config.py`, `make_gapped_dataset.py`, and `graph.py`. Keep new Python modules under `src/audio_infill/`.

`configs/train/*.yaml` and `configs/data/*.yaml` store experiment settings; prefer new YAML over hard-coded values. `scripts/*.sh` are thin wrappers that set `PYTHONPATH=src` and pick a config. `tests/test_*.py` covers config parsing, dataset logic, validation, and decoded-loss behavior. Exploratory work belongs in `notebooks/`, figures in `docs/figures/`, and prompt assets in `assets/prompts/`.

## Build, Test, and Development Commands
Run all commands from the repo root inside the `research-project` conda environment. For interactive work, use `conda activate research-project`. For scripted or agent execution, prefer `conda run -n research-project <command>`.

- `conda run -n research-project pip install -r requirements.txt`: install or refresh dependencies in the shared environment.
- `conda run -n research-project env PYTHONPATH=src python -m pytest -q`: run the full test suite.
- `conda run -n research-project env PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'`: fallback if `pytest` is unavailable.
- `conda run -n research-project scripts/smoke_test.sh`: run the smallest training config for a fast pipeline check.
- `conda run -n research-project scripts/run_train.sh` or `conda run -n research-project scripts/resume.sh`: start the default long run or resume flow.
- `conda run -n research-project env PYTHONPATH=src python -m audio_infill.make_gapped_dataset --help`: inspect dataset-generation options.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, snake_case for functions, variables, files, and YAML keys, and PascalCase for dataclasses and `unittest.TestCase` classes. Keep imports readable, use type hints where surrounding code does, and match local logging/config patterns.

No formatter or linter config is checked in, so keep changes small, consistent, and easy to diff. Shell scripts should stay minimal and use `set -euo pipefail`.

## Testing Guidelines
Add tests as `tests/test_<feature>.py`. The suite uses `unittest` and should stay `pytest`-compatible. When changing config schema, update both config dataclasses and add parser coverage. For training-loop changes, include at least one smoke-level assertion around validation, checkpoints, or scheduling.

## Commit & Pull Request Guidelines
Recent commits use short, imperative subjects such as `validation set` and `big refactor + activity sampling`. Keep commit titles brief and specific. Pull requests should summarize the purpose, list the configs/scripts/modules touched, and include the exact test command run. Attach plots or screenshots only when training behavior or generated figures changed. Do not commit generated `outputs/`, checkpoints, or `data/` artifacts.
