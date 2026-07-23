# Repository Setup with uv

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git.
An NVIDIA GPU is optional, but GPU training requires a working host NVIDIA
driver. Confirm host GPU access with `nvidia-smi` before troubleshooting
PyTorch.

## Clone and Initialize

```bash
git clone <repository-url>
cd research_project
uv sync --locked
```

`uv sync --locked` reads `pyproject.toml` and `uv.lock`, installs the Python
version selected by `.python-version` when necessary, and creates the
project-local `.venv`. Do not create or activate a Conda environment.

Use `uv run` for Python commands. It selects `.venv` and checks that its
packages match the lockfile:

```bash
uv run python --version
uv run pytest -q
uv run audio-infill-train --help
uv run audio-infill-dataset --help
uv run audio-infill-graph --help
```

Some tests and training configurations expect audio and annotation files under
`data/processed/`. These files are not tracked by Git and must be generated or
copied into place separately.

## Launch Scripts

The shell wrappers can be launched directly from any working directory. Each
wrapper changes to the repository root and uses the uv-managed environment:

```bash
scripts/smoke_test.sh
scripts/run_train.sh
scripts/run_multigap_train.sh
scripts/run_multigap_boundary_train.sh
scripts/resume.sh
```

Pass configuration overrides after the script name:

```bash
scripts/run_train.sh --device cpu --total-steps 1000
```

## Updating Dependencies

After changing dependencies in `pyproject.toml`, regenerate and apply the lock:

```bash
uv lock
uv sync
```

Commit both `pyproject.toml` and `uv.lock`. Never commit `.venv/`, generated
datasets, checkpoints, or files under `outputs/`.
