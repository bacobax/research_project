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
`data/processed/`. These files are not tracked by Git; the next section
regenerates them.

## Reproduce From Scratch

Heavy artifacts (`data/`, `dataset/`, `outputs/`, checkpoints) are never
committed. Everything downloaded from the internet is fetched by one
idempotent, checksum-aware tool, `audio-infill-download`
(`src/audio_infill/download_data.py`), driven by `configs/data/download.yaml`
(YouTube video IDs, output filenames, EnCodec weight URL + sha256) and
`configs/data/dataset_manifest.json` (reference sha256/frames of each WAV).
System requirement: `ffmpeg` on `PATH` (`sudo apt install -y ffmpeg`).

```bash
uv sync --locked
uv run audio-infill-download --help
scripts/download_data.sh                       # 17 playlist WAVs -> dataset/, EnCodec weights -> torch hub cache
uv run audio-infill-download --verify          # offline checksum check of everything (exit 1 on problems)

# Pivot recording: not downloadable automatically (upstream URL was not recorded).
# Place the exact file at data/nuvole_bianche.mp3 (sha256 in configs/data/download.yaml; --verify checks it).

uv run audio-infill-dataset --config configs/data/nuvole_bianche_short_gaps.yaml   # -> data/processed/nuvole_bianche_short_gaps/
scripts/smoke_test.sh                          # fast pipeline check
uv run audio-infill-train --config configs/train/phase6_final_songs17_seed43.yaml  # final run
uv run pytest -q
```

Useful flags: `--only {playlist,weights}`, `--dry-run`, `--limit N`,
`--strict` (treat audio checksum mismatches as errors), `--update-manifest`
(re-pin checksums of files on disk).

Caveats on bit-exactness:

- The weights download is hard-verified by sha256. Playlist audio is re-encoded
  by YouTube/ffmpeg and videos can change, so a fresh download may differ from
  the manifest; mismatches are warnings by default. Playlist entry 008 already
  drifted upstream (it is excluded from training, so results are unaffected).
- Some legacy tests (`test_config_parsing`, `test_dataset_logic`,
  `test_decoded_loss`, `test_train_config_schema`, `test_validation_config`)
  need the old fixtures `data/processed/single_gap/` and
  `data/processed/multigap/` built from `data/raw/wav_test.wav`, whose source
  was never recorded; they fail on a clean checkout (8 failures, the rest pass).

## Seeds and Determinism

- Every train config has a `seed` (default 42; the multi-seed phase 3/6 runs use
  43/44). `set_seed` (`train.py`) seeds `random`, `numpy`, `torch` and all CUDA
  devices; the validation pool uses `seed + 1009`. Override with `--seed`.
- Dataset generation (`audio-infill-dataset`) draws gap positions from
  `random.Random(seed)`; the seed is in each `configs/data/*.yaml` (42).
- cuDNN deterministic mode is not forced, and AMP/fp16 plus nondeterministic CUDA
  kernels mean reruns on the same seed match closely but not bit-for-bit,
  and results can differ across GPU models and torch versions (the lockfile pins
  the latter). Phase 3 quantifies seed-to-seed spread.

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
