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
(YouTube video IDs, output filenames, the pivot recording, EnCodec weight URL + sha256) and
`configs/data/dataset_manifest.json` (reference sha256/frames of each WAV).
System requirement: `ffmpeg` on `PATH` (`sudo apt install -y ffmpeg`).

```bash
uv sync --locked
uv run audio-infill-download --help
scripts/download_data.sh                       # EnCodec weights -> torch hub cache, pivot -> data/nuvole_bianche.mp3,
                                               # 17 playlist WAVs -> dataset/ (~3 min)
uv run audio-infill-download --verify          # offline checksum check of everything (exit 1 on problems)
uv run audio-infill-dataset --config configs/data/nuvole_bianche_short_gaps.yaml   # -> data/processed/nuvole_bianche_short_gaps/
scripts/smoke_test.sh                          # fast pipeline check (30 steps, writes outputs/runs_test/)
scripts/check_replicability.sh                 # every script runs on this checkout (see below)
uv run audio-infill-train --config configs/train/phase6_final_songs17_seed43.yaml  # final run (= scripts/run_train.sh)
uv run pytest -q
```

`scripts/check_replicability.sh` runs `tests/test_replicability.py`, which
registers every script under `scripts/` and `phase_*/` and checks what
"runnable after setup" means for each: checksums verify, the gapped dataset
rebuilds bit-exactly, every research config parses against the setup data,
every script imports or answers `--help`, standalone scripts run, and every
offline report mode (`make_summary.py`, `--figures-only`, `--report-only`)
regenerates the tracked text results byte-identically (in a scratch copy, so
the working tree is untouched). `scripts/check_replicability.sh full` also
trains each launch wrapper for a few steps and runs the Phase 6 evaluation on
those tiny checkpoints (~10 min on a GPU, ~5 GB in `$TMPDIR`). Scripts that
evaluate the trained runs (`phase_0`-`phase_5` experiments, export scripts)
need the corresponding `outputs/runs/` from training with the listed configs;
the suite checks that every run path they read is exactly where its config
trains to.

Useful flags: `--only {playlist,weights}`, `--dry-run`, `--limit N`,
`--strict` (treat audio checksum mismatches as errors), `--update-manifest`
(re-pin checksums of files on disk).

Caveats on bit-exactness:

- The pivot recording (`data/nuvole_bianche.mp3`, Nuvole Bianche piano cover by
  Rousseau, YouTube `4VR-6AS0-l4`) is downloaded from the same m4a stream the
  reference copy was made from and re-encoded to a 192 kb/s mp3 the same way.
  The new encode is not byte-identical (`--verify` reports `DRIFT`), but it
  decodes to exactly the same length, so the gaps are identical (checked by
  `tests/test_replicability.py`); the audio is near-identical (corr 0.99999,
  gapped WAV 48 dB SNR, 93.7% identical EnCodec tokens; 2026-10-19). Retraining
  should land within the seed spread of Phase 3; re-scoring the original
  checkpoints exactly as reported needs the reference copy (sha256 in
  `configs/data/download.yaml`): put it at `data/nuvole_bianche.mp3` before
  running the download, and it is kept.
- The weights download is hard-verified by sha256. Playlist audio is re-encoded
  by YouTube/ffmpeg and videos can change, so a fresh download may differ from
  the manifest; mismatches are warnings by default. Playlist entry 008 already
  drifted upstream (it is excluded from training, so results are unaffected);
  `--verify` reports it as `DRIFT` and still passes (`--strict` makes it fail).
  On a fresh clone (2026-10-19) the other 16 WAVs and the weights matched
  their checksums bit-for-bit, and with the reference pivot so did the gapped
  dataset.
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
scripts/smoke_test.sh        # configs/train/smoke.yaml (30 steps, 2 songs)
scripts/run_train.sh         # configs/train/phase6_final_songs17_seed43.yaml (final run)
scripts/resume.sh            # final run from outputs/runs/phase6_final/.../latest.pt
RESUME_CKPT=path/to/x.pt scripts/resume.sh
```

Pass configuration overrides after the script name (`CONFIG_PATH=...` picks
another config):

```bash
scripts/run_train.sh --device cpu --total-steps 1000
```

`scripts/run_multigap_train.sh` and `scripts/run_multigap_boundary_train.sh`
are legacy (pre-Phase 0), like the configs `base.yaml`, `longrun.yaml`,
`multigap*.yaml` and `resume_multigap.yaml`: they need `data/processed/multigap/`
or `single_gap/` and `data/interim/gapped_audio.wav`, whose source was never
recorded, so on a fresh clone they stop with an "input ... not found" error.

## Updating Dependencies

After changing dependencies in `pyproject.toml`, regenerate and apply the lock:

```bash
uv lock
uv sync
```

Commit both `pyproject.toml` and `uv.lock`. Never commit `.venv/`, generated
datasets, checkpoints, or files under `outputs/`.
