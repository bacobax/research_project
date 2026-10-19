# Supervisor guide: how to navigate this repository

This guide is for readers who want to find results, understand how a number was produced, or
re-run something, without reading the code first. Everything below was taken from the repository
itself (git history, READMEs, configs, scripts). Related documents:

- `SETUP.md`: installing uv, syncing the environment, launch scripts, data requirements.
- `docs/interpreting_results.md`: how to read the metrics, tables and figures.
- `docs/experiment_log.md`: one row per training run, in chronological order.
- `AGENTS.md`: repository conventions (layout, style, test expectations).

## 1. The project in three sentences

A Transformer inpaints short gaps (1-4 EnCodec frames, about 13-53 ms) in solo-piano audio by
predicting the missing RVQ codec tokens. The central finding of Phases 0-1 is that the model
underfits even as training data grows from 1 to 17 songs (training NLL rises from 1.32 to 4.51 nats).
Phases 2-6 then ask why (regularization, seeds, mechanism diagnostics, loss ablation, train/inference
mismatches) and end with one final run that fixes the mismatches found.

## 2. Repository map

| Path | What it holds |
|---|---|
| `src/audio_infill/` | All runtime code. `train.py` (model, trainer, validation, losses, inference), `config.py` (config dataclasses and YAML parsing), `make_gapped_dataset.py` (dataset generation, CLI `audio-infill-dataset`), `graph.py` (architecture diagram, CLI `audio-infill-graph`). |
| `configs/train/*.yaml` | One YAML per training run; every parameter is listed explicitly. Names tell the phase (`phase1_songs17.yaml`, `phase4_stft_off_seed43.yaml`, `phase6_final_songs17_seed43.yaml`). `smoke.yaml` is a 30-step pipeline check; `base.yaml`, `longrun.yaml`, `multigap*.yaml`, `resume_multigap.yaml` are legacy (pre-Phase 0, their input audio cannot be rebuilt). The two `nuvole_bianche_short_gaps_encoder_decoder*.yaml` files are the Phase 0 single-song runs. |
| `configs/data/*.yaml` | Dataset-generation templates (single gap, multigap, short gaps). |
| `scripts/` | Thin shell wrappers that call `uv run audio-infill-train --config ...`: `smoke_test.sh` (pipeline check), `run_train.sh` (final run), `resume.sh` (resume it); `run_multigap*_train.sh` are legacy. `check_replicability.sh` checks that every script runs after setup. Also `scaffold_phase.py` (creates a new phase skeleton), `export_gap_audio_samples.py` (curated audio examples), (playlist download now lives in `audio-infill-download`, `src/audio_infill/download_data.py`). |
| `phase_0/` ... `phase_6/` | One directory per research phase: analysis/export scripts, a `README.md`, and a `results/` folder with the portable outputs. See section 5. |
| `docs/experiment_log.md` | Table of every training run (changes made, hypothesis, key result, conclusion), plus notes on analysis-only phases. |
| `docs/RERUN_PLAN.md` | Audit of the STFT-loss `eps` bug that confounded some Phase 1-3 runs, the empirical evidence, and the rerun plan. Also records the answers to the supervisor's three questions (Q1 STFT impact, Q2 argmax decoding, Q3 gap-only vs gap+context loss). |
| `docs/archive/HANDOFF_2026-10-06.md` (archived, stale) | Session handoff note written mid-project (Phase 4 era); good for context, but `docs/experiment_log.md` and the phase READMEs are more current. |
| `notebooks/` | Exploratory notebooks (activity map, codec test, decoder reconstruction debugging, post-training validation inspection, same-song retrieval baselines). Not part of the reproducible pipeline. |
| `docs/figures/` | Architecture diagram (`network.png`, `network.dot`) and comparison figures. Result figures live in `phase_N/results/`, not here. |
| `tests/` | `unittest`-style tests, also run by pytest: config parsing and schema, dataset logic, validation, decoded loss, early stopping, codebook conditioning, zeroed context, splice inference, phase export and scaffolding. |
| `assets/prompts/` | Prompt assets. |
| `data/`, `dataset/`, `outputs/` | Local only, not committed (see section 4). |

## 3. How the logistics work

### Config-driven runs

Every training run is defined by one YAML in `configs/train/`. Shell scripts only choose a config;
the `audio-infill-train` console script (`audio_infill.train:main`) does the rest. CLI flags such as
`--total-steps` or `--device cuda:1` override the YAML for a single launch. The YAML also sets
`output_dir`, `run_name` and `seed`, so a run's location is determined by its config, for example
`outputs/runs/phase1_data_scaling/phase1_songs17` or `outputs/runs/phase6_final/phase6_final_songs17_seed43`.
New behaviours are added as new YAML keys whose defaults reproduce old behaviour, so older configs
keep working (the config schema is covered by `tests/test_train_config_schema.py` and
`tests/test_config_parsing.py`).

### The phase pipeline

```
configs/train/phaseN_*.yaml
        |  uv run audio-infill-train --config ...
        v
outputs/runs/<output_dir>/<run_name>/     (checkpoints, TensorBoard tb/, samples; NOT committed)
        |  phase_N/export_results.py, exp*.py, make_summary.py
        v
phase_N/results/                          (CSV, JSON, PNG+PDF, Markdown; committed)
```

- Phases 1 and 2 are training-then-export: `phase_1/export_results.py` and `phase_2/export_results.py`
  parse the TensorBoard logs of finished runs and write `metrics.csv`, `run_summary.csv`,
  `best_by_group.csv`, `tables/`, `figures/`, `configs/` (copies of the YAMLs used), `song_manifests/`,
  `provenance.json` (hashes of the raw inputs) and `source_state.patch` (code state at export).
- Phases 0, 3 (gap-length sweep), 4 (H1-H4, exp02) and 5 are evaluation-only: `exp*.py` scripts
  rebuild a trainer from a run's YAML plus one checkpoint (shared code in `phase_0/common.py`),
  evaluate on held-out examples, and write JSON/CSV; `make_summary.py` turns the JSON into
  `FINDINGS.md`. Raw scratch output is redirected away from `outputs/runs/**`, so original runs are
  never modified.
- Phase 3 (seeds) and Phase 4 exp03 are log-parsing comparisons of extra training runs (`export_seed_check.py`,
  `exp03_stft_ablation.py`; no GPU needed).
- Phase 6 trains the final run; `phase_6/monitor.py` reports its progress and
  `phase_6/preflight_zeroed_context.py` is the check behind its design.

### Where things live and what is not committed

`.gitignore` excludes `outputs/`, `data/`, `dataset/`, checkpoints (`*.pt`), TensorBoard event
files and `phase_N/results/**/*.wav` (except the small curated `results/audio_samples/**/*.wav`).
So a fresh clone contains code, configs, and all exported results, but cannot re-run evaluation
phases until checkpoints and audio data exist locally. Phase 1/2/3/4/5 evaluation scripts read
checkpoints from `outputs/runs/...`; these were produced on the project machine (GPU `cuda:1`).
Audio inputs are expected under `data/` (for example `data/nuvole_bianche.mp3`, `data/processed/`);
see `SETUP.md`.

### Setting up a fresh clone

`SETUP.md` ("Reproduce From Scratch") has the full sequence, checked on a fresh clone on
2026-10-19. Everything, including the pivot recording, is downloaded; a re-downloaded pivot
gives exactly the same gaps but near-identical (not byte-identical) audio, see `SETUP.md`.
In short, from the repository root:

```bash
uv sync --locked
scripts/download_data.sh                      # weights, pivot recording, 17 playlist WAVs (~3 min)
uv run audio-infill-download --verify         # checksums; DRIFT lines for entry 008 and the pivot are expected
uv run audio-infill-dataset --config configs/data/nuvole_bianche_short_gaps.yaml
scripts/smoke_test.sh                         # 30-step training check
scripts/check_replicability.sh                # ~2 min: every script runs on this checkout
```

`scripts/check_replicability.sh` (`tests/test_replicability.py`) verifies the checksums, that
the gapped dataset rebuilds bit-exactly, that every script imports or answers `--help`, and
that every offline report mode (`make_summary.py`, `--figures-only`, `--report-only`)
regenerates the committed text results byte-identically. `scripts/check_replicability.sh full`
also trains each launch wrapper for a few steps and runs the Phase 6 evaluation on those tiny
checkpoints (~10 min on a GPU). The phase experiment scripts themselves need the trained
`outputs/runs/` (retrain with the listed configs); the check confirms each one reads exactly
where its config writes.

### Reproducing a phase with uv

Run everything from the repository root, after the setup above:

```bash
uv run pytest -q                                                   # tests (9 known legacy-fixture failures)
uv run audio-infill-train --config configs/train/phase1_songs17.yaml --device cuda:0
uv run python phase_1/export_results.py                            # raw runs -> phase_1/results/
uv run python phase_0/exp01_train_region_control.py                # or: bash phase_0/run_all.sh
uv run python phase_3/export_seed_check.py
uv run python phase_3/exp01_gap_length_activity_sweep.py && uv run python phase_3/make_summary.py
uv run python phase_4/exp01_underfitting_diagnostics.py            # H1-H4
uv run python phase_4/exp02_gradient_spike_sources.py --device cuda:1
uv run python phase_4/exp03_stft_ablation.py && uv run python phase_4/make_summary.py
uv run python phase_5/exp01_context_contamination.py --device cuda:1
uv run python phase_5/exp02_gap_position.py --device cuda:1
uv run python phase_5/exp03_consistency_checks.py --device cuda:1
uv run audio-infill-train --config configs/train/phase6_final_songs17_seed43.yaml
uv run python phase_6/monitor.py
```

Each phase README lists the exact commands and approximate runtimes; check it before launching.
`phase_4` should only be run against checkpoints that are free of the STFT-eps bug (see
`docs/RERUN_PLAN.md`). To start a new phase, `uv run python scripts/scaffold_phase.py N` creates
`phase_N/results/` (with `figures`, `tables`, `configs`, `manifests`), the raw directory
`outputs/runs/phase_N/`, and a starter README.

### Naming conventions

- Runs: `phase<N>_<description>[_seed<S>]`, for example `phase3_regularization_seed43`.
- Configs: `configs/train/<run_name>.yaml`; the run name equals the YAML `run_name`.
- Experiment-log IDs: `<phase><letter>` (1A = first Phase 1 run, 4A = STFT-off twin, 6A = final run).
  `0A`/`0B` are the two single-song runs that predate the phase structure. A rerun keeps its letter.
- Scripts: `expNN_<topic>.py` for experiments, `make_summary.py` for JSON-to-Markdown, `export_*.py`
  for TensorBoard exports.
- Result files: `FINDINGS.md` (generated, do not edit by hand), `METHODS.md` (how it was measured),
  `results/README.md` (write-up). Figures are saved as matching `.png` and `.pdf` pairs.
- Checkpoints: `best_val.pt` (lowest validation loss), `best.pt` (lowest training loss), `latest.pt`,
  `final.pt`, periodic `step_*.pt`; Phase 6 adds `best_audio.pt` (selected on SI-SDR). Note that the
  Phase 0 `baseline` run's `latest.pt` has non-finite weights; always use `best_val`/`best`.

## 4. Which phase answers which question

| Phase | Question | Type | Answer in one line |
|---|---|---|---|
| 0 | Is the train/validation gap real or a measurement bug? Does the model beat trivial baselines? | evaluation only | Real gap (train-region control tracks train accuracy; holdout near random). Model beats token baselines only inconsistently. |
| 1 | Does more data (1, 4, 17 songs) close the gap? | training + export | Held-out metrics improve with data, but training NLL rises 1.32 to 4.51 nats, so underfitting is not just data starvation. |
| 2 | Can regularization substitute for data (same single song)? | training + export | Weaker than adding data. |
| 3 | Is the data-vs-regularization result seed-robust? How does quality degrade with gap length and activity? | extra seeds + evaluation | Holds for decoded audio SI-SDR, not clearly for token loss. Degradation visible in and beyond the trained gap range (8/16 frames are out of distribution). |
| 4 | Why does it underfit (H1 mode averaging, H2 RVQ hierarchy, H3 context use, H4 convergence)? What does the STFT loss do? | evaluation + one ablation run | H1 not supported; H2 consistent (accuracy falls with codebook index); H3 context is used; H4 mild overfitting; STFT term is 1.5-6.5% of the token gradient and ablating it has no detectable effect. |
| 5 | Supervisor Q3: is training consistent with inference? | evaluation only | Mostly yes; the real mismatch is how context is encoded at a real gap (zeroed gap changes later tokens), costing 2-4 dB SI-SDR. A pipeline issue, not a model issue. |
| 6 | Does a run that fixes all found mismatches do better? | final training run | In progress at time of writing (launched 2026-10-12, 300k steps, 17 songs). Compare against the `songs17` seed family. |

The `docs/RERUN_PLAN.md` audit (grad-norm spikes from the STFT log-eps floor) affected some Phase 1-3 runs;
the Phase B reruns showed that it did not change validation loss for the configs rerun.

## 5. Chronological narrative

1. **March 2026 (first commits to the multi-gap work):** initial training code, refactor with activity sampling,
   validation set, decoded (STFT/waveform) loss, boundary conditioning, encoder-decoder architecture.
2. **July 2026:** migration to uv.
3. **September 2026, Phase 0:** measurement validation on the two single-song runs.
4. **September 2026, Phases 1-3:** multi-song training with early stopping and audio validation metrics;
   data scaling (Phase 1), regularization ablation (Phase 2), seed robustness and gap-length sweep (Phase 3),
   plus curated audio samples for the paper.
5. **Early October 2026, audit:** supervisor questions led to the discovery of the STFT-eps confound;
   `docs/RERUN_PLAN.md` and `docs/archive/HANDOFF_2026-10-06.md` (archived, stale) written; affected runs rerun (2026-10-07 to 10-10) and Phases 1-3
   re-exported; `docs/experiment_log.md` started.
6. **10-11 October 2026, Phase 4:** H1-H4 diagnostics, gradient attribution (exp02), methods report.
7. **12 October 2026, Phase 4 exp03 and Phase 5:** STFT-loss ablation write-up and the train/inference
   consistency experiments (Q3).
8. **12 October 2026, Phase 6:** final 17-song run with zeroed-gap context, depth-autoregressive codebook heads,
   STFT loss off, fp32 validation, and splice inference. Training was launched the same day.

## 6. Quick start: "I want to see X, go to Y"

| I want to see | Go to |
|---|---|
| The overall story and the key numbers per run | `docs/experiment_log.md` |
| Phase 0 evidence that the measurements are valid | `phase_0/results/FINDINGS.md`, `phase_0/README.md` |
| The data-scaling table and curves | `phase_1/results/README.md`, `phase_1/results/figures/`, `phase_1/results/tables/` |
| Regularization vs. data | `phase_2/results/`, `phase_3/results/seed_check/README.md` |
| Gap-length and activity sweep | `phase_3/results/FINDINGS.md`, `phase_3/results/figures/` |
| Why the model underfits (H1-H4) | `phase_4/results/FINDINGS.md`, `phase_4/results/figures/`, how-it-was-done in `phase_4/METHODS.md` |
| STFT-loss impact (Q1) | `phase_4/results/exp03_stft_ablation/` (`stft_ablation.png`, `comparison.csv`), `phase_4/results/exp02/` |
| Train vs. inference consistency (Q3) | `phase_5/README.md` (aspect table), `phase_5/METHODS.md`, `phase_5/results/*/figures/` |
| Argmax decoding and gap-only loss audit (Q2, Q3 code audit) | `docs/RERUN_PLAN.md` section 6 |
| The STFT-eps bug and the reruns | `docs/RERUN_PLAN.md`, `docs/experiment_log.md` (rerun rows) |
| The final run's design and status | `phase_6/README.md`, `configs/train/phase6_final_songs17_seed43.yaml`, `phase_6/monitor.py` |
| Listenable examples | `phase_N/results/audio_samples/` (small curated WAVs; phases 1-4) |
| How to interpret metrics and figures | `docs/interpreting_results.md` |
| Model architecture | `docs/figures/network.png`; `src/audio_infill/train.py` |
| Which parameters a run used | `configs/train/<run_name>.yaml` (also copied to `phase_N/results/configs/` for Phases 1-2) |
| How a result was exported and from which raw files | `phase_N/results/provenance.json` (Phases 1-2) and the exporting script |
| Environment, data and launch instructions | `SETUP.md` |
| That the code still works | `uv run pytest -q`, `scripts/smoke_test.sh`, `scripts/check_replicability.sh` |
| Prototyping code and one-off investigations | `notebooks/` |

Note: Phases 1 and 2 have their narrative in `phase_N/results/README.md` rather than a top-level `phase_N/README.md`.
