# Prompt for the remote-machine session: generate demo audio samples

You are working in the audio-infill research repo (read `AGENTS.md` first; use `uv` only; never glob or commit `outputs/`, checkpoints or `data/`). Checkpoints are in `outputs/runs/<run_dir>/checkpoints/<tag>.pt` on this machine. Check `nvidia-smi` and `df -h /` before running. Write the code, run it, and report what was produced.

## Goal
Produce demonstration audio samples for the project supervisor: for each chosen gap, several short WAVs (same window, different fills) so the supervisor can listen and compare models.

## Models to include (one WAV each per gap)
1. `original`: the true audio.
2. `zeroed`: the input with the gap silenced (the "do nothing" reference).
3. `phase6_final`: run `phase6_final_songs17_seed43`, tag `final` (300k). Optionally also tag `best_audio` (210k) as `phase6_best_audio`.
4. `songs17_s43` and `songs17_s44`: runs `phase3_songs17_seed43` and `phase3_songs17_seed44`, tag `final` (300k).
5. `songs17_1M`: run `phase1_songs17`, tag `best_val` (step 950k).
6. `single_song`: run `phase1_songs01`, tag `final`.

Do not include early best_val checkpoints (Phase 6 step 30k, songs17 s43 step 20k).

## Which gaps
- The 4 real gaps (15/20/30/50 ms), as in `scripts/export_gap_audio_samples.py`. Extend its `RUN_TABLE` (or write a sibling script) so the Phase 6 run is supported, with `inpaint_output_mode: splice`.
- 6-8 held-out gaps from the 128-gap set used by `phase_6/eval_final.py` (reuse its gap sampling and `Trainer` reconstruction via `phase_0/common.py::build_trainer`; do not reimplement). Choose two per length band, picked by gap-only SI-SDR averaged over all models: best, median and worst. This keeps the demo honest, not cherry-picked.

## How to fill
Use the repo's own inference path (`Trainer.inpaint_all_gaps` for real gaps; the `eval_final.py` condition C protocol for held-out gaps: whole song re-encoded with a real-like zeroed span, fp32, greedy codebook decoding). For every model use splice output (`splice_gap_audio`, 2.5 ms raised-cosine cross-fade), so only the gap's samples differ from the original.

## Output format
Each WAV is 24 kHz mono, a crop of gap +/- 1 s (about 2 s), loudness-matched across versions of the same gap. Layout:
```
demo_samples/
  README.md        # how to listen, what each file is, caveats
  index.csv        # gap_id, kind (real|heldout), gap_ms, model, run, checkpoint, step, gap_si_sdr_db, region_si_sdr_db
  gap_<id>/original.wav zeroed.wav phase6_final.wav songs17_s43.wav songs17_s44.wav songs17_1M.wav single_song.wav
  figures/gap_<id>.png/.pdf   # waveform + spectrogram of original, zeroed, phase6_final
```
Put code in `scripts/make_demo_samples.py` (docstring, `set -euo pipefail` for any shell wrapper, small diff). Save the output under `phase_6/results/demo_samples/` (small WAV/PNG/PDF/CSV only).

## README caveats to state
Gaps are 13-53 ms. Under the common protocol no 300k-step model beats the zeroed input on region SI-SDR, and Phase 6 final is 5.6 +/- 1.8 dB worse than the songs17 seeds on gap-only SI-SDR (`phase_6/results/eval/FINDINGS.md`). The samples illustrate this, they do not hide it.

## When done
Run `uv run pytest -q` only if you changed shared code; report the exact command used to generate the samples, the list of files, and any checkpoint that was missing. Do not commit `outputs/`, checkpoints or `data/`.
