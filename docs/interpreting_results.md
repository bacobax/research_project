# Interpreting the results: index and shared glossary

One guide per phase explains every result file in that phase. This page holds what is common to all of them.
Everything here was checked against `src/audio_infill/train.py`, the phase scripts and the committed result files.
When a guide quotes a number, it was read from a file in `phase_N/results/`; when two documents disagree, the guide
says which one to trust (generally: the generated CSV/JSON, not hand-written README prose).

| Phase | Question | Guide |
|---|---|---|
| 0 | Is the evaluation pipeline trustworthy, and does the model beat trivial fillers? | [phase_0/INTERPRETING_RESULTS.md](../phase_0/INTERPRETING_RESULTS.md) |
| 1 | Does more training data (1, 4, 17 songs) help? | [phase_1/INTERPRETING_RESULTS.md](../phase_1/INTERPRETING_RESULTS.md) |
| 2 | Does regularization (smaller model, more dropout/weight decay) help on one song? | [phase_2/INTERPRETING_RESULTS.md](../phase_2/INTERPRETING_RESULTS.md) |
| 3 | Is the data-vs-regularization contrast seed-robust? How does quality fall with gap length? | [phase_3/INTERPRETING_RESULTS.md](../phase_3/INTERPRETING_RESULTS.md) |
| 4 | Why does the model underfit (H1-H4), what produces the gradients, does the STFT loss matter? | [phase_4/INTERPRETING_RESULTS.md](../phase_4/INTERPRETING_RESULTS.md) |
| 5 | Is training consistent with how the model is used on a real gap (supervisor Q3)? | [phase_5/INTERPRETING_RESULTS.md](../phase_5/INTERPRETING_RESULTS.md) |
| 6 | One final run applying the fixes; no results exported yet | [phase_6/INTERPRETING_RESULTS.md](../phase_6/INTERPRETING_RESULTS.md) |

## The task and the data

- Audio is 24 kHz mono, tokenised by EnCodec at 6 kbps: **75 frames/s (13.33 ms, 320 samples per frame), 8 codebooks, vocabulary 1024**.
  A "gap" of L frames therefore hides 8 x L tokens.
- Key material ("pivot song"): `nuvole_bianche_multigap_4x_15ms_20ms_30ms_50ms`, 400.2 s = 30,017 frames, with 4 real carved gaps.
  Gap index 0..3 = 50, 20, 30, 15 ms of zeroed samples (`phase_5/results/exp01_context_contamination/real_gap_coverage.csv`),
  which round to a mask of 3, 1, 2, 1 frames (40, 13.3, 26.7, 13.3 ms) at frames [3164,3167), [4275,4276), [7427,7429), [22279,22280).
- Training masks are 1-4 frames (13-53 ms). Anything longer (Phase 3: 8 and 16 frames) is out of distribution.
- Chance level: NLL = ln(1024) = 6.93 nats, top-1 = 1/1024 = 0.098%.

## Validation set (identical in Phases 1-5 for a given run family)

`validation_strategy: holdout_regions`. From the pivot song only, two held-out regions of 1,536 frames (about 20.5 s each, 3,072 frames =
10.2% of the song) are removed from training: one drawn for the **high-activity** band and one for the **low-activity** band.
In each, 8 examples per gap length (1,2,3,4) are built (`validation_examples_per_length_band: 8`): 32 per band, 64 total, each a 512-frame window.
Consequences for interpretation:

- Validation never contains the 17-song extras, so Phase 1's "17 songs" are evaluated on one song's held-out passages only.
- n = 8 per (band, length) cell. Single cells are noisy; trust trends over many cells.
- The two regions are fixed, so "validation" is one piece of music; there is no confidence interval from resampling music.
- Activity band: `activity` = 0.65 x normalised frame RMS + 0.35 x normalised token-change rate, smoothed (kernel 9), clipped to [0,1].
  High band = mean activity over the gap above the 0.70 quantile; low band = below the 0.30 quantile. "High" is loud/busy material, "low" is quiet/sustained.
  Tokens in the high band are far less predictable (NLL about 6.4-6.8 vs 3.0-3.6).

## Metrics (all produced by `Trainer.run_validation` unless stated)

| Metric | Meaning | Unit / direction |
|---|---|---|
| `val/<band>_len_<L>_loss` = `_nll` | Mean cross-entropy over all masked frames x 8 codebooks (`_compute_loss`), averaged over the examples of that cell. `loss` and `nll` are identical numbers. | nats/token, lower is better; chance 6.93 |
| `_ppl` | exp(NLL), capped at exp(20) | lower is better |
| `_acc_top1` / `_acc_top5` | share of masked tokens whose argmax / top-5 contains the true token. Counts a frame-codebook token as one item. | 0-1, higher is better; chance 0.001 / 0.005 |
| `val/<band>_*` | count-weighted mean over the 4 gap lengths | |
| **`val/combined_loss`** | **mean of the high-band and low-band NLL** (equal weight per band). This is the checkpoint-selection key (`best_val.pt`). | lower is better |
| `val/si_sdr_db` | Scale-invariant SDR between the decoded filled tokens and the decoded **true tokens** (codec reference, so codec error is excluded), on the gap plus a **16-frame margin on each side**. | dB, higher is better. **Dominated by the identical margin tokens**: it reads 10-18 dB even when the gap is poor. |
| `val/si_sdr_db_gap_only` | same, on the gap frames only (margin 0). Much more sensitive to fill quality; negative values are common. | dB, higher is better |
| `val/spectral_convergence` / `_gap_only` | ||\|S_pred\|-\|S_true\|||_F / ||\|S_true\|||_F, averaged over STFT sizes 512/1024/2048 | lower is better, 0 = identical; about 0.6-0.9 gap-only is poor. Short crops are zero-padded, so gap-only values are noisy. |
| `val_clean/*` | (Phase 6 only) the same examples with clean-token context instead of zeroed-gap context | |
| `train/avg_*` | training aggregates over the last `log_every` (1000) steps. `train/avg_loss` includes the weighted decoded loss, `train/avg_nll` is the token term only. | |

Rules of thumb that hold across phases:

1. **NLL, accuracy and audio metrics can disagree** (Phase 5 aspect 8). Never rank checkpoints on one of them alone.
2. Prefer `*_gap_only` audio metrics; the margin versions mostly measure the unchanged context.
3. fp16 autocast during training-time validation can move NLL by up to 0.33 nats for some checkpoints (Phase 4), comparable to the data-scaling effect. Compare numbers produced under the same precision.
4. `best_val` is the minimum of noisy points, so metrics at that step are optimistically biased.
5. One seed per condition unless a section says otherwise; Phase 3 gives the only seed spread (see its guide).

## Run names

`baseline`, `bigmodel` (Phase 0 single-song runs, 4 layers d=256 and 6 layers d=384, manually stopped); `phase1_songs01/04/17`
(d=256, 4 layers, 11.84 M params; 100k/300k/1M steps); `phase2_regularization` (d=128, dropout 0.3, weight decay 0.1, 4.09 M params, 600k);
`phase3_*_seed43/44` (300k steps); `phase4_stft_off_seed43` (decoded loss disabled twin); `phase6_final_songs17_seed43`.

## Audio samples

Every `results/audio_samples/<run>/{latest,best_val}/gap{0..3}_step_<N>.wav` is a 6.61 s mono 24 kHz crop (496 frames) centred on real gap i, from
`scripts/export_gap_audio_samples.py` -> `Trainer.inpaint_all_gaps(save_crop_samples=True)`. The gap sits about 3.3 s into the file. The file is the codec decode of the **filled
token window**: context tokens come from encoding the gapped audio (so they carry EnCodec contamination, see Phase 5), the gap tokens are the model's argmax. There is no reference
file in the folder; compare against the same gap from another run or from `data/nuvole_bianche.mp3`. Gap 0 is the longest (50 ms), gap 3 the shortest (15 ms).
`latest` = last trained checkpoint, `best_val` = min `val/combined_loss`. For runs with `inpaint_output_mode: splice` (Phase 6) only the gap samples are replaced in the original waveform.
These are 4 examples per checkpoint: illustrative, not a statistic.
