"""Glue that lets the unchanged Phase 5 experiments run on the Phase 6 model.

Phase 5 scripts call `common.build_trainer(run, "best_val")` and score single-song checkpoints with
independent codebook heads. This module patches that, in-process only (no Phase 5 file is
rewritten):

* a "run" may be written `<run>@<checkpoint tag>` (e.g. `phase6_final_songs17_seed43@final`); the
  trainer of each base run is built once and the requested weights are swapped in;
* every run is scored on ONE shared held-out example set (the Phase 6 validation construction,
  16 gaps per band x length 1-4 = 128), with the holdout ranges asserted equal across runs;
* `depth_ar` heads: tokens come from the greedy pass (what inference does), token NLL from the
  teacher-forced pass (exact joint likelihood), as in `phase_6/eval_final.py`. For independent heads
  both passes are the same call, so the numbers equal the Phase 5 definition.

Usage: `import p6_protocol as p; p.install()` before importing a phase_5 experiment module.
"""
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "phase_0"))
sys.path.insert(0, str(REPO_ROOT / "phase_5"))
import common as c  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
P6 = "phase6_final_songs17_seed43"
S43, S44 = "phase3_songs17_seed43", "phase3_songs17_seed44"
# Same layout as Phase 5 (4 checkpoints): Phase 6 final + best_val (the Phase 5 convention) against the
# matched songs17 baselines (same data, seed family and 300k-step budget).
RUNS = (f"{P6}@final", f"{P6}@best_val", f"{S43}@final", f"{S44}@final")
MARGIN = 16

_orig_build_trainer = c.build_trainer
_orig_build_examples = c.build_real_validation_examples
_trainers: Dict[str, object] = {}
_shared: Dict[str, object] = {}


def build_trainer(run_tag: str, checkpoint_tag: str = "best_val", device: str = None):
    """Return the base run's trainer (built once) with the weights of `<run>@<tag>` loaded; the tag in the name wins over `checkpoint_tag`."""
    run, _, tag = run_tag.partition("@")
    tag = tag or checkpoint_tag
    if run not in _trainers:
        _trainers[run] = _orig_build_trainer(run, tag, device)
        # zeroed-context runs make the dataset return an extra `meta` item; the Phase 5 scripts unpack 3 items
        for ds in getattr(_trainers[run].dataset, "datasets", [_trainers[run].dataset]):
            ds.return_meta = False
    tr = _trainers[run]
    ck = torch.load(c.RUNS[run]["checkpoints"][tag], map_location=tr.device, weights_only=False)
    tr.model.load_state_dict(ck["model"])
    tr.model.eval()
    tr.global_step = int(ck["step"])
    return tr


def build_real_validation_examples(trainer, *args, **kwargs):
    """Phase 6's example set for every trainer; asserts each trainer's own holdout ranges equal Phase 6's."""
    if "p6" not in _shared:
        _shared["p6"] = _orig_build_examples(_trainers[P6] if P6 in _trainers else trainer, *args, **kwargs)
    grouped, holdout = _shared["p6"]
    own = id(trainer)
    if own not in _shared:
        _, h = _orig_build_examples(trainer, *args, **kwargs)
        assert [tuple(r) for r in h] == [tuple(r) for r in holdout], "holdout ranges differ between runs"
        _shared[own] = True
    return grouped, holdout


def install() -> None:
    """Patch phase_0/common so the Phase 5 modules use the helpers above."""
    c.build_trainer = build_trainer
    c.build_real_validation_examples = build_real_validation_examples


@torch.no_grad()
def predict_fill(trainer, x_win: torch.Tensor, mask: torch.Tensor, y_win: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32, eval, batch of 1. Returns (greedy tokens [K,T], log-probs [K,T,V] for the NLL: teacher-forced for depth_ar)."""
    dev = trainer.device
    xb = x_win.clone().long()
    xb[:, mask] = trainer.mask_token
    xb = xb.unsqueeze(0).to(dev)
    seg, left, right = trainer._build_model_boundary_tensors(mask.unsqueeze(0).to(dev))
    greedy = trainer.model(xb, seg, left, right)[0]
    tf = greedy
    if trainer.cfg.codebook_conditioning == "depth_ar":
        tf = trainer.model(xb, seg, left, right, gap_targets=y_win.long().unsqueeze(0).to(dev))[0]
    return greedy.argmax(-1).cpu(), F.log_softmax(tf.float(), dim=-1).cpu()


def score_exp01(trainer, enc_win, clean_win, mask, eval_frames, crop, region, fill=True, out_win=None) -> Dict[str, float]:
    """Replacement for phase_5 exp01 `score` (same arguments and metrics) that uses `predict_fill`."""
    out: Dict[str, float] = {}
    filled = (enc_win if out_win is None else out_win).clone()
    if fill:
        pred, logp = predict_fill(trainer, enc_win, mask, clean_win)
        filled[:, mask] = pred[:, mask]
        tgt = clean_win[:, eval_frames]
        out["nll"] = float(-logp[:, eval_frames, :].gather(-1, tgt.unsqueeze(-1)).mean())
        out["acc_top1"] = float((pred[:, eval_frames] == tgt).float().mean())
    lo, hi = crop
    hop = trainer.cfg.target_sr // trainer.encoder.frame_rate
    pred_audio = trainer._decode_validation_window_audio(filled[:, lo:hi])
    ref_audio = trainer._decode_validation_window_audio(clean_win[:, lo:hi])
    n = min(len(pred_audio), len(ref_audio))
    pred_audio, ref_audio = pred_audio[:n], ref_audio[:n]
    r0, r1 = (region[0] - lo) * hop, (region[1] - lo) * hop
    out["si_sdr_crop"] = c.si_sdr_db(pred_audio, ref_audio)
    out["si_sdr_gap"] = c.si_sdr_db(pred_audio[r0:r1], ref_audio[r0:r1])
    out["spectral_convergence_crop"] = trainer._spectral_convergence(pred_audio, ref_audio)
    out["snr_gap"] = c.snr_db(pred_audio[r0:r1], ref_audio[r0:r1])
    out["log_mel_l1_crop"] = c.log_mel_l1(pred_audio, ref_audio, trainer.cfg.target_sr)
    return out


def token_metrics_exp02(trainer, codes: torch.Tensor, w0: int, w1: int, g0: int, g1: int) -> Dict[str, float]:
    """Replacement for phase_5 exp02 `token_metrics`: NLL (teacher-forced for depth_ar) and greedy top-1 of gap [g0, g1) in window [w0, w1)."""
    x = codes[:, w0:w1].clone().long()
    mask = torch.zeros(w1 - w0, dtype=torch.bool)
    mask[g0 - w0:g1 - w0] = True
    tgt = x[:, mask].clone()
    pred, logp = predict_fill(trainer, x, mask, x)
    return {"nll": float(-logp[:, mask, :].gather(-1, tgt.unsqueeze(-1)).mean()),
            "acc_top1": float((pred[:, mask] == tgt).float().mean())}
