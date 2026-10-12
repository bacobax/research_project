"""Pre-flight for training on zeroed-gap context (Phase 6).

(1) Token agreement between a windowed zeroed encoding (warm-up W frames + window) and the
    whole-song zeroed encoding (what inference does), as a function of W.
(2) Cost of re-encoding a batch of 16 windows on the GPU.

    uv run python phase_6/preflight_zeroed_context.py --device cuda:1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from audio_infill.train import AudioEncoder, reallike_span  # noqa: E402

SONG = REPO / "dataset"
HOP, SEQ = 320, 512


def load(path: Path) -> torch.Tensor:
    """Read a wav as mono float, require 24 kHz and peak-normalise; returns a [1, T] tensor."""
    a, sr = sf.read(str(path), always_2d=True)
    a = a.astype(np.float32).mean(axis=1)
    assert sr == 24000, f"{path} sr={sr}"
    w = torch.from_numpy(a).unsqueeze(0)
    return w / (w.abs().max() + 1e-9)


def main() -> None:
    """CLI: measure (1) token agreement between windowed (warm-up + window) and whole-song zeroed encodings versus warm-up length and (2) the GPU cost of re-encoding a batch of windows."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--songs", type=int, default=3)
    ap.add_argument("--windows", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device(args.device)
    enc = AudioEncoder(6.0, dev)
    rng = np.random.default_rng(0)
    warmups = [0, 75, 150, 300, 750, 1500]
    paths = sorted(SONG.glob("*.wav"))[: args.songs]
    agree = {w: {"before": [], "after": []} for w in warmups}
    clean_after, clean_before = [], []
    for path in paths:
        wav = load(path)
        n_frames = wav.shape[1] // HOP
        for _ in range(args.windows):
            s = int(rng.integers(400, n_frames - SEQ - 1))
            L = int(rng.integers(1, 5))
            m0 = int(rng.integers(100, SEQ - 100))
            g0, g1 = s + m0, s + m0 + L
            zs, ze = reallike_span(g0, g1, HOP, rng)
            full = wav.clone()
            full[:, zs:ze] = 0.0
            ref = enc.model.encode(full.unsqueeze(0).to(dev))[0][0][0, :, s:s + SEQ].cpu()
            clean = enc.model.encode(wav.unsqueeze(0).to(dev))[0][0][0, :, s:s + SEQ].cpu()
            ceq = (clean == ref).float().mean(0)
            clean_before.append(float(ceq[:m0].mean())); clean_after.append(float(ceq[m0 + L:].mean()))
            for w in warmups:
                w0 = max(0, s - w)
                seg = wav[:, w0 * HOP:(s + SEQ) * HOP].clone()
                seg[:, zs - w0 * HOP:ze - w0 * HOP] = 0.0
                got = enc.model.encode(seg.unsqueeze(0).to(dev))[0][0][0, :, s - w0:s - w0 + SEQ].cpu()
                eq = (got == ref).float().mean(0)  # per frame
                agree[w]["before"].append(float(eq[:m0].mean()))
                agree[w]["after"].append(float(eq[m0 + L:].mean()))
    out = {"clean_tokens_vs_whole_song_zeroed": {"before": float(np.mean(clean_before)), "after": float(np.mean(clean_after))},
           "songs": [p.name for p in paths], "windows_per_song": args.windows,
           "agreement_with_whole_song_zeroed_encoding": {
               str(w): {k: float(np.mean(v)) for k, v in d.items()} for w, d in agree.items()}}

    # (2) cost of one batch of 16
    wav = load(paths[0])
    for w in (150, 750, 1500):
        segs = torch.stack([wav[:, (1600 + 37 * i - w) * HOP:(1600 + 37 * i + SEQ) * HOP] for i in range(16)]).to(dev)
        enc.model.encode(segs)
        torch.cuda.synchronize(dev)
        t = time.time()
        for _ in range(5):
            enc.model.encode(segs)
        torch.cuda.synchronize(dev)
        out[f"encode_batch16_seconds_W{w}"] = (time.time() - t) / 5
    (REPO / "phase_6/results/preflight_zeroed_context.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
