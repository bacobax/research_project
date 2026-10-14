#!/usr/bin/env python3
"""Phase 5 experiment 02_gap_position on the Phase 6 model (and the matched songs17 baselines).

Runs the unchanged `phase_5/exp02_gap_position.py` with its checkpoints, example set and output folder
swapped (see `p6_protocol.py`); results go to `phase_6/phase5_protocol/results/exp02_gap_position/`.

    uv run python phase_6/phase5_protocol/run_exp02.py --device cuda:1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p6_protocol as p  # noqa: E402

p.install()
import exp02_gap_position as m  # noqa: E402

m.OUT = p.RESULTS / "exp02_gap_position"
m.FIG = m.OUT / "figures"
m.DEFAULT_RUNS = p.RUNS
m.token_metrics = p.token_metrics_exp02
if __name__ == "__main__":
    m.main()
