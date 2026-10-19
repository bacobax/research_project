"""Replicability suite: does every script run on a fresh clone right after the SETUP.md steps?

Every runnable file under ``scripts/`` and ``phase_*/`` is registered in ``SCRIPTS`` with what
"runs after setup" means for it, and the suite checks exactly that. Levels, selected with the
``AUDIO_INFILL_REPRO`` environment variable (``scripts/check_replicability.sh`` sets it):

* unset (plain ``uv run pytest -q``): static checks only, a few seconds, no data needed.
  Registry complete, shell scripts strict, Python compiles, every hard-coded run path matches a
  training config, launch wrappers default to configs whose inputs SETUP.md can rebuild.
* ``setup``: needs the SETUP.md steps done (download, pivot mp3, dataset); a few minutes.
  Checksums verify, the dataset rebuilds bit-exactly, every research config parses against the
  setup data, every script imports / answers ``--help``, standalone scripts run, and every
  offline report mode regenerates the tracked text results byte-identically (in a scratch copy
  of the checkout, so the working tree is never touched).
* ``full``: ``setup`` plus a few-step training run through each launch wrapper (smoke, train,
  resume) and the checkpoint-consuming Phase 6 scripts on tiny checkpoints; ~10 min on a GPU.

Nothing is written to ``outputs/runs/`` or ``data/``: training goes to a temp dir and offline
scripts run in a scratch copy. Override the device with ``AUDIO_INFILL_TEST_DEVICE`` (default: the
GPU with most free memory, else cpu) and the temp location with ``TMPDIR``.
"""

import hashlib
import json
import math
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
LEVEL = os.environ.get("AUDIO_INFILL_REPRO", "").strip().lower()
AT_SETUP = LEVEL in ("setup", "full")
AT_FULL = LEVEL == "full"
needs_setup = unittest.skipUnless(AT_SETUP, "set AUDIO_INFILL_REPRO=setup (or run scripts/check_replicability.sh)")
needs_full = unittest.skipUnless(AT_FULL, "set AUDIO_INFILL_REPRO=full (or run scripts/check_replicability.sh full)")

# --------------------------------------------------------------------------------------------
# Script registry
# --------------------------------------------------------------------------------------------
LIBRARY = "library"        # imported by other scripts; import-checked only
STANDALONE = "standalone"  # runs on the SETUP.md outputs alone (executed at level setup)
OFFLINE = "offline"        # has a mode regenerating tracked results from tracked inputs (level setup)
TRAIN = "train"            # launches training (executed for a few steps at level full)
NEEDS_RUNS = "needs_runs"  # evaluates trained checkpoints / TensorBoard logs under outputs/runs
LEGACY = "legacy"          # needs pre-Phase-0 inputs SETUP.md cannot rebuild; must fail fast and clearly
RUNNER = "runner"          # runs this suite


@dataclass(frozen=True)
class Script:
    """How one runnable file is expected to behave after setup."""
    kind: str
    has_help: bool = False                      # `--help` exits 0 without doing any work
    offline_args: Optional[Tuple[str, ...]] = None  # args of the offline regeneration mode (OFFLINE only)


SCRIPTS: Dict[str, Script] = {
    "scripts/check_replicability.sh": Script(RUNNER),
    "scripts/download_data.sh": Script(STANDALONE, has_help=True),
    "scripts/package_submission_samples.sh": Script(STANDALONE),
    "scripts/smoke_test.sh": Script(TRAIN, has_help=True),
    "scripts/run_train.sh": Script(TRAIN, has_help=True),
    "scripts/resume.sh": Script(TRAIN),
    "scripts/run_multigap_train.sh": Script(LEGACY),
    "scripts/run_multigap_boundary_train.sh": Script(LEGACY),
    "scripts/export_gap_audio_samples.py": Script(NEEDS_RUNS, has_help=True),
    "scripts/make_demo_samples.py": Script(OFFLINE, has_help=True, offline_args=("--scores-figure-only",)),
    "scripts/scaffold_phase.py": Script(STANDALONE, has_help=True),
    "phase_0/common.py": Script(LIBRARY),
    "phase_0/exp01_train_region_control.py": Script(NEEDS_RUNS),
    "phase_0/exp02_baselines.py": Script(NEEDS_RUNS),
    "phase_0/exp03_decoded_metrics.py": Script(NEEDS_RUNS),
    "phase_0/exp04_real_gaps.py": Script(NEEDS_RUNS),
    "phase_0/make_summary.py": Script(OFFLINE, offline_args=()),
    "phase_0/run_all.sh": Script(NEEDS_RUNS),
    "phase_1/export_results.py": Script(NEEDS_RUNS, has_help=True),
    "phase_2/export_results.py": Script(NEEDS_RUNS, has_help=True),
    "phase_3/exp01_gap_length_activity_sweep.py": Script(NEEDS_RUNS, has_help=True),
    "phase_3/export_seed_check.py": Script(NEEDS_RUNS, has_help=True),
    "phase_3/make_summary.py": Script(OFFLINE, offline_args=()),
    "phase_4/exp01_underfitting_diagnostics.py": Script(NEEDS_RUNS, has_help=True),
    "phase_4/exp02_gradient_spike_sources.py": Script(NEEDS_RUNS, has_help=True),
    "phase_4/exp03_stft_ablation.py": Script(NEEDS_RUNS, has_help=True),
    "phase_4/make_summary.py": Script(OFFLINE, offline_args=()),
    "phase_5/plot_style.py": Script(LIBRARY),
    "phase_5/exp01_context_contamination.py": Script(OFFLINE, has_help=True, offline_args=("--figures-only",)),
    "phase_5/exp02_gap_position.py": Script(NEEDS_RUNS, has_help=True),
    "phase_5/exp03_consistency_checks.py": Script(OFFLINE, has_help=True, offline_args=("--figures-only",)),
    "phase_6/eval_final.py": Script(OFFLINE, has_help=True, offline_args=("--report-only",)),
    "phase_6/monitor.py": Script(NEEDS_RUNS, has_help=True),
    "phase_6/preflight_zeroed_context.py": Script(STANDALONE, has_help=True),
    "phase_6/phase5_protocol/p6_protocol.py": Script(LIBRARY),
    "phase_6/phase5_protocol/run_exp01.py": Script(OFFLINE, has_help=True, offline_args=("--figures-only",)),
    "phase_6/phase5_protocol/run_exp02.py": Script(NEEDS_RUNS, has_help=True),
    "phase_6/phase5_protocol/run_exp03.py": Script(OFFLINE, has_help=True, offline_args=("--figures-only",)),
}

# Train configs whose inputs (data/processed/{single_gap,multigap}, data/interim/gapped_audio.wav,
# built from data/raw/wav_test.wav) predate Phase 0 and cannot be rebuilt from SETUP.md.
LEGACY_CONFIGS = {
    "base.yaml", "longrun.yaml", "multigap.yaml", "multigap_decoded_loss.yaml",
    "multigap_decoded_loss_val_v2.yaml", "multigap_decoded_loss_val_v3_boundary.yaml",
    "multigap_encoder_decoder_decoded_loss_val_v1_boundary.yaml",
    "multigap_encoder_decoder_decoded_loss_val_v1_boundary_lighter.yaml", "resume_multigap.yaml",
}
# Data configs whose output the SETUP.md "Reproduce From Scratch" steps build.
SETUP_DATA_CONFIGS = ("configs/data/nuvole_bianche_short_gaps.yaml",)
# sha256 of the gapped WAV built by configs/data/nuvole_bianche_short_gaps.yaml from the pinned
# data/nuvole_bianche.mp3 (identical on the original machine and a fresh clone, 2026-10-19).
GAPPED_WAV = "nuvole_bianche_multigap_4x_15ms_20ms_30ms_50ms.wav"
GAPPED_WAV_SHA256 = "c0cb6487bc08d8f85a5ac9c4858f9f4230834df0739db120cb1086206a4fb6a1"
# Hard-coded `outputs/runs/...` strings that are not a run directory or an output root.
NON_RUN_OUTPUT_PATHS = {"outputs/runs/phase6_final/train.log", "outputs/runs/phase_N"}
TEXT_RESULT_SUFFIXES = {".md", ".csv", ".json", ".txt"}


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------
def discover_scripts() -> List[str]:
    """Every runnable file under scripts/ and phase_*/ (repo-relative, excluding package __init__)."""
    found = []
    for pattern in ("scripts/*.py", "scripts/*.sh", "phase_*/**/*.py", "phase_*/**/*.sh"):
        for p in REPO.glob(pattern):
            if p.name != "__init__.py" and "__pycache__" not in p.parts:
                found.append(p.relative_to(REPO).as_posix())
    return sorted(set(found))


def load_yaml(path: Path) -> dict:
    """Read one YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def train_configs() -> Dict[str, dict]:
    """configs/train/*.yaml by file name."""
    return {p.name: load_yaml(p) for p in sorted((REPO / "configs/train").glob("*.yaml"))}


def wrapper_default_config(script: str) -> Optional[str]:
    """The CONFIG_PATH default of a launch wrapper (None if it has none)."""
    m = re.search(r'CONFIG_PATH="\$\{CONFIG_PATH:-([^}]+)\}"', (REPO / script).read_text())
    return m.group(1) if m else None


def sha256(path: Path) -> str:
    """Hex sha256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def child_env(**extra: str) -> Dict[str, str]:
    """Environment for subprocesses: no foreign VIRTUAL_ENV, unbuffered, headless matplotlib."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    env.update(extra)
    return env


def run(cmd: Sequence[str], cwd: Path = REPO, timeout: float = 900, **env: str) -> subprocess.CompletedProcess:
    """Run a command, capturing combined output as text."""
    return subprocess.run(list(map(str, cmd)), cwd=cwd, env=child_env(**env), stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, timeout=timeout)


def tail(proc: subprocess.CompletedProcess, n: int = 25) -> str:
    """Last lines of a process's output, for assertion messages."""
    return "\n".join(proc.stdout.splitlines()[-n:])


def pick_device() -> str:
    """AUDIO_INFILL_TEST_DEVICE, else the GPU with most free memory (via nvidia-smi), else cpu."""
    if os.environ.get("AUDIO_INFILL_TEST_DEVICE"):
        return os.environ["AUDIO_INFILL_TEST_DEVICE"]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        rows = [tuple(int(x) for x in line.split(",")) for line in out.strip().splitlines()]
        if rows:
            return f"cuda:{max(rows, key=lambda r: r[1])[0]}"
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return "cpu"


def make_temp_dir(prefix: str) -> Path:
    """A fresh temp directory (honours TMPDIR)."""
    return Path(tempfile.mkdtemp(prefix=f"audio_infill_{prefix}_"))


def make_scratch_copy() -> Path:
    """Copy the checkout's tracked files (working-tree versions) plus the setup data into a temp dir.

    Scripts there resolve REPO from their own path, so they write into the copy; `dataset/` is
    symlinked (read-only use), the small `data/` inputs are copied.
    """
    root = make_temp_dir("scratch") / "repo"
    files = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout
    for rel in filter(None, files.decode().split("\0")):
        src = REPO / rel
        if src.is_file():
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    (root / "dataset").symlink_to(REPO / "dataset")
    (root / "data").mkdir(exist_ok=True)
    shutil.copy2(REPO / "data/nuvole_bianche.mp3", root / "data/nuvole_bianche.mp3")
    for cfg in SETUP_DATA_CONFIGS:
        outdir = load_yaml(REPO / cfg)["outdir"]
        shutil.copytree(REPO / outdir, root / outdir)
    return root


def tb_scalars(tb_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    """All TensorBoard scalars under tb_dir as {tag: [(step, value)]}."""
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util import tensor_util

    out: Dict[str, List[Tuple[int, float]]] = {}
    for f in sorted(tb_dir.rglob("events.out.tfevents.*")):
        for ev in EventFileLoader(str(f)).Load():
            for v in ev.summary.value:
                if v.HasField("simple_value"):
                    val = float(v.simple_value)
                elif v.HasField("tensor"):
                    arr = tensor_util.make_ndarray(v.tensor)
                    if arr.size != 1 or arr.dtype.kind not in "biuf":  # text / hparams summaries
                        continue
                    val = float(arr.reshape(()))
                else:
                    continue
                out.setdefault(v.tag, []).append((ev.step, val))
    return out


def load_run_table():
    """Import phase_0/common.py (its RUNS table maps run names to configs and checkpoint paths)."""
    sys.path.insert(0, str(REPO / "phase_0"))
    try:
        import common
    finally:
        sys.path.pop(0)
    return common


def checkpoint_step(path: Path) -> int:
    """Training step stored in a checkpoint."""
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=False)["step"])


def tiny_train_args(out_dir: Path, run_name: str, device: str, steps: int) -> List[str]:
    """CLI overrides that shrink a training config to a few steps written only under out_dir."""
    return ["--device", device, "--total-steps", str(steps), "--save-every", str(steps),
            "--validation-every", "2", "--log-every", "1", "--test-fill-every", "0",
            "--output-dir", str(out_dir), "--run-name", run_name,
            "--inpaint-output", str(out_dir / f"{run_name}_infilled.wav")]


# --------------------------------------------------------------------------------------------
# Level 0: static checks (always run)
# --------------------------------------------------------------------------------------------
class TestScriptInventory(unittest.TestCase):
    """Static checks that need neither data nor GPU."""

    def test_every_script_is_registered(self):
        found = set(discover_scripts())
        self.assertEqual(sorted(found - SCRIPTS.keys()), [],
                         "new script(s): register them in SCRIPTS in tests/test_replicability.py")
        self.assertEqual(sorted(SCRIPTS.keys() - found), [], "SCRIPTS lists files that no longer exist")

    def test_registry_entries_are_consistent(self):
        for path, s in SCRIPTS.items():
            with self.subTest(path=path):
                self.assertEqual(s.offline_args is not None, s.kind == OFFLINE)

    def test_shell_scripts_are_strict_executable_and_parse(self):
        for path in (p for p in SCRIPTS if p.endswith(".sh")):
            with self.subTest(path=path):
                text = (REPO / path).read_text()
                self.assertTrue(text.startswith("#!/usr/bin/env bash"), "missing bash shebang")
                self.assertIn("set -euo pipefail", text)
                self.assertTrue(os.access(REPO / path, os.X_OK), "not executable (chmod +x)")
                proc = subprocess.run(["bash", "-n", str(REPO / path)], capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_python_scripts_compile(self):
        with tempfile.TemporaryDirectory() as d:
            for path in (p for p in SCRIPTS if p.endswith(".py")):
                with self.subTest(path=path):
                    py_compile.compile(str(REPO / path), cfile=str(Path(d) / "x.pyc"), doraise=True)

    def test_train_configs_are_legacy_or_built_by_setup(self):
        """Non-legacy configs only read data the SETUP.md steps produce."""
        setup_outdirs = {load_yaml(REPO / c)["outdir"] for c in SETUP_DATA_CONFIGS}
        download_dir = load_yaml(REPO / "configs/data/download.yaml")["output_dir"]
        configs = train_configs()
        self.assertEqual(sorted(LEGACY_CONFIGS - configs.keys()), [], "LEGACY_CONFIGS lists missing files")
        for name, cfg in configs.items():
            if name in LEGACY_CONFIGS:
                continue
            with self.subTest(config=name):
                self.assertIn(cfg["ds_dir"], setup_outdirs)
                self.assertEqual(Path(cfg["wav_path"]).parent.as_posix(), cfg["ds_dir"])
                if cfg.get("extra_wavs_dir"):
                    self.assertEqual(cfg["extra_wavs_dir"], download_dir)
                self.assertFalse(cfg.get("resume"), "a committed config must not depend on a local checkpoint")

    def test_setup_md_builds_every_setup_data_config(self):
        setup_md = (REPO / "SETUP.md").read_text()
        for cfg in SETUP_DATA_CONFIGS:
            self.assertIn(f"audio-infill-dataset --config {cfg}", setup_md)

    def test_launch_wrappers_default_to_rebuildable_configs(self):
        for path, s in SCRIPTS.items():
            if s.kind not in (TRAIN, LEGACY):
                continue
            with self.subTest(path=path):
                cfg = wrapper_default_config(path)
                self.assertIsNotNone(cfg, "wrapper has no CONFIG_PATH default")
                self.assertTrue((REPO / cfg).is_file(), f"{cfg} does not exist")
                self.assertEqual(Path(cfg).name in LEGACY_CONFIGS, s.kind == LEGACY,
                                 f"{path} defaults to {cfg}; legacy configs only belong to LEGACY wrappers")

    def test_hard_coded_run_paths_match_training_configs(self):
        """Every `outputs/runs/...` path a script reads is where some training config writes."""
        run_dirs, roots = set(), set()
        for cfg in train_configs().values():
            run_dirs.add((Path(cfg["output_dir"]) / cfg["run_name"]).as_posix())
            roots.add(Path(cfg["output_dir"]).as_posix())
        pat = re.compile(r"outputs/runs/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)?")
        for path in SCRIPTS:
            for hit in pat.findall((REPO / path).read_text()):
                with self.subTest(script=path, path=hit):
                    self.assertTrue(hit in run_dirs or hit in roots or hit in NON_RUN_OUTPUT_PATHS,
                                    f"{hit} is produced by no configs/train/*.yaml (output_dir/run_name)")

    def test_checkpoint_table_matches_training_configs(self):
        """phase_0/common.py RUNS: config exists and its checkpoints land at the paths scripts load."""
        for run_name, run_info in load_run_table().RUNS.items():
            with self.subTest(run=run_name):
                cfg = load_yaml(run_info["config"])
                ckpt_dir = REPO / cfg["output_dir"] / cfg["run_name"] / "checkpoints"
                for tag, path in run_info["checkpoints"].items():
                    self.assertEqual(Path(path), ckpt_dir / f"{tag}.pt")


# --------------------------------------------------------------------------------------------
# Level setup: the SETUP.md outputs exist and every script starts
# --------------------------------------------------------------------------------------------
@needs_setup
class TestSetupArtifacts(unittest.TestCase):
    """The data, weights and environment the SETUP.md steps produce."""

    def test_download_verify_passes(self):
        proc = run([PY, "-m", "audio_infill.download_data", "--verify"], timeout=600)
        self.assertEqual(proc.returncode, 0, tail(proc, 40))

    def test_setup_dataset_matches_pinned_checksum(self):
        outdir = load_yaml(REPO / SETUP_DATA_CONFIGS[0])["outdir"]
        self.assertEqual(sha256(REPO / outdir / GAPPED_WAV), GAPPED_WAV_SHA256)

    def test_dataset_rebuild_is_bit_exact(self):
        with tempfile.TemporaryDirectory() as d:
            proc = run([PY, "-m", "audio_infill.make_gapped_dataset", "--config", SETUP_DATA_CONFIGS[0], "--outdir", d])
            self.assertEqual(proc.returncode, 0, tail(proc))
            self.assertEqual(sha256(Path(d) / GAPPED_WAV), GAPPED_WAV_SHA256)
            ann = json.loads((Path(d) / GAPPED_WAV.replace(".wav", ".json")).read_text())
            ref = json.loads((REPO / load_yaml(REPO / SETUP_DATA_CONFIGS[0])["outdir"]
                              / GAPPED_WAV.replace(".wav", ".json")).read_text())
            drop = {"source_wav", "generated_wav"}  # absolute paths
            self.assertEqual({k: v for k, v in ann.items() if k not in drop},
                             {k: v for k, v in ref.items() if k not in drop})

    def test_research_configs_parse_against_setup_data(self):
        """Both config parsers accept every non-legacy config with the setup data on disk."""
        from audio_infill.config import parse_args as parse_shared
        from audio_infill.train import parse_args as parse_runtime

        for name, cfg in train_configs().items():
            if name in LEGACY_CONFIGS:
                continue
            with self.subTest(config=name):
                argv = ["--config", f"configs/train/{name}"]
                cwd = os.getcwd()
                os.chdir(REPO)
                try:
                    parse_shared(argv)
                    parse_runtime(argv)
                finally:
                    os.chdir(cwd)
                if cfg.get("extra_wavs_dir"):
                    wavs = list((REPO / cfg["extra_wavs_dir"]).glob("*.wav"))
                    self.assertGreaterEqual(len(wavs), int(cfg.get("extra_wavs_limit") or 1))

    def test_cli_entry_points_answer_help(self):
        for cli in ("audio-infill-train", "audio-infill-dataset", "audio-infill-graph", "audio-infill-download"):
            with self.subTest(cli=cli):
                proc = run(["uv", "run", "--project", REPO, cli, "--help"], timeout=300)
                self.assertEqual(proc.returncode, 0, tail(proc))
                self.assertIn("usage", proc.stdout)


@needs_setup
class TestEveryScriptStarts(unittest.TestCase):
    """Each Python script imports (module level, no main) and each `has_help` script answers --help."""

    def test_python_scripts_import_or_answer_help(self):
        def check(path: str) -> Tuple[str, subprocess.CompletedProcess]:
            if SCRIPTS[path].has_help:
                cmd = [PY, path, "--help"] if path.endswith(".py") else ["bash", path, "--help"]
            else:
                cmd = [PY, "-c", "import runpy, sys; sys.argv=[sys.argv[1]]; runpy.run_path(sys.argv[0], run_name='__replicability__')", path]
            return path, run(cmd, timeout=600)

        targets = [p for p, s in SCRIPTS.items() if s.kind != RUNNER and (p.endswith(".py") or s.has_help)]
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            results = list(pool.map(check, targets))
        for path, proc in results:
            with self.subTest(path=path):
                self.assertEqual(proc.returncode, 0, tail(proc))
                if SCRIPTS[path].has_help:
                    self.assertIn("usage", proc.stdout.lower())

    def test_legacy_wrappers_fail_fast_with_a_clear_message(self):
        for path in (p for p, s in SCRIPTS.items() if s.kind == LEGACY):
            cfg = load_yaml(REPO / wrapper_default_config(path))
            if (REPO / cfg["wav_path"]).is_file() or (REPO / cfg["ds_dir"]).is_dir():
                continue  # legacy fixtures present on this machine: nothing to check
            with self.subTest(path=path):
                t0 = time.time()
                proc = run([REPO / path, "--device", "cpu"], timeout=600)
                self.assertNotEqual(proc.returncode, 0)
                self.assertRegex(proc.stdout, r"not found")
                self.assertNotIn("Traceback", proc.stdout, "should be a clean CLI error, not a crash")
                self.assertLess(time.time() - t0, 300)


@needs_setup
class TestStandaloneScripts(unittest.TestCase):
    """Scripts that need nothing beyond the SETUP.md outputs run to completion."""

    def test_download_dry_run(self):
        proc = run([REPO / "scripts/download_data.sh", "--dry-run"], timeout=600)
        self.assertEqual(proc.returncode, 0, tail(proc))

    def test_scaffold_phase(self):
        with tempfile.TemporaryDirectory() as d:
            proc = run([PY, "scripts/scaffold_phase.py", "99", "--repo-root", d])
            self.assertEqual(proc.returncode, 0, tail(proc))
            self.assertTrue((Path(d) / "phase_99/results").is_dir())
            self.assertTrue((Path(d) / "outputs/runs/phase_99").is_dir())

    def test_preflight_zeroed_context(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "preflight.json"
            proc = run([PY, "phase_6/preflight_zeroed_context.py", "--device", pick_device(),
                        "--songs", "1", "--windows", "1", "--out", out], timeout=1200)
            self.assertEqual(proc.returncode, 0, tail(proc))
            res = json.loads(out.read_text())
            agree = res["agreement_with_whole_song_zeroed_encoding"]
            self.assertTrue(all(0.0 <= v["after"] <= 1.0 for v in agree.values()))
            self.assertIn("encode_batch16_seconds_W150", res)

    def test_package_submission_samples(self):
        scratch = make_scratch_copy()
        try:
            proc = run(["bash", scratch / "scripts/package_submission_samples.sh", "pkg_test"], cwd=scratch)
            self.assertEqual(proc.returncode, 0, tail(proc))
            self.assertTrue((scratch / "pkg_test.zip").is_file())
            self.assertGreater(len(list((scratch / "pkg_test").rglob("*.wav"))), 0)
        finally:
            shutil.rmtree(scratch.parent, ignore_errors=True)


@needs_setup
class TestOfflineRegeneration(unittest.TestCase):
    """Every OFFLINE mode rebuilds tracked results from tracked inputs, reproducing the text exactly."""

    @classmethod
    def setUpClass(cls):
        cls.scratch = make_scratch_copy()
        cls.t0 = time.time() - 1
        cls.procs = {}
        for path, s in SCRIPTS.items():
            if s.kind == OFFLINE:
                cls.procs[path] = run([PY, cls.scratch / path, *s.offline_args], cwd=cls.scratch, timeout=1800)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.scratch.parent, ignore_errors=True)

    def test_offline_modes_succeed(self):
        for path, proc in self.procs.items():
            with self.subTest(path=path):
                self.assertEqual(proc.returncode, 0, tail(proc))

    def test_tracked_text_results_are_reproduced(self):
        changed, created = [], []
        for p in self.scratch.glob("phase_*/**/results/**/*"):
            if not p.is_file() or p.suffix not in TEXT_RESULT_SUFFIXES or p.stat().st_mtime < self.t0:
                continue
            rel = p.relative_to(self.scratch)
            if not (REPO / rel).is_file():
                created.append(rel.as_posix())
            elif p.read_bytes() != (REPO / rel).read_bytes():
                changed.append(rel.as_posix())
        self.assertEqual(changed, [], "regenerated results differ from the tracked ones")
        self.assertEqual(created, [], "offline modes wrote untracked result files")

    def test_regenerated_figures_come_as_png_and_pdf(self):
        rewritten = [p for p in self.scratch.glob("phase_*/**/figures/**/*")
                     if p.suffix in (".png", ".pdf") and p.stat().st_mtime >= self.t0]
        self.assertGreater(len(rewritten), 0)
        for p in rewritten:
            with self.subTest(figure=p.relative_to(self.scratch).as_posix()):
                self.assertGreater(p.stat().st_size, 0)
                if p.suffix == ".png":  # every PNG chart has its PDF twin (some figures are PDF-only)
                    self.assertTrue(p.with_suffix(".pdf").is_file(), f"missing {p.with_suffix('.pdf').name}")


# --------------------------------------------------------------------------------------------
# Level full: training wrappers and checkpoint-consuming scripts on tiny runs
# --------------------------------------------------------------------------------------------
@needs_full
class TestTrainingAndEvaluation(unittest.TestCase):
    """Launch wrappers train for a few steps; Phase 6 evaluation runs on the resulting checkpoints."""

    P6 = "phase6_final_songs17_seed43"
    BASELINES = ("phase3_songs17_seed43", "phase3_songs17_seed44")

    @classmethod
    def setUpClass(cls):
        cls.device = pick_device()
        cls.tmp = make_temp_dir("train")
        cls.procs: Dict[str, subprocess.CompletedProcess] = {}
        out = cls.tmp / "runs"
        cls.procs["smoke"] = run([REPO / "scripts/smoke_test.sh", "--device", cls.device, "--output-dir", out,
                                  "--inpaint-output", out / "smoke_infilled.wav"], timeout=3600)
        cls.procs["train"] = run([REPO / "scripts/run_train.sh", *tiny_train_args(out, cls.P6, cls.device, 2)], timeout=3600)
        (out / "train.log").write_text(cls.procs["train"].stdout)
        cls.procs["resume"] = run([REPO / "scripts/resume.sh", *tiny_train_args(out, cls.P6, cls.device, 3)], timeout=3600,
                                  RESUME_CKPT=str(out / cls.P6 / "checkpoints/latest.pt"))
        for name in cls.BASELINES:
            cls.procs[name] = run([PY, "-m", "audio_infill.train", "--config", f"configs/train/{name}.yaml",
                                   *tiny_train_args(out, name, cls.device, 2)], timeout=3600)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def assert_ok(self, key: str):
        """The named setUpClass process succeeded."""
        self.assertEqual(self.procs[key].returncode, 0, tail(self.procs[key], 40))

    def test_smoke_test_trains_and_validates(self):
        self.assert_ok("smoke")
        cfg = load_yaml(REPO / wrapper_default_config("scripts/smoke_test.sh"))
        run_dir = self.tmp / "runs" / cfg["run_name"]
        self.assertEqual(checkpoint_step(run_dir / "checkpoints/final.pt"), cfg["total_steps"])
        val = tb_scalars(run_dir / "tb").get("val/combined_loss", [])
        self.assertGreater(len(val), 0, "no val/combined_loss logged")
        self.assertTrue(all(math.isfinite(v) for _, v in val))
        self.assertTrue((self.tmp / "runs/smoke_infilled.wav").is_file())

    def test_run_train_then_resume_continues_the_run(self):
        self.assert_ok("train")
        self.assert_ok("resume")
        ckpts = self.tmp / "runs" / self.P6 / "checkpoints"
        self.assertEqual(checkpoint_step(ckpts / "step_2.pt"), 2)
        self.assertEqual(checkpoint_step(ckpts / "final.pt"), 3)

    def test_resume_without_checkpoint_fails_clearly(self):
        proc = run([REPO / "scripts/resume.sh"], RESUME_CKPT=str(self.tmp / "missing.pt"))
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("checkpoint not found", proc.stdout)

    def _scratch_with_runs(self) -> Path:
        """Scratch copy whose outputs/runs holds the tiny runs at the paths the scripts load."""
        c = load_run_table()
        scratch = make_scratch_copy()
        for name in (self.P6, *self.BASELINES):
            src = self.tmp / "runs" / name
            cfg = load_yaml(c.RUNS[name]["config"])
            dst = scratch / cfg["output_dir"] / cfg["run_name"]
            dst.mkdir(parents=True)
            (dst / "tb").symlink_to(src / "tb")
            (dst / "checkpoints").mkdir()
            for tag in c.RUNS[name]["checkpoints"]:
                have = src / "checkpoints" / f"{tag}.pt"
                (dst / "checkpoints" / f"{tag}.pt").symlink_to(have if have.is_file() else src / "checkpoints/final.pt")
        shutil.copy2(self.tmp / "runs/train.log", scratch / "outputs/runs/phase6_final/train.log")
        return scratch

    def test_phase6_scripts_run_on_tiny_checkpoints(self):
        for key in ("train", "resume", *self.BASELINES):
            self.assert_ok(key)
        scratch = self._scratch_with_runs()
        try:
            proc = run([PY, scratch / "phase_6/monitor.py"], cwd=scratch)
            self.assertEqual(proc.returncode, 0, tail(proc))
            self.assertIn("step 3/", proc.stdout)
            proc = run([PY, scratch / "phase_6/eval_final.py", "--smoke", "--device", self.device], cwd=scratch, timeout=3600)
            self.assertEqual(proc.returncode, 0, tail(proc, 40))
            out = scratch / "phase_6/results/eval_smoke"
            self.assertTrue(any(out.glob("*.csv")), "eval_final --smoke wrote no CSV")
            self.assertTrue(any(out.glob("figures/*.png")) and any(out.glob("figures/*.pdf")))
        finally:
            shutil.rmtree(scratch.parent, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
