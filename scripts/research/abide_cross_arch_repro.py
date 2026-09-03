#!/usr/bin/env python3
"""
Sub-PTX B2 follow-on: cross-arch / re-emit reproducibility test for the
per-subject ORC matrices produced by abide_cohort_orc_sweep.py.

Modes:
  --mode local
      Re-emit the Sinkhorn-16 kernel from source, re-run the per-subject
      pipeline on a small subset on the local GPU, save outputs under
      artifacts/research/abide_orc_repro/<arch_tag>/<subject>_orc.npy,
      then diff bytewise (md5) against the canonical
      artifacts/research/abide_orc/<subject>_orc.npy from the cohort sweep.
      This validates "re-emit on the same GPU is bit-identical."

  --mode diff
      Skip the re-run; just diff two existing output trees.
      Reports per-subject md5, max |Δ|, NaN-mask equality.

Default subjects (one per representative site):
    CMU_a_0050642   NYU_0050959   Pitt_0050026   UCLA_1_0051214   Yale_0050628

Usage:
    # Reproduce on local GPU (~15 s for 5 subjects on RTX 4000 Ada).
    python3 scripts/research/abide_cross_arch_repro.py --mode local

    # Diff two existing trees (e.g. one local, one from slurm).
    python3 scripts/research/abide_cross_arch_repro.py --mode diff \\
        --baseline artifacts/research/abide_orc \\
        --candidate artifacts/research/abide_orc_repro/sm_89_remit
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from abide_cohort_orc_sweep import (read_subject_ts, compute_subject_orc)  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
ABIDE_DIR = ROOT / "artifacts" / "research" / "abide"
COHORT_DIR = ROOT / "artifacts" / "research" / "abide_orc"
REPRO_DIR = ROOT / "artifacts" / "research" / "abide_orc_repro"
N = 16
EDGE_SLOTS = 320
CHUNK = 256

DEFAULT_SUBJECTS = [
    "CMU_a_0050642",
    "NYU_0050959",
    "Pitt_0050026",
    "UCLA_1_0051214",
    "Yale_0050628",
]


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_arch_tag() -> str:
    """Compose a tag from the local GPU + driver."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
             "--format=csv,noheader"],
            text=True).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        name = parts[0].replace(" ", "_").replace("/", "_")
        cc = parts[1].replace(".", "")
        drv = parts[2]
        return f"{name}_sm{cc}_drv{drv}"
    except Exception:
        return "unknown_arch"


def emit_kernel(out_ptx: Path) -> str:
    """Re-emit the Sinkhorn-16 PTX from Sounio source; return md5 of the PTX."""
    print(f"emitting kernel -> {out_ptx}")
    t0 = time.time()
    subprocess.run(
        [str(ROOT / "bin" / "kretikos"),
         "kaxi-emit-ptx", "sinkhorn16", "--f32",
         "-o", str(out_ptx)],
        check=True)
    digest = md5_file(out_ptx)
    print(f"  emitted in {time.time() - t0:.1f}s, md5={digest}, size={out_ptx.stat().st_size} B")
    return digest


def build_runner(out_runner: Path) -> str:
    """Build the CUDA driver-API host runner; return md5."""
    print(f"building runner -> {out_runner}")
    src = ROOT / "scripts" / "gpu" / "kaxi_ptx_runner.c"
    subprocess.run(
        ["cc", "-O2", str(src), "-ldl", "-lm", "-o", str(out_runner)],
        check=True)
    digest = md5_file(out_runner)
    print(f"  built md5={digest}")
    return digest


def run_subject(subject: str, ptx: Path, runner: Path) -> tuple[np.ndarray, float]:
    """Run the per-subject pipeline using abide_cohort_orc_sweep's helpers."""
    path = ABIDE_DIR / f"{subject}_rois_cc200.1D"
    ts = read_subject_ts(path)
    return compute_subject_orc(ts, k=15, lam=1.0,
                               ptx=str(ptx), runner=str(runner))


def diff_trees(baseline_dir: Path, candidate_dir: Path,
               subjects: list[str]) -> int:
    """Report per-subject md5 + numeric diff. Returns failure count."""
    print(f"baseline:  {baseline_dir}")
    print(f"candidate: {candidate_dir}")
    print()
    fails = 0
    header = f"{'subject':<24s}  {'baseline md5':<34s}  {'candidate md5':<34s}  identical  max|Δ|  nan_eq"
    print(header)
    print("-" * len(header))
    for subj in subjects:
        b = baseline_dir / f"{subj}_orc.npy"
        c = candidate_dir / f"{subj}_orc.npy"
        if not b.is_file() or not c.is_file():
            print(f"{subj:<24s}  MISSING ({b.is_file()=}, {c.is_file()=})")
            fails += 1
            continue
        bd, cd = md5_file(b), md5_file(c)
        ba = np.load(b)
        ca = np.load(c)
        nan_eq = bool(np.array_equal(np.isnan(ba), np.isnan(ca)))
        finite = np.isfinite(ba) & np.isfinite(ca)
        max_diff = float(np.abs(ba[finite] - ca[finite]).max()) if finite.any() else float("nan")
        identical = (bd == cd)
        print(f"{subj:<24s}  {bd:<34s}  {cd:<34s}  "
              f"{'YES' if identical else 'NO ':<9s}  {max_diff:8.2e}  {nan_eq}")
        if not identical:
            fails += 1
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "diff"], default="local")
    ap.add_argument("--subjects", nargs="*", default=DEFAULT_SUBJECTS)
    ap.add_argument("--baseline", default=str(COHORT_DIR),
                    help="diff mode: baseline tree of <subject>_orc.npy")
    ap.add_argument("--candidate", default=None,
                    help="diff mode: candidate tree of <subject>_orc.npy")
    ap.add_argument("--arch-tag", default=None,
                    help="local mode: override the GPU/driver tag for output dir")
    ap.add_argument("--ptx", default=None,
                    help="local mode: PTX path (re-emitted if not supplied)")
    ap.add_argument("--runner", default=None,
                    help="local mode: runner path (re-built if not supplied)")
    args = ap.parse_args()

    if args.mode == "diff":
        if not args.candidate:
            sys.exit("--candidate is required in diff mode")
        fails = diff_trees(Path(args.baseline), Path(args.candidate),
                           args.subjects)
        print()
        if fails == 0:
            print(f"PASS: all {len(args.subjects)} subjects bit-identical")
            sys.exit(0)
        print(f"FAIL: {fails} subject(s) diverged")
        sys.exit(1)

    # local mode: re-emit kernel, re-run, diff against cohort baseline.
    arch_tag = args.arch_tag or detect_arch_tag()
    out_dir = REPRO_DIR / arch_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== cross-arch repro: local re-emit on {arch_tag} ==")
    print(f"out dir: {out_dir}")
    print()

    if args.ptx is None:
        ptx_path = out_dir / "sinkhorn16iter.ptx"
        ptx_md5 = emit_kernel(ptx_path)
    else:
        ptx_path = Path(args.ptx)
        ptx_md5 = md5_file(ptx_path)
        print(f"using pre-emitted PTX {ptx_path} md5={ptx_md5}")

    if args.runner is None:
        runner_path = out_dir / "kaxi_runner"
        runner_md5 = build_runner(runner_path)
    else:
        runner_path = Path(args.runner)
        runner_md5 = md5_file(runner_path)
        print(f"using pre-built runner {runner_path} md5={runner_md5}")
    print()

    print(f"running {len(args.subjects)} subjects...")
    for i, subj in enumerate(args.subjects, 1):
        t0 = time.time()
        orc, gpu_s = run_subject(subj, ptx_path, runner_path)
        out_npy = out_dir / f"{subj}_orc.npy"
        np.save(out_npy, orc)
        print(f"  [{i}/{len(args.subjects)}] {subj:<24s} {time.time()-t0:5.1f}s "
              f"(gpu {gpu_s:.1f}s)")
    print()

    # Diff against the cohort baseline.
    fails = diff_trees(COHORT_DIR, out_dir, args.subjects)
    print()

    # Provenance line.
    print(f"provenance:")
    print(f"  arch_tag:   {arch_tag}")
    print(f"  PTX md5:    {ptx_md5}")
    print(f"  runner md5: {runner_md5}")
    print(f"  out dir:    {out_dir}")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
