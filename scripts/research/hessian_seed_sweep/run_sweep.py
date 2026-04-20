#!/usr/bin/env python3
"""
P1.3 seed-robustness sweep for examples/seizure_hessian_ossm.sio.

For each SEED_OFFSET in SEED_LIST:
  1. write a copy of the source with `var SEED_OFFSET: i64 = <seed>`
  2. compile+run via `./bin/souc run <copy>`
  3. capture full stdout under <out-dir>/hess_seed<seed>.stdout

Intended to be invoked either locally or inside a Slurm job; all paths
are relative to the repo root unless absolutized by flags.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import time


SEED_LINE_RE = re.compile(r"^var SEED_OFFSET: i64 = \d+$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="examples/seizure_hessian_ossm.sio")
    p.add_argument("--souc", default="./bin/souc")
    p.add_argument("--out-dir", default="artifacts/research/hessian_seed_sweep")
    p.add_argument(
        "--seeds",
        default="0,1000,2000,3000,4000,5000,6000,7000,8000,9000,"
        "10000,11000,12000,13000,14000,15000,16000,17000,18000,19000",
        help="Comma-separated list of SEED_OFFSET values to sweep.",
    )
    p.add_argument("--tmpdir", default="/tmp/hess_seed_sweep")
    return p.parse_args()


def render_seeded(src: str, seed: int) -> str:
    replacement = f"var SEED_OFFSET: i64 = {seed}"
    new_src, n = SEED_LINE_RE.subn(replacement, src, count=1)
    if n != 1:
        raise RuntimeError(
            "Could not locate the `var SEED_OFFSET: i64 = <n>` line; "
            "was the source file edited in an unexpected way?"
        )
    return new_src


def main() -> int:
    args = parse_args()
    repo = pathlib.Path.cwd()
    src_path = repo / args.source
    out_dir = repo / args.out_dir
    tmp_dir = pathlib.Path(args.tmpdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src = src_path.read_text()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"[seed-sweep] source={src_path} out={out_dir} n_seeds={len(seeds)}")

    t_total_start = time.time()
    for seed in seeds:
        seeded_src = render_seeded(src, seed)
        seeded_path = tmp_dir / f"seizure_hessian_ossm_seed{seed:05d}.sio"
        seeded_path.write_text(seeded_src)
        out_path = out_dir / f"hess_seed{seed:05d}.stdout"
        err_path = out_dir / f"hess_seed{seed:05d}.stderr"
        t0 = time.time()
        with out_path.open("w") as out_f, err_path.open("w") as err_f:
            rc = subprocess.call(
                [args.souc, "run", str(seeded_path)],
                stdout=out_f,
                stderr=err_f,
                cwd=str(repo),
            )
        dt = time.time() - t0
        print(f"[seed-sweep] seed={seed:5d}  rc={rc}  t={dt:5.2f}s  -> {out_path.name}")
        if rc != 0:
            print(f"  WARN: non-zero exit; see {err_path}")

    dt_total = time.time() - t_total_start
    print(f"[seed-sweep] total elapsed {dt_total:.1f}s across {len(seeds)} seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
