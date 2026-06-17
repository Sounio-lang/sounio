#!/usr/bin/env python3
"""Emit a disk-safe Slurm launcher for the chi6 colour-guided beam campaign.

The generated script targets the Darwin `gpu-orangefs` path and defaults to the
RTX 4000 Ada lane (`constraint=rtx4000ada`).  It is only a search launcher:
GPU/worker output remains untrusted until exact geometry plus checked SAT/LRAT
artifacts are replayed by the proof side.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shlex
import sys
from pathlib import Path


CAMPAIGN = "examples/erdos/chi6_colour_guided_beam_campaign.py"
SAFE_SLURM_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
SAFE_JOB_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_slurm_token(raw: str, name: str) -> str:
    if not SAFE_SLURM_TOKEN_RE.fullmatch(raw):
        raise ValueError(f"{name} has unsafe characters: {raw!r}")
    return raw


def safe_job_name(raw: str) -> str:
    if not SAFE_JOB_NAME_RE.fullmatch(raw):
        raise ValueError(f"--job-name has unsafe characters: {raw!r}")
    if len(raw) > 128:
        raise ValueError("--job-name must be <= 128 characters")
    return raw


def safe_csv_ints(raw: str, name: str, *, allow_zero: bool = False) -> str:
    if not re.fullmatch(r"[0-9]+(,[0-9]+)*", raw):
        raise ValueError(f"{name} must be a comma-separated integer list")
    for token in raw.split(","):
        value = int(token)
        if value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"{name} values must be {'non-negative' if allow_zero else 'positive'}")
    return raw


def positive_int(raw: int, name: str) -> int:
    if raw <= 0:
        raise ValueError(f"{name} must be positive")
    return raw


def nonnegative_int(raw: int, name: str) -> int:
    if raw < 0:
        raise ValueError(f"{name} must be non-negative")
    return raw


def orangefs_path(raw: str, name: str) -> str:
    if not raw.startswith("/orangefs/"):
        raise ValueError(f"{name} must be under /orangefs to avoid node-local scratch")
    return raw.rstrip("/")


def absolute_path_or_env(raw: str, name: str) -> str:
    if raw.startswith("$"):
        if not re.fullmatch(r"\$[A-Za-z_][A-Za-z0-9_]*", raw):
            raise ValueError(f"{name} env reference is unsafe: {raw!r}")
        return raw
    if not raw.startswith("/"):
        raise ValueError(f"{name} must be absolute or an env reference like $CHI6_COORDS_CSV")
    return raw


def q(value: str | int | Path) -> str:
    return shlex.quote(str(value))


def add_input_args(args: argparse.Namespace, campaign_cmd: list[str]) -> None:
    if args.satfanout_json:
        campaign_cmd.extend(["--satfanout-json", args.satfanout_json])
    else:
        campaign_cmd.extend(["--coords-csv", args.coords_csv])
        campaign_cmd.extend(["--colourings-file", args.colourings_file])


def render_campaign_command(args: argparse.Namespace) -> str:
    campaign_cmd = [
        "python3",
        CAMPAIGN,
        "$RUN_DIR/campaign",
        "--candidate-prefix",
        args.candidate_prefix,
        "--generations-list",
        args.generations_list,
        "--beam-width-list",
        args.beam_width_list,
        "--branch-width-list",
        args.branch_width_list,
        "--mutation-max-den-list",
        args.mutation_max_den_list,
        "--mutation-top-points-list",
        args.mutation_top_points_list,
        "--dsatur-node-limit-list",
        args.dsatur_node_limit_list,
        "--mutation-max-candidates",
        str(args.mutation_max_candidates),
        "--mutation-emit-mutations",
        str(args.mutation_emit_mutations),
        "--mutation-add-points",
        str(args.mutation_add_points),
        "--mutation-min-neighbor-count",
        str(args.mutation_min_neighbor_count),
        "--mutation-edge-gain-pool-points",
        str(args.mutation_edge_gain_pool_points),
        "--mutation-edge-gain-max-combinations",
        str(args.mutation_edge_gain_max_combinations),
        "--mutation-edge-gain-combination-offset",
        str(args.mutation_edge_gain_combination_offset),
        "--mutation-edge-gain-combination-stride",
        str(args.mutation_edge_gain_combination_stride),
        "--mutation-edge-gain-emit-mutations",
        str(args.mutation_edge_gain_emit_mutations),
        "--max-vertices",
        str(args.max_vertices),
        "--min-vertices",
        str(args.min_vertices),
        "--min-edges",
        str(args.min_edges),
        "--split-depth",
        str(args.split_depth),
        "--min-split-degree",
        str(args.min_split_degree),
        "--max-cubes",
        str(args.max_cubes),
        "--sample-hard-cubes",
        str(args.sample_hard_cubes),
        "--refute-limit",
        str(args.refute_limit),
        "--refute-timeout-seconds",
        str(args.refute_timeout_seconds),
        "--max-carried-colourings",
        str(args.max_carried_colourings),
        "--shard-index",
        "$SHARD_INDEX",
        "--shard-count",
        str(args.array_count),
        "--cell-budget",
        str(args.cell_budget),
        "--resume",
    ]
    add_input_args(args, campaign_cmd)
    if args.run_refute_ready:
        campaign_cmd.append("--run-refute-ready")
    rendered: list[str] = []
    for part in campaign_cmd:
        if part in ("$RUN_DIR/campaign", "$SHARD_INDEX") or re.fullmatch(
            r"\$[A-Za-z_][A-Za-z0-9_]*",
            part,
        ):
            rendered.append(part)
        else:
            rendered.append(q(part))
    return " ".join(rendered)


def render_script(args: argparse.Namespace) -> str:
    array_line = f"#SBATCH --array=0-{args.array_count - 1}" if args.array_count > 1 else ""
    campaign_command = render_campaign_command(args)
    return f"""#!/usr/bin/env bash
# Generated by examples/erdos/make_chi6_colour_guided_beam_slurm_job.py
# Search-only chi6 colour-guided beam campaign.
# Trust boundary: worker/GPU output is untrusted until DRAT/LRAT/Lean verification.
# Darwin lane: partition={args.partition} constraint={args.constraint} gres={args.gres}
# Scratch contract: write campaign ledgers only under OrangeFS, never node-local disk.
#SBATCH --job-name={args.job_name}
#SBATCH --partition={args.partition}
#SBATCH --constraint={args.constraint}
#SBATCH --gres={args.gres}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --mem={args.mem}
#SBATCH --time={args.time}
#SBATCH --output={args.scratch_root}/slurm-%x-%A-%a.out
{array_line}

set -euo pipefail
echo "chi6_colour_guided_beam_slurm_job: start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
SHARD_INDEX="${{SLURM_ARRAY_TASK_ID:-0}}"
SOUNIO_REPO="${{SOUNIO_REPO:-{args.repo_path}}}"
SCRATCH_ROOT="${{CHI6_SCRATCH_ROOT:-{args.scratch_root}}}"
RUN_STAMP="${{SLURM_JOB_ID:-manual}}-${{SHARD_INDEX}}"
RUN_DIR="$SCRATCH_ROOT/$RUN_STAMP-{args.candidate_prefix}"

case "$SCRATCH_ROOT" in
  /orangefs/*) ;;
  *) echo "error: CHI6_SCRATCH_ROOT must stay under /orangefs" >&2; exit 2 ;;
esac
if [[ ! -d "$SOUNIO_REPO/examples/erdos" ]]; then
  echo "error: SOUNIO_REPO does not contain examples/erdos: $SOUNIO_REPO" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
cd "$SOUNIO_REPO"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,memory.total,uuid --format=csv,noheader \
    | tee "$RUN_DIR/nvidia-smi.csv"
else
  echo "nvidia-smi=unavailable" | tee "$RUN_DIR/nvidia-smi.csv"
fi
echo "trust_boundary=worker_untrusted__drat_lrat_lean_verified_required" \
  | tee "$RUN_DIR/TRUST_BOUNDARY.txt"
echo "run_dir=$RUN_DIR"
echo "shard_index=$SHARD_INDEX"
echo "shard_count={args.array_count}"
{campaign_command} 2>&1 | tee "$RUN_DIR/beam-campaign.out"
echo "chi6_colour_guided_beam_slurm_job: done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"""


def validate_args(args: argparse.Namespace) -> None:
    args.job_name = safe_job_name(args.job_name)
    args.partition = safe_slurm_token(args.partition, "--partition")
    args.constraint = safe_slurm_token(args.constraint, "--constraint")
    args.gres = safe_slurm_token(args.gres, "--gres")
    args.candidate_prefix = safe_job_name(args.candidate_prefix)
    if not re.fullmatch(r"[0-9]+[GMK]?", args.mem):
        raise ValueError("--mem must look like 32G")
    if not re.fullmatch(r"[0-9]{1,2}:[0-9]{2}:[0-9]{2}", args.time):
        raise ValueError("--time must look like HH:MM:SS")
    args.scratch_root = orangefs_path(args.scratch_root, "--scratch-root")
    args.repo_path = absolute_path_or_env(args.repo_path, "--repo-path")
    if args.satfanout_json:
        args.satfanout_json = absolute_path_or_env(args.satfanout_json, "--satfanout-json")
    else:
        if not args.coords_csv or not args.colourings_file:
            raise ValueError("provide --satfanout-json or both --coords-csv and --colourings-file")
        args.coords_csv = absolute_path_or_env(args.coords_csv, "--coords-csv")
        args.colourings_file = absolute_path_or_env(args.colourings_file, "--colourings-file")
    for field, allow_zero in (
        ("generations_list", False),
        ("beam_width_list", False),
        ("branch_width_list", False),
        ("mutation_max_den_list", False),
        ("mutation_top_points_list", True),
        ("dsatur_node_limit_list", True),
    ):
        setattr(
            args,
            field,
            safe_csv_ints(getattr(args, field), f"--{field.replace('_', '-')}", allow_zero=allow_zero),
        )
    branch_widths = [int(token) for token in args.branch_width_list.split(",")]
    for field in (
        "array_count",
        "cpus_per_task",
        "mutation_max_candidates",
        "mutation_emit_mutations",
        "mutation_add_points",
        "mutation_min_neighbor_count",
        "mutation_edge_gain_max_combinations",
        "mutation_edge_gain_combination_stride",
        "max_vertices",
        "min_vertices",
        "min_split_degree",
        "max_cubes",
        "max_carried_colourings",
        "cell_budget",
    ):
        positive_int(getattr(args, field), f"--{field.replace('_', '-')}")
    for field in (
        "min_edges",
        "split_depth",
        "mutation_edge_gain_pool_points",
        "mutation_edge_gain_emit_mutations",
        "mutation_edge_gain_combination_offset",
        "sample_hard_cubes",
        "refute_limit",
        "refute_timeout_seconds",
    ):
        nonnegative_int(getattr(args, field), f"--{field.replace('_', '-')}")
    if max(branch_widths) > args.mutation_emit_mutations:
        raise ValueError("--branch-width-list cannot exceed --mutation-emit-mutations")
    if args.max_vertices < args.min_vertices:
        raise ValueError("--max-vertices cannot be smaller than --min-vertices")
    if args.sample_hard_cubes > args.max_cubes:
        raise ValueError("--sample-hard-cubes cannot exceed --max-cubes")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-name", default="chi6-beam-ada")
    parser.add_argument("--partition", default="gpu-orangefs")
    parser.add_argument("--constraint", default="rtx4000ada")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--array-count", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--repo-path", default="/orangefs/training/chi6-payloads/latest/sounio")
    parser.add_argument("--scratch-root", default="/orangefs/training/chi6-colour-guided-beam")
    parser.add_argument("--satfanout-json")
    parser.add_argument("--coords-csv")
    parser.add_argument("--colourings-file")
    parser.add_argument("--candidate-prefix", default="cgada")
    parser.add_argument("--generations-list", default="2")
    parser.add_argument("--beam-width-list", default="2")
    parser.add_argument("--branch-width-list", default="2")
    parser.add_argument("--mutation-max-den-list", default="5")
    parser.add_argument("--mutation-top-points-list", default="50")
    parser.add_argument("--dsatur-node-limit-list", default="1,100000")
    parser.add_argument("--mutation-max-candidates", type=int, default=20_000)
    parser.add_argument("--mutation-emit-mutations", type=int, default=4)
    parser.add_argument("--mutation-add-points", type=int, default=4)
    parser.add_argument("--mutation-min-neighbor-count", type=int, default=1)
    parser.add_argument("--mutation-edge-gain-pool-points", type=int, default=0)
    parser.add_argument("--mutation-edge-gain-max-combinations", type=int, default=250_000)
    parser.add_argument("--mutation-edge-gain-combination-offset", type=int, default=0)
    parser.add_argument("--mutation-edge-gain-combination-stride", type=int, default=1)
    parser.add_argument("--mutation-edge-gain-emit-mutations", type=int, default=1)
    parser.add_argument("--max-vertices", type=int, default=4096)
    parser.add_argument("--min-vertices", type=int, default=2)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--split-depth", type=int, default=1)
    parser.add_argument("--min-split-degree", type=int, default=2)
    parser.add_argument("--max-cubes", type=int, default=1_000_000)
    parser.add_argument("--sample-hard-cubes", type=int, default=5)
    parser.add_argument("--run-refute-ready", action="store_true")
    parser.add_argument("--refute-limit", type=int, default=1)
    parser.add_argument("--refute-timeout-seconds", type=int, default=0)
    parser.add_argument("--max-carried-colourings", type=int, default=8)
    parser.add_argument("--cell-budget", type=int, default=1_000_000)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    try:
        validate_args(args)
        script = render_script(args)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(script, encoding="ascii")
            script_path = args.output
        else:
            sys.stdout.write(script)
            script_path = Path("STDOUT")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.output is not None:
        print("chi6_colour_guided_beam_slurm_job v1")
        print(f"sbatch_script={script_path}")
        print(f"sbatch_script_sha256={sha256_file(script_path)}")
        print(f"partition={args.partition}")
        print(f"constraint={args.constraint}")
        print(f"gres={args.gres}")
        print(f"array_count={args.array_count}")
        print("claim_scope=colour_guided_beam_slurm_launcher_only")
        print("sat_claim=none")
        print("chromatic_claim=none")
        print("global_unsat_claim=none")
        print("verified_claim=none")
        print("promotable=0")
        print("status=COLOUR_GUIDED_BEAM_SLURM_JOB_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
