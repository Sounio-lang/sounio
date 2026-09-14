#!/usr/bin/env bash
# a64_run.sh — run an aarch64 ELF on a real arm64 node and return its exit code.
#
# WHY THIS EXISTS
# ---------------
# Everything the compiler asserts about --target aarch64-linux was, until this
# script, checked by compiling: exit codes and diagnostics. That verifies the
# compiler's VERDICT. It does not verify the CODE it emits, and the difference
# is not academic. Three defects found on 2026-09-04 by running the output:
#
#   emit_heap_realloc_a64 overwrote the caller's pointer with `sub x10, x0, #8`
#     (comment broke off mid-sentence). Every aarch64 heap_realloc segfaulted.
#     Disassembly had passed it: a valid instruction in the wrong place.
#   the Seq grow path, which only that realloc reaches
#   a mis-scoped LAST_EXPR_VAR_IDX that made move tracking inert
#
# Encoding-correct and semantics-wrong is a real state, and only execution
# separates the two.
#
# THE ROUTE, MEASURED 2026-09-04
# ------------------------------
# docs/ops/SLURM_LAUNCH_REPAIR_2026-08-17.md names `cpu-ops` as the proven
# partition. That is no longer true and fails in a way that looks like nothing
# at all:
#
#   slurm.conf sets Prolog=prolog-90-dcgm.sh cluster-wide. On cpu-ops, which has
#   no GPU, the prolog fails; the job comes back JobState=CANCELLED
#   Reason=Prolog; and srun is never told, so it sits in "Waiting for resource
#   configuration" until it is killed, having printed nothing. It reads as a
#   mute srun. It is a cancelled job.
#
# The arm64 nodes (gpuorangefs-multi-spark-*, features arm64/gb10) are in
# gpu-orangefs, where the DCGM prolog has a GPU to talk to and succeeds.
#
# /orangefs is mounted on NEITHER this host NOR the arm64 node, so the staging
# step the seed-refresh recipe uses is not available here. srun --bcast carries
# the binary to the node instead, which needs no shared filesystem.
#
# Never sbatch: user_env_retrieval_failed for this submitter is a controller-side
# issue and leaves held corpses. See the ops doc.
#
# USAGE
#   bash scripts/dev/a64_run.sh /path/to/program.elf [args...]
#   A64_TIME=00:10:00 A64_TIMEOUT=600 bash scripts/dev/a64_run.sh prog.elf
#
# The program's stdout/stderr come back on this terminal and its exit code is
# this script's exit code, so it composes with `diff` against an x86 run:
#
#   ./bin/souc-lean-single-x86_64 t.sio /tmp/t.x86
#   ./bin/souc-lean-single-x86_64 t.sio /tmp/t.a64 --target aarch64-linux
#   diff <(/tmp/t.x86) <(bash scripts/dev/a64_run.sh /tmp/t.a64)
set -uo pipefail

export SLURM_CONF="${SLURM_CONF:-/tmp/slurm-direct.conf}"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <aarch64.elf> [args...]" >&2
    exit 2
fi

ELF="$1"; shift

[[ -f "$ELF" ]] || { echo "a64_run: no such file: $ELF" >&2; exit 2; }

# Refuse to ship something that is not an aarch64 ELF: the failure on the node
# would otherwise be an opaque exec error.
if command -v file >/dev/null 2>&1; then
    kind="$(file -b "$ELF" 2>/dev/null || true)"
    case "$kind" in
        *"ARM aarch64"*) ;;
        *) echo "a64_run: not an aarch64 ELF: $ELF" >&2
           echo "a64_run:   file says: $kind" >&2
           exit 2 ;;
    esac
fi

command -v srun >/dev/null 2>&1 || { echo "a64_run: srun not found" >&2; exit 2; }

exec timeout "${A64_TIMEOUT:-300}" srun \
    --partition="${A64_PARTITION:-gpu-orangefs}" \
    --constraint="${A64_CONSTRAINT:-arm64}" \
    --nodes=1 --ntasks=1 \
    --time="${A64_TIME:-00:05:00}" \
    --chdir=/tmp \
    --export=NONE,PATH=/usr/bin:/bin:/usr/local/bin,TMPDIR=/tmp,TMP=/tmp,TEMP=/tmp,HOME=/tmp \
    --bcast=/tmp/a64_run.elf \
    "$ELF" "$@"
