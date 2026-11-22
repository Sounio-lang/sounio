#!/usr/bin/env bash
# Hardware verification gate for Pillar #4:
#   1. Page-Straddle #PF vs #UD vs Retirement Differential (>= 50,000 cases +
#      exhaustive k = 1..15 boundary sweep at 0x100000000 - k right before a
#      PROT_NONE guard page at 0x100000000) against X86Dec.decode on the
#      truncated byte stream (b i = none for i >= k).
#   2. 64B Adversarial Cache-Line Burst & Decision-50 Arena Injection on the
#      canonical 415-byte x86-64 machine-code kernel (memExposureKernelCode)
#      across 3 page-separated 4096B arenas (0x100000, 0x200000, 0x300000)
#      surrounded by PROT_NONE guard pages, checked via exact Rat arithmetic
#      by formal/Bootstrap/TMRBurstCheck.lean.
#
#   bootstrap/model/tmr_burst_straddle_gate.sh [seed_groups] [mc_trials] [seed] [--neg-straddle|--neg-packed]
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"

seed_groups="3400"
mc_trials="20000"
seed="1"
neg_flag=""

pos_idx=0
for arg in "$@"; do
  if [[ "$arg" == "--neg-straddle" || "$arg" == "--neg-packed" ]]; then
    neg_flag="$arg"
  else
    pos_idx=$((pos_idx + 1))
    if [[ $pos_idx -eq 1 ]]; then
      seed_groups="$arg"
    elif [[ $pos_idx -eq 2 ]]; then
      mc_trials="$arg"
    elif [[ $pos_idx -eq 3 ]]; then
      seed="$arg"
    fi
  fi
done

lake="${LEAN_BIN:+$LEAN_BIN/}lake"
if ! command -v "$lake" >/dev/null 2>&1 && [[ -x "$HOME/.elan/bin/lake" ]]; then
  lake="$HOME/.elan/bin/lake"
fi
cxx="${CXX:-g++}"

work="$(mktemp -d /tmp/sounio-p3-burst-gate-XXXXXX)"
trap 'rm -rf "$work"' EXIT

cpu="$work/tmr_burst_straddle_cpu"
"$cxx" -std=c++23 -O2 \
  -o "$cpu" "$root/bootstrap/model/tmr_burst_straddle_cpu.cpp"

command -v "$lake" >/dev/null 2>&1 || { echo "FAIL: lake not found" >&2; exit 1; }
(cd "$root/formal" && "$lake" build tmr-burst-check >/dev/null)
check="$root/formal/.lake/build/bin/tmr-burst-check"
"$check" gen "$seed_groups" "$seed" "$work/straddle_cases.txt" "$work/straddle_stats.txt"
cpu_rc=0
"$cpu" "$seed" "$work/straddle_cases.txt" "$mc_trials" ${neg_flag:+"$neg_flag"} > "$work/cpu_report.txt" || cpu_rc=$?
"$check" verify "$work/cpu_report.txt" "$work/straddle_stats.txt"
if [[ $cpu_rc -ne 0 ]]; then
  exit "$cpu_rc"
fi
