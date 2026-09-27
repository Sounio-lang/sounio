#!/usr/bin/env bash
# Measure the stack floor of the shipped Madaros compiler.
#
# WHY THIS EXISTS
# ---------------
# `bin/madaros:67` runs `ulimit -s unlimited` before every compile. That line
# is not a tuning knob: without it, the compiler SIGSEGVs on a one-function
# program under the default 8 MB Linux stack. This script measures how much
# stack the compiler actually needs, so the number is a re-runnable measurement
# rather than a claim.
#
# Anything that invokes the raw ELF without the launcher -- an editor, a CI
# step, a user who found `bin/madaros-linux-x86_64` -- gets the segfault.
#
# Usage:
#   bash scripts/dev/measure_madaros_stack_floor.sh [--reps N]
#
# Output: CSV on stdout, `program,stack_mb,failures,reps`.
# Exit status: 0 if the measurement completed (regardless of the floor found).

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPS=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps) REPS="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; exit 2 ;;
  esac
done

RAW=""
for cand in "${MADAROS_RAW_BIN:-}" "$ROOT_DIR/artifacts/self-hosted/madaros" "$ROOT_DIR/bin/madaros-linux-x86_64"; do
  [[ -n "$cand" && -x "$cand" ]] && { RAW="$cand"; break; }
done
if [[ -z "$RAW" ]]; then
  echo "error: no raw Madaros ELF found" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cat > "$WORK/minimal.sio" <<'EOF'
fn main() with IO { print("E\n") }
EOF

cat > "$WORK/helper.sio" <<'EOF'
fn add3(a: i64, b: i64, c: i64) -> i64 { a + b + c }
fn main() with IO {
    let r = add3(1, 2, 3)
    if r == 6 { print("ADD3_OK\n") } else { print("ADD3_WRONG\n") }
}
EOF

echo "# raw compiler: $RAW"
echo "# reps per cell: $REPS"
echo "program,stack_mb,failures,reps"

for prog in minimal helper; do
  for kb in 8192 16384 32768 65536 131072 262144; do
    fails=0
    for _ in $(seq 1 "$REPS"); do
      # Two things this line is careful about:
      #   - `$?` is read before any command substitution, which would reset it;
      #   - the compile runs under an inner `bash -c` so that the job-control
      #     "Segmentation fault" notice is written to the inner shell's stderr
      #     and discarded, instead of interleaving with the CSV.
      # No `exec` here on purpose: the subshell must outlive the compiler so
      # that *it*, not this script, reports the SIGSEGV -- and its stderr is
      # discarded, keeping the job-control notice out of the CSV.
      ( ulimit -s "$kb" 2>/dev/null || true
        timeout 300 "$RAW" "$WORK/$prog.sio" -o "$WORK/$prog.elf" >/dev/null 2>&1
      ) 2>/dev/null
      [[ $? -ne 0 ]] && fails=$((fails + 1))
    done
    mb=$((kb / 1024))
    echo "$prog,$mb,$fails,$REPS"
  done
done
