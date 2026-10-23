#!/usr/bin/env bash
# scripts/ci/sedenion_hessian_demo_gate.sh
#
# examples/sedenion_hessian_brain_demo.sio runs to completion on both engines
# (#1489).
#
# The demo used to print its own "no value" sentinel: with every eigenvalue
# null (Experiment 3, ASD norm 0.7) min_eig_nz kept 1.0e30, pr_f64 cast it to
# i32, and pr_i32 recursed forever on the overflowed negative (0 - i32::MIN is
# i32::MIN). lean_single streamed dashes until killed (85 MB in 120 s);
# Madaros died with SIGSEGV after ~600 kB. Nothing ran the example, so it
# stayed broken.
#
# For each engine: compile, run, require rc=0, the end marker, the explicit
# "all null" line for the norm-0.7 case, and a bounded output size (the
# failure mode is unbounded output).
#
# GATE_CONTRACT: v0
# GATE_ID: sedenion_hessian_demo
# GATE_CLAIMS: the sedenion Hessian demo terminates with its end marker and a
#              bounded output on Madaros and lean_single
# GATE_ENGINE: madaros, lean_single
# GATE_RESULT_ON_SKIP: fail
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "sedenion_hessian_demo"

SRC="examples/sedenion_hessian_brain_demo.sio"
SOUC="${SOUC:-./bin/souc}"
LEAN="${SOUNIO_LEAN_SINGLE_BIN:-./bin/souc-lean-single-x86_64}"
TIMEOUT="${SEDENION_DEMO_TIMEOUT:-300}"
MAX_BYTES=65536
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
ulimit -s unlimited 2>/dev/null || true
require_file "$SRC"
require_executable "$LEAN"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

fails=0
check_run() {
  local engine="$1" elf="$2"
  local out="$work/$engine.out"
  # head -c caps the capture so the old unbounded stream cannot fill the disk.
  timeout "$TIMEOUT" "$elf" 2>&1 | head -c $((MAX_BYTES + 1)) > "$out"
  local rc="${PIPESTATUS[0]}"
  local bytes
  bytes="$(wc -c < "$out" | tr -d ' ')"
  if [[ "$rc" -ne 0 ]]; then
    echo "SEDENION_DEMO_FAIL $engine: rc=$rc (124 = timeout, >=128 = signal)"; fails=$((fails + 1)); return
  fi
  if [[ "$bytes" -gt "$MAX_BYTES" ]]; then
    echo "SEDENION_DEMO_FAIL $engine: output exceeds $MAX_BYTES bytes"; fails=$((fails + 1)); return
  fi
  if ! grep -qx 'SEDENION_HESSIAN_DEMO_DONE' "$out"; then
    echo "SEDENION_DEMO_FAIL $engine: no end marker"; tail -5 "$out"; fails=$((fails + 1)); return
  fi
  if ! grep -A6 -- '--- ASD norm=0.7 ---' "$out" | grep -q 'Min |eig|:  n/a (all 16 eigenvalues null)'; then
    echo "SEDENION_DEMO_FAIL $engine: the all-null norm-0.7 case did not print its n/a line"
    grep -A6 -- '--- ASD norm=0.7 ---' "$out"; fails=$((fails + 1)); return
  fi
  echo "SEDENION_DEMO_OK $engine: rc=0, ${bytes} bytes, end marker present"
}

# lean_single: raw `SRC OUT` CLI.
rm -f "$work/ls.elf"
timeout "$TIMEOUT" "$LEAN" "$SRC" "$work/ls.elf" > "$work/ls.compile.log" 2>&1
if [[ -s "$work/ls.elf" ]] && grep -q '^compile: fns=' "$work/ls.compile.log"; then
  chmod +x "$work/ls.elf"; check_run lean_single "$work/ls.elf"
else
  echo "SEDENION_DEMO_FAIL lean_single: did not compile"; tail -5 "$work/ls.compile.log"; fails=$((fails + 1))
fi

# Madaros: the default engine behind bin/souc.
rm -f "$work/md.elf"
timeout "$TIMEOUT" "$SOUC" compile "$SRC" -o "$work/md.elf" > "$work/md.compile.log" 2>&1
crc=$?
if [[ "$crc" -eq 0 && -s "$work/md.elf" ]] && head -c 4 "$work/md.elf" | grep -q 'ELF'; then
  chmod +x "$work/md.elf"; check_run madaros "$work/md.elf"
else
  echo "SEDENION_DEMO_FAIL madaros: did not compile (rc=$crc)"; tail -5 "$work/md.compile.log"; fails=$((fails + 1))
fi

if [[ "$fails" -ne 0 ]]; then
  gate_fail "$fails engine run(s) failed"
fi
gate_pass "demo terminates on both engines"
