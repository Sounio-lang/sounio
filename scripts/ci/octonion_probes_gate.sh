#!/usr/bin/env bash
# Octonion / O-SSM research probes — keep them live (math-bearing .sio rots if
# nothing runs it). Each probe is self-validating; this gate asserts scientific
# invariants in stdout, not just compile success.
#
# Engine: lean_single (committed ELF). Exact sentinels were re-measured
# 2026-08-18 on origin/main; if the seed is rebuilt, re-pin from a fresh run.
#
# GATE_CONTRACT: v0
# GATE_ID: octonion_probes
# GATE_CLAIMS: octonion algebra + O-SSM separation/recover + correlated RK4 sd
# GATE_ENGINE: lean_single
# GATE_RESULT_ON_SKIP: fail
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"
SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
OUT="$(mktemp -d /tmp/sounio-octonion-probes.XXXXXX)"
trap 'rm -rf "$OUT"' EXIT

fail=0
fail() { echo "[octonion-probes] FAIL: $*" >&2; fail=1; }
pass() { echo "[octonion-probes] PASS: $*"; }

[[ -x "$SOUC" ]] || { echo "[octonion-probes] FAIL: souc missing: $SOUC" >&2; exit 1; }

run() {
  local f="$1"; shift
  echo "[octonion-probes] == $f =="
  if [[ ! -f "$f" ]]; then
    fail "missing fixture $f"
    return
  fi
  local log="$OUT/$(basename "$f").log"
  set +e
  timeout 90 env SOUNIO_SOUC_ENGINE=lean_single SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
    MADAROS_STACK_KB="$MADAROS_STACK_KB" \
    "$SOUC" run "$f" >"$log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    fail "run rc=$rc $f (log $log)"
    tail -20 "$log" >&2 || true
    return
  fi
  local s
  for s in "$@"; do
    if ! grep -qF -- "$s" "$log"; then
      fail "invariant missing in $f: «$s»"
      return
    fi
  done
  pass "$f"
}

# Positive control: a string that must appear so an empty-log green is impossible.
run scripts/research/oct_truth.sio \
  "alternative (x,x,y) ||.||^2 (*1e6): 0" \
  "norm-mult |xy|^2-|x|^2|y|^2 (*1e6): 0"

run scripts/research/oct_algebra.sio \
  "antisymmetry violations = 0" \
  "alternative violations = 0" \
  "flexible ||.||^2 (*1e6) = 0"

run scripts/research/ossm_separation.sio \
  "permil): 500"

# Recover curve: alpha=400 reaches 987 permil under lean_single (measured 2026-08-18).
run scripts/research/ossm_recover.sio \
  "cc     987"

run examples/epistemic/rk4_correlated_uncertainty.sio \
  "sd = 3.279153"

if [[ "$fail" -ne 0 ]]; then
  echo "[octonion-probes] GATE_RECEIPT id=octonion_probes result=fail"
  exit 1
fi
echo "[octonion-probes] GATE_RECEIPT id=octonion_probes result=pass measured=1 inputs=5 assertions=5"
echo "OCTONION_PROBES_GATE_OK"
exit 0
