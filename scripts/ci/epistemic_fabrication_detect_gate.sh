#!/usr/bin/env bash
# epistemic_fabrication_detect_gate.sh
#
# Detect silent epistemic fabrication on the dissertation surface:
#   F1 — GUM variance collapsed to exactly 0.0 while concentrations are live
#        (rapamycin_epistemic_adaptive under default Madaros).
#   F2 — confidence printed / stored as a huge magnitude (bit-pattern-as-integer
#        class), outside (0,1] (epistemic_pbpk28 TEST 6 under Madaros).
#
# Positive controls (designed to fire if F1/F2 recur on current Madaros;
# as of 2026-09-22 both witnesses measure healthy — see the "engine healthy"
# pass branches below, not "must currently fire"):
#   - adaptive run must NOT exit 0 with var(blood)=0.000000
#   - ep28 run must emit EPISTEMIC_FABRICATION or FAIL on TEST 6 if conf corrupted
#
# lean_single is the non-fabricating reference for F1 (non-zero variance).
# GATE_CONTRACT: v0
# GATE_ID: epistemic_fabrication_detect
# GATE_CLAIMS: refuse silent zero GUM variance and out-of-range confidence
# GATE_ENGINE: both
# GATE_RESULT_ON_SKIP: fail
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"
# Clear ambient stdlib pin (often /workspace/sounio/stdlib on this pod) so the
# worktree's stdlib/ is used when we run from ROOT_DIR as cwd.
unset SOUNIO_STDLIB_PATH || true
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
# Never inherit ambient SOUC_BIN (often /workspace/sounio/bin/souc on this pod).
SOUC="$ROOT_DIR/bin/souc"
# Relative paths from ROOT_DIR — absolute paths into stdlib/ dual-load the
# package as main+import and trip Madaros E012/E137 on this surface.
ADAPTIVE="tests/run-pass/rapamycin_epistemic_adaptive.sio"
EP28="stdlib/darwin_pbpk/epistemic_pbpk28.sio"
TMP="${GATE_EPISTEMIC_FAB_DIR:-$(mktemp -d /tmp/sounio-epistemic-fab.XXXXXX)}"
mkdir -p "$TMP"

fail() { echo "[epistemic-fab] FAIL: $*" >&2; exit 1; }
pass() { echo "[epistemic-fab] PASS: $*"; }

[[ -x "$SOUC" ]] || fail "souc missing: $SOUC"
[[ -f "$ADAPTIVE" ]] || fail "missing $ADAPTIVE"
[[ -f "$EP28" ]] || fail "missing $EP28"

# --- F1: adaptive variance under Madaros must fail-closed if zero ---
set +e
timeout 120 "$SOUC" run "$ADAPTIVE" \
  >"$TMP/adaptive_madaros.log" 2>&1
arc=$?
set -e

f1_madaros_collapsed=0
if grep -Eq 'var\(blood\)[[:space:]]*=[[:space:]]*0\.0+$' "$TMP/adaptive_madaros.log" \
   || grep -Fq 'var(blood)  = 0.000000' "$TMP/adaptive_madaros.log"; then
  f1_madaros_collapsed=1
  if [[ "$arc" -eq 0 ]] && grep -Fq 'PASS' "$TMP/adaptive_madaros.log"; then
    fail "F1 silent fabrication: zero variance with PASS and rc=0 (see $TMP/adaptive_madaros.log)"
  fi
  if [[ "$arc" -eq 0 ]]; then
    fail "F1 zero variance must exit non-zero (rc=$arc); log=$TMP/adaptive_madaros.log"
  fi
  grep -Fq 'EPISTEMIC_FABRICATION' "$TMP/adaptive_madaros.log" \
    || grep -Fq 'FABRICATED_ZERO' "$TMP/adaptive_madaros.log" \
    || fail "F1 zero variance without FABRICATED_ZERO/EPISTEMIC_FABRICATION marker"
  pass "F1 Madaros zero-variance is fail-closed (rc=$arc)"
else
  # Non-zero variance path: must PASS with rc=0
  [[ "$arc" -eq 0 ]] || fail "F1 unexpected adaptive rc=$arc without zero-var (log $TMP/adaptive_madaros.log)"
  grep -Fq 'PASS' "$TMP/adaptive_madaros.log" || fail "F1 non-zero var but no PASS"
  pass "F1 Madaros reports non-zero variance (engine healthy on this path)"
fi

# --- F2: confidence magnitude under Madaros (before lean_single — engine env
# isolation has been flaky when lean_single is interposed mid-gate) ---
# Timeout raised 180 -> 600 on 2026-09-27, reasoning from the 2026-09-26
# PBPK28 sprint's documented slowdown elsewhere. TESTED AND WRONG: CI (job
# 108667241904, PR #2501) shows the whole gate -- F1 AND F2 -- completing in
# under 2s, so nothing here is timing out; the failure is immediate, not a
# budget problem. Left at 600 anyway since it is a harmless ceiling either
# way, but it is not what makes this gate pass or fail. The real cause is
# still open: either the captured EP28 run compiles/executes to completion
# and prints something neither pattern below matches, or it fails to compile
# or run at all and prints neither. The diagnostic dump right before the
# final `fail` exists to settle that from the next real CI run's own visible
# log, since the captured log itself was never previously surfaced (only its
# /tmp path, gone when the job ends).
set +e
timeout 600 "$SOUC" run "$EP28" \
  >"$TMP/ep28_madaros.log" 2>&1
erc=$?
set -e

# Huge integer-like confidence (bit-pattern class) or explicit fabrication line.
# Match both "4604….000000" (Madaros print_f64 of bitcast bits) and markers.
f2_huge=0
f2_mark=0
grep -E 'AUC confidence:[[:space:]]*[0-9]{8,}' "$TMP/ep28_madaros.log" >/dev/null && f2_huge=1
grep -Fq 'EPISTEMIC_FABRICATION' "$TMP/ep28_madaros.log" && f2_mark=1
grep -Fq 'bit-pattern-as-integer' "$TMP/ep28_madaros.log" && f2_mark=1

if [[ "$f2_huge" -eq 1 || "$f2_mark" -eq 1 ]]; then
  [[ "$f2_mark" -eq 1 ]] \
    || fail "F2 huge/out-of-range confidence without EPISTEMIC_FABRICATION marker (log $TMP/ep28_madaros.log)"
  if grep -Fq 'ALL 9 TESTS PASSED' "$TMP/ep28_madaros.log"; then
    fail "F2 fabrication marked but suite still claims all passed"
  fi
  pass "F2 Madaros bit-pattern-scale confidence is detected and not silent-pass (huge=$f2_huge mark=$f2_mark)"
elif grep -E 'AUC confidence:[[:space:]]*0\.[0-9]+' "$TMP/ep28_madaros.log" >/dev/null; then
  pass "F2 Madaros confidence print looks like a probability (engine healthy)"
else
  # 2026-09-27: neither pattern matched, and both submitted follow-up fixes
  # (raising this section's timeout; a real arity-mismatch fix elsewhere in
  # the merged tree) left this FAIL unchanged -- CI shows the whole gate
  # completing in under 2s, ruling out a timeout. The captured log has never
  # been visible in CI output (only its /tmp path, gone when the job ends),
  # so the actual cause was never confirmed. Print it here instead of
  # guessing again.
  echo "[epistemic-fab] F2 diagnostic: rc=$erc, captured EP28 output follows" >&2
  echo "[epistemic-fab] ---- ep28_madaros.log (last 60 lines) ----" >&2
  tail -n 60 "$TMP/ep28_madaros.log" >&2 || true
  echo "[epistemic-fab] ---- end ep28_madaros.log ----" >&2
  fail "F2 no confidence line recognised (log $TMP/ep28_madaros.log, rc=$erc)"
fi

# lean_single reference for F1: physics is not actually zero variance
set +e
timeout 120 env SOUNIO_SOUC_ENGINE=lean_single SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" MADAROS_STACK_KB="$MADAROS_STACK_KB" \
  "$SOUC" run "$ADAPTIVE" \
  >"$TMP/adaptive_lean.log" 2>&1
lrc=$?
set -e
[[ "$lrc" -eq 0 ]] || fail "F1 lean_single adaptive rc=$lrc (reference broken)"
if grep -Eq 'var\(blood\)[[:space:]]*=[[:space:]]*0\.0+$' "$TMP/adaptive_lean.log" \
   || grep -Fq 'var(blood)  = 0.000000' "$TMP/adaptive_lean.log"; then
  fail "F1 lean_single also zero — would refute Madaros-only fabrication diagnosis"
fi
if [[ "$f1_madaros_collapsed" -eq 1 ]]; then
  pass "F1 lean_single reference has non-zero variance (Madaros-only collapse stands)"
else
  pass "F1 lean_single reference has non-zero variance too (no collapse on this witness to compare against)"
fi

echo "[epistemic-fab] RECEIPT f1=checked f2=checked log_dir=$TMP"
echo "[epistemic-fab] GATE_RECEIPT id=epistemic_fabrication_detect result=pass measured=1 inputs=2 assertions=4"
echo "[epistemic-fab] OK"
exit 0
