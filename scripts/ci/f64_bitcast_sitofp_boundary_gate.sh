#!/usr/bin/env bash
# f64_bitcast_sitofp_boundary_gate.sh
#
# Trigger-boundary gate for the f64 bitcast/sitofp family that produces F2
# epistemic confidence fabrication (ep28 TEST 6 ~4.6e18).
#
# Mechanism (named): Knowledge.confidence f64-payload / i64-layout kind-split
#   → FieldGet of .confidence leaves integer kind → f64 arithmetic emits
#   sitofp (cvtsi2sd) on the IEEE bit pattern → magnitude ~4e18.
# Audit: docs/audit/MADAROS_F64_BITCAST_SITOFP_BOUNDARY_2026-08-17.md
#
# Rows:
#   CONTROLS  — plain f64 cast/print/struct (must PASS rc=0, CONTROLS_ALL_OK)
#   USER     — MiniF.confidence:f64 + MiniI.confidence:i64 (must PASS; #1496 twin)
#   KCONF    — Knowledge.confidence arithmetic (Madaros today: BITCAST_SITOFP + rc!=0;
#              when fixed: KCONF_ALL_OK + rc=0). Never silent rc=0 with huge conf.
#   lean_single KCONF must be KCONF_ALL_OK (refutes "physics is 4e18").
#
# GATE_CONTRACT: v0
# GATE_ID: f64_bitcast_sitofp_boundary
# GATE_CLAIMS: name F2 bitcast/sitofp trigger boundary; refuse silent fabrication
# GATE_ENGINE: both
# GATE_RESULT_ON_SKIP: fail
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export MADAROS_STACK_KB="${MADAROS_STACK_KB:-524288}"
unset SOUNIO_STDLIB_PATH || true
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
SOUC="$ROOT_DIR/bin/souc"

CONTROLS="tests/run-pass/f64_bitcast_boundary_controls.sio"
USER="tests/run-pass/f64_bitcast_boundary_user_conf.sio"
KCONF="tests/run-pass/f64_bitcast_boundary_knowledge_conf.sio"
TMP="${GATE_F64_BITCAST_DIR:-$(mktemp -d /tmp/sounio-f64-bitcast.XXXXXX)}"
mkdir -p "$TMP"

fail() { echo "[f64-bitcast-boundary] FAIL: $*" >&2; exit 1; }
pass() { echo "[f64-bitcast-boundary] PASS: $*"; }

[[ -x "$SOUC" ]] || fail "souc missing: $SOUC"
[[ -f "$CONTROLS" ]] || fail "missing $CONTROLS"
[[ -f "$USER" ]] || fail "missing $USER"
[[ -f "$KCONF" ]] || fail "missing $KCONF"

run_one() {
  local label="$1" engine="$2" src="$3" out="$4"
  set +e
  if [[ "$engine" == "lean_single" ]]; then
    timeout 60 env SOUNIO_SOUC_ENGINE=lean_single SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
      MADAROS_STACK_KB="$MADAROS_STACK_KB" "$SOUC" run "$src" >"$out" 2>&1
  else
    timeout 60 env -u SOUNIO_SOUC_ENGINE SOUNIO_STDLIB_PATH="$SOUNIO_STDLIB_PATH" \
      MADAROS_STACK_KB="$MADAROS_STACK_KB" "$SOUC" run "$src" >"$out" 2>&1
  fi
  local rc=$?
  set -e
  echo "$rc" >"${out}.rc"
  echo "[f64-bitcast-boundary] ran label=$label engine=$engine rc=$rc log=$out"
}

# --- CONTROLS (Madaros): instrument health ---
run_one controls madaros "$CONTROLS" "$TMP/controls_madaros.log"
crc="$(cat "$TMP/controls_madaros.log.rc")"
[[ "$crc" -eq 0 ]] || fail "CONTROLS rc=$crc (plain f64 path broken; log $TMP/controls_madaros.log)"
grep -Fq 'CONTROLS_ALL_OK' "$TMP/controls_madaros.log" \
  || fail "CONTROLS missing CONTROLS_ALL_OK"
# every R01..R13 OK
for r in 01 02 03 04 05 06 07 08 09 10 11 12 13; do
  grep -Fq "R${r}_OK" "$TMP/controls_madaros.log" \
    || fail "CONTROLS missing R${r}_OK"
done
pass "CONTROLS Madaros plain f64 cast/print/struct matrix green (13 rows)"

# --- USER conf positive controls ---
run_one user madaros "$USER" "$TMP/user_madaros.log"
urc="$(cat "$TMP/user_madaros.log.rc")"
[[ "$urc" -eq 0 ]] || fail "USER rc=$urc (log $TMP/user_madaros.log)"
grep -Fq 'R23_OK' "$TMP/user_madaros.log" || fail "R23 user f64 confidence broken"
grep -Fq 'R24_OK' "$TMP/user_madaros.log" || fail "R24 user i64 confidence broken (#1496 twin)"
grep -Fq 'USER_CONF_ALL_OK' "$TMP/user_madaros.log" || fail "USER_CONF_ALL_OK missing"
# refuse a "fix" that sitofp's ordinary f64 confidence
if grep -Fq 'R23_BITCAST_SITOFP' "$TMP/user_madaros.log"; then
  fail "R23 user f64 confidence hit BITCAST_SITOFP — name-global confidence float mark regressed"
fi
pass "USER Madaros MiniF f64 + MiniI i64 confidence both healthy"

# --- KCONF smoking gun (Madaros) ---
run_one kconf madaros "$KCONF" "$TMP/kconf_madaros.log"
krc="$(cat "$TMP/kconf_madaros.log.rc")"
k_ok=0
k_fab=0
grep -Fq 'KCONF_ALL_OK' "$TMP/kconf_madaros.log" && k_ok=1
grep -Fq 'BITCAST_SITOFP' "$TMP/kconf_madaros.log" && k_fab=1
grep -Fq 'EPISTEMIC_FABRICATION' "$TMP/kconf_madaros.log" && k_fab=1

if [[ "$k_ok" -eq 1 ]]; then
  [[ "$krc" -eq 0 ]] || fail "KCONF_ALL_OK but rc=$krc"
  pass "KCONF Madaros Knowledge.confidence arithmetic healthy (mechanism closed)"
elif [[ "$k_fab" -eq 1 ]]; then
  [[ "$krc" -ne 0 ]] \
    || fail "KCONF fabricates but exited 0 (silent green); log $TMP/kconf_madaros.log"
  # require the ep28-shaped row specifically
  grep -Fq 'R25_BITCAST_SITOFP' "$TMP/kconf_madaros.log" \
    || fail "KCONF fabrication without R25_BITCAST_SITOFP (ep28-shaped row)"
  pass "KCONF Madaros Knowledge.confidence sitofp is fail-closed (rc=$krc, defect live)"
else
  fail "KCONF neither OK nor BITCAST_SITOFP/FABRICATION (log $TMP/kconf_madaros.log)"
fi

# --- lean_single KCONF reference: must be healthy probability ---
run_one kconf_lean lean_single "$KCONF" "$TMP/kconf_lean.log"
lrc="$(cat "$TMP/kconf_lean.log.rc")"
[[ "$lrc" -eq 0 ]] || fail "lean_single KCONF rc=$lrc (reference broken)"
grep -Fq 'KCONF_ALL_OK' "$TMP/kconf_lean.log" \
  || fail "lean_single KCONF missing KCONF_ALL_OK"
if grep -Fq 'BITCAST_SITOFP' "$TMP/kconf_lean.log"; then
  fail "lean_single also BITCAST_SITOFP — would refute Madaros-only diagnosis"
fi
# R25 must be ~0.66
grep -E 'R25[[:space:]]+0\.6' "$TMP/kconf_lean.log" >/dev/null \
  || fail "lean_single R25 not ~0.66 (log $TMP/kconf_lean.log)"
pass "lean_single KCONF reference is probability-scale (Madaros-only sitofp stands)"

# Positive control of the detector: Madaros KCONF log must be capable of showing
# non-OK when defect live OR OK when fixed — already branched above.
# Extra: ensure we did not vacuous-pass on empty logs.
wc -l <"$TMP/kconf_madaros.log" | awk '$1 < 5 { exit 1 }' \
  || fail "kconf_madaros.log suspiciously short"

echo "[f64-bitcast-boundary] RECEIPT controls=ok user=ok kconf=checked lean_kconf=ok log_dir=$TMP"
echo "[f64-bitcast-boundary] GATE_RECEIPT id=f64_bitcast_sitofp_boundary result=pass measured=1 inputs=3 assertions=4"
echo "[f64-bitcast-boundary] OK"
exit 0
