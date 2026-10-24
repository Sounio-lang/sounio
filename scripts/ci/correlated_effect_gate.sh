#!/usr/bin/env bash
# R6 Correlated effect acceptance controls (founder 2026-08-20).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# shellcheck source=../lib/classify_compile.sh
source "$ROOT/scripts/lib/classify_compile.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

if [[ -n "${MADAROS_RAW_BIN:-}" && -x "${MADAROS_RAW_BIN}" ]]; then
  run_check() { "$MADAROS_RAW_BIN" check "$1" 2>&1; }
elif [[ -x "$ROOT/bin/souc" ]]; then
  run_check() { "$ROOT/bin/souc" check "$1" 2>&1; }
else
  echo "CORRELATED_GATE_FAIL: no checker" >&2; exit 1
fi

fail() { echo "CORRELATED_GATE_FAIL: $*" >&2; exit 1; }
pass() { echo "CORRELATED_GATE_OK: $*"; }

# A refusal is a clean rc 1..123 with a diagnostic, not any non-zero status:
# the old `|| true` read a crash or a timeout as the check firing (#2425).
sounio_capture_rc rc1 "$WORK/out1.log" -- run_check tests/compile-fail/correlated_slot_identity_requires_effect.sio
out1="$(cat "$WORK/out1.log")"
sounio_expect_refusal "$rc1" "$WORK/out1.log" "" || fail "slot identity is not a clean refusal: $(sounio_refusal_reason): $out1"
echo "$out1" | grep -qE 'Correlated|E221|error\[E221\]' || fail "slot identity did not refuse: $out1"
pass "slot-identity refuse"

set +e
out2=$(run_check tests/run-pass/correlated_slot_identity_with_effect.sio)
rc2=$?
set -e
[[ $rc2 -eq 0 ]] || fail "with Correlated failed: $out2"
echo "$out2" | grep -qE 'error\[E' && fail "unexpected error: $out2" || true
pass "slot-identity with effect"

set +e
out3=$(run_check tests/run-pass/correlated_independent_measures_ok.sio)
rc3=$?
set -e
[[ $rc3 -eq 0 ]] || fail "independent failed: $out3"
echo "$out3" | grep -qE 'error\[E221\]' && fail "false positive: $out3" || true
pass "independent negative control"

SOUNIO_FORCE_CORRELATED=1 sounio_capture_rc rc4 "$WORK/out4.log" -- run_check tests/compile-fail/correlated_force_fires_on_independent.sio
out4="$(cat "$WORK/out4.log")"
sounio_expect_refusal "$rc4" "$WORK/out4.log" "" || fail "FORCE is not a clean refusal: $(sounio_refusal_reason): $out4"
echo "$out4" | grep -qE 'Correlated|E221|error\[E221\]' || fail "FORCE did not fire: $out4"
pass "FORCE positive control"

out5=$(run_check tests/compile-fail/correlated_force_fires_on_independent.sio || true)
echo "$out5" | grep -qE 'error\[E221\]' && fail "false positive without FORCE: $out5" || true
pass "force fixture clean without FORCE"

set +e
out6=$(run_check tests/run-pass/effect_known_names_regression.sio)
rc6=$?
set -e
[[ $rc6 -eq 0 ]] || fail "known names: $out6"
pass "known names include Correlated"

bash scripts/ci/correlated_slot_ratchet_gate.sh
echo "CORRELATED_EFFECT_GATE_OK"
