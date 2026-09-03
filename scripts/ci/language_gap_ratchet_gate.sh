#!/usr/bin/env bash
# Ratchet gate for measured language gaps (issues #2387, #2388 and the IEEE
# special-value findings in tests/known-gaps/numerics/).  Each witness asserts
# the DEFECTIVE behaviour as measured on 2026-09-02.  When a fix lands, the
# corresponding line flips and this gate fails on purpose: that is the signal to
# (1) move the reproduction to tests/run-pass or tests/compile-fail, (2) update
# docs/compiler/KNOWN_LIMITATIONS.md, and (3) write the migration note, because
# every one of these fixes is source-breaking for code that relies on the gap.
set -uo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
lean() { SOUNIO_SOUC_ENGINE=lean_single "$SOUC" "$@"; }
mad()  { "$SOUC" "$@"; }
FAILS=0; N=0
# Madaros lines are measured against whatever bin/souc resolves to (CLAUDE.md §6.15:
# a prebuilt binary is not a baseline) -- record it so a flip can be attributed.
echo "[ratchet] madaros: $("$SOUC" --version 2>&1 | head -1)"
echo "[ratchet] lean_single: $(md5sum "$ROOT_DIR/bin/souc-lean-single-x86_64" 2>/dev/null | cut -c1-8) bin/souc-lean-single-x86_64"
expect() { # label, want, got
  N=$((N+1))
  if [[ "$2" == "$3" ]]; then echo "[ratchet] ok   $1: $3"
  else echo "[ratchet] FLIP $1: expected '$2', got '$3'" >&2; FAILS=$((FAILS+1)); fi
}
rc() { "$@" >/dev/null 2>&1; echo $?; }
UNITS=tests/known-gaps/units
NUM=tests/known-gaps/numerics

# #2387 -- f128 is f64 on lean_single (53 halvings until 1+e == 1); refused by Madaros
h=$(lean run examples/numerics/f128_is_f64_probe.sio 2>/dev/null | tail -1 | tr -d '[:space:]')
expect "f128 halvings on lean_single (f64 would be 53, binary128 113)" "53" "$h"
expect "f128 refused by Madaros check (exit != 0)" "1" "$([[ $(rc mad check examples/numerics/f128_is_f64_probe.sio) -ne 0 ]] && echo 1 || echo 0)"

# #2388 (1) -- derived unit annotations do not parse on either engine
expect "derived unit annotation refused by lean_single" "1" "$([[ $(rc lean check $UNITS/derived_unit_annotation_unparsed.sio) -ne 0 ]] && echo 1 || echo 0)"
expect "derived unit annotation refused by Madaros" "1" "$([[ $(rc mad check $UNITS/derived_unit_annotation_unparsed.sio) -ne 0 ]] && echo 1 || echo 0)"
# control: a direct mol + K mismatch IS caught (if this flips, units are off entirely)
expect "control: direct unit mismatch caught by lean_single" "1" "$([[ $(rc lean check $UNITS/direct_unit_mismatch_is_caught.sio) -ne 0 ]] && echo 1 || echo 0)"
# #2388 (2),(3) -- lean_single accepts a dimensionless quotient + K, and K into an f64 parameter
expect "quotient loses dimension (lean_single accepts)" "0" "$(rc lean check $UNITS/derived_unit_dropped_by_inference.sio)"
expect "unit lost at call boundary (lean_single accepts)" "0" "$(rc lean check $UNITS/unit_lost_at_call_boundary.sio)"

# IEEE special values -- NaN compares as ordered on both engines
for e in lean mad; do
  got=$($e run $NUM/nan_compare_is_not_ieee.sio 2>/dev/null | grep -E '^[01]$' | tr '\n' ' ' | sed 's/ $//')
  expect "nan ==,!=,< on $e (IEEE would be '0 1 0')" "1 0 1" "$got"
done
# println(inf): lean_single never returns (timeout 20s -> 124); Madaros prints 2^63
expect "println(inf) hangs on lean_single" "124" "$(timeout 20 env SOUNIO_SOUC_ENGINE=lean_single "$SOUC" run $NUM/print_inf_never_returns.sio >/dev/null 2>&1; echo $?)"
expect "println(inf) on Madaros" "9223372036854775808.000000" "$(timeout 60 "$SOUC" run $NUM/print_inf_never_returns.sio 2>/dev/null | sed -n '/^START$/{n;p;}')"
expect "println(nan) on Madaros" "-9223372036854775808.000000" "$(timeout 60 "$SOUC" run $NUM/print_nan_is_garbage.sio 2>/dev/null | sed -n '/^START$/{n;p;}')"

echo "[ratchet] $((N-FAILS))/$N witnesses hold the measured gap"
[[ $FAILS -eq 0 ]]
