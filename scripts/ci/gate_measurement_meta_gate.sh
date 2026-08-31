#!/usr/bin/env bash
# Meta-gate: success without a positive measurement receipt is not success.
#
# General form (see docs/audit/CI_ABSENCE_AS_SUCCESS_CONTRACT_2026-08-17.md):
#   A CI system reports success for work it did not do whenever the *absence*
#   of a measurement signal is treated as a *positive* outcome.
#
# This gate enforces the minimal contract that makes pass-without-work impossible
# for enrolled subjects: exit 0 ⇒ GATE_MEASURED assertions>=1.
#
# It is stronger than "green": it demands a countable unit of work. Pilot
# enrollment is the ratchet — start with fixtures + gate_vacuity_gate, grow.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 9
# shellcheck source=scripts/lib/gate_assert.sh
. "$ROOT_DIR/scripts/lib/gate_assert.sh"
# shellcheck source=scripts/lib/gate_measurement_receipt.sh
. "$ROOT_DIR/scripts/lib/gate_measurement_receipt.sh"
gate_name "gate_measurement_meta_gate"
gate_measurement_reset

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-gate-measurement-meta.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

assertions_run=0
tick() { assertions_run=$((assertions_run + 1)); gate_measurement_add 1; }

# --- A. Receipt contract unit tests (no external gates) --------------------

# A1: emit pass with zero assertions must fail
tick
set +e
(
  gate_name "unit_zero_pass"
  gate_measurement_reset
  gate_measurement_emit pass
) >"$WORK/a1.out" 2>"$WORK/a1.err"
a1=$?
set -e
[[ "$a1" -ne 0 ]] || gate_fail "A1: emit pass with assertions=0 was accepted — contract broken"
echo "  A1 OK: pass-with-zero refused (rc=$a1)"

# A2: emit pass with positive count must succeed and print GATE_MEASURED
tick
set +e
(
  gate_name "unit_ok"
  gate_measurement_reset
  gate_measurement_set 4
  gate_measurement_emit pass
) >"$WORK/a2.out" 2>"$WORK/a2.err"
a2=$?
set -e
[[ "$a2" -eq 0 ]] || gate_fail "A2: emit pass with assertions=4 failed rc=$a2"
grep -q 'GATE_MEASURED schema=sounio.gate.measurement.v1 gate=unit_ok assertions=4 status=pass' "$WORK/a2.out" \
  || gate_fail "A2: missing exact GATE_MEASURED line"
echo "  A2 OK: positive pass receipt emitted"

# A3: require_positive_receipt rejects exit-0 log without receipt
# (run in subshell: gate_fail exits the process by design)
tick
printf 'ALL 0 TESTS PASSED\n' >"$WORK/a3.log"
set +e
(
  gate_measurement_require_positive_receipt "$WORK/a3.log" 0 "a3_vacuous"
) >"$WORK/a3.out" 2>"$WORK/a3.err"
a3=$?
set -e
[[ "$a3" -ne 0 ]] || gate_fail "A3: vacuous exit-0 log was accepted"
echo "  A3 OK: exit-0 without GATE_MEASURED rejected"

# A4: require_positive_receipt accepts exit-0 log with assertions>=1
tick
printf 'GATE_MEASURED schema=sounio.gate.measurement.v1 gate=demo assertions=2 status=pass\n' >"$WORK/a4.log"
set +e
(
  gate_measurement_require_positive_receipt "$WORK/a4.log" 0 "a4_ok"
) >"$WORK/a4.out" 2>"$WORK/a4.err"
a4=$?
set -e
[[ "$a4" -eq 0 ]] || gate_fail "A4: valid receipt rejected rc=$a4"
echo "  A4 OK: exit-0 with assertions=2 accepted"

# A5: skip without reason refused at emit
tick
set +e
(
  gate_name "unit_skip"
  gate_measurement_reset
  gate_measurement_emit skip
) >"$WORK/a5.out" 2>"$WORK/a5.err"
a5=$?
set -e
[[ "$a5" -ne 0 ]] || gate_fail "A5: skip with assertions=0 and no reason was accepted"
echo "  A5 OK: silent skip refused"

# --- B. Pilot fixtures -------------------------------------------------------

PILOT_OK="$ROOT_DIR/scripts/ci/fixtures/gate_measurement_pilot_ok.sh"
PILOT_VAC="$ROOT_DIR/scripts/ci/fixtures/gate_measurement_pilot_vacuous.sh"
require_file "$PILOT_OK"
require_file "$PILOT_VAC"

tick
set +e
bash "$PILOT_OK" >"$WORK/pilot_ok.log" 2>&1
pok=$?
set -e
[[ "$pok" -eq 0 ]] || { cat "$WORK/pilot_ok.log" >&2; gate_fail "pilot_ok exited $pok"; }
gate_measurement_require_positive_receipt "$WORK/pilot_ok.log" 0 "pilot_ok" \
  || { cat "$WORK/pilot_ok.log" >&2; gate_fail "pilot_ok missing positive receipt"; }
pok_n="$(gate_measurement_last_pass_assertions "$WORK/pilot_ok.log")"
[[ "${pok_n:-0}" -ge 1 ]] || gate_fail "pilot_ok assertions=$pok_n"
echo "  B1 OK: pilot_ok measured assertions=$pok_n"

tick
set +e
bash "$PILOT_VAC" >"$WORK/pilot_vac.log" 2>&1
pvac=$?
set -e
[[ "$pvac" -eq 0 ]] || gate_fail "pilot_vacuous should exit 0 (it is the defect specimen)"
set +e
(
  gate_measurement_require_positive_receipt "$WORK/pilot_vac.log" 0 "pilot_vacuous"
) >"$WORK/pilot_vac_check.out" 2>"$WORK/pilot_vac_check.err"
pvac_check=$?
set -e
[[ "$pvac_check" -ne 0 ]] || gate_fail "B2: vacuous pilot was NOT caught — meta-gate is itself vacuous"
echo "  B2 OK: vacuous pilot exit-0 rejected by receipt contract"

# --- C. Live enrolled gate: gate_vacuity_gate (already fail-closed on empty) -

VACUITY="$ROOT_DIR/scripts/ci/gate_vacuity_gate.sh"
require_file "$VACUITY"
tick
set +e
bash "$VACUITY" >"$WORK/vacuity.log" 2>&1
vrc=$?
set -e
if [[ "$vrc" -ne 0 ]]; then
  # Still require that a failing run is not silent: log must be non-empty.
  require_nonempty_file "$WORK/vacuity.log" "vacuity gate failed with empty log"
  echo "  C1 NOTE: gate_vacuity_gate exited $vrc (debt/baseline); log non-empty"
  # Count the non-empty log as measurement of the failure path.
  gate_measurement_add 1
else
  set +e
  (
    gate_measurement_require_positive_receipt "$WORK/vacuity.log" 0 "gate_vacuity_gate"
  ) >"$WORK/vacuity_receipt.out" 2>"$WORK/vacuity_receipt.err"
  vrecv=$?
  set -e
  if [[ "$vrecv" -ne 0 ]]; then
    cat "$WORK/vacuity.log" >&2
    cat "$WORK/vacuity_receipt.err" >&2
    gate_fail "gate_vacuity_gate exited 0 without valid GATE_MEASURED receipt"
  fi
  vn="$(gate_measurement_last_pass_assertions "$WORK/vacuity.log")"
  [[ "${vn:-0}" -ge 1 ]] || gate_fail "gate_vacuity_gate assertions=$vn"
  echo "  C1 OK: gate_vacuity_gate GATE_MEASURED assertions=$vn"
fi

# --- D. Enrollment static check (ratchet list) -------------------------------
# Gates that have opted into the receipt contract must mention the emit symbol.
ENROLLED=(
  scripts/ci/fixtures/gate_measurement_pilot_ok.sh
  scripts/ci/gate_measurement_meta_gate.sh
  scripts/ci/gate_vacuity_gate.sh
)
tick
for g in "${ENROLLED[@]}"; do
  require_file "$g"
  grep -q 'gate_measurement_emit\|GATE_MEASURED' "$g" \
    || gate_fail "enrolled $g does not reference gate_measurement_emit/GATE_MEASURED"
  gate_measurement_add 1
  echo "  D OK: enrolled $g"
done

# --- Emit our own receipt ----------------------------------------------------
# assertions_run tracked via tick/add; ensure floor
n="$(gate_measurement_count)"
[[ "$n" -ge 8 ]] || gate_fail "meta-gate internal assertions=$n (expected >= 8 unit+pilot checks)"
gate_measurement_emit pass
gate_pass "absence-as-success refused; assertions=$(gate_measurement_count)"
