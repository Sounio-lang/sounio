#!/usr/bin/env bash
# Oracle for the E250 engine-split. The theorem is a Madaros judgment.
# The default suite binary is lean_single and skips the fixture
# (`requires: madaros`). Two greens that do not touch are not an oracle.
#
# This gate scores the DISAGREEMENT, the Lean `unimplemented_disagrees`
# at CI: Madaros has the constructor and the suite skip is named; the
# seed has no E250 constructor, implements abs, and emits an ELF for
# the distinguishing program. A cosmetic `tc_error("E250")` on the seed
# is a fail, not a close. A real seed refuse is also a fail — update
# the theorem, do not silence the gate.
#
# Shape matches #1780 / e219_refusal_correspondence_gate: check_grep,
# check_absent, not_run is fail, JSON metrics, --control-child mutants
# that MUST be rejected.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

CHECKER_SOURCE="self-hosted/check/check.sio"
SEED_SOURCE="self-hosted/compiler/lean_single.sio"
SUITE_SOURCE="scripts/dev/run_sio_test_suite.sh"
FIXTURE="tests/compile-fail/extern_c_unimplemented_builtin.sio"
LIVE_FIXTURE="tests/compile-fail/extern_c_unimplemented_builtin.sio"
# The file and legacy E219 env name remain compatibility surfaces for CI jobs.
SEED_ELF="${E250_ORACLE_SEED_ELF:-${E219_ORACLE_SEED_ELF:-$ROOT_DIR/bin/souc-lean-single-x86_64}}"
ARTIFACT="${TMPDIR:-/tmp}/e219_engine_oracle_gate.v1.json"
RUN_POSITIVE_CONTROLS=1
RUN_SEED_LIVE=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checker) CHECKER_SOURCE="${2:?missing path after --checker}"; shift 2 ;;
    --seed-source) SEED_SOURCE="${2:?missing path after --seed-source}"; shift 2 ;;
    --suite) SUITE_SOURCE="${2:?missing path after --suite}"; shift 2 ;;
    --fixture) FIXTURE="${2:?missing path after --fixture}"; shift 2 ;;
    --live-fixture) LIVE_FIXTURE="${2:?missing path after --live-fixture}"; shift 2 ;;
    --seed-elf) SEED_ELF="${2:?missing path after --seed-elf}"; shift 2 ;;
    --artifact) ARTIFACT="${2:?missing path after --artifact}"; shift 2 ;;
    --control-child) RUN_POSITIVE_CONTROLS=0; RUN_SEED_LIVE=0; shift ;;
    *) echo "e219_engine_oracle_gate: unknown argument: $1" >&2; exit 2 ;;
  esac
done

TOTAL=0
PASSED=0
FAILED=0
NOT_RUN=0
FAILURES=""
WORK="$(mktemp -d "${TMPDIR:-/tmp}/e219-engine-oracle.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

record_failure() {
  local label="$1"
  if [[ -n "$FAILURES" ]]; then
    FAILURES="$FAILURES,$label"
  else
    FAILURES="$label"
  fi
  echo "[e219-oracle] FAIL: $label" >&2
}

check_grep() {
  local label="$1"
  local pattern="$2"
  local path="$3"
  TOTAL=$((TOTAL + 1))
  if grep -qE "$pattern" "$path"; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    record_failure "$label"
  fi
}

check_absent() {
  local label="$1"
  local pattern="$2"
  local path="$3"
  TOTAL=$((TOTAL + 1))
  if grep -qE "$pattern" "$path"; then
    FAILED=$((FAILED + 1))
    record_failure "$label"
  else
    PASSED=$((PASSED + 1))
  fi
}

check_fixed() {
  local label="$1"
  local needle="$2"
  local path="$3"
  TOTAL=$((TOTAL + 1))
  if grep -qF "$needle" "$path"; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    record_failure "$label"
  fi
}

check_count_ge() {
  local label="$1"
  local pattern="$2"
  local path="$3"
  local min="$4"
  TOTAL=$((TOTAL + 1))
  local n
  n="$(grep -cE "$pattern" "$path" || true)"
  if [[ "$n" -ge "$min" ]]; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    record_failure "$label:got_$n"
  fi
}

for req in \
  "checker:$CHECKER_SOURCE" \
  "seed:$SEED_SOURCE" \
  "suite:$SUITE_SOURCE" \
  "fixture:$FIXTURE"; do
  kind="${req%%:*}"
  path="${req#*:}"
  if [[ ! -f "$path" ]]; then
    TOTAL=$((TOTAL + 1))
    NOT_RUN=$((NOT_RUN + 1))
    record_failure "${kind}_source_missing:$path"
  fi
done

if [[ "$NOT_RUN" -eq 0 ]]; then
  # M — Madaros has the judgment (source). Live compile is the Witness arm.
  check_count_ge "madaros_e250_reports" \
    ', 250, 0, 0, 0\)' "$CHECKER_SOURCE" 3
  check_count_ge "madaros_refuse_infects" \
    'refused_unimplemented' "$CHECKER_SOURCE" 3
  check_grep "madaros_allowlist_predicate" \
    'fn name_is_native_backend_builtin\(' "$CHECKER_SOURCE"

  # S — seed has no constructor and still implements the distinguishing name.
  check_absent "seed_must_not_spell_e250" \
    'E250|refused_unimplemented' "$SEED_SOURCE"
  check_grep "seed_implements_abs" \
    '__native_abs_i64' "$SEED_SOURCE"

  # K — the suite skip is named, not coverage.
  check_grep "fixture_requires_madaros_annotation" \
    '^//@ requires: madaros' "$FIXTURE"
  check_fixed "suite_skips_requires_madaros" \
    'reason\":\"requires:madaros\"' "$SUITE_SOURCE"
  check_grep "suite_madaros_skip_branch" \
    'madaros\) \[\[ -z "\$\{SOUNIO_MADAROS_AVAILABLE:-\}" \]\]' "$SUITE_SOURCE"
fi

if [[ "$RUN_SEED_LIVE" -eq 1 && "$NOT_RUN" -eq 0 ]]; then
  TOTAL=$((TOTAL + 1))
  if [[ ! -x "$SEED_ELF" ]]; then
    NOT_RUN=$((NOT_RUN + 1))
    record_failure "seed_elf_missing:$SEED_ELF"
  else
    seed_out="$WORK/seed-live.elf"
    rm -f "$seed_out"
    "$SEED_ELF" "$LIVE_FIXTURE" "$seed_out" >"$WORK/seed-live.log" 2>&1
    seed_rc=$?
    if [[ "$seed_rc" -ne 0 ]]; then
      FAILED=$((FAILED + 1))
      record_failure "seed_live_refused_rc_${seed_rc}"
      sed 's/^/[seed-live] /' "$WORK/seed-live.log" >&2
    elif [[ ! -e "$seed_out" ]]; then
      FAILED=$((FAILED + 1))
      record_failure "seed_live_no_elf"
    elif grep -qiE 'error\[E250\]|call to an `extern "C"` function the native backend does not implement' "$WORK/seed-live.log"; then
      FAILED=$((FAILED + 1))
      record_failure "seed_live_printed_e250"
    else
      PASSED=$((PASSED + 1))
      echo "[e219-oracle] seed_live: compile_rc=0 elf_exists=1 (split reported)"
    fi
  fi
fi

if [[ "$RUN_POSITIVE_CONTROLS" -eq 1 && "$NOT_RUN" -eq 0 ]]; then
  MADAROS_MUTANT="scripts/ci/fixtures/e219_engine_oracle/madaros_without_e219.sio"
  SEED_MUTANT="scripts/ci/fixtures/e219_engine_oracle/seed_cosmetic_e219.sio"
  FIXTURE_MUTANT="scripts/ci/fixtures/e219_engine_oracle/fixture_without_requires_madaros.sio"

  TOTAL=$((TOTAL + 1))
  if "$0" --control-child --checker "$MADAROS_MUTANT" --seed-source "$SEED_SOURCE" \
      --suite "$SUITE_SOURCE" --fixture "$FIXTURE" \
      --artifact "$WORK/madaros-mutant.json" \
      >"$WORK/madaros-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_madaros_without_e219_was_not_rejected"
  else
    PASSED=$((PASSED + 1))
    echo "[e219-oracle] POSITIVE_CONTROL_FIRED: madaros_without_e219 rejected"
    sed 's/^/[madaros-control] /' "$WORK/madaros-mutant.log"
  fi

  TOTAL=$((TOTAL + 1))
  if "$0" --control-child --checker "$CHECKER_SOURCE" --seed-source "$SEED_MUTANT" \
      --suite "$SUITE_SOURCE" --fixture "$FIXTURE" \
      --artifact "$WORK/seed-mutant.json" \
      >"$WORK/seed-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_seed_cosmetic_e219_was_not_rejected"
  else
    PASSED=$((PASSED + 1))
    echo "[e219-oracle] POSITIVE_CONTROL_FIRED: seed_cosmetic_e219 rejected"
    sed 's/^/[seed-control] /' "$WORK/seed-mutant.log"
  fi

  TOTAL=$((TOTAL + 1))
  if "$0" --control-child --checker "$CHECKER_SOURCE" --seed-source "$SEED_SOURCE" \
      --suite "$SUITE_SOURCE" --fixture "$FIXTURE_MUTANT" \
      --artifact "$WORK/fixture-mutant.json" \
      >"$WORK/fixture-mutant.log" 2>&1; then
    FAILED=$((FAILED + 1))
    record_failure "positive_control_fixture_without_requires_was_not_rejected"
  else
    PASSED=$((PASSED + 1))
    echo "[e219-oracle] POSITIVE_CONTROL_FIRED: fixture_without_requires_madaros rejected"
    sed 's/^/[fixture-control] /' "$WORK/fixture-mutant.log"
  fi
fi

STATUS="pass"
if [[ "$FAILED" -gt 0 || "$NOT_RUN" -gt 0 ]]; then
  STATUS="fail"
fi

mkdir -p "$(dirname "$ARTIFACT")"
cat > "$ARTIFACT" <<JSON
{
  "schema": "sounio.e219-engine-oracle-gate.v1",
  "status": "$STATUS",
  "checker_source": "$CHECKER_SOURCE",
  "seed_source": "$SEED_SOURCE",
  "suite_source": "$SUITE_SOURCE",
  "fixture": "$FIXTURE",
  "metrics": {
    "total": $TOTAL,
    "passed": $PASSED,
    "failed": $FAILED,
    "not_run": $NOT_RUN
  },
  "failures_csv": "$FAILURES"
}
JSON

echo "e219_engine_oracle_gate: status=$STATUS total=$TOTAL passed=$PASSED failed=$FAILED not_run=$NOT_RUN artifact=$ARTIFACT"
if [[ "$STATUS" != "pass" ]]; then
  exit 1
fi
