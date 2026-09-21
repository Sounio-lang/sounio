#!/usr/bin/env bash
# Verify forward inverse-role enforcement under both compiler engines.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib/gate_artifact.sh"

# Require an explicit source-current binary. The checked-in Madaros artefact
# predates this fix and would turn a source gate into a stale-binary measurement.
MADAROS="${MADAROS_BIN:-}"
LEAN_WRAPPER="$ROOT_DIR/bin/souc"
ARTIFACT="${SOUNIO_ARTIFACT_DIR:-$ROOT_DIR/artifacts/gates}/madaros_ontology_enforcement.json"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/madaros-ontology-enforcement.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

WITNESSES=(
  tests/fixtures/madaros_ontology_enforcement/madaros_inverse_forward_reference.sio
  tests/fixtures/madaros_ontology_enforcement/lean_single_inverse_forward_reference.sio
)
VALID_CONTROL="tests/run-pass/ontology_roles_basic.sio"

TOTAL=0
PASSED=0
FAILED=0
NOT_RUN=0
FAILURES=""

record_failure() {
  local label="$1"
  if [[ -n "$FAILURES" ]]; then
    FAILURES="$FAILURES,$label"
  else
    FAILURES="$label"
  fi
  echo "[madaros-ontology-enforcement] FAIL: $label" >&2
}

run_engine_check() {
  local engine="$1"
  local source="$2"
  local log="$3"
  if [[ "$engine" == "madaros" ]]; then
    (ulimit -s 524288 2>/dev/null || true; \
      env -u SOUC_BIN -u SOUNIO_SOUC_BIN -u SOUNIO_SOUC_ENGINE \
      "$MADAROS" check "$source") >"$log" 2>&1
  else
    (ulimit -s 524288 2>/dev/null || true; \
      env -u SOUC_BIN -u SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE=lean_single \
      "$LEAN_WRAPPER" check "$source") >"$log" 2>&1
  fi
}

check_rejects_e158() {
  local engine="$1"
  local source="$2"
  local label="${engine}:$(basename "$source")"
  local log="$WORK/${engine}.$(basename "$source").log"
  local rc
  TOTAL=$((TOTAL + 1))
  run_engine_check "$engine" "$source" "$log"
  rc=$?
  echo "$label exit_code=$rc"
  grep -m1 -E '^error\[E158\].*inverse_of target role not found' "$log" || true
  if [[ $rc -ne 0 ]] && grep -qE '^error\[E158\].*inverse_of target role not found' "$log"; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    record_failure "$label:expected_E158_rejection"
    sed -n '1,120p' "$log" >&2
  fi
}

check_accepts_control() {
  local engine="$1"
  local label="${engine}:valid_control"
  local log="$WORK/${engine}.valid-control.log"
  local rc
  TOTAL=$((TOTAL + 1))
  run_engine_check "$engine" "$VALID_CONTROL" "$log"
  rc=$?
  echo "$label exit_code=$rc"
  if [[ $rc -eq 0 ]]; then
    PASSED=$((PASSED + 1))
  else
    FAILED=$((FAILED + 1))
    record_failure "$label:unexpected_rejection"
    sed -n '1,120p' "$log" >&2
  fi
}

if [[ ! -x "$MADAROS" ]]; then
  TOTAL=$((TOTAL + 1))
  NOT_RUN=$((NOT_RUN + 1))
  record_failure "madaros_binary_missing:$MADAROS"
fi
if [[ ! -x "$LEAN_WRAPPER" ]]; then
  TOTAL=$((TOTAL + 1))
  NOT_RUN=$((NOT_RUN + 1))
  record_failure "lean_wrapper_missing:$LEAN_WRAPPER"
fi
for source in "${WITNESSES[@]}" "$VALID_CONTROL"; do
  if [[ ! -f "$source" ]]; then
    TOTAL=$((TOTAL + 1))
    NOT_RUN=$((NOT_RUN + 1))
    record_failure "source_missing:$source"
  fi
done

if [[ $NOT_RUN -eq 0 ]]; then
  for source in "${WITNESSES[@]}"; do
    check_rejects_e158 madaros "$source"
    check_rejects_e158 lean_single "$source"
  done
  check_accepts_control madaros
  check_accepts_control lean_single
fi

STATUS="pass"
if [[ $FAILED -gt 0 || $NOT_RUN -gt 0 ]]; then
  STATUS="fail"
fi

mkdir -p "$(dirname "$ARTIFACT")"
cat <<JSON | gate_write_artifact "$ARTIFACT"
{
  "schema": "sounio.madaros-ontology-enforcement-gate.v1",
  "status": "$STATUS",
  "madaros_bin": "$MADAROS",
  "metrics": {
    "total": $TOTAL,
    "passed": $PASSED,
    "failed": $FAILED,
    "not_run": $NOT_RUN
  },
  "failures_csv": "$FAILURES"
}
JSON

echo "madaros_ontology_enforcement_gate: status=$STATUS total=$TOTAL passed=$PASSED failed=$FAILED not_run=$NOT_RUN artifact=$ARTIFACT"
if [[ "$STATUS" != "pass" ]]; then
  exit 1
fi
