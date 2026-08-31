#!/usr/bin/env bash
# Live Madaros arm of E241. The static census in Contracts is grep.
# This script asks the engine built from THIS commit: a bare unknown
# Knowledge component must refuse, and a real epsilon bound must not.
#
# Do not point this at the committed bin/souc ELF. That ELF still swallows
# unknown components (measured 2026-08-23). Putting this in Contracts would
# fail every PR until the ELF is rebuilt — same class as E219 / E158.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

IDENT_FIXTURE="${E241_IDENT_FIXTURE:-tests/compile-fail/knowledge_unknown_component_ident.sio}"
INT_FIXTURE="${E241_INT_FIXTURE:-tests/compile-fail/knowledge_unknown_component_int.sio}"
CONTROL="${E241_CONTROL:-docs/audit/probes/knowledge-annotation-parser-coverage-2026-08-19/source_eps.sio}"
SOUC="${E241_SOUC:-$ROOT_DIR/bin/souc}"
ARTIFACT="${TMPDIR:-/tmp}/e241_madaros_live_refuse.v1.json"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/e241-live-refuse.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

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
  echo "[e241-live-refuse] FAIL: $label" >&2
}

if [[ ! -x "$SOUC" ]]; then
  echo "[e241-live-refuse] missing souc wrapper: $SOUC" >&2
  exit 2
fi
for f in "$IDENT_FIXTURE" "$INT_FIXTURE" "$CONTROL"; do
  if [[ ! -f "$f" ]]; then
    echo "[e241-live-refuse] missing file: $f" >&2
    exit 2
  fi
done

if [[ ! -x "${MADAROS_RAW_BIN:-}" ]]; then
  echo "[e241-live-refuse] MADAROS_RAW_BIN is not an executable ELF" >&2
  TOTAL=1
  NOT_RUN=1
  record_failure "madaros_elf_missing"
  STATUS="fail"
else
  export MADAROS_RAW_BIN
  export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

  check_one() {
    local label="$1" file="$2"
    ( ulimit -s 524288
      env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
        "$SOUC" check "$file"
    ) >"$WORK/${label}.out" 2>&1
  }

  # Positive control: Ident + comparison is an epsilon bound, must still check OK.
  TOTAL=$((TOTAL + 1))
  check_one control "$CONTROL"
  control_rc=$?
  if [[ "$control_rc" -eq 0 ]] && grep -qF 'check: OK' "$WORK/control.out"; then
    PASSED=$((PASSED + 1))
    echo "[e241-live-refuse] POSITIVE_CONTROL_FIRED: source_eps checks clean"
  else
    FAILED=$((FAILED + 1))
    record_failure "control_source_eps_rc_${control_rc}"
    sed 's/^/[control] /' "$WORK/control.out" >&2
  fi

  refuse_one() {
    local label="$1" file="$2"
    TOTAL=$((TOTAL + 1))
    check_one "$label" "$file"
    local rc=$?
    if [[ "$rc" -eq 0 ]]; then
      FAILED=$((FAILED + 1))
      record_failure "${label}_checked_ok"
      sed "s/^/[${label}] /" "$WORK/${label}.out" >&2
    elif ! grep -qF 'error[E241]' "$WORK/${label}.out"; then
      FAILED=$((FAILED + 1))
      record_failure "${label}_missing_e241"
      sed "s/^/[${label}] /" "$WORK/${label}.out" >&2
    else
      PASSED=$((PASSED + 1))
      echo "[e241-live-refuse] $label refused: check_rc=$rc e241=1"
    fi
  }

  refuse_one ident "$IDENT_FIXTURE"
  refuse_one int "$INT_FIXTURE"
fi

STATUS="pass"
if [[ "$FAILED" -gt 0 || "$NOT_RUN" -gt 0 ]]; then
  STATUS="fail"
fi

mkdir -p "$(dirname "$ARTIFACT")"
cat > "$ARTIFACT" <<JSON
{
  "schema": "sounio.e241-madaros-live-refuse-gate.v1",
  "status": "$STATUS",
  "madaros_raw_bin": "${MADAROS_RAW_BIN:-}",
  "ident_fixture": "$IDENT_FIXTURE",
  "int_fixture": "$INT_FIXTURE",
  "control": "$CONTROL",
  "metrics": {
    "total": $TOTAL,
    "passed": $PASSED,
    "failed": $FAILED,
    "not_run": $NOT_RUN
  },
  "failures_csv": "$FAILURES"
}
JSON

echo "e241_madaros_live_refuse: status=$STATUS total=$TOTAL passed=$PASSED failed=$FAILED not_run=$NOT_RUN artifact=$ARTIFACT"
if [[ "$STATUS" != "pass" ]]; then
  exit 1
fi
exit 0
