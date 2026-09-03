#!/usr/bin/env bash
# Live Madaros arm of E219. The grep correspondence gate does not compile
# anything. This script asks the engine that has the judgment: a well-typed
# call to an unimplemented extern must refuse, and must not become an ELF.
#
# Do not point this at lean_single. The seed rewrites externs into stubs,
# implements abs, and emits an ELF for the same fixture (measured 2026-08-17).
# A pass that required the seed to keep compiling would freeze the #1798
# split as a contract. The seed observation below is printed, not scored.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

FIXTURE="${E219_LIVE_FIXTURE:-tests/compile-fail/extern_c_unimplemented_builtin.sio}"
CONTROL="${E219_LIVE_CONTROL:-tests/run-pass/ffi_libm_call.sio}"
SOUC="${E219_LIVE_SOUC:-$ROOT_DIR/bin/souc}"
ARTIFACT="${TMPDIR:-/tmp}/e219_madaros_live_refuse.v1.json"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/e219-live-refuse.XXXXXX")"
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
  echo "[e219-live-refuse] FAIL: $label" >&2
}

if [[ ! -x "$SOUC" ]]; then
  echo "[e219-live-refuse] missing souc wrapper: $SOUC" >&2
  exit 2
fi
if [[ ! -f "$FIXTURE" ]]; then
  echo "[e219-live-refuse] missing fixture: $FIXTURE" >&2
  exit 2
fi
if [[ ! -f "$CONTROL" ]]; then
  echo "[e219-live-refuse] missing control: $CONTROL" >&2
  exit 2
fi

if [[ ! -x "${MADAROS_RAW_BIN:-}" ]]; then
  for cand in "${SOUNIO_MADAROS_BIN:-}" \
              "$ROOT_DIR/artifacts/self-hosted/madaros" \
              "$ROOT_DIR/bin/madaros-linux-x86_64"; do
    if [[ -n "$cand" && -x "$cand" && "$(head -c2 "$cand" 2>/dev/null)" != '#!' ]]; then
      MADAROS_RAW_BIN="$cand"
      break
    fi
  done
fi

if [[ ! -x "${MADAROS_RAW_BIN:-}" ]]; then
  TOTAL=$((TOTAL + 1))
  NOT_RUN=$((NOT_RUN + 1))
  record_failure "madaros_elf_missing"
else
  export MADAROS_RAW_BIN
  export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

  # Positive control: declare unimplemented names, call only an allow-listed
  # one. If this fails, the instrument is broken and a fixture refuse is
  # unreadable.
  TOTAL=$((TOTAL + 1))
  "$SOUC" check "$CONTROL" >"$WORK/control.out" 2>&1
  control_rc=$?
  if [[ "$control_rc" -eq 0 ]] && grep -qF 'check: OK' "$WORK/control.out"; then
    PASSED=$((PASSED + 1))
    echo "[e219-live-refuse] POSITIVE_CONTROL_FIRED: ffi_libm_call checks clean"
  else
    FAILED=$((FAILED + 1))
    record_failure "control_ffi_libm_check_rc_${control_rc}"
    sed 's/^/[control] /' "$WORK/control.out" >&2
  fi

  # The distinguishing program. Never read rc through a pipe.
  TOTAL=$((TOTAL + 1))
  fixture_elf="$WORK/must-not-exist.elf"
  rm -f "$fixture_elf"
  "$SOUC" compile "$FIXTURE" -o "$fixture_elf" >"$WORK/fixture.out" 2>&1
  fixture_rc=$?
  if [[ "$fixture_rc" -eq 0 ]]; then
    FAILED=$((FAILED + 1))
    record_failure "fixture_compiled_exit_0"
  elif [[ -e "$fixture_elf" ]]; then
    FAILED=$((FAILED + 1))
    record_failure "fixture_emitted_elf"
  elif ! grep -qiF 'call to an `extern "C"` function the native backend does not implement' "$WORK/fixture.out"; then
    FAILED=$((FAILED + 1))
    record_failure "missing_e219_pattern"
    sed 's/^/[fixture] /' "$WORK/fixture.out" >&2
  elif ! grep -qiF 'no dynamic linker in this backend' "$WORK/fixture.out"; then
    FAILED=$((FAILED + 1))
    record_failure "missing_no_dynamic_linker_pattern"
    sed 's/^/[fixture] /' "$WORK/fixture.out" >&2
  else
    PASSED=$((PASSED + 1))
    echo "[e219-live-refuse] fixture refused: compile_rc=$fixture_rc no_elf=1"
  fi
fi

# Seed disagreement is scored by e219_engine_oracle_gate.sh, not here.
# Printing it without a verdict was the same hole as the suite skip.

STATUS="pass"
if [[ "$FAILED" -gt 0 || "$NOT_RUN" -gt 0 ]]; then
  STATUS="fail"
fi

mkdir -p "$(dirname "$ARTIFACT")"
cat > "$ARTIFACT" <<JSON
{
  "schema": "sounio.e219-madaros-live-refuse-gate.v1",
  "status": "$STATUS",
  "madaros_raw_bin": "${MADAROS_RAW_BIN:-}",
  "fixture": "$FIXTURE",
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

echo "e219_madaros_live_refuse: status=$STATUS total=$TOTAL passed=$PASSED failed=$FAILED not_run=$NOT_RUN artifact=$ARTIFACT"
if [[ "$STATUS" != "pass" ]]; then
  exit 1
fi
