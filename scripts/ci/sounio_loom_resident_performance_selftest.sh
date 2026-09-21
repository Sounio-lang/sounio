#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-performance.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
FRAME="$TEST_ROOT/valid.frame"
RECEIPTS="$TEST_ROOT/resident.tsv"
RUNTIME="$TEST_ROOT/sounio-loom-resident-membrane-runtime"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-resident-performance-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local output="$1" key="$2" token
  for token in $output; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "benchmark field is missing: $key"
}

SOUNIO_LOOM_RESIDENT_MEMBRANE_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane.sh" >/dev/null
mkdir -p "$ROOT_DIR/tools/loom/_build"
(
  flock -x 8
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 8>"$ROOT_DIR/tools/loom/_build/.resident-gate-build.lock"

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'
printf '%s\n' \
  "9023 3 1 3 10 1 1 1 1 1 1 1 0 1 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 $one $one $one $one $one $one $one $one $one $zero $one $zero $zero" \
  > "$FRAME"

benchmark="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_RESIDENT_RECEIPT_LOG="$RECEIPTS" \
  SOUNIO_LOOM_RESIDENT_MEMBRANE_RUNTIME="$RUNTIME" \
  "$LOOM" resident-authority-probe --root "$ROOT_DIR" --mode benchmark \
    --frame "$FRAME" --deadline-ms 5000 --iterations 20)"

[[ "$(field "$benchmark" mode)" == benchmark ]] || fail "wrong benchmark mode: $benchmark"
[[ "$(field "$benchmark" semantic_authority)" == Sounio ]] || fail 'semantic authority drifted'
[[ "$(field "$benchmark" iterations)" == 20 ]] || fail 'iteration count drifted'
[[ "$(field "$benchmark" process_identity)" == stable ]] || fail 'resident identity drifted'
[[ "$(field "$benchmark" decisions)" == parity ]] || fail 'decision parity drifted'
[[ "$(field "$benchmark" performance_gate)" == PASS ]] || fail "resident path lost: $benchmark"

resident_transport_total="$(field "$benchmark" resident_transport_total_us)"
resident_audited_total="$(field "$benchmark" resident_audited_total_us)"
resident_audit_overhead="$(field "$benchmark" resident_audit_overhead_us)"
single_transport_total="$(field "$benchmark" single_transport_total_us)"
resident_p50="$(field "$benchmark" resident_p50_us)"
resident_p95="$(field "$benchmark" resident_p95_us)"
single_p50="$(field "$benchmark" single_p50_us)"
single_p95="$(field "$benchmark" single_p95_us)"
speedup_milli="$(field "$benchmark" speedup_milli)"
[[ "$resident_transport_total" =~ ^[0-9]+$ && \
  "$resident_audited_total" =~ ^[0-9]+$ && \
  "$resident_audit_overhead" =~ ^[0-9]+$ && \
  "$single_transport_total" =~ ^[0-9]+$ && \
  "$speedup_milli" =~ ^[0-9]+$ ]] || fail "non-numeric benchmark receipt: $benchmark"
(( resident_transport_total < single_transport_total )) || \
  fail "resident transport is not lower: $benchmark"
(( resident_audited_total >= resident_transport_total )) || \
  fail "audited total lost its receipt overhead: $benchmark"
[[ "$(field "$benchmark" receipt_policy)" == fsync-per-event ]] || \
  fail "receipt policy is not explicit: $benchmark"
(( speedup_milli > 1000 )) || fail "resident speedup is not positive: $benchmark"

printf 'sounio-loom-resident-performance-selftest: PASS semantic_authority=Sounio operational_realization=OCaml+resident-Sounio iterations=20 resident_transport_total_us=%s resident_audited_total_us=%s resident_audit_overhead_us=%s resident_p50_us=%s resident_p95_us=%s single_transport_total_us=%s single_p50_us=%s single_p95_us=%s speedup_milli=%s receipt_policy=fsync-per-event process_identity=stable decisions=parity performance_gate=PASS membrane_integration=false\n' \
  "$resident_transport_total" "$resident_audited_total" \
  "$resident_audit_overhead" "$resident_p50" "$resident_p95" \
  "$single_transport_total" "$single_p50" "$single_p95" "$speedup_milli"
