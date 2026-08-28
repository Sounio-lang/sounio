#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-process-witness-principal-cell.XXXXXX")"
CELL_ONE="$TEST_ROOT/one/cell"
CELL_TWO="$TEST_ROOT/two/cell"
PAYLOAD="$TEST_ROOT/payload"
MANIFEST="$ROOT_DIR/tools/loom/process_witness_handshake_payload.freeze.v1"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-principal-cell-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

mkdir -p "$(dirname "$CELL_ONE")" "$(dirname "$CELL_TWO")"
for output in "$CELL_ONE" "$CELL_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_PRINCIPAL_CELL_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_loom_process_witness_principal_cell.sh" >/dev/null
done
SOUNIO_LOOM_PROCESS_WITNESS_HANDSHAKE_OUTPUT="$PAYLOAD" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_handshake_payload.sh" >/dev/null

cmp "$CELL_ONE" "$CELL_TWO" || fail 'two ProcessWitness cell builds differ'
[[ "$(file_hash "$MANIFEST")" == 624ccd7297778803eff8d9972a33d5e55fb022f9e7e37f444f0aee13c22fb4da ]] ||
  fail 'frozen Sounio payload manifest hash drifted'
[[ "$(file_hash "$PAYLOAD")" == 7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d ]] ||
  fail 'source-fresh Sounio payload hash drifted'

receipt="$TEST_ROOT/receipt.txt"
"$CELL_ONE" --selftest --payload "$PAYLOAD" --payload-manifest "$MANIFEST" > "$receipt"
grep -Fq 'LOOM_PROCESS_WITNESS_PRINCIPAL_CELL_SELFTEST PASS' "$receipt" ||
  fail 'cell selftest did not pass'
for fact in \
  'semantic_authority=Sounio' \
  'local_execveat=true' \
  'same_pid=true' \
  'start_tick=true' \
  'pidfd=true' \
  'pre_exec=cell' \
  'post_exec=Sounio' \
  'treatment=closed' \
  'positive=done' \
  'wrong_generation=closed' \
  'extra_release=closed' \
  'payload_hash_mismatch=closed' \
  'causal_bypass=done' \
  'causal_sabotage=PASS' \
  'same_descriptor_hash_and_exec=true' \
  'no_read_ahead=true' \
  'empty_env=true' \
  'host_internal_mode=bounded' \
  'dynamic_user_required=true' \
  'principal_distinct_uid=false' \
  'material_grant=true' \
  'material_execution=false' \
  'host_execveat=false' \
  'launch_open=false' \
  'python_executed=false' \
  'rust_executed=false'; do
  grep -Fq "$fact" "$receipt" || fail "cell receipt omitted $fact"
done

for executable in "$CELL_ONE" "$PAYLOAD"; do
  dependencies="$(ldd "$executable" 2>&1 || true)"
  if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
    fail "prohibited runtime dependency in $executable"
  fi
done

printf 'sounio-loom-process-witness-principal-cell-selftest: PASS semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY transitory=true action=9030 builds=2 deterministic_binary=true source_fresh_sounio_payload=true local_execveat=true same_pid=true start_tick=true pidfd=true pre_exec=cell post_exec=Sounio treatment=closed positive=done wrong_generation=closed extra_release=closed payload_hash_mismatch=closed causal_bypass=done causal_sabotage=PASS two_phase=true same_descriptor_hash_and_exec=true no_read_ahead=true empty_env=true host_internal_mode=bounded dynamic_user_required=true principal_distinct_uid=false material_grant=true material_execution=false host_execveat=false launch_open=false recycle_open=false commit_attached=false ci_attached=false python_executed=false rust_executed=false runtime_dependencies=clean cell_sha256=%s payload_manifest_sha256=%s payload_sha256=%s\n' \
  "$(file_hash "$CELL_ONE")" "$(file_hash "$MANIFEST")" "$(file_hash "$PAYLOAD")"
