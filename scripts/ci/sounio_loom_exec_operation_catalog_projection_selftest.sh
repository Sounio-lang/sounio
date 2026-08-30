#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-operation-catalog-projection.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-exec-operation-catalog-projection-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

bash "$ROOT_DIR/scripts/ci/sounio_loom_exec_operation_catalog_freeze_selftest.sh" >/dev/null
dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_catalog_fixture.sh" >/dev/null

LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
RUNTIME="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-exec-operation-catalog"
MANIFEST="$ROOT_DIR/tools/loom/exec_operation_catalog.freeze.v1"
SOURCE="tests/verify-ir/call_b.sio"
SOURCE_SHA256="$(sha256sum "$ROOT_DIR/$SOURCE" | cut -d ' ' -f 1)"

run_probe() {
  SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" exec-catalog-probe --root "$ROOT_DIR" "$@"
}

calibration="$(run_probe --operation calibration)"
sounio_check="$(run_probe --operation sounio-check --source "$SOURCE")"
[[ "$calibration" == *'semantic_authority=Sounio action=9035 operational_kernel=OCaml operation=calibration'* &&
   "$calibration" == *'source_path=- source_sha256=-'* &&
   "$calibration" == *'semantic_event_sha256=113b0f7d2a1f7adc5b68a92ef37be3689accfe4bbb00cb4e8c15fc3ab7b70013'* &&
   "$calibration" == *'arbitrary_shell=false ocaml_catalog_projection_attached=true'* ]] ||
  fail "calibration projection diverged: $calibration"
[[ "$sounio_check" == *'semantic_authority=Sounio action=9035 operational_kernel=OCaml operation=sounio-check'* &&
   "$sounio_check" == *"source_path=$SOURCE source_sha256=$SOURCE_SHA256"* &&
   "$sounio_check" == *'semantic_event_sha256=6017e4c6e745560696f78836f9cc07ec71a9106f13ad1bfdb16d7e342f0840a9'* &&
   "$sounio_check" == *'command_template_sha256=3b195a8030465bb96245039053fb529ecb39ed330ded747fd5d71188fe4ca0d9'* &&
   "$sounio_check" == *'arbitrary_shell=false ocaml_catalog_projection_attached=true'* ]] ||
  fail "sounio-check projection diverged: $sounio_check"

set +e
invalid="$(run_probe --operation sounio-check --source ../escape.sio 2>&1)"
invalid_code=$?
unknown="$(run_probe --operation shell --source "$SOURCE" 2>&1)"
unknown_code=$?
set -e
[[ $invalid_code -eq 1 && "$invalid" == \
   'error: exec-catalog-invalid-argument-denied:SOUNIO_EXEC_OPERATION_CATALOG DENY563 semantic_authority=Sounio action=9035' ]] ||
  fail "invalid-argument control diverged: $invalid"
[[ $unknown_code -eq 1 && "$unknown" == \
   'error: exec-catalog-unknown-operation-denied:SOUNIO_EXEC_OPERATION_CATALOG DENY562 semantic_authority=Sounio action=9035' ]] ||
  fail "unknown-operation control diverged: $unknown"

tampered_manifest="$TEST_ROOT/manifest"
sed 's/^stage=SEMANTICS_FROZEN$/stage=PARITY_OPEN/' "$MANIFEST" > "$tampered_manifest"
set +e
manifest_tamper="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_OPERATION_CATALOG_MANIFEST="$tampered_manifest" \
  "$LOOM" exec-catalog-probe --root "$ROOT_DIR" --operation calibration 2>&1)"
manifest_tamper_code=$?
set -e
[[ $manifest_tamper_code -eq 1 && "$manifest_tamper" == \
   'error: exec-catalog-manifest-hash-mismatch' ]] ||
  fail "manifest tamper did not fail closed: $manifest_tamper"

tampered_runtime="$TEST_ROOT/runtime"
cp "$RUNTIME" "$tampered_runtime"
printf 'tamper' >> "$tampered_runtime"
chmod 0755 "$tampered_runtime"
set +e
runtime_tamper="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_OPERATION_CATALOG_RUNTIME="$tampered_runtime" \
  "$LOOM" exec-catalog-probe --root "$ROOT_DIR" --operation calibration 2>&1)"
runtime_tamper_code=$?
set -e
[[ $runtime_tamper_code -eq 1 && "$runtime_tamper" == \
   'error: exec-catalog-runtime-hash-mismatch' ]] ||
  fail "runtime tamper did not fail closed: $runtime_tamper"

oracle_executed="$TEST_ROOT/oracle-executed"
for name in python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$oracle_executed" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
PATH="$TEST_ROOT:$PATH" run_probe --operation sounio-check --source "$SOURCE" >/dev/null
[[ ! -e "$oracle_executed" ]] || fail 'a prohibited oracle executed'
dependencies="$(ldd "$LOOM" 2>&1 || true; ldd "$RUNTIME" 2>&1 || true)"
printf '%s\n' "$dependencies" | grep -Eqi 'python|rust' &&
  fail 'projection runtime has a prohibited dependency'

printf 'sounio-loom-exec-operation-catalog-projection-selftest: PASS semantic_authority=Sounio action=9035 stage=SEMANTICS_FROZEN operational_kernel=OCaml entries=calibration+sounio-check source_path=%s source_sha256=%s catalog_sha256=31089d39690c8e008dd7ea9bfecefd7e51e4257cd96e091f7fb492c240ed342c invalid_argument=DENY563 unknown_operation=DENY562 manifest_tamper=REFUSED runtime_tamper=REFUSED causal_sabotage=PASS arbitrary_shell=false python_executed=false rust_executed=false ocaml_catalog_projection_attached=true host_payload_selection_attached=false provider_lifecycle_attached=false general_exec_attached=false production_activation=false parity_open=false claim_ready=false operational_source_sha256=%s loom_executable_sha256=%s\n' \
  "$SOURCE" "$SOURCE_SHA256" \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_exec_catalog.ml" | cut -d ' ' -f 1)" \
  "$(sha256sum "$LOOM" | cut -d ' ' -f 1)"
