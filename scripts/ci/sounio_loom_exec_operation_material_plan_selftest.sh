#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-operation-material.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-exec-operation-material-plan-selftest: FAIL: %s\n' "$*" >&2
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
OUTPUT_NAME="loom-sounio-check-${SOURCE_SHA256:0:16}.elf"

run_material() {
  local output="$1"
  SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" exec-catalog-material-probe \
    --root "$ROOT_DIR" --source "$SOURCE" --output "$output"
}

mkdir -p "$TEST_ROOT/run-a" "$TEST_ROOT/run-b"
OUTPUT_A="$TEST_ROOT/run-a/$OUTPUT_NAME"
OUTPUT_B="$TEST_ROOT/run-b/$OUTPUT_NAME"
RESULT_A="$(run_material "$OUTPUT_A")"
RESULT_B="$(run_material "$OUTPUT_B")"

[[ "$RESULT_A" == *'semantic_authority=Sounio action=9035 operational_kernel=OCaml material_selector=OCaml operation=sounio-check'* &&
   "$RESULT_A" == *"source_path=$SOURCE source_sha256=$SOURCE_SHA256"* &&
   "$RESULT_A" == *'compiler_sha256=81c4929823460a807762abd4a878ca9adbd053313e7ac6662fa489456269a4a3'* &&
   "$RESULT_A" == *'direct_exec=true shell=false artifact_executed=false direct_exec_material_plan_attached=true'* ]] ||
  fail "material result diverged: $RESULT_A"

ARTIFACT_A_SHA256="$(sha256sum "$OUTPUT_A" | cut -d ' ' -f 1)"
ARTIFACT_B_SHA256="$(sha256sum "$OUTPUT_B" | cut -d ' ' -f 1)"
[[ "$ARTIFACT_A_SHA256" == "$ARTIFACT_B_SHA256" ]] ||
  fail 'repeated source-built compilation was not deterministic'
[[ "$(stat -c '%a' "$OUTPUT_A")" == '600' ]] ||
  fail 'material artifact was not private'
file "$OUTPUT_A" | grep -q 'ELF 64-bit LSB executable' ||
  fail 'material artifact is not the expected ELF result'

set +e
invalid="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" exec-catalog-material-probe \
  --root "$ROOT_DIR" --source ../escape.sio \
  --output "$TEST_ROOT/run-a/loom-sounio-check-invalid.elf" 2>&1)"
invalid_code=$?
set -e
[[ $invalid_code -eq 1 && "$invalid" == \
   'error: exec-catalog-invalid-argument-denied:SOUNIO_EXEC_OPERATION_CATALOG DENY563 semantic_authority=Sounio action=9035' ]] ||
  fail "invalid source was not refused by Sounio: $invalid"

mkdir -p "$TEST_ROOT/preexisting"
PREEXISTING="$TEST_ROOT/preexisting/$OUTPUT_NAME"
printf 'sentinel' > "$PREEXISTING"
set +e
preexisting="$(run_material "$PREEXISTING" 2>&1)"
preexisting_code=$?
set -e
[[ $preexisting_code -eq 1 && "$preexisting" == \
   'error: exec-catalog-output-exists' && "$(cat "$PREEXISTING")" == 'sentinel' ]] ||
  fail "preexisting output was not preserved: $preexisting"

set +e
wrong_name="$(run_material "$TEST_ROOT/run-a/not-authorized.elf" 2>&1)"
wrong_name_code=$?
set -e
[[ $wrong_name_code -eq 1 && "$wrong_name" == \
   'error: exec-catalog-output-name-mismatch' ]] ||
  fail "non-derived output name was not refused: $wrong_name"

TAMPER_ROOT="$TEST_ROOT/tamper-root"
mkdir -p "$TAMPER_ROOT/tools/loom"
cp "$MANIFEST" "$TAMPER_ROOT/tools/loom/exec_operation_catalog.freeze.v1"
while IFS='=' read -r _ relative; do
  [[ "$relative" == 'bin/souc-lean-single-x86_64' ]] && continue
  mkdir -p "$TAMPER_ROOT/$(dirname "$relative")"
  cp "$ROOT_DIR/$relative" "$TAMPER_ROOT/$relative"
done < <(grep -E '^[a-z0-9_]+_path=' "$MANIFEST")
mkdir -p "$TAMPER_ROOT/bin"
cp "$ROOT_DIR/bin/souc-lean-single-x86_64" \
  "$TAMPER_ROOT/bin/souc-lean-single-x86_64"
printf 'tamper' >> "$TAMPER_ROOT/bin/souc-lean-single-x86_64"
chmod 0755 "$TAMPER_ROOT/bin/souc-lean-single-x86_64"
set +e
toolchain_tamper="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_OPERATION_CATALOG_RUNTIME="$RUNTIME" \
  "$LOOM" exec-catalog-material-probe --root "$TAMPER_ROOT" \
  --source "$SOURCE" --output "$TEST_ROOT/run-a/$OUTPUT_NAME" 2>&1)"
toolchain_tamper_code=$?
set -e
[[ $toolchain_tamper_code -eq 1 && "$toolchain_tamper" == \
   'error: exec-catalog-toolchain-compiler-hash-mismatch' ]] ||
  fail "toolchain tamper did not fail closed: $toolchain_tamper"

ORACLE_EXECUTED="$TEST_ROOT/prohibited-executable-ran"
for name in sh bash python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
mkdir -p "$TEST_ROOT/run-c"
PATH="$TEST_ROOT:/usr/bin:/bin" run_material \
  "$TEST_ROOT/run-c/$OUTPUT_NAME" >/dev/null
[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a shell, Python, or Rust oracle executed'

DEPENDENCIES="$(ldd "$LOOM" 2>&1 || true; ldd "$RUNTIME" 2>&1 || true)"
printf '%s\n' "$DEPENDENCIES" | grep -Eqi 'python|rust' &&
  fail 'material-plan runtime has a prohibited dependency'

RESULT="$(printf 'sounio-loom-exec-operation-material-plan-selftest: PASS semantic_authority=Sounio action=9035 stage=SEMANTICS_FROZEN operational_kernel=OCaml material_selector=OCaml operation=sounio-check source_path=%s source_sha256=%s compiler_sha256=81c4929823460a807762abd4a878ca9adbd053313e7ac6662fa489456269a4a3 artifact_sha256=%s deterministic_artifact=true artifact_private=true artifact_executed=false invalid_argument=DENY563 preexisting_output=REFUSED output_name_mismatch=REFUSED toolchain_tamper=REFUSED direct_exec=true shell_executed=false python_executed=false rust_executed=false runtime_dependencies=clean direct_exec_material_plan_attached=true host_payload_selection_attached=false provider_lifecycle_attached=false general_exec_attached=false production_activation=false operational_source_sha256=%s loom_executable_sha256=%s' \
  "$SOURCE" "$SOURCE_SHA256" "$ARTIFACT_A_SHA256" \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_exec_catalog.ml" | cut -d ' ' -f 1)" \
  "$(sha256sum "$LOOM" | cut -d ' ' -f 1)")"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-exec-operation-material-plan-v1-20260830.txt"
[[ "$(cat "$EVIDENCE")" == "$RESULT" ]] || fail 'checked-in evidence drifted'
printf '%s\n' "$RESULT"
