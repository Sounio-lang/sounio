#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/exec_grant_controller.runtime.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-exec-grant-controller-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-grant-controller-freeze.XXXXXX")"
CONTROLLER_ONE="$TEST_ROOT/controller-one"
CONTROLLER_TWO="$TEST_ROOT/controller-two"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-exec-grant-controller-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "record field $key occurs $count times in $path"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

stream_hash() {
  local sum
  sum="$(sha256sum)"
  printf '%s' "${sum%% *}"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'controller manifest is missing or linked'
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'controller evidence is missing or linked'
[[ "$(field schema)" == loom-exec-grant-controller-runtime-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == EFFECT_PARITY_FROZEN ]] || fail 'wrong controller stage'
[[ "$(field producing_language)" == OCaml ]] || fail 'controller producer is not OCaml'
[[ "$(field language_role)" == EFFECT_PARITY ]] || fail 'controller has the wrong language role'
[[ "$(field semantic_authority)" == Sounio ]] || fail 'Sounio is not semantic authority'
[[ "$(field action)" == 9030 ]] || fail 'wrong semantic action'
[[ "$(field single_resident_controller)" == true ]] || fail 'controller is not single-resident'
[[ "$(field non_bearer_transport)" == pending ]] || fail 'transport was promoted without material evidence'
for boundary in expected_results_encoded_in_ocaml python_executable_invoked rust_executable_invoked material_grant material_execution barrier_release exec_attached parity_open claim_ready; do
  [[ "$(field "$boundary")" == false ]] || fail "$boundary was promoted during freeze"
done

controller_commit="$(field controller_commit)"
freeze_commit="$(field freeze_gate_commit)"
git -C "$ROOT_DIR" cat-file -e "${controller_commit}^{commit}" || fail 'controller commit is absent'
git -C "$ROOT_DIR" cat-file -e "${freeze_commit}^{commit}" || fail 'freeze gate commit is absent'
for pair in \
  controller_source_path:controller_source_sha256 \
  resident_source_path:resident_source_sha256 \
  cell_source_path:cell_source_sha256 \
  build_script_path:build_script_sha256 \
  selftest_path:selftest_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git -C "$ROOT_DIR" show "$controller_commit:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the frozen controller commit"
done
freeze_path="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$freeze_path")" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze selftest drifted'
[[ "$(git -C "$ROOT_DIR" show "$freeze_commit:$freeze_path" | stream_hash)" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze selftest differs from its commit'

garden="$ROOT_DIR/$(field garden_path)"
authority="$ROOT_DIR/$(field authority_manifest_path)"
fixtures="$ROOT_DIR/$(field fixture_manifest_path)"
kernel="$ROOT_DIR/$(field kernel_runtime_manifest_path)"
resident="$ROOT_DIR/$(field resident_runtime_manifest_path)"
[[ "$(file_hash "$garden")" == "$(field garden_sha256)" ]] || fail 'Garden preregistration drifted'
[[ "$(file_hash "$authority")" == "$(field authority_manifest_sha256)" ]] || fail 'action 9030 manifest drifted'
[[ "$(record_field "$authority" stage)" == SEMANTICS_FROZEN ]] || fail 'action 9030 semantics are not frozen'
[[ "$(record_field "$authority" producing_language)" == Sounio ]] || fail 'action 9030 was not produced by Sounio'
[[ "$(record_field "$authority" language_role)" == SEMANTIC_AUTHORITY ]] || fail 'action 9030 is not semantic authority'
[[ "$(file_hash "$fixtures")" == "$(field fixture_manifest_sha256)" ]] || fail 'Sounio fixture manifest drifted'
[[ "$(record_field "$fixtures" stage)" == SEMANTICS_FROZEN ]] || fail 'Sounio fixtures are not frozen'
[[ "$(record_field "$fixtures" producing_language)" == Sounio ]] || fail 'fixtures were not produced by Sounio'
[[ "$(record_field "$fixtures" authority_manifest_sha256)" == "$(field authority_manifest_sha256)" ]] ||
  fail 'fixture and controller authority roots differ'
[[ "$(file_hash "$kernel")" == "$(field kernel_runtime_manifest_sha256)" ]] || fail 'OCaml cell runtime manifest drifted'
[[ "$(record_field "$kernel" semantic_authority)" == Sounio ]] || fail 'OCaml cell runtime lost its authority root'
[[ "$(file_hash "$resident")" == "$(field resident_runtime_manifest_sha256)" ]] || fail 'resident v4 runtime manifest drifted'
[[ "$(record_field "$resident" runtime_frozen)" == true ]] || fail 'resident v4 is not frozen'

for source in \
  "$ROOT_DIR/$(field controller_source_path)" \
  "$ROOT_DIR/$(field cell_source_path)"; do
  if grep -Eq 'DENY49[1-9]|DENY500|DENY501|ALLOW code=0 reason=allow|code=491|code=499' "$source"; then
    fail "$source encodes a Sounio expected result"
  fi
done

ocamlfind="$(field ocamlfind_path)"
ocamlopt="$(field ocamlopt_path)"
objcopy="$(field objcopy_path)"
[[ "$(file_hash "$ocamlfind")" == "$(field ocamlfind_sha256)" ]] || fail 'ocamlfind binary drifted'
[[ "$(file_hash "$ocamlopt")" == "$(field ocamlopt_sha256)" ]] || fail 'ocamlopt binary drifted'
[[ "$(file_hash "$objcopy")" == "$(field objcopy_sha256)" ]] || fail 'objcopy binary drifted'
[[ "$($ocamlopt -version)" == "$(field ocaml_version)" ]] || fail 'OCaml version drifted'
[[ "$($objcopy --version | sed -n '1p')" == "$(field objcopy_version)" ]] || fail 'objcopy version drifted'

for output in "$CONTROLLER_ONE" "$CONTROLLER_TWO"; do
  SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_OUTPUT="$output" \
    bash "$ROOT_DIR/$(field build_script_path)" >/dev/null
done
cmp "$CONTROLLER_ONE" "$CONTROLLER_TWO" || fail 'two controller rebuilds differ'
[[ "$(file_hash "$CONTROLLER_ONE")" == "$(field runtime_sha256)" ]] || fail 'controller runtime hash differs'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_exec_grant_controller_selftest.sh' ]] || fail 'unexpected gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
result="$(bash "$ROOT_DIR/$(field selftest_path)")"
[[ "$result" == "$(field result)" ]] || fail 'controller gate result differs'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'controller gate result hash differs'

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'controller evidence hash drifted'
grep -Fxq "controller_runtime_sha256=$(field runtime_sha256)" "$EVIDENCE" || fail 'evidence does not bind the controller runtime'
grep -Fxq "result_sha256=$(field result_sha256)" "$EVIDENCE" || fail 'evidence does not bind the gate result'
grep -Fxq 'expected_results_encoded_in_ocaml=false' "$EVIDENCE" || fail 'evidence omits semantic non-laundering'
grep -Fxq 'non_bearer_transport=pending' "$EVIDENCE" || fail 'evidence overstates non-bearer transport'
grep -Fxq 'material_grant=false' "$EVIDENCE" || fail 'evidence overstates material grant'
grep -Fxq 'barrier_release=false' "$EVIDENCE" || fail 'evidence overstates barrier release'

manifest_hash="$(file_hash "$MANIFEST")"
printf 'sounio-loom-exec-grant-controller-freeze-selftest: PASS semantic_authority=Sounio producer=OCaml role=EFFECT_PARITY action=9030 manifest_sha256=%s controller_runtime_sha256=%s rebuilds=2 single_resident_controller=true lifecycle=ISSUE-CONSUME-CLOSE replay=refused out_of_order=refused parent_identity=bound generation=bound expected_results_encoded_in_ocaml=false non_bearer_transport=pending material_grant=false material_execution=false barrier_release=false exec_attached=false parity_open=false claim_ready=false\n' \
  "$manifest_hash" "$(field runtime_sha256)"
