#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell.runtime.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-kernel-exec-grant-cell-ocaml-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-exec-grant-cell-ocaml-freeze.XXXXXX")"
RUNTIME_ONE="$TEST_ROOT/loom-one"
RUNTIME_TWO="$TEST_ROOT/loom-two"
DUNE_BUILD="$TEST_ROOT/dune-build"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-kernel-exec-grant-cell-ocaml-freeze-selftest: FAIL: %s\n' "$*" >&2
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

[[ -f "$MANIFEST" ]] || fail 'OCaml runtime manifest is missing'
[[ -f "$EVIDENCE" ]] || fail 'OCaml runtime evidence is missing'
[[ "$(field schema)" == loom-kernel-exec-grant-cell-ocaml-runtime-v1 ]] || fail 'unknown manifest schema'
[[ "$(field stage)" == OPERATIONAL_KERNEL_FROZEN ]] || fail 'wrong operational stage'
[[ "$(field producing_language)" == OCaml ]] || fail 'producer is not OCaml'
[[ "$(field language_role)" == OPERATIONAL_KERNEL ]] || fail 'wrong language role'
[[ "$(field semantic_authority)" == Sounio ]] || fail 'Sounio is not semantic authority'
[[ "$(field action)" == 9030 ]] || fail 'wrong semantic action'
for boundary in expected_results_encoded_in_ocaml python_executable_invoked rust_executable_invoked material_grant material_coverage same_uid_peer_isolation exec_attached commit_attached ci_attached parity_open claim_ready; do
  [[ "$(field "$boundary")" == false ]] || fail "$boundary was promoted during freeze"
done

ocaml_commit="$(field ocaml_kernel_commit)"
freeze_commit="$(field freeze_gate_commit)"
git -C "$ROOT_DIR" cat-file -e "${ocaml_commit}^{commit}" || fail 'OCaml kernel commit is absent'
git -C "$ROOT_DIR" cat-file -e "${freeze_commit}^{commit}" || fail 'OCaml freeze gate commit is absent'
for pair in \
  grant_source_path:grant_source_sha256 \
  resident_source_path:resident_source_sha256 \
  cli_source_path:cli_source_sha256 \
  dune_path:dune_sha256 \
  gate_script_path:gate_script_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git -C "$ROOT_DIR" show "$ocaml_commit:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the frozen OCaml commit"
done
freeze_path="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$freeze_path")" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze selftest drifted'
[[ "$(git -C "$ROOT_DIR" show "$freeze_commit:$freeze_path" | stream_hash)" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze selftest differs from its commit'

for source in \
  "$ROOT_DIR/$(field grant_source_path)" \
  "$ROOT_DIR/$(field resident_source_path)" \
  "$ROOT_DIR/$(field cli_source_path)"; do
  if grep -Eq 'DENY49[1-9]|DENY500|DENY501|ALLOW code=0 reason=allow' "$source"; then
    fail "$source encodes a semantic expected result"
  fi
done

action_manifest="$ROOT_DIR/$(field parent_9030_manifest_path)"
resident_manifest="$ROOT_DIR/$(field parent_resident_v4_manifest_path)"
[[ "$(file_hash "$action_manifest")" == "$(field parent_9030_manifest_sha256)" ]] || fail 'frozen action 9030 manifest drifted'
[[ "$(record_field "$action_manifest" stage)" == SEMANTICS_FROZEN ]] || fail 'action 9030 semantics are not frozen'
[[ "$(record_field "$action_manifest" producing_language)" == Sounio ]] || fail 'action 9030 was not produced by Sounio'
[[ "$(record_field "$action_manifest" language_role)" == SEMANTIC_AUTHORITY ]] || fail 'action 9030 is not semantic authority'
[[ "$(file_hash "$resident_manifest")" == "$(field parent_resident_v4_manifest_sha256)" ]] || fail 'resident v4 manifest drifted'
[[ "$(record_field "$resident_manifest" parent_9030_sha256)" == "$(field parent_9030_manifest_sha256)" ]] || fail 'resident v4 does not bind the same action 9030 freeze'
[[ "$(record_field "$resident_manifest" runtime_frozen)" == true ]] || fail 'resident v4 is not frozen'

ocamlc="$(field ocamlc_path)"
dune="$(field dune_executable_path)"
[[ "$(file_hash "$ocamlc")" == "$(field ocamlc_sha256)" ]] || fail 'ocamlc binary drifted'
[[ "$(file_hash "$dune")" == "$(field dune_executable_sha256)" ]] || fail 'dune binary drifted'
[[ "$($ocamlc -version)" == "$(field ocaml_version)" ]] || fail 'OCaml version drifted'
[[ "$($dune --version)" == "$(field dune_version)" ]] || fail 'Dune version drifted'

(
  flock -x 9
  "$dune" build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
  [[ "$(file_hash "$ROOT_DIR/tools/loom/_build/default/src/loom.exe")" == \
    "$(field runtime_sha256)" ]] || fail 'standard OCaml runtime hash differs'
  "$dune" build --root "$ROOT_DIR/tools/loom" --build-dir "$DUNE_BUILD" \
    src/loom.exe >/dev/null
  cp "$DUNE_BUILD/default/src/loom.exe" "$RUNTIME_ONE"
  "$dune" clean --root "$ROOT_DIR/tools/loom" --build-dir "$DUNE_BUILD"
  "$dune" build --root "$ROOT_DIR/tools/loom" --build-dir "$DUNE_BUILD" \
    src/loom.exe >/dev/null
  cp "$DUNE_BUILD/default/src/loom.exe" "$RUNTIME_TWO"
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two OCaml rebuilds differ'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_kernel_exec_grant_cell_ocaml_selftest.sh' ]] || fail 'unexpected gate command'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash differs'
result="$(bash "$ROOT_DIR/$(field gate_script_path)")"
[[ "$result" == "$(field result)" ]] || fail 'adversarial gate result differs'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'gate result hash differs'

manifest_hash="$(file_hash "$MANIFEST")"
grep -Fq "manifest_sha256=$manifest_hash" "$EVIDENCE" || fail 'evidence does not bind the OCaml manifest'
grep -Fq "runtime_sha256=$(field runtime_sha256)" "$EVIDENCE" || fail 'evidence does not bind the OCaml runtime'

printf '%s\n' \
  "sounio-loom-kernel-exec-grant-cell-ocaml-freeze-selftest: PASS semantic_authority=Sounio operational_kernel=OCaml action=9030 manifest_sha256=$manifest_hash runtime_sha256=$(field runtime_sha256) rebuilds=2 expected_results_encoded_in_ocaml=false happy=ALLOWx3 semantic_deny=DENY493+STATE_PRESERVED revoke=ALLOWx2 current_material=DENY491 python_oracle=DENY499 replay=POISON mismatch=POISON timeout=POISON eof=POISON material_grant=false material_coverage=false same_uid_peer_isolation=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false"
