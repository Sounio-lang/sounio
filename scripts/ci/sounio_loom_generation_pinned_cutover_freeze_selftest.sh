#!/usr/bin/env bash
set -euo pipefail
umask 077
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT/tools/loom/generation_pinned_cutover.freeze.v2"

fail() { printf 'sounio-loom-generation-pinned-cutover-freeze-selftest: FAIL: %s\n' "$*" >&2; exit 1; }
value() {
  local count
  count="$(grep -c "^$1=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "$1 occurs $count times"
  sed -n "s/^$1=//p" "$MANIFEST"
}
expect() { [[ "$(value "$1")" == "$2" ]] || fail "$1 diverged"; }
expect_hash() {
  local path="$(value "$1_path")" expected="$(value "$1_sha256")"
  [[ -f "$ROOT/$path" && ! -L "$ROOT/$path" ]] || fail "$path absent or linked"
  [[ "$(sha256sum "$ROOT/$path" | cut -d' ' -f1)" == "$expected" ]] || fail "$path drifted"
}
expect_commit() {
  local commit="$1" key="$2" path="$(value "$2_path")" expected="$(value "$2_sha256")"
  git -C "$ROOT" cat-file -e "${commit}^{commit}" || fail "$commit absent"
  [[ "$(git -C "$ROOT" show "${commit}:$path" | sha256sum | cut -d' ' -f1)" == "$expected" ]] ||
    fail "$path not bound to $commit"
}

expect schema loom-generation-pinned-cutover-freeze-v2
expect stage SEMANTICS_FROZEN
expect semantic_authority Sounio
expect producing_language Sounio
expect language_role SEMANTIC_AUTHORITY
expect action 9048
expect parent_actions 9046-frozen+9047-frozen
expect fail_closed_required true
expect immutable_runtime_target_required true
expect kernel_generation_identity_required true
expect complete_inventory_required true
expect pre_cutover_legacy_only true
expect one_hop_forward_only true
expect decision_receipt_required true
expect python_oracle_attempt DENY696
expect python_executed false
expect rust_executed false
expect disposable_oracle_executed false
expect parity_open false
expect claim_ready false
expect global_cutover_complete false

expect change_class ENTRYPOINT_INPUT_ROBUSTNESS
expect semantics_module_changed false
expect_hash predecessor_manifest
[[ "$(value source_sha256)" == "$(sed -n 's/^source_sha256=//p' "$ROOT/$(value predecessor_manifest_path)")" ]] ||
  fail 'semantic module differs from predecessor'
for key in garden source entrypoint build_script selftest freeze_selftest \
  first_manifest first_evidence frozen_evidence parent_9046_freeze \
  parent_9047_freeze toolchain_wrapper toolchain_compiler; do expect_hash "$key"; done

expect_commit "$(value garden_commit)" garden
source_commit="$(value sounio_executable_commit)"
for key in source entrypoint build_script selftest; do expect_commit "$source_commit" "$key"; done
first_commit="$(value first_receipt_commit)"
for key in first_manifest first_evidence; do expect_commit "$first_commit" "$key"; done

[[ "$(cat "$ROOT/$(value source_path)" "$ROOT/$(value entrypoint_path)" | sha256sum | cut -d' ' -f1)" == \
  "$(value semantics_sha256)" ]] || fail 'semantics drifted'
grep -qx 'action=9046' "$ROOT/$(value parent_9046_freeze_path)" || fail 'parent 9046 drifted'
grep -qx 'action=9047' "$ROOT/$(value parent_9047_freeze_path)" || fail 'parent 9047 drifted'

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-generation-pin-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for n in one two; do
  SOUNIO_LOOM_GENERATION_PINNED_CUTOVER_OUTPUT="$work/$n" \
    bash "$ROOT/scripts/dev/build_sounio_loom_generation_pinned_cutover.sh" >/dev/null
done
cmp "$work/one" "$work/two" || fail 'Sounio rebuild nondeterministic'
[[ "$(sha256sum "$work/one" | cut -d' ' -f1)" == "$(value executable_sha256)" ]] ||
  fail 'executable drifted'

allow() {
  local observed
  observed="$(printf '%s\n' "$1" | "$work/one")"
  [[ "$observed" == "$2" ]] || fail "ALLOW diverged: $observed"
}
deny() {
  local observed rc
  set +e; observed="$(printf '%s\n' "$1" | "$work/one")"; rc=$?; set -e
  [[ $rc -eq 42 && "$observed" == "$2" ]] || fail "DENY diverged: $observed"
}
allow '9048 1 3 33546239 1 2 3 4 5 14 14' 'SOUNIO_GENERATION_PINNED_CUTOVER SEALED_CAPABILITY semantic_authority=Sounio action=9048'
allow '9048 2 3 33546239 1 2 3 4 5 14 14' 'SOUNIO_GENERATION_PINNED_CUTOVER SEALED_LEGACY semantic_authority=Sounio action=9048'
allow '9048 3 3 33550335 1 2 3 4 5 14 14' 'SOUNIO_GENERATION_PINNED_CUTOVER CONTINUE semantic_authority=Sounio action=9048'
allow '9048 4 3 33550335 1 2 3 4 5 14 14' 'SOUNIO_GENERATION_PINNED_CUTOVER FORWARD semantic_authority=Sounio action=9048'
allow '9048 5 3 33021951 1 2 3 4 5 14 14' 'SOUNIO_GENERATION_PINNED_CUTOVER BIRTH_PINNED semantic_authority=Sounio action=9048'
allow '9048 6 3 33554431 1 2 3 4 5 14 14' 'SOUNIO_GENERATION_PINNED_CUTOVER CUTOVER_READY semantic_authority=Sounio action=9048'
deny '9048 1 3 33480703 1 2 3 4 5 14 14' 'SOUNIO_GENERATION_PINNED_CUTOVER DENY696 semantic_authority=Sounio action=9048'
deny '9048 3 3 29356031 1 2 3 4 5 14 14' 'SOUNIO_GENERATION_PINNED_CUTOVER DENY698 semantic_authority=Sounio action=9048'

printf 'sounio-loom-generation-pinned-cutover-freeze-selftest: PASS semantic_authority=Sounio action=9048 stage=SEMANTICS_FROZEN cases=16 deterministic=true python_oracle_attempt=PRE_EXEC_REFUSED python_executed=false rust_executed=false disposable_oracle_executed=false parity_open=false claim_ready=false global_cutover_complete=false\n'
