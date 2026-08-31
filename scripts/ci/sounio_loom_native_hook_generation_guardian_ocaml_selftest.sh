#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/native-hook-generation-guardian-ocaml.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
LIVE_CURRENT="$ROOT_DIR/.git"
FORBIDDEN_BIN="$TEST_ROOT/forbidden-bin"
FORBIDDEN_LOG="$TEST_ROOT/forbidden-exec.log"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-native-hook-generation-guardian-ocaml-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_native_hook_generation_drain.sh" >/dev/null

common_dir="$(git -C "$ROOT_DIR" rev-parse --git-common-dir)"
if [[ "$common_dir" != /* ]]; then
  common_dir="$ROOT_DIR/$common_dir"
fi
common_dir="$(cd "$common_dir" && pwd -P)"
LIVE_CURRENT="$common_dir/sounio-coord-runtime/current"
current_before="$(readlink -f "$LIVE_CURRENT")"

mkdir -p "$FORBIDDEN_BIN"
for executable in python python3 rustc cargo; do
  printf '%s\n' '#!/bin/sh' \
    'printf "%s\n" "$0" >> "$SOUNIO_FORBIDDEN_EXEC_LOG"' \
    'exit 97' > "$FORBIDDEN_BIN/$executable"
  chmod 700 "$FORBIDDEN_BIN/$executable"
done
export PATH="$FORBIDDEN_BIN:$PATH"
export SOUNIO_FORBIDDEN_EXEC_LOG="$FORBIDDEN_LOG"

guardian() {
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR="$STATE_DIR" \
    "$LOOM" hook-generation-guardian --cwd "$ROOT_DIR" "$@"
}

drain() {
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR="$STATE_DIR" \
    "$LOOM" hook-generation-drain-snapshot --cwd "$ROOT_DIR"
}

plan_output="$(guardian)"
[[ "$plan_output" == *'"state":"PLAN_READY"'* && \
  "$plan_output" == *'"applied":false'* && \
  "$plan_output" == *'"semantic_authority":"Sounio"'* ]] ||
  fail "dry run was not plan-only: $plan_output"
[[ ! -e "$STATE_DIR/final-config.v1" && \
  ! -e "$STATE_DIR/rollback-pair-tested.v1" && \
  ! -e "$STATE_DIR/guardian-ed25519-key.v1" ]] ||
  fail 'dry run wrote guardian markers'

apply_output="$(guardian --apply)"
[[ "$apply_output" == *'"state":"PREPARED"'* && \
  "$apply_output" == *'"applied":true'* && \
  "$apply_output" == *'"live_runtime_unchanged":true'* ]] ||
  fail "guardian apply did not prepare receipts: $apply_output"
[[ -f "$STATE_DIR/final-config.v1" && \
  -f "$STATE_DIR/rollback-pair-tested.v1" && \
  -f "$STATE_DIR/guardian-ed25519-private.pem" && \
  -f "$STATE_DIR/guardian-ed25519-public.pem" && \
  -f "$STATE_DIR/guardian-ed25519-key.v1" ]] ||
  fail 'guardian markers are absent'
[[ "$(stat -c %a "$STATE_DIR/final-config.v1")" == 600 && \
  "$(stat -c %a "$STATE_DIR/rollback-pair-tested.v1")" == 600 ]] ||
  fail 'guardian markers are not private regular files'
[[ "$(stat -c %a "$STATE_DIR/guardian-ed25519-private.pem")" == 600 && \
  "$(stat -c %a "$STATE_DIR/guardian-ed25519-public.pem")" == 600 && \
  "$(stat -c %a "$STATE_DIR/guardian-ed25519-key.v1")" == 600 ]] ||
  fail 'guardian signing material is not private'
grep -Fxq 'schema=loom-native-hook-final-config-v1' "$STATE_DIR/final-config.v1" ||
  fail 'final config marker schema is absent'
grep -Fxq 'state=FINAL_CONFIG_BOUND' "$STATE_DIR/final-config.v1" ||
  fail 'final config marker state is absent'
grep -Fxq 'schema=loom-native-hook-rollback-pair-v1' \
  "$STATE_DIR/rollback-pair-tested.v1" || fail 'rollback marker schema is absent'
grep -Fxq 'forward_result=PASS' "$STATE_DIR/rollback-pair-tested.v1" ||
  fail 'forward probe did not pass'
grep -Fxq 'rollback_result=PASS' "$STATE_DIR/rollback-pair-tested.v1" ||
  fail 'rollback probe did not pass'

current_after="$(readlink -f "$LIVE_CURRENT")"
[[ "$current_after" == "$current_before" ]] ||
  fail "live runtime changed during isolated proof: before=$current_before after=$current_after"

set +e
drain_output="$(drain 2>&1)"
drain_rc=$?
set -e
[[ "$drain_rc" -eq 42 && \
  "$drain_output" == *'"authority_observed":true'* && \
  "$drain_output" == *'"candidate_config_bound":true'* && \
  "$drain_output" == *'"final_config_bound":true'* && \
  "$drain_output" == *'"rollback_pair_tested":true'* && \
  "$drain_output" == *'"ui_attestation":{"schema":"loom-native-hook-ui-attestation-v1"'* && \
  "$drain_output" == *'"verified":true'* && \
  "$drain_output" == *'"same_uid_peer_isolation":false'* && \
  "$drain_output" != *'"decision":"DENY672"'* ]] ||
  fail "prepared markers did not advance Sounio 9046: $drain_output"

cp "$STATE_DIR/final-config.v1" "$TEST_ROOT/final-config.good"
printf '%s\n' \
  'schema=loom-native-hook-final-config-v1' \
  'state=FINAL_CONFIG_BOUND' \
  'runtime_id=tampered-runtime' \
  'runtime_manifest_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' \
  'config_bundle_sha256=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' \
  'semantic_authority=Sounio' \
  'action=9046' > "$STATE_DIR/final-config.v1"
set +e
tampered_final="$(drain 2>&1)"
tampered_final_rc=$?
set -e
[[ "$tampered_final_rc" -eq 42 && \
  "$tampered_final" == *'"final_config_bound":false'* && \
  "$tampered_final" == *'"decision":"DENY672"'* ]] ||
  fail "tampered final marker did not close Sounio 9046: $tampered_final"
cp "$TEST_ROOT/final-config.good" "$STATE_DIR/final-config.v1"

cp "$STATE_DIR/guardian-ed25519-public.pem" "$TEST_ROOT/guardian-public.good"
printf '\n' >> "$STATE_DIR/guardian-ed25519-public.pem"
set +e
tampered_key="$(drain 2>&1)"
tampered_key_rc=$?
set -e
[[ "$tampered_key_rc" -eq 42 && \
  "$tampered_key" == *'"decision":"FAIL_CLOSED"'* && \
  "$tampered_key" == *'guardian-key-manifest-drift'* ]] ||
  fail "tampered guardian key did not fail closed: $tampered_key"
cp "$TEST_ROOT/guardian-public.good" "$STATE_DIR/guardian-ed25519-public.pem"

cp "$STATE_DIR/rollback-pair-tested.v1" "$TEST_ROOT/rollback.good"
sed 's/^rollback_result=PASS$/rollback_result=FAIL/' \
  "$TEST_ROOT/rollback.good" > "$STATE_DIR/rollback-pair-tested.v1"
set +e
tampered_rollback="$(drain 2>&1)"
tampered_rollback_rc=$?
set -e
[[ "$tampered_rollback_rc" -eq 42 && \
  "$tampered_rollback" == *'"rollback_pair_tested":false'* && \
  "$tampered_rollback" != *'"decision":"CUTOVER_READY"'* ]] ||
  fail "tampered rollback marker did not close Sounio 9046: $tampered_rollback"

[[ "$(readlink -f "$LIVE_CURRENT")" == "$current_before" ]] ||
  fail 'live runtime changed after sabotage controls'
[[ ! -e "$FORBIDDEN_LOG" ]] ||
  fail "forbidden Python or Rust executable ran: $(tr '\n' ' ' < "$FORBIDDEN_LOG")"

printf '%s\n' \
  'sounio-loom-native-hook-generation-guardian-ocaml-selftest: PASS semantic_authority=Sounio operational_realization=OCaml plan=no-write isolated_sequence=old-candidate-old atomic_markers=true live_runtime=unchanged final_binding=true rollback_binding=true ui_attestation=ed25519-verified same_uid_peer_isolation=false final_tamper=DENY672 key_tamper=fail_closed rollback_tamper=refused forbidden_python_rust_exec=absent'
