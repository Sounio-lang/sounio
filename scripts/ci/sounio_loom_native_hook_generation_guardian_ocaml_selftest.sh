#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/native-hook-generation-guardian-ocaml.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
ROTATION_STATE_DIR="$TEST_ROOT/rotation-state"
TAMPER_STATE_DIR="$TEST_ROOT/tamper-state"
ARCHIVE_TAMPER_STATE_DIR="$TEST_ROOT/archive-tamper-state"
CANDIDATE_ONE="$TEST_ROOT/candidate-one"
CANDIDATE_TWO="$TEST_ROOT/candidate-two"
ROTATION_CANARY_ROOT="$TEST_ROOT/rotation-canary"
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

sha256_file() {
  local digest remainder
  read -r digest remainder < <(sha256sum "$1")
  printf '%s\n' "$digest"
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

write_candidate() {
  local directory="$1" runtime_id="$2"
  local loom_sha256 codex_sha256 claude_sha256 cursor_sha256 grok_sha256
  mkdir -p "$directory/bin"
  cp "$LOOM" "$directory/bin/sounio-loom-runtime"
  chmod 700 "$directory/bin/sounio-loom-runtime"
  loom_sha256="$(sha256_file "$directory/bin/sounio-loom-runtime")"
  codex_sha256="$(sha256_file "$ROOT_DIR/.codex/hooks.json")"
  claude_sha256="$(sha256_file "$ROOT_DIR/.claude/settings.json")"
  cursor_sha256="$(sha256_file "$ROOT_DIR/.cursor/hooks.json")"
  grok_sha256="$(sha256_file "$ROOT_DIR/.grok/hooks/loom-native.json")"
  printf '%s\n' \
    'schema=sounio-coord-runtime-manifest-v1' \
    'protocol_major=3' \
    "runtime_id=$runtime_id" \
    "loom_runtime_sha256=$loom_sha256" \
    'loom_native_hook_cutover_python_bridge_absent=true' \
    "loom_native_hook_cutover_codex_config_sha256=$codex_sha256" \
    "loom_native_hook_cutover_claude_config_sha256=$claude_sha256" \
    "loom_native_hook_cutover_cursor_config_sha256=$cursor_sha256" \
    "loom_native_hook_cutover_grok_config_sha256=$grok_sha256" \
    > "$directory/manifest"
}

rotation_guardian() {
  local state_directory="$1" candidate="$2"
  shift 2
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_NATIVE_HOOK_CANDIDATE="$candidate" \
    SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR="$state_directory" \
    "$LOOM" hook-generation-guardian --cwd "$ROOT_DIR" "$@"
}

write_rotation_canary_fixture() {
  local candidate="$1"
  local common="$ROTATION_CANARY_ROOT/.git"
  local decisions="$common/sounio-loom-language-authority/agent-hook.tsv"
  local lifecycle="$common/sounio-coord-state/hook-session-lifecycle/events.tsv"
  local loom_sha256 codex_sha256
  loom_sha256="$(sha256_file "$candidate/bin/sounio-loom-runtime")"
  codex_sha256="$(sha256_file "$ROOT_DIR/.codex/hooks.json")"
  git init -q "$ROTATION_CANARY_ROOT"
  mkdir -p "$(dirname "$decisions")" "$(dirname "$lifecycle")"
  : > "$decisions"
  for event in SessionStart UserPromptSubmit Stop; do
    printf '%b\n' \
      "schema=loom-agent-hook-receipt-v1\tdecision=ALLOW\tprovider=codex\tevent=$event\tsemantics_sha256=27c5fd758d161026c5c41d0cd0be0f1aa90bd4e3f4287da3c60fb748d1334882\tsemantic_authority_language=Sounio\tsemantic_authority_role=SEMANTIC_AUTHORITY\ttoolchain_sha256=$loom_sha256\tprovider_config_sha256=$codex_sha256\tresult=SOUNIO_NATIVE_HOOK_CUTOVER HOOK_EVENT_ADMIT semantic_authority=Sounio action=9045" \
      >> "$decisions"
  done
  printf '%b\n' \
    'schema=loom-hook-session-lifecycle-v1\tagent=codex\taction=PROCESS_EXIT_CLOSED' \
    > "$lifecycle"
  printf '%s\n' 'LOOM ROTATION CANARY CODEX' > "$ROTATION_CANARY_ROOT/provider.out"
}

issue_rotation_canary() {
  local state_directory="$1" candidate="$2"
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_NATIVE_HOOK_CANDIDATE="$candidate" \
    SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR="$state_directory" \
    "$LOOM" hook-generation-canary --cwd "$ROOT_DIR" --provider codex \
      --canary-root "$ROTATION_CANARY_ROOT" \
      --output "$ROTATION_CANARY_ROOT/provider.out" \
      --expect 'LOOM ROTATION CANARY CODEX' --apply
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

write_candidate "$CANDIDATE_ONE" fixture-native-candidate-one
write_candidate "$CANDIDATE_TWO" fixture-native-candidate-two
write_rotation_canary_fixture "$CANDIDATE_ONE"

rotation_first="$(rotation_guardian "$ROTATION_STATE_DIR" "$CANDIDATE_ONE" --apply)"
[[ "$rotation_first" == *'"state":"PREPARED"'* && \
  "$rotation_first" == *'"prior_generation_archived":false'* ]] ||
  fail "first rotation generation was not prepared: $rotation_first"
rotation_canary="$(issue_rotation_canary "$ROTATION_STATE_DIR" "$CANDIDATE_ONE")"
[[ "$rotation_canary" == *'"state":"RECORDED"'* ]] ||
  fail "rotation canary was not recorded: $rotation_canary"
rotation_final_before="$(sha256_file "$ROTATION_STATE_DIR/final-config.v1")"
rotation_plan="$(rotation_guardian "$ROTATION_STATE_DIR" "$CANDIDATE_TWO")"
[[ "$rotation_plan" == *'"state":"PLAN_READY"'* && \
  "$rotation_plan" == *'"rotation_required":true'* && \
  "$(sha256_file "$ROTATION_STATE_DIR/final-config.v1")" == "$rotation_final_before" ]] ||
  fail "rotation dry run mutated or missed the prior generation: $rotation_plan"
rotation_apply="$(rotation_guardian "$ROTATION_STATE_DIR" "$CANDIDATE_TWO" --apply)"
[[ "$rotation_apply" == *'"state":"PREPARED"'* && \
  "$rotation_apply" == *'"prior_generation_archived":true'* ]] ||
  fail "successor generation did not archive its predecessor: $rotation_apply"
mapfile -t rotation_archives < <(find "$ROTATION_STATE_DIR/archives" -mindepth 1 \
  -maxdepth 1 -type d -print)
[[ "${#rotation_archives[@]}" -eq 1 && \
  -f "${rotation_archives[0]}/generation-archive.v1" && \
  -f "${rotation_archives[0]}/final-config.v1" && \
  -f "${rotation_archives[0]}/rollback-pair-tested.v1" && \
  -f "${rotation_archives[0]}/canaries/codex.canary.v1" && \
  ! -e "$ROTATION_STATE_DIR/canaries/codex.canary.v1" ]] ||
  fail 'prior generation archive is incomplete or its active canary survived'
grep -Fxq 'state=ARCHIVED' "${rotation_archives[0]}/generation-archive.v1" ||
  fail 'generation archive receipt is not bound'
grep -Fxq 'runtime_id=fixture-native-candidate-two' \
  "$ROTATION_STATE_DIR/final-config.v1" ||
  fail 'successor final marker was not installed'

cp "${rotation_archives[0]}/final-config.v1" \
  "$ROTATION_STATE_DIR/final-config.v1"
cp "${rotation_archives[0]}/rollback-pair-tested.v1" \
  "$ROTATION_STATE_DIR/rollback-pair-tested.v1"
cp "${rotation_archives[0]}/canaries/codex.canary.v1" \
  "$ROTATION_STATE_DIR/canaries/codex.canary.v1"
rotation_retry="$(rotation_guardian "$ROTATION_STATE_DIR" "$CANDIDATE_TWO" --apply)"
[[ "$rotation_retry" == *'"state":"PREPARED"'* && \
  "$rotation_retry" == *'"prior_generation_archived":true'* && \
  ! -e "$ROTATION_STATE_DIR/canaries/codex.canary.v1" && \
  "$(find "$ROTATION_STATE_DIR/archives" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1 ]] ||
  fail "committed archive retry was not idempotent: $rotation_retry"

cp "${rotation_archives[0]}/final-config.v1" \
  "$ROTATION_STATE_DIR/final-config.v1"
rm -f "$ROTATION_STATE_DIR/rollback-pair-tested.v1" \
  "$ROTATION_STATE_DIR/canaries/codex.canary.v1"
rotation_partial_cleanup_retry="$(rotation_guardian "$ROTATION_STATE_DIR" \
  "$CANDIDATE_TWO" --apply)"
[[ "$rotation_partial_cleanup_retry" == *'"state":"PREPARED"'* && \
  "$rotation_partial_cleanup_retry" == *'"prior_generation_archived":true'* && \
  -f "$ROTATION_STATE_DIR/rollback-pair-tested.v1" && \
  "$(find "$ROTATION_STATE_DIR/archives" -mindepth 1 -maxdepth 1 -type d | wc -l)" -eq 1 ]] ||
  fail "partially cleaned archive retry was not idempotent: $rotation_partial_cleanup_retry"

cp -a "$ROTATION_STATE_DIR" "$ARCHIVE_TAMPER_STATE_DIR"
archive_tamper_dir="$(find "$ARCHIVE_TAMPER_STATE_DIR/archives" -mindepth 1 \
  -maxdepth 1 -type d -print -quit)"
cp "$archive_tamper_dir/final-config.v1" \
  "$ARCHIVE_TAMPER_STATE_DIR/final-config.v1"
cp "$archive_tamper_dir/rollback-pair-tested.v1" \
  "$ARCHIVE_TAMPER_STATE_DIR/rollback-pair-tested.v1"
mkdir -p "$ARCHIVE_TAMPER_STATE_DIR/canaries"
cp "$archive_tamper_dir/canaries/codex.canary.v1" \
  "$ARCHIVE_TAMPER_STATE_DIR/canaries/codex.canary.v1"
archive_signature_character="$(sed -n 's/^signature_base64=\(.\).*/\1/p' \
  "$archive_tamper_dir/generation-archive.v1")"
archive_replacement=A
[[ "$archive_signature_character" == A ]] && archive_replacement=B
sed -i "s/^signature_base64=./signature_base64=$archive_replacement/" \
  "$archive_tamper_dir/generation-archive.v1"
set +e
archive_tamper="$(rotation_guardian "$ARCHIVE_TAMPER_STATE_DIR" \
  "$CANDIDATE_TWO" --apply 2>&1)"
archive_tamper_rc=$?
set -e
[[ "$archive_tamper_rc" -eq 42 && \
  "$archive_tamper" == *'guardian-generation-archive-signature-invalid'* ]] ||
  fail "archive collision sabotage did not fail closed: $archive_tamper"
grep -Fxq 'runtime_id=fixture-native-candidate-one' \
  "$ARCHIVE_TAMPER_STATE_DIR/final-config.v1" ||
  fail 'archive collision sabotage mutated active predecessor state'

tamper_first="$(rotation_guardian "$TAMPER_STATE_DIR" "$CANDIDATE_ONE" --apply)"
[[ "$tamper_first" == *'"state":"PREPARED"'* ]] ||
  fail "tamper fixture generation was not prepared: $tamper_first"
issue_rotation_canary "$TAMPER_STATE_DIR" "$CANDIDATE_ONE" >/dev/null
first_signature_character="$(sed -n 's/^signature_base64=\(.\).*/\1/p' \
  "$TAMPER_STATE_DIR/canaries/codex.canary.v1")"
replacement=A
[[ "$first_signature_character" == A ]] && replacement=B
sed -i "s/^signature_base64=./signature_base64=$replacement/" \
  "$TAMPER_STATE_DIR/canaries/codex.canary.v1"
set +e
tamper_rotation="$(rotation_guardian "$TAMPER_STATE_DIR" "$CANDIDATE_TWO" --apply 2>&1)"
tamper_rotation_rc=$?
set -e
[[ "$tamper_rotation_rc" -eq 42 && \
  "$tamper_rotation" == *'canary-receipt-signature-invalid:codex'* && \
  ! -e "$TAMPER_STATE_DIR/archives" ]] ||
  fail "tampered predecessor was not preserved fail-closed: $tamper_rotation"
grep -Fxq 'runtime_id=fixture-native-candidate-one' \
  "$TAMPER_STATE_DIR/final-config.v1" ||
  fail 'tampered predecessor marker changed during refused rotation'

[[ "$(readlink -f "$LIVE_CURRENT")" == "$current_before" ]] ||
  fail 'live runtime changed after sabotage controls'
[[ ! -e "$FORBIDDEN_LOG" ]] ||
  fail "forbidden Python or Rust executable ran: $(tr '\n' ' ' < "$FORBIDDEN_LOG")"

printf '%s\n' \
  'sounio-loom-native-hook-generation-guardian-ocaml-selftest: PASS semantic_authority=Sounio operational_realization=OCaml plan=no-write isolated_sequence=old-candidate-old atomic_markers=true live_runtime=unchanged final_binding=true rollback_binding=true generation_rotation=signed-archive successor_prepare=true rotation_retry=crash-idempotent archive_collision_tamper=fail_closed tampered_predecessor=fail_closed ui_attestation=ed25519-verified same_uid_peer_isolation=false final_tamper=DENY672 key_tamper=fail_closed rollback_tamper=refused forbidden_python_rust_exec=absent'
