#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/native-hook-generation-canary-ocaml.XXXXXX")"
STATE_DIR="$TEST_ROOT/state"
CANDIDATE_DIR="$TEST_ROOT/candidate"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
FORBIDDEN_BIN="$TEST_ROOT/forbidden-bin"
FORBIDDEN_LOG="$TEST_ROOT/forbidden-exec.log"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-native-hook-generation-canary-ocaml-selftest: FAIL: %s\n' "$*" >&2
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
[[ -x "$LOOM" ]] || fail 'OCaml Loom executable is absent'

common_dir="$(git -C "$ROOT_DIR" rev-parse --git-common-dir)"
if [[ "$common_dir" != /* ]]; then
  common_dir="$ROOT_DIR/$common_dir"
fi
common_dir="$(cd "$common_dir" && pwd -P)"
live_current="$common_dir/sounio-coord-runtime/current"
current_before="$(readlink -f "$live_current")"

mkdir -p "$FORBIDDEN_BIN"
for executable in python python3 rustc cargo; do
  printf '%s\n' '#!/bin/sh' \
    'printf "%s\n" "$0" >> "$SOUNIO_FORBIDDEN_EXEC_LOG"' \
    'exit 97' > "$FORBIDDEN_BIN/$executable"
  chmod 700 "$FORBIDDEN_BIN/$executable"
done
export PATH="$FORBIDDEN_BIN:$PATH"
export SOUNIO_FORBIDDEN_EXEC_LOG="$FORBIDDEN_LOG"

mkdir -p "$CANDIDATE_DIR/bin" "$STATE_DIR"
cp "$LOOM" "$CANDIDATE_DIR/bin/sounio-loom-runtime"
chmod 700 "$CANDIDATE_DIR/bin/sounio-loom-runtime"
loom_sha256="$(sha256_file "$CANDIDATE_DIR/bin/sounio-loom-runtime")"
codex_config_sha256="$(sha256_file "$ROOT_DIR/.codex/hooks.json")"
claude_config_sha256="$(sha256_file "$ROOT_DIR/.claude/settings.json")"
cursor_config_sha256="$(sha256_file "$ROOT_DIR/.cursor/hooks.json")"
grok_config_sha256="$(sha256_file "$ROOT_DIR/.grok/hooks/loom-native.json")"

printf '%s\n' \
  'schema=sounio-coord-runtime-manifest-v1' \
  'protocol_major=3' \
  'runtime_id=fixture-native-candidate-v1' \
  "loom_runtime_sha256=$loom_sha256" \
  'loom_native_hook_cutover_python_bridge_absent=true' \
  "loom_native_hook_cutover_codex_config_sha256=$codex_config_sha256" \
  "loom_native_hook_cutover_claude_config_sha256=$claude_config_sha256" \
  "loom_native_hook_cutover_cursor_config_sha256=$cursor_config_sha256" \
  "loom_native_hook_cutover_grok_config_sha256=$grok_config_sha256" \
  > "$CANDIDATE_DIR/manifest"

/usr/bin/openssl genpkey -algorithm ED25519 \
  -out "$STATE_DIR/guardian-ed25519-private.pem" >/dev/null 2>&1
/usr/bin/openssl pkey -in "$STATE_DIR/guardian-ed25519-private.pem" -pubout \
  -out "$STATE_DIR/guardian-ed25519-public.pem" >/dev/null 2>&1
/usr/bin/openssl pkey -pubin -in "$STATE_DIR/guardian-ed25519-public.pem" \
  -outform DER -out "$TEST_ROOT/guardian-public.der" >/dev/null 2>&1
private_sha256="$(sha256_file "$STATE_DIR/guardian-ed25519-private.pem")"
public_sha256="$(sha256_file "$STATE_DIR/guardian-ed25519-public.pem")"
key_id="$(sha256_file "$TEST_ROOT/guardian-public.der")"
printf '%s\n' \
  'schema=loom-native-hook-guardian-key-v1' \
  'algorithm=ed25519' \
  "key_id=$key_id" \
  "private_key_sha256=$private_sha256" \
  "public_key_sha256=$public_sha256" \
  'created_utc=2026-09-01T00:00:00Z' \
  > "$STATE_DIR/guardian-ed25519-key.v1"
chmod 600 "$STATE_DIR"/guardian-ed25519-*.pem \
  "$STATE_DIR/guardian-ed25519-key.v1"

provider_config_sha256() {
  case "$1" in
    codex) printf '%s\n' "$codex_config_sha256" ;;
    claude) printf '%s\n' "$claude_config_sha256" ;;
    cursor) printf '%s\n' "$cursor_config_sha256" ;;
    grok) printf '%s\n' "$grok_config_sha256" ;;
    *) fail "unsupported fixture provider: $1" ;;
  esac
}

write_decision() {
  local provider="$1" event="$2" config_sha256="$3" path="$4"
  printf '%b\n' \
    "schema=loom-agent-hook-receipt-v1\tdecision=ALLOW\tprovider=$provider\tevent=$event\tsemantics_sha256=27c5fd758d161026c5c41d0cd0be0f1aa90bd4e3f4287da3c60fb748d1334882\tsemantic_authority_language=Sounio\tsemantic_authority_role=SEMANTIC_AUTHORITY\ttoolchain_sha256=$loom_sha256\tprovider_config_sha256=$config_sha256\tresult=SOUNIO_NATIVE_HOOK_CUTOVER HOOK_EVENT_ADMIT semantic_authority=Sounio action=9045" \
    >> "$path"
}

write_canary_fixture() {
  local provider="$1" expected="$2"
  local root="$TEST_ROOT/canary-$provider"
  local common="$root/.git"
  local decisions="$common/sounio-loom-language-authority/agent-hook.tsv"
  local lifecycle="$common/sounio-coord-state/hook-session-lifecycle/events.tsv"
  local config_sha256
  config_sha256="$(provider_config_sha256 "$provider")"
  git init -q "$root"
  mkdir -p "$(dirname "$decisions")" "$(dirname "$lifecycle")"
  : > "$decisions"
  write_decision "$provider" SessionStart "$config_sha256" "$decisions"
  if [[ "$provider" == codex ]]; then
    write_decision "$provider" UserPromptSubmit "$config_sha256" "$decisions"
    write_decision "$provider" Stop "$config_sha256" "$decisions"
    printf '%b\n' \
      "schema=loom-hook-session-lifecycle-v1\tagent=$provider\taction=PROCESS_EXIT_CLOSED" \
      > "$lifecycle"
  else
    write_decision "$provider" SessionEnd "$config_sha256" "$decisions"
    printf '%b\n' \
      "schema=loom-hook-session-lifecycle-v1\tagent=$provider\taction=CLOSED" \
      > "$lifecycle"
  fi
  printf '%s\n' "$expected" > "$root/provider.out"
}

canary() {
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_NATIVE_HOOK_CANDIDATE="$CANDIDATE_DIR" \
    SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR="$STATE_DIR" \
    "$LOOM" hook-generation-canary --cwd "$ROOT_DIR" "$@"
}

verify_receipts() {
  SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_NATIVE_HOOK_CANDIDATE="$CANDIDATE_DIR" \
    SOUNIO_LOOM_NATIVE_HOOK_DRAIN_STATE_DIR="$STATE_DIR" \
    "$LOOM" hook-generation-canary --verify --cwd "$ROOT_DIR"
}

for provider in codex claude cursor grok; do
  write_canary_fixture "$provider" "LOOM CANARY ${provider^^}"
done

plan_output="$(canary --provider codex \
  --canary-root "$TEST_ROOT/canary-codex" \
  --output "$TEST_ROOT/canary-codex/provider.out" \
  --expect 'LOOM CANARY CODEX')"
[[ "$plan_output" == *'"state":"PLAN_READY"'* && \
  "$plan_output" == *'"applied":false'* && \
  ! -e "$STATE_DIR/canaries/codex.canary.v1" ]] ||
  fail "canary dry run wrote state: $plan_output"

set +e
mismatch_output="$(canary --provider codex \
  --canary-root "$TEST_ROOT/canary-codex" \
  --output "$TEST_ROOT/canary-codex/provider.out" \
  --expect 'WRONG OUTPUT' --apply 2>&1)"
mismatch_rc=$?
set -e
[[ "$mismatch_rc" -eq 42 && \
  "$mismatch_output" == *'canary-expected-output-absent'* && \
  ! -e "$STATE_DIR/canaries/codex.canary.v1" ]] ||
  fail "output mismatch was not refused before receipt creation: $mismatch_output"

for provider in codex claude cursor; do
  apply_output="$(canary --provider "$provider" \
    --canary-root "$TEST_ROOT/canary-$provider" \
    --output "$TEST_ROOT/canary-$provider/provider.out" \
    --expect "LOOM CANARY ${provider^^}" --apply)"
  [[ "$apply_output" == *'"state":"RECORDED"'* && \
    "$apply_output" == *'"same_uid_peer_isolation":false'* ]] ||
    fail "$provider receipt was not recorded: $apply_output"
done

partial_output="$(verify_receipts)"
[[ "$partial_output" == *'"mask":7'* && \
  "$partial_output" == *'"four_provider_complete":false'* ]] ||
  fail "missing Grok receipt did not keep the set incomplete: $partial_output"

grok_output="$(canary --provider grok \
  --canary-root "$TEST_ROOT/canary-grok" \
  --output "$TEST_ROOT/canary-grok/provider.out" \
  --expect 'LOOM CANARY GROK' --apply)"
[[ "$grok_output" == *'"state":"RECORDED"'* ]] ||
  fail "Grok receipt was not recorded: $grok_output"

complete_output="$(verify_receipts)"
[[ "$complete_output" == *'"mask":15'* && \
  "$complete_output" == *'"four_provider_complete":true'* && \
  "$complete_output" == *'"same_uid_peer_isolation":false'* ]] ||
  fail "four-provider receipt set did not verify: $complete_output"

cp "$STATE_DIR/canaries/codex.canary.v1" "$TEST_ROOT/codex.canary.good"
first_signature_character="$(sed -n 's/^signature_base64=\(.\).*/\1/p' \
  "$STATE_DIR/canaries/codex.canary.v1")"
replacement=A
[[ "$first_signature_character" == A ]] && replacement=B
sed -i "s/^signature_base64=./signature_base64=$replacement/" \
  "$STATE_DIR/canaries/codex.canary.v1"
set +e
signature_output="$(verify_receipts 2>&1)"
signature_rc=$?
set -e
[[ "$signature_rc" -eq 42 && \
  "$signature_output" == *'canary-receipt-signature-invalid:codex'* && \
  "$signature_output" == *'"state":"FAIL_CLOSED"'* ]] ||
  fail "signature-only sabotage did not fail closed: $signature_output"
cp "$TEST_ROOT/codex.canary.good" "$STATE_DIR/canaries/codex.canary.v1"

restored_output="$(verify_receipts)"
[[ "$restored_output" == *'"mask":15'* ]] ||
  fail "restored receipt set did not verify: $restored_output"
[[ "$(readlink -f "$live_current")" == "$current_before" ]] ||
  fail 'shared current runtime changed during isolated canary proof'
[[ ! -e "$FORBIDDEN_LOG" ]] ||
  fail "forbidden Python or Rust executable ran: $(tr '\n' ' ' < "$FORBIDDEN_LOG")"

printf '%s\n' \
  'sounio-loom-native-hook-generation-canary-ocaml-selftest: PASS semantic_authority=Sounio operational_realization=OCaml signed_receipts=codex+claude+cursor+grok partial_mask=7 complete_mask=15 output_mismatch=refused signature_sabotage=fail_closed shared_runtime=unchanged same_uid_peer_isolation=false forbidden_python_rust_exec=absent'
