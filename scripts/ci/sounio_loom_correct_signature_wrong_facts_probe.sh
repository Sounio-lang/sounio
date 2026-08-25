#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CANARY="$ROOT_DIR/scripts/ci/sounio_loom_pod_replay_canary.sh"
LOOM="$ROOT_DIR/bin/sounio-loom"
ADAPTER="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-wrong-facts.XXXXXX")"
CANARY_ROOT="$TEST_ROOT/canary"
KEY_ROOT="$TEST_ROOT/keys"

fail() {
  printf 'sounio-loom-wrong-facts-probe: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

kill_generation() {
  local status pid bridge_file bridge_pid
  status="$(
    SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" status \
      --state-dir "$CANARY_ROOT/loom" --cwd "$ROOT_DIR" \
      --agent beagle-workbench \
      --lane pane-6c6f6f6d2d706f642d7265706c61793a7465726d696e616c \
      2>/dev/null || true
  )"
  for pid in "$(field daemon_pid "$status")" "$(field guardian_pid "$status")" \
    "$(field harness_pid "$status")"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -9 "$pid" 2>/dev/null || true
  done
  for bridge_file in "$CANARY_ROOT"/bridge-*.pid; do
    [[ -f "$bridge_file" ]] || continue
    bridge_pid="$(cat "$bridge_file")"
    [[ "$bridge_pid" =~ ^[1-9][0-9]*$ ]] && kill -9 "$bridge_pid" 2>/dev/null || true
  done
}

cleanup() {
  kill_generation || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

run_phase() {
  local uid="$1" phase="$2"
  env \
    POD_UID="$uid" \
    POD_NAME=loom-wrong-facts-0 \
    SOUNIO_CANARY_SOURCE_ROOT="$ROOT_DIR" \
    SOUNIO_LOOM_POD_CANARY_ROOT="$CANARY_ROOT" \
    SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS=1 \
    SOUNIO_LOOM_REQUIRE_INDEPENDENT_OBSERVER=1 \
    SOUNIO_LOOM_SIGNING_KEY="$KEY_ROOT/private.pem" \
    SOUNIO_LOOM_VERIFY_KEY="$KEY_ROOT/public.pem" \
    SOUNIO_LOOM_OBSERVER_VERIFY_KEY="$KEY_ROOT/observer-public.pem" \
    bash "$CANARY" "$phase"
}

receipt_path() {
  local matches=()
  mapfile -t matches < <(
    find "$CANARY_ROOT/loom" -name sounio-continuity.receipt -type f -print
  )
  [[ "${#matches[@]}" -eq 1 ]] || \
    fail "expected one predecessor receipt, found ${#matches[@]}"
  printf '%s\n' "${matches[0]}"
}

key_value() {
  local key="$1" path="$2" value
  value="$(sed -n "s/^${key}=//p" "$path")"
  [[ -n "$value" && "$value" != *$'\n'* ]] || \
    fail "$path must contain exactly one $key"
  printf '%s\n' "$value"
}

command -v openssl >/dev/null || fail 'OpenSSL is required'
mkdir -p "$KEY_ROOT"
openssl genpkey -algorithm ED25519 -out "$KEY_ROOT/private.pem"
openssl pkey -in "$KEY_ROOT/private.pem" -pubout -out "$KEY_ROOT/public.pem"
openssl genpkey -algorithm ED25519 -out "$KEY_ROOT/observer-private.pem"
openssl pkey -in "$KEY_ROOT/observer-private.pem" -pubout \
  -out "$KEY_ROOT/observer-public.pem"
"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

phase_one="$(run_phase wrong-facts-pod-one phase-one)"
[[ "$phase_one" == *'CANARY_PHASE_ONE'* ]] || fail 'signed predecessor did not start'
receipt="$(receipt_path)"
original_receipt_sha256="$(sha256sum "$receipt" | awk '{print $1}')"
attestation="$(dirname "$receipt")/sounio-continuity.observer-attestation"
SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" attest-continuity-receipt \
  --receipt "$receipt" --subject-public-key "$KEY_ROOT/public.pem" \
  --observer-private-key "$KEY_ROOT/observer-private.pem" \
  --observer-public-key "$KEY_ROOT/observer-public.pem" \
  --out "$attestation" --adapter "$ADAPTER" >/dev/null
[[ -s "$attestation" ]] || fail 'independent predecessor commitment was not created'
facts="$(key_value facts "$receipt")"
original_semantic_head="$(awk '{print $5}' <<< "$facts")"
forged_facts="$(awk '{$5="777777777777777777"; print}' <<< "$facts")"
forged_semantic_head="$(awk '{print $5}' <<< "$forged_facts")"
[[ "$forged_semantic_head" != "$original_semantic_head" ]] || \
  fail 'semantic-head intervention did not change the predecessor fact'

key_id="$(sha256sum "$KEY_ROOT/public.pem" | awk '{print $1}')"
adapter_sha256="$(sha256sum "$ADAPTER" | awk '{print $1}')"
facts_sha256="$(printf '%s\n' "$forged_facts" | sha256sum | awk '{print $1}')"
verdict='SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v2 authenticity=ed25519'
payload="$TEST_ROOT/payload.txt"
signature="$TEST_ROOT/signature.bin"
{
  printf 'schema=loom-native-continuity-signed-payload-v1\n'
  printf 'algorithm=ed25519\n'
  printf 'key_id=%s\n' "$key_id"
  printf 'adapter_sha256=%s\n' "$adapter_sha256"
  printf 'facts_sha256=%s\n' "$facts_sha256"
  printf 'facts=%s\n' "$forged_facts"
  printf 'verdict=%s\n' "$verdict"
} > "$payload"
payload_sha256="$(sha256sum "$payload" | awk '{print $1}')"
openssl pkeyutl -sign -rawin -inkey "$KEY_ROOT/private.pem" \
  -in "$payload" -out "$signature"
signature_base64="$(openssl base64 -A -in "$signature")"
{
  printf 'schema=loom-native-continuity-receipt-v2\n'
  printf 'algorithm=ed25519\n'
  printf 'key_id=%s\n' "$key_id"
  printf 'adapter_sha256=%s\n' "$adapter_sha256"
  printf 'facts_sha256=%s\n' "$facts_sha256"
  printf 'facts=%s\n' "$forged_facts"
  printf 'verdict=%s\n' "$verdict"
  printf 'signed_payload_sha256=%s\n' "$payload_sha256"
  printf 'signature_base64=%s\n' "$signature_base64"
} > "$receipt"
forged_receipt_sha256="$(sha256sum "$receipt" | awk '{print $1}')"

verification="$(
  SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" verify-continuity-receipt \
    --receipt "$receipt" --public-key "$KEY_ROOT/public.pem" --adapter "$ADAPTER"
)"
[[ "$verification" == LOOM_CONTINUITY_RECEIPT_VERIFIED* ]] || \
  fail "legitimate signature did not verify: $verification"

kill_generation
generation_count_before="$(find "$CANARY_ROOT/loom" \
  -name sounio-continuity.receipt -type f -print | wc -l)"
set +e
phase_two="$(run_phase wrong-facts-pod-two phase-two 2>&1)"
phase_two_rc=$?
set -e
[[ "$phase_two_rc" -ne 0 && \
   "$phase_two" == *'sounio-continuity-independent-observation-mismatch'* ]] || \
  fail "pre-spawn independent observation did not catch the semantic forgery: rc=$phase_two_rc output=$phase_two"
generation_count_after="$(find "$CANARY_ROOT/loom" \
  -name sounio-continuity.receipt -type f -print | wc -l)"
[[ "$generation_count_after" -eq "$generation_count_before" ]] || \
  fail 'wrong-facts refusal created a successor generation'

printf '%s\n' 'SOUNIO_LOOM_CORRECT_SIGNATURE_WRONG_FACTS_FALSIFIER=REFUSED_PRESPAWN'
printf 'signature=valid independent_observation=receipt-digest-mismatch successor_created=0\n'
printf 'pre_spawn_control=refused_before-successor-creation\n'
printf 'wrong_fact=predecessor_semantic_head original=%s forged=%s\n' \
  "$original_semantic_head" "$forged_semantic_head"
printf 'original_receipt_sha256=%s forged_receipt_sha256=%s\n' \
  "$original_receipt_sha256" "$forged_receipt_sha256"
printf 'scope=post-observation-faulty-keyholder bounded_precommitted-observation-claim=supported signer-correctness-claim=unsupported\n'
