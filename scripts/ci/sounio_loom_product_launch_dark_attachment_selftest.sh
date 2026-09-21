#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_PARENT="$ROOT_DIR/tools/loom/_build"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/product-launch-dark.XXXXXX")"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
PROJECTION="$ROOT_DIR/tools/loom/kernel_peer_activation_capsule.current.v1"
STATE_DIR="$TEST_ROOT/state"
AGENT=product-launch-dark

cleanup() {
  for lane in direct provider-start provider-open recover; do
    "$LOOM" stop --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$lane" \
      --cwd "$TEST_ROOT" >/dev/null 2>&1 || true
  done
  if [[ "${SOUNIO_LOOM_KEEP_TEST_ROOT:-0}" != 1 ]]; then
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-product-launch-dark-attachment-selftest: FAIL: %s test_root=%s\n' \
    "$*" "$TEST_ROOT" >&2
  exit 1
}

projection_code() {
  local label="$1" line rest
  line="$(grep -m1 "^CASE label=${label} EXPECT code=" "$PROJECTION")"
  [[ -n "$line" ]] || fail "Sounio projection omitted $label"
  rest="${line#* EXPECT code=}"
  printf '%s' "${rest%% *}"
}

receipt_field() {
  local line="$1" key="$2" field
  while IFS= read -r field; do
    if [[ "$field" == "$key="* ]]; then
      printf '%s' "${field#*=}"
      return 0
    fi
  done < <(printf '%s\n' "$line" | tr '\t' '\n')
  fail "receipt field is missing: $key"
}

wait_inactive() {
  local lane="$1"
  for _ in $(seq 1 120); do
    if ! "$LOOM" status --state-dir "$STATE_DIR" --agent "$AGENT" \
      --lane "$lane" --cwd "$TEST_ROOT" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.05
  done
  fail "$lane remained active"
}

expect_pre_session_refusal() {
  local tag="$1" lane="$2" marker="$3"
  shift 3
  local output rc
  set +e
  output="$({ "$@"; } 2>&1)"
  rc=$?
  set -e
  [[ "$rc" -eq 1 && "$output" == *"$marker"* ]] ||
    fail "$tag was not refused by $marker: rc=$rc output=$output"
  [[ ! -e "$STATE_DIR/sessions/$AGENT--$lane" ]] ||
    fail "$tag created a session before refusing"
}

SOUNIO_LOOM_KERNEL_PEER_ACTIVATION_CURRENT_OUTPUT="$PROJECTION" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_peer_activation_capsule_current_frame.sh" \
  >/dev/null
(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"

current_code="$(projection_code current_material)"
seal_code="$(projection_code seal)"

mkdir -p "$TEST_ROOT/untrusted-cwd/tools/loom"
printf 'schema=attacker-selected-policy\n' > \
  "$TEST_ROOT/untrusted-cwd/tools/loom/kernel_peer_activation_capsule_authority.freeze.v1"

direct_output="$($LOOM start --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane direct --session-id direct-session --cwd "$TEST_ROOT/untrusted-cwd" \
  -- /usr/bin/tail -f /dev/null)"
[[ "$direct_output" == *"launch_source=start"* && \
   "$direct_output" == *"launch_dark_code=$current_code"* && \
   "$direct_output" == *"authorizing=false"* && \
   "$direct_output" == *"production_activation=false"* ]] ||
  fail "direct start lost its Sounio dark observation: $direct_output"

launch_log="$STATE_DIR/product-launch-dark.tsv"
[[ -f "$launch_log" ]] || fail 'direct start omitted the durable launch receipt'
direct_receipt="$(grep -m1 $'\tlaunch_source=start\t' "$launch_log")"
[[ "$direct_receipt" == *$'\tdecision=DENY\tcode='"$current_code"$'\t'* && \
   "$direct_receipt" == *$'\tproducing_language=Sounio\tlanguage_role=SEMANTIC_AUTHORITY\t'* && \
   "$direct_receipt" == *$'\toperational_language=OCaml\toperational_role=OPERATIONAL_ATTACHMENT\t'* ]] ||
  fail 'direct receipt lost the authority boundary'
direct_digest="$(receipt_field "$direct_receipt" command_sha256)"
descriptor_digest="$(sed -n 's/^argv_digest=//p' \
  "$STATE_DIR/sessions/$AGENT--direct/session.state")"
[[ "$direct_digest" == "$descriptor_digest" ]] ||
  fail 'direct receipt did not bind the launched argv digest'
"$LOOM" stop --state-dir "$STATE_DIR" --agent "$AGENT" --lane direct \
  --cwd "$TEST_ROOT" >/dev/null
wait_inactive direct

printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -euo pipefail' \
  'trap "exit 0" TERM INT HUP' \
  'while :; do sleep 1; done' > "$TEST_ROOT/fake-codex"
chmod +x "$TEST_ROOT/fake-codex"
provider_start_output="$(SOUNIO_LOOM_PROVIDER_CODEX="$TEST_ROOT/fake-codex" \
  "$LOOM" provider-start --provider codex --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane provider-start --session-id provider-start-session \
  --cwd "$TEST_ROOT" --prompt turn-witness)"
[[ "$provider_start_output" == *"launch_source=provider-start"* && \
   "$provider_start_output" == *"LOOM_PROVIDER_STARTED"* ]] ||
  fail "provider-start omitted its launch observation: $provider_start_output"
grep -Fq $'\tlaunch_source=provider-start\tdecision=DENY\tcode='"$current_code"$'\t' \
  "$launch_log" || fail 'provider-start receipt is missing'
"$LOOM" stop --state-dir "$STATE_DIR" --agent "$AGENT" --lane provider-start \
  --cwd "$TEST_ROOT" >/dev/null
wait_inactive provider-start

provider_open_output="$(SOUNIO_LOOM_PROVIDER_CODEX="$TEST_ROOT/fake-codex" \
  "$LOOM" provider-open --provider codex --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane provider-open --session-id provider-open-session \
  --cwd "$TEST_ROOT" --prompt persistent-witness)"
[[ "$provider_open_output" == *"launch_source=provider-open"* && \
   "$provider_open_output" == *"LOOM_PROVIDER_OPENED"* ]] ||
  fail "provider-open omitted its launch observation: $provider_open_output"
grep -Fq $'\tlaunch_source=provider-open\tdecision=DENY\tcode='"$current_code"$'\t' \
  "$launch_log" || fail 'provider-open receipt is missing'
"$LOOM" stop --state-dir "$STATE_DIR" --agent "$AGENT" --lane provider-open \
  --cwd "$TEST_ROOT" >/dev/null
wait_inactive provider-open

recover_start="$($LOOM start --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane recover --session-id recover-session --cwd "$TEST_ROOT" \
  -- /usr/bin/tail -f /dev/null)"
[[ "$recover_start" == *"launch_source=start"* ]] ||
  fail 'recovery fixture did not cross the start observation'
recover_descriptor="$STATE_DIR/sessions/$AGENT--recover/session.state"
recover_digest="$(sed -n 's/^argv_digest=//p' "$recover_descriptor")"
"$LOOM" crash-kernel --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane recover --cwd "$TEST_ROOT" --at now >/dev/null
wait_inactive recover
guardian_during="$($LOOM guardian-status --state-dir "$STATE_DIR" \
  --agent "$AGENT" --lane recover --cwd "$TEST_ROOT")"
[[ "$guardian_during" == *"state=active"* && \
   "$guardian_during" == *"argv_digest=$recover_digest"* ]] ||
  fail 'Guardian did not preserve the recoverable generation identity'
recover_output="$($LOOM recover --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane recover --cwd "$TEST_ROOT")"
[[ "$recover_output" == *"launch_source=recover"* && \
   "$recover_output" == *"launch_dark_code=$current_code"* ]] ||
  fail "recover omitted its launch observation: $recover_output"
recover_receipt="$(grep -m1 $'\tlaunch_source=recover\t' "$launch_log")"
[[ "$(receipt_field "$recover_receipt" command_sha256)" == "$recover_digest" ]] ||
  fail 'recover receipt did not preserve the original argv digest'
"$LOOM" stop --state-dir "$STATE_DIR" --agent "$AGENT" --lane recover \
  --cwd "$TEST_ROOT" >/dev/null
wait_inactive recover

sabotage_log="$TEST_ROOT/sabotage.tsv"
expect_pre_session_refusal sabotage-start sabotage \
  product-launch-dark-unexpected-allow \
  env SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_ACTIVATION_DARK_LABEL=seal \
    SOUNIO_LOOM_PRODUCT_LAUNCH_DARK_LOG="$sabotage_log" \
    "$LOOM" start --state-dir "$STATE_DIR" --agent "$AGENT" --lane sabotage \
    --session-id sabotage-session --cwd "$TEST_ROOT" -- /usr/bin/tail -f /dev/null
grep -Fq $'\tlaunch_source=start\tdecision=ALLOW\tcode='"$seal_code"$'\t' \
  "$sabotage_log" || fail 'causal sabotage did not record the Sounio ALLOW'

expect_pre_session_refusal sabotage-provider sabotage-provider \
  product-launch-dark-unexpected-allow \
  env SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_ACTIVATION_DARK_LABEL=seal \
    SOUNIO_LOOM_PRODUCT_LAUNCH_DARK_LOG="$TEST_ROOT/provider-sabotage.tsv" \
    SOUNIO_LOOM_PROVIDER_CODEX="$TEST_ROOT/fake-codex" \
    "$LOOM" provider-open --provider codex --state-dir "$STATE_DIR" \
    --agent "$AGENT" --lane sabotage-provider \
    --session-id sabotage-provider-session --cwd "$TEST_ROOT" \
    --prompt must-not-launch

tampered_projection="$TEST_ROOT/tampered-projection"
cp "$PROJECTION" "$tampered_projection"
printf '\n' >> "$tampered_projection"
expect_pre_session_refusal projection-tamper projection-tamper \
  activation-dark-projection-hash-mismatch \
  env SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    SOUNIO_LOOM_ACTIVATION_DARK_PROJECTION="$tampered_projection" \
    "$LOOM" start --state-dir "$STATE_DIR" --agent "$AGENT" \
    --lane projection-tamper --session-id projection-tamper-session \
    --cwd "$TEST_ROOT" -- /usr/bin/tail -f /dev/null

set +e
receipt_failure_output="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_PRODUCT_LAUNCH_DARK_LOG="$TEST_ROOT" \
  "$LOOM" start --state-dir "$STATE_DIR" --agent "$AGENT" \
  --lane receipt-failure --session-id receipt-failure-session \
  --cwd "$TEST_ROOT" -- /usr/bin/tail -f /dev/null 2>&1)"
receipt_failure_rc=$?
set -e
[[ "$receipt_failure_rc" -eq 1 && \
   ! -e "$STATE_DIR/sessions/$AGENT--receipt-failure" ]] ||
  fail "receipt failure did not stop before session creation: $receipt_failure_output"

[[ "$(grep -c '^schema=loom-product-launch-dark-decision-v1' "$launch_log")" -eq 5 ]] ||
  fail 'real paths did not emit exactly five launch observations'

printf '%s\n' \
  "sounio-loom-product-launch-dark-attachment-selftest: PASS semantic_authority=Sounio operational_attachment=OCaml action=9031 real_start=DENY${current_code}+CONTINUE provider_start=DENY${current_code}+CONTINUE provider_open=DENY${current_code}+CONTINUE recover=DENY${current_code}+CONTINUE causal_sabotage=ALLOW${seal_code}+NO_SESSION provider_sabotage=ALLOW${seal_code}+NO_SESSION projection_tamper=NO_SESSION receipt_failure=NO_SESSION cwd_policy_injection=IGNORED receipts=hash-bound authorizing=false production_activation=false exec_attached=false commit_attached=false ci_attached=false python_executed=false rust_executed=false"
