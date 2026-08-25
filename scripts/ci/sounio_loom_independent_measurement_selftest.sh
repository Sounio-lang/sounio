#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CANARY="$ROOT_DIR/scripts/ci/sounio_loom_pod_replay_canary.sh"
LOOM="$ROOT_DIR/bin/sounio-loom"
NORMAL_ADAPTER="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime"
MODULE="$ROOT_DIR/stdlib/coordination/loom_continuity.sio"
BUILD_ADAPTER="$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-independent-measurement.XXXXXX")"
KEY_ROOT="$TEST_ROOT/keys"
PANE_ID='loom-pod-replay:terminal'
LANE='pane-6c6f6f6d2d706f642d7265706c61793a7465726d696e616c'

fail() {
  printf 'sounio-loom-independent-measurement-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

json_string_value() {
  local path="$1" key="$2"
  sed -n "s/.*\"${key}\":\"\([^\"]*\)\".*/\1/p" "$path" | head -1
}

key_value() {
  local key="$1" path="$2" value
  value="$(sed -n "s/^${key}=//p" "$path")"
  [[ -n "$value" && "$value" != *$'\n'* ]] || \
    fail "$path must contain exactly one $key"
  printf '%s\n' "$value"
}

kill_generation() {
  local root="$1" status pid bridge_file bridge_pid
  status="$(
    SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" status \
      --state-dir "$root/loom" --cwd "$ROOT_DIR" \
      --agent beagle-workbench --lane "$LANE" 2>/dev/null || true
  )"
  for pid in "$(field daemon_pid "$status")" "$(field guardian_pid "$status")" \
    "$(field harness_pid "$status")"; do
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -9 "$pid" 2>/dev/null || true
  done
  for bridge_file in "$root"/bridge-*.pid; do
    [[ -f "$bridge_file" ]] || continue
    bridge_pid="$(cat "$bridge_file")"
    [[ "$bridge_pid" =~ ^[1-9][0-9]*$ ]] && kill -9 "$bridge_pid" 2>/dev/null || true
  done
}

cleanup() {
  local root
  for root in "$TEST_ROOT/treatment" "$TEST_ROOT/control"; do
    kill_generation "$root" || true
  done
  chmod -R u+w "$TEST_ROOT" 2>/dev/null || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

run_phase() {
  local root="$1" uid="$2" phase="$3" adapter="$4"
  env \
    POD_UID="$uid" \
    POD_NAME=loom-independent-measurement-0 \
    SOUNIO_CANARY_SOURCE_ROOT="$ROOT_DIR" \
    SOUNIO_LOOM_POD_CANARY_ROOT="$root" \
    SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS=1 \
    SOUNIO_LOOM_REQUIRE_INDEPENDENT_MEASUREMENT=1 \
    SOUNIO_LOOM_SIGNING_KEY="$KEY_ROOT/signer-private.pem" \
    SOUNIO_LOOM_VERIFY_KEY="$KEY_ROOT/signer-public.pem" \
    SOUNIO_LOOM_OBSERVER_VERIFY_KEY="$KEY_ROOT/observer-public.pem" \
    SOUNIO_LOOM_CONTINUITY_ADAPTER="$adapter" \
    bash "$CANARY" "$phase"
}

receipt_path() {
  local root="$1" matches=()
  mapfile -t matches < <(
    find "$root/loom" -name sounio-continuity.receipt -type f -print
  )
  [[ "${#matches[@]}" -eq 1 ]] || \
    fail "$root contains ${#matches[@]} predecessor receipts instead of one"
  printf '%s\n' "${matches[0]}"
}

generation_count() {
  local root="$1"
  find "$root/loom" -name sounio-continuity.receipt -type f -print \
    2>/dev/null | wc -l
}

forge_semantic_fact() {
  local receipt="$1" adapter="$2" facts key_id adapter_sha256 facts_sha256
  local verdict payload signature signature_base64 payload_sha256
  local fields=()
  facts="$(key_value facts "$receipt")"
  read -r -a fields <<< "$facts"
  [[ "${#fields[@]}" -eq 15 ]] || fail 'genesis receipt did not contain 15 facts'
  ORIGINAL_SEMANTIC_TOKEN="${fields[4]}"
  FORGED_SEMANTIC_TOKEN="$((ORIGINAL_SEMANTIC_TOKEN + 1))"
  fields[4]="$FORGED_SEMANTIC_TOKEN"
  FORGED_FACTS="${fields[*]}"
  key_id="$(sha256sum "$KEY_ROOT/signer-public.pem" | awk '{print $1}')"
  adapter_sha256="$(sha256sum "$adapter" | awk '{print $1}')"
  facts_sha256="$(printf '%s\n' "$FORGED_FACTS" | sha256sum | awk '{print $1}')"
  verdict='SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v2 authenticity=ed25519'
  payload="$TEST_ROOT/payload-$RANDOM.txt"
  signature="$TEST_ROOT/signature-$RANDOM.bin"
  {
    printf 'schema=loom-native-continuity-signed-payload-v1\n'
    printf 'algorithm=ed25519\n'
    printf 'key_id=%s\n' "$key_id"
    printf 'adapter_sha256=%s\n' "$adapter_sha256"
    printf 'facts_sha256=%s\n' "$facts_sha256"
    printf 'facts=%s\n' "$FORGED_FACTS"
    printf 'verdict=%s\n' "$verdict"
  } > "$payload"
  payload_sha256="$(sha256sum "$payload" | awk '{print $1}')"
  openssl pkeyutl -sign -rawin -inkey "$KEY_ROOT/signer-private.pem" \
    -in "$payload" -out "$signature"
  signature_base64="$(openssl base64 -A -in "$signature")"
  {
    printf 'schema=loom-native-continuity-receipt-v2\n'
    printf 'algorithm=ed25519\n'
    printf 'key_id=%s\n' "$key_id"
    printf 'adapter_sha256=%s\n' "$adapter_sha256"
    printf 'facts_sha256=%s\n' "$facts_sha256"
    printf 'facts=%s\n' "$FORGED_FACTS"
    printf 'verdict=%s\n' "$verdict"
    printf 'signed_payload_sha256=%s\n' "$payload_sha256"
    printf 'signature_base64=%s\n' "$signature_base64"
  } > "$receipt"
}

bind_canary_to_subject_receipt() {
  local root="$1" receipt="$2" state="$root/phase.env" temporary
  local digest="$(sha256sum "$receipt" | awk '{print $1}')"
  [[ -f "$state" ]] || fail 'canary phase state is missing'
  temporary="$state.$$"
  awk -v digest="$digest" '
    /^receipt_one=/ { print "receipt_one=" digest; changed=1; next }
    { print }
    END { if (changed != 1) exit 42 }
  ' "$state" > "$temporary" || fail 'could not bind canary state to subject receipt'
  mv "$temporary" "$state"
}

measure_generation() {
  local root="$1" generation="$2" receipt="$3" adapter="$4"
  local output="$(dirname "$receipt")/sounio-continuity.observer-attestation"
  SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" measure-continuity-generation \
    --state-dir "$root/loom" --pane-id "$PANE_ID" --generation "$generation" \
    --receipt "$receipt" --subject-public-key "$KEY_ROOT/signer-public.pem" \
    --observer-private-key "$KEY_ROOT/observer-private.pem" \
    --observer-public-key "$KEY_ROOT/observer-public.pem" \
    --out "$output" --adapter "$adapter" >/dev/null
  [[ -s "$output" ]] || fail 'independent measurement attestation was not created'
  printf '%s\n' "$output"
}

assert_raw_measurement() {
  local receipt="$1" attestation="$2" raw_head measured_head
  raw_head="$(awk -F '\t' 'NF { head=$3 } END { print head }' \
    "$(dirname "$receipt")/journal.tsv")"
  measured_head="$(key_value measured_semantic_head "$attestation")"
  [[ "${#raw_head}" -eq 64 && "$measured_head" == "$raw_head" ]] || \
    fail 'observer measurement did not come from the retained semantic journal'
}

make_measurement_mutant() {
  local mutated="$TEST_ROOT/loom_continuity_measurement_mutated.sio"
  local mutant="$TEST_ROOT/sounio-loom-continuity-measurement-mutant"
  awk '
    BEGIN { in_function=0; skip_body=0; changed=0 }
    $0 == "fn measurement_tokens_agree(" {
      in_function=1
      print
      next
    }
    in_function && !skip_body {
      print
      if ($0 == ") -> bool {") {
        print "    true"
        skip_body=1
        changed=changed + 1
      }
      next
    }
    skip_body {
      if ($0 == "}") {
        print
        in_function=0
        skip_body=0
      }
      next
    }
    { print }
    END { if (changed != 1) exit 42 }
  ' "$MODULE" > "$mutated" || fail 'could not apply measurement-agreement sabotage'
  SOUNIO_LOOM_CONTINUITY_PREBUILT= \
  SOUNIO_LOOM_CONTINUITY_MODULE="$mutated" \
  SOUNIO_LOOM_CONTINUITY_OUTPUT="$mutant" \
    "$BUILD_ADAPTER" >/dev/null
  printf '%s\n' "$mutant"
}

run_treatment() {
  local root="$TEST_ROOT/treatment" phase receipt generation attestation
  local before after output rc
  phase="$(run_phase "$root" measurement-one phase-one "$NORMAL_ADAPTER")"
  [[ "$phase" == *CANARY_PHASE_ONE* ]] || fail 'treatment genesis failed'
  receipt="$(receipt_path "$root")"
  generation="$(json_string_value "$root/spawn-measurement-one.json" loomInstanceId)"
  [[ -n "$generation" ]] || fail 'treatment genesis omitted generation identity'
  forge_semantic_fact "$receipt" "$NORMAL_ADAPTER"
  bind_canary_to_subject_receipt "$root" "$receipt"
  attestation="$(measure_generation "$root" "$generation" "$receipt" "$NORMAL_ADAPTER")"
  assert_raw_measurement "$receipt" "$attestation"
  kill_generation "$root"
  before="$(generation_count "$root")"
  set +e
  output="$(run_phase "$root" measurement-two phase-two "$NORMAL_ADAPTER" 2>&1)"
  rc=$?
  set -e
  [[ "$rc" -ne 0 && \
     "$output" == *sounio-continuity-pre-spawn-measurement-policy-refused* && \
     "$output" == *independent-measurement-disagreement* ]] || \
    fail "measurement disagreement was not refused by Sounio: rc=$rc output=$output"
  after="$(generation_count "$root")"
  [[ "$after" -eq "$before" ]] || \
    fail 'measurement disagreement created a successor receipt'
  printf '%s\n' \
    'SOUNIO_LOOM_INDEPENDENT_MEASUREMENT_TREATMENT=PASS signer-semantic=forged observer-semantic=raw-journal sounio-agreement=refused-before-spawn successor_created=0'
}

run_control() {
  local root="$TEST_ROOT/control" mutant phase receipt generation attestation spawn
  mutant="$(make_measurement_mutant)"
  phase="$(run_phase "$root" control-one phase-one "$mutant")"
  [[ "$phase" == *CANARY_PHASE_ONE* ]] || fail 'control genesis failed'
  receipt="$(receipt_path "$root")"
  generation="$(json_string_value "$root/spawn-control-one.json" loomInstanceId)"
  [[ -n "$generation" ]] || fail 'control genesis omitted generation identity'
  forge_semantic_fact "$receipt" "$mutant"
  bind_canary_to_subject_receipt "$root" "$receipt"
  attestation="$(measure_generation "$root" "$generation" "$receipt" "$mutant")"
  assert_raw_measurement "$receipt" "$attestation"
  kill_generation "$root"
  phase="$(run_phase "$root" control-two phase-two "$mutant")"
  [[ "$phase" == *CANARY_PHASE_TWO* ]] || \
    fail 'targeted measurement-agreement sabotage did not admit the control'
  spawn="$root/spawn-control-two.json"
  grep -q '"sounioPolicyIndependentMeasurementVerified":true' "$spawn" || \
    fail 'control successor omitted independent-measurement status'
  [[ "$(generation_count "$root")" -eq 2 ]] || \
    fail 'control successor did not create its continuity receipt'
  printf '%s\n' \
    'SOUNIO_LOOM_INDEPENDENT_MEASUREMENT_CONTROL=PASS intervention=measurement_tokens_agree-always-true signer-semantic=forged observer-semantic=raw-journal successor=admitted'
}

command -v openssl >/dev/null || fail 'OpenSSL is required'
mkdir -p "$KEY_ROOT"
openssl genpkey -algorithm ED25519 -out "$KEY_ROOT/signer-private.pem"
openssl pkey -in "$KEY_ROOT/signer-private.pem" -pubout \
  -out "$KEY_ROOT/signer-public.pem"
openssl genpkey -algorithm ED25519 -out "$KEY_ROOT/observer-private.pem"
openssl pkey -in "$KEY_ROOT/observer-private.pem" -pubout \
  -out "$KEY_ROOT/observer-public.pem"
chmod a-w "$KEY_ROOT" "$KEY_ROOT"/*.pem
"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

case "${1:-all}" in
  treatment) run_treatment ;;
  control) run_control ;;
  all) run_treatment; run_control ;;
  *) fail 'usage: sounio_loom_independent_measurement_selftest.sh [treatment|control|all]' ;;
esac
