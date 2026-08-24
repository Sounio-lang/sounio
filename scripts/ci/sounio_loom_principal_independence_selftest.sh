#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CANARY="$ROOT_DIR/scripts/ci/sounio_loom_pod_replay_canary.sh"
LOOM="$ROOT_DIR/bin/sounio-loom"
NORMAL_ADAPTER="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime"
MODULE="$ROOT_DIR/stdlib/coordination/loom_continuity.sio"
BUILD_ADAPTER="$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-principal-independence.XXXXXX")"
KEY_ROOT="$TEST_ROOT/keys"

fail() {
  printf 'sounio-loom-principal-independence-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

kill_generation() {
  local root="$1" status pid bridge_file bridge_pid
  status="$(
    SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" status \
      --state-dir "$root/loom" --cwd "$ROOT_DIR" \
      --agent beagle-workbench \
      --lane pane-6c6f6f6d2d706f642d7265706c61793a7465726d696e616c \
      2>/dev/null || true
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
  for root in "$TEST_ROOT"/distinct "$TEST_ROOT"/collapsed "$TEST_ROOT"/control; do
    kill_generation "$root" || true
  done
  chmod -R u+w "$TEST_ROOT" 2>/dev/null || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

run_phase() {
  local root="$1" uid="$2" phase="$3" adapter="$4" observer_public="$5"
  env \
    POD_UID="$uid" \
    POD_NAME=loom-principal-independence-0 \
    SOUNIO_CANARY_SOURCE_ROOT="$ROOT_DIR" \
    SOUNIO_LOOM_POD_CANARY_ROOT="$root" \
    SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS=1 \
    SOUNIO_LOOM_REQUIRE_INDEPENDENT_OBSERVER=1 \
    SOUNIO_LOOM_SIGNING_KEY="$KEY_ROOT/signer-private.pem" \
    SOUNIO_LOOM_VERIFY_KEY="$KEY_ROOT/signer-public.pem" \
    SOUNIO_LOOM_OBSERVER_VERIFY_KEY="$observer_public" \
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

attest_receipt() {
  local receipt="$1" observer_private="$2" observer_public="$3" adapter="$4"
  local output
  output="$(dirname "$receipt")/sounio-continuity.observer-attestation"
  SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" attest-continuity-receipt \
    --receipt "$receipt" \
    --subject-public-key "$KEY_ROOT/signer-public.pem" \
    --observer-private-key "$observer_private" \
    --observer-public-key "$observer_public" \
    --out "$output" --adapter "$adapter" >/dev/null
  [[ -s "$output" ]] || fail 'observer attestation was not created'
}

generation_count() {
  local root="$1"
  find "$root/loom" -name sounio-continuity.receipt -type f -print \
    2>/dev/null | wc -l
}

make_principal_mutant() {
  local mutated="$TEST_ROOT/loom_continuity_principal_mutated.sio"
  local mutant="$TEST_ROOT/sounio-loom-continuity-principal-mutant"
  awk '
    BEGIN { changed=0 }
    !changed && $0 == "    if signer_authority_token == observer_authority_token { return false }" {
      print "    if false { return false }"
      changed=1
      next
    }
    { print }
    END { if (changed != 1) exit 42 }
  ' "$MODULE" > "$mutated" || fail 'could not apply principal-disjointness sabotage'
  SOUNIO_LOOM_CONTINUITY_PREBUILT= \
  SOUNIO_LOOM_CONTINUITY_MODULE="$mutated" \
  SOUNIO_LOOM_CONTINUITY_OUTPUT="$mutant" \
    "$BUILD_ADAPTER" >/dev/null
  printf '%s\n' "$mutant"
}

run_treatment() {
  local distinct_root="$TEST_ROOT/distinct" collapsed_root="$TEST_ROOT/collapsed"
  local receipt phase output before after signer_principal_id observer_principal_id spawn

  phase="$(run_phase "$distinct_root" distinct-one phase-one \
    "$NORMAL_ADAPTER" "$KEY_ROOT/observer-public.pem")"
  [[ "$phase" == *CANARY_PHASE_ONE* ]] || fail 'distinct-principal genesis failed'
  receipt="$(receipt_path "$distinct_root")"
  attest_receipt "$receipt" "$KEY_ROOT/observer-private.pem" \
    "$KEY_ROOT/observer-public.pem" "$NORMAL_ADAPTER"
  kill_generation "$distinct_root"
  phase="$(run_phase "$distinct_root" distinct-two phase-two \
    "$NORMAL_ADAPTER" "$KEY_ROOT/observer-public.pem")"
  [[ "$phase" == *CANARY_PHASE_TWO* ]] || fail 'distinct principals were not admitted'
  spawn="$distinct_root/spawn-distinct-two.json"
  grep -q '"sounioPolicyIndependentObservationVerified":true' "$spawn" || \
    fail 'distinct successor omitted independent-observation authority'
  signer_principal_id="$(sed -n 's/.*"sounioPolicySignerPrincipalId":"\([^"]*\)".*/\1/p' "$spawn" | head -1)"
  observer_principal_id="$(sed -n 's/.*"sounioPolicyObserverPrincipalId":"\([^"]*\)".*/\1/p' "$spawn" | head -1)"
  [[ -n "$signer_principal_id" && -n "$observer_principal_id" && \
     "$signer_principal_id" != "$observer_principal_id" ]] || \
    fail 'distinct successor omitted disjoint canonical principal identities'
  kill_generation "$distinct_root"

  phase="$(run_phase "$collapsed_root" collapsed-one phase-one \
    "$NORMAL_ADAPTER" "$KEY_ROOT/signer-public-reserialized.pem")"
  [[ "$phase" == *CANARY_PHASE_ONE* ]] || fail 'collapsed-principal genesis failed'
  receipt="$(receipt_path "$collapsed_root")"
  attest_receipt "$receipt" "$KEY_ROOT/signer-private.pem" \
    "$KEY_ROOT/signer-public-reserialized.pem" "$NORMAL_ADAPTER"
  kill_generation "$collapsed_root"
  before="$(generation_count "$collapsed_root")"
  set +e
  output="$(run_phase "$collapsed_root" collapsed-two phase-two \
    "$NORMAL_ADAPTER" "$KEY_ROOT/signer-public-reserialized.pem" 2>&1)"
  local rc=$?
  set -e
  [[ "$rc" -ne 0 && "$output" == *sounio-continuity-pre-spawn-policy-refused* ]] || \
    fail "collapsed principals were not refused by the pre-spawn policy: rc=$rc output=$output"
  after="$(generation_count "$collapsed_root")"
  [[ "$after" -eq "$before" ]] || \
    fail 'collapsed-principal refusal created a successor generation'

  printf '%s\n' \
    'SOUNIO_LOOM_PRINCIPAL_INDEPENDENCE_TREATMENT=PASS distinct=admitted collapsed-reserialized-key=refused-before-spawn successor_created=0'
}

run_control() {
  local root="$TEST_ROOT/control" mutant receipt phase spawn signer_key_id observer_key_id
  local signer_principal_id observer_principal_id
  mutant="$(make_principal_mutant)"
  phase="$(run_phase "$root" control-one phase-one "$mutant" \
    "$KEY_ROOT/signer-public-reserialized.pem")"
  [[ "$phase" == *CANARY_PHASE_ONE* ]] || fail 'sabotage-control genesis failed'
  receipt="$(receipt_path "$root")"
  attest_receipt "$receipt" "$KEY_ROOT/signer-private.pem" \
    "$KEY_ROOT/signer-public-reserialized.pem" "$mutant"
  kill_generation "$root"
  phase="$(run_phase "$root" control-two phase-two "$mutant" \
    "$KEY_ROOT/signer-public-reserialized.pem")"
  [[ "$phase" == *CANARY_PHASE_TWO* ]] || \
    fail 'targeted principal-disjointness sabotage did not admit the control'
  spawn="$root/spawn-control-two.json"
  grep -q '"sounioPolicyIndependentObservationVerified":true' "$spawn" || \
    fail 'sabotage control omitted independent-observation verification'
  signer_key_id="$(sed -n 's/.*"sounioPolicySignerKeyId":"\([^"]*\)".*/\1/p' "$spawn" | head -1)"
  observer_key_id="$(sed -n 's/.*"sounioPolicyObserverKeyId":"\([^"]*\)".*/\1/p' "$spawn" | head -1)"
  signer_principal_id="$(sed -n 's/.*"sounioPolicySignerPrincipalId":"\([^"]*\)".*/\1/p' "$spawn" | head -1)"
  observer_principal_id="$(sed -n 's/.*"sounioPolicyObserverPrincipalId":"\([^"]*\)".*/\1/p' "$spawn" | head -1)"
  [[ -n "$signer_key_id" && -n "$observer_key_id" && \
     "$signer_key_id" != "$observer_key_id" ]] || \
    fail 'sabotage control did not exercise distinct PEM serializations'
  [[ -n "$signer_principal_id" && \
     "$signer_principal_id" == "$observer_principal_id" ]] || \
    fail 'sabotage control did not preserve the collapsed canonical principal'
  printf '%s\n' \
    'SOUNIO_LOOM_PRINCIPAL_INDEPENDENCE_CONTROL=PASS intervention=remove-only-disjointness-guard raw-key-ids=different canonical-principal=collapsed collapsed=admitted'
}

command -v openssl >/dev/null || fail 'OpenSSL is required'
mkdir -p "$KEY_ROOT"
openssl genpkey -algorithm ED25519 -out "$KEY_ROOT/signer-private.pem"
openssl pkey -in "$KEY_ROOT/signer-private.pem" -pubout \
  -out "$KEY_ROOT/signer-public.pem"
cp "$KEY_ROOT/signer-public.pem" "$KEY_ROOT/signer-public-reserialized.pem"
printf '\n' >> "$KEY_ROOT/signer-public-reserialized.pem"
openssl pkey -pubin -in "$KEY_ROOT/signer-public-reserialized.pem" -noout \
  >/dev/null 2>&1 || fail 'OpenSSL rejected the alternate PEM serialization'
[[ "$(sha256sum "$KEY_ROOT/signer-public.pem" | awk '{print $1}')" != \
   "$(sha256sum "$KEY_ROOT/signer-public-reserialized.pem" | awk '{print $1}')" ]] || \
  fail 'alternate PEM serialization did not change the raw key identity'
openssl genpkey -algorithm ED25519 -out "$KEY_ROOT/observer-private.pem"
openssl pkey -in "$KEY_ROOT/observer-private.pem" -pubout \
  -out "$KEY_ROOT/observer-public.pem"
chmod a-w "$KEY_ROOT" "$KEY_ROOT"/*.pem
"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

case "${1:-all}" in
  treatment) run_treatment ;;
  control) run_control ;;
  all) run_treatment; run_control ;;
  *) fail 'usage: sounio_loom_principal_independence_selftest.sh [treatment|control|all]' ;;
esac
