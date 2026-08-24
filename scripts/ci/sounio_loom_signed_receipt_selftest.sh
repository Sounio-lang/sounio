#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
CANARY="$ROOT_DIR/scripts/ci/sounio_loom_pod_replay_canary.sh"
LOOM="$ROOT_DIR/bin/sounio-loom"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-signed-receipt.XXXXXX")"
MAIN_ROOT="$TEST_ROOT/main"
SPLICE_ROOT="$TEST_ROOT/splice"
MISMATCH_ROOT="$TEST_ROOT/mismatch"
KEY_ROOT="$TEST_ROOT/keys"

fail() {
  printf 'sounio-loom-signed-receipt-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

kill_generation() {
  local root="$1" loom_dir status pid bridge_file bridge_pid
  loom_dir="$root/loom"
  status="$(
    SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" status --state-dir "$loom_dir" \
      --cwd "$ROOT_DIR" --agent beagle-workbench \
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
  kill_generation "$MAIN_ROOT" || true
  kill_generation "$SPLICE_ROOT" || true
  kill_generation "$MISMATCH_ROOT" || true
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

run_phase() {
  local root="$1" uid="$2" phase="$3"
  env \
    POD_UID="$uid" \
    POD_NAME=loom-signed-receipt-0 \
    SOUNIO_CANARY_SOURCE_ROOT="$ROOT_DIR" \
    SOUNIO_LOOM_POD_CANARY_ROOT="$root" \
    SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS=1 \
    SOUNIO_LOOM_SIGNING_KEY="$KEY_ROOT/private.pem" \
    SOUNIO_LOOM_VERIFY_KEY="$KEY_ROOT/public.pem" \
    bash "$CANARY" "$phase"
}

receipt_path() {
  local root="$1" matches=()
  mapfile -t matches < <(find "$root/loom" -name sounio-continuity.receipt -type f -print)
  [[ "${#matches[@]}" -eq 1 ]] || \
    fail "$root contains ${#matches[@]} receipts instead of one"
  printf '%s\n' "${matches[0]}"
}

verify_receipt() {
  local receipt="$1" public_key="$2"
  SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" verify-continuity-receipt \
    --receipt "$receipt" --public-key "$public_key" \
    --adapter "$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime"
}

expect_verify_refusal() {
  local label="$1" receipt="$2" public_key="$3" reason="$4" output rc=0
  set +e
  output="$(verify_receipt "$receipt" "$public_key" 2>&1)"
  rc=$?
  set -e
  [[ "$rc" -ne 0 && "$output" == *"$reason"* ]] || \
    fail "$label did not refuse for $reason: rc=$rc output=$output"
}

command -v openssl >/dev/null || fail 'OpenSSL is required'
mkdir -p "$KEY_ROOT"
openssl genpkey -algorithm ED25519 -out "$KEY_ROOT/private.pem"
openssl pkey -in "$KEY_ROOT/private.pem" -pubout -out "$KEY_ROOT/public.pem"
openssl genpkey -algorithm ED25519 -out "$KEY_ROOT/wrong-private.pem"
openssl pkey -in "$KEY_ROOT/wrong-private.pem" -pubout -out "$KEY_ROOT/wrong-public.pem"
"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

if env POD_UID=missing-key POD_NAME=loom-signed-receipt-0 \
  SOUNIO_CANARY_SOURCE_ROOT="$ROOT_DIR" \
  SOUNIO_LOOM_POD_CANARY_ROOT="$TEST_ROOT/missing-key" \
  SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS=1 \
  bash "$CANARY" phase-one >"$TEST_ROOT/missing-key.log" 2>&1; then
  fail 'signed-required mode accepted missing keys'
fi
grep -q 'signed receipts require mounted private and public keys' \
  "$TEST_ROOT/missing-key.log" || fail 'missing-key control failed for the wrong reason'

if env POD_UID=mismatched-key POD_NAME=loom-signed-receipt-0 \
  SOUNIO_CANARY_SOURCE_ROOT="$ROOT_DIR" \
  SOUNIO_LOOM_POD_CANARY_ROOT="$MISMATCH_ROOT" \
  SOUNIO_LOOM_REQUIRE_SIGNED_RECEIPTS=1 \
  SOUNIO_LOOM_SIGNING_KEY="$KEY_ROOT/wrong-private.pem" \
  SOUNIO_LOOM_VERIFY_KEY="$KEY_ROOT/public.pem" \
  bash "$CANARY" phase-one >"$TEST_ROOT/mismatched-key.log" 2>&1; then
  fail 'signed receipt emission accepted a mismatched keypair'
fi
grep -q 'sounio-continuity-signing-keypair-mismatch' \
  "$TEST_ROOT/mismatched-key.log" || {
  cat "$TEST_ROOT/mismatched-key.log" >&2
  fail 'mismatched-key control failed for the wrong reason'
}

main_phase="$(run_phase "$MAIN_ROOT" main-pod-one phase-one)"
[[ "$main_phase" == *'CANARY_PHASE_ONE'* ]] || fail 'main signed genesis did not start'
main_receipt="$(receipt_path "$MAIN_ROOT")"
valid="$(verify_receipt "$main_receipt" "$KEY_ROOT/public.pem")"
[[ "$valid" == LOOM_CONTINUITY_RECEIPT_VERIFIED*algorithm=ed25519* ]] || \
  fail "public verifier omitted its receipt: $valid"

tampered_payload="$TEST_ROOT/tampered-payload.receipt"
awk '
  BEGIN { changed=0 }
  !changed && /^facts=/ { sub(/^facts=/, "facts=9 "); changed=1 }
  { print }
  END { if (changed != 1) exit 42 }
' "$main_receipt" > "$tampered_payload"
expect_verify_refusal payload-tamper "$tampered_payload" "$KEY_ROOT/public.pem" \
  sounio-continuity-signed-receipt-mismatch

tampered_signature="$TEST_ROOT/tampered-signature.receipt"
awk '
  BEGIN { changed=0 }
  !changed && /^signature_base64=/ {
    value=substr($0, 18)
    first=substr(value, 1, 1)
    replacement=(first == "A" ? "B" : "A")
    print "signature_base64=" replacement substr(value, 2)
    changed=1
    next
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$main_receipt" > "$tampered_signature"
expect_verify_refusal signature-tamper "$tampered_signature" "$KEY_ROOT/public.pem" \
  sounio-continuity-signature-invalid
expect_verify_refusal wrong-key "$main_receipt" "$KEY_ROOT/wrong-public.pem" \
  sounio-continuity-signed-receipt-mismatch

splice_phase="$(run_phase "$SPLICE_ROOT" splice-pod-one phase-one)"
[[ "$splice_phase" == *'CANARY_PHASE_ONE'* ]] || fail 'splice control genesis did not start'
splice_receipt="$(receipt_path "$SPLICE_ROOT")"
verify_receipt "$splice_receipt" "$KEY_ROOT/public.pem" >/dev/null
[[ "$(sha256sum "$main_receipt" | awk '{print $1}')" != \
   "$(sha256sum "$splice_receipt" | awk '{print $1}')" ]] || \
  fail 'independent signed generations produced the same receipt'

kill_generation "$MAIN_ROOT"
cp "$splice_receipt" "$main_receipt"
if run_phase "$MAIN_ROOT" main-pod-two phase-two >"$TEST_ROOT/splice.log" 2>&1; then
  fail 'successor accepted a valid receipt from the wrong predecessor generation'
fi
grep -q 'sounio-continuity-predecessor-receipt-splice' "$TEST_ROOT/splice.log" || {
  cat "$TEST_ROOT/splice.log" >&2
  fail 'predecessor splice control failed for the wrong reason'
}

echo 'sounio-loom-signed-receipt-selftest: PASS algorithm=ed25519 public_only_verify=pass payload_tamper=refused signature_tamper=refused wrong_key=refused mismatched_keypair=refused predecessor_splice=refused missing_key=refused'
