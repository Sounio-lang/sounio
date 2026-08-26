#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-epoch.XXXXXX")"
WORLD="epoch-handoff"
BASE_ROOT="$WORK/base-root"
OLD_ROOT="$WORK/old-root"
NEW_ROOT="$WORK/new-root"
REUSE_ROOT="$WORK/reuse-root"
OLD_MEMBERSHIP="$WORK/old-membership.tsv"
NEW_MEMBERSHIP="$WORK/new-membership.tsv"
OLD_ENDPOINTS="$WORK/old-endpoints.tsv"
NEW_ENDPOINTS="$WORK/new-endpoints.tsv"
declare -A WITNESS_PIDS=()
declare -A WITNESS_PORTS=()

cleanup() {
  local key pid
  for key in "${!WITNESS_PIDS[@]}"; do
    pid="${WITNESS_PIDS[$key]:-}"
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'loom-witness-epoch-handoff: FAIL: %s\n' "$*" >&2
  exit 1
}

digest() {
  printf '%s' "$1" | sha256sum | awk '{print $1}'
}

membership_for() {
  case "$1" in
    old) printf '%s\n' "$OLD_MEMBERSHIP" ;;
    new) printf '%s\n' "$NEW_MEMBERSHIP" ;;
    *) fail "unknown witness side: $1" ;;
  esac
}

endpoints_for() {
  case "$1" in
    old) printf '%s\n' "$OLD_ENDPOINTS" ;;
    new) printf '%s\n' "$NEW_ENDPOINTS" ;;
    *) fail "unknown endpoint side: $1" ;;
  esac
}

start_witness() {
  local side="$1" index="$2" key="${1}${2}"
  local membership log pid attempt=0
  membership="$(membership_for "$side")"
  log="$WORK/$key.log"
  : > "$log"
  "$LOOM" witness-serve \
    --witness-state-dir "$WORK/$key-state" \
    --membership "$membership" --witness "$key" \
    --private-key "$WORK/$key-private.pem" \
    --bind 127.0.0.1 --port 0 >"$log" 2>&1 &
  pid=$!
  WITNESS_PIDS[$key]="$pid"
  until rg -q 'LOOM_WITNESS_READY schema=loom-witness-service-v1 ' "$log"; do
    kill -0 "$pid" 2>/dev/null || {
      sed -n '1,220p' "$log" >&2
      fail "$key exited before readiness"
    }
    attempt=$((attempt + 1))
    [[ "$attempt" -lt 240 ]] || fail "$key readiness timed out"
    sleep 0.025
  done
  WITNESS_PORTS[$key]="$(sed -n 's/.* port=\([0-9][0-9]*\) .*/\1/p' "$log")"
  [[ -n "${WITNESS_PORTS[$key]}" ]] || fail "$key omitted its port"
}

stop_witness() {
  local side="$1" index="$2" key="${1}${2}" pid="${WITNESS_PIDS[${1}${2}]:-}"
  if [[ -n "$pid" ]]; then
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    WITNESS_PIDS[$key]=""
  fi
}

write_endpoints() {
  local side="$1" endpoints index key
  endpoints="$(endpoints_for "$side")"
  printf 'witness_id\thost\tport\n' > "$endpoints"
  for index in 1 2 3 4; do
    key="${side}${index}"
    printf '%s\t127.0.0.1\t%s\n' "$key" "${WITNESS_PORTS[$key]}" \
      >> "$endpoints"
  done
}

restart_pair() {
  local side="$1"
  start_witness "$side" 3
  start_witness "$side" 4
  write_endpoints "$side"
}

expect_refusal() {
  local label="$1" expected="$2" rc=0
  shift 2
  set +e
  "$@" >"$WORK/$label.out" 2>"$WORK/$label.err"
  rc=$?
  set -e
  [[ "$rc" -eq 1 ]] || fail "$label returned rc=$rc"
  rg -q "$expected" "$WORK/$label.err" || {
    sed -n '1,240p' "$WORK/$label.err" >&2
    fail "$label was refused by an unrelated rule"
  }
}

observe() {
  local root="$1" knowledge="$2" value="$3"
  "$LOOM" knowledge-observe --state-dir "$root" --world "$WORLD" \
    --knowledge "$knowledge" --value "$value" --error 0 \
    --uncertainty bounded --confidence 1 \
    --provenance "$(digest "$knowledge-provenance")" >/dev/null
}

anchor() {
  local side="$1" root="$2"
  "$LOOM" witness-mesh-anchor --state-dir "$root" --world "$WORLD" \
    --membership "$(membership_for "$side")" \
    --endpoints "$(endpoints_for "$side")" \
    --anchor-private-key "$WORK/$side-anchor-private.pem"
}

verify_mesh() {
  local side="$1" root="$2"
  "$LOOM" witness-mesh-verify --state-dir "$root" --world "$WORLD" \
    --membership "$(membership_for "$side")" \
    --endpoints "$(endpoints_for "$side")"
}

handoff_command() {
  local epoch_state="$1" from_epoch="$2" to_epoch="$3"
  local old_root="$4" old_membership="$5" old_endpoints="$6"
  local new_root="$7" new_membership="$8" new_endpoints="$9"
  "$LOOM" witness-epoch-handoff --epoch-state-dir "$epoch_state" \
    --world "$WORLD" --from-epoch "$from_epoch" --to-epoch "$to_epoch" \
    --old-state-dir "$old_root" --old-membership "$old_membership" \
    --old-endpoints "$old_endpoints" --new-state-dir "$new_root" \
    --new-membership "$new_membership" --new-endpoints "$new_endpoints"
}

verify_active() {
  local epoch_state="$1" root="$2" membership="$3" endpoints="$4"
  "$LOOM" witness-epoch-verify --epoch-state-dir "$epoch_state" \
    --world "$WORLD" --active-state-dir "$root" \
    --membership "$membership" --endpoints "$endpoints"
}

"$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null

for side in old new; do
  openssl genpkey -algorithm ED25519 \
    -out "$WORK/$side-anchor-private.pem" 2>/dev/null
  openssl pkey -in "$WORK/$side-anchor-private.pem" -pubout \
    -out "$WORK/$side-anchor-public.pem" 2>/dev/null
  for index in 1 2 3 4; do
    key="${side}${index}"
    openssl genpkey -algorithm ED25519 \
      -out "$WORK/$key-private.pem" 2>/dev/null
    openssl pkey -in "$WORK/$key-private.pem" -pubout \
      -out "$WORK/$key-public.pem" 2>/dev/null
  done
  membership="$(membership_for "$side")"
  printf 'schema\tloom-witness-membership-v1\n' > "$membership"
  printf 'anchor_public_key\t%s\n' "$WORK/$side-anchor-public.pem" >> "$membership"
  printf 'witness_id\tpublic_key\n' >> "$membership"
  for index in 1 2 3 4; do
    key="${side}${index}"
    printf '%s\t%s\n' "$key" "$WORK/$key-public.pem" >> "$membership"
  done
done

for side in old new; do
  for index in 1 2 3 4; do
    start_witness "$side" "$index"
  done
  write_endpoints "$side"
done

"$LOOM" world-create --state-dir "$BASE_ROOT" --world "$WORLD" \
  --agent codex --lane witness-epoch-handoff >/dev/null
observe "$BASE_ROOT" checkpoint shared
cp -a "$BASE_ROOT" "$OLD_ROOT"
cp -a "$BASE_ROOT" "$NEW_ROOT"
cp -a "$BASE_ROOT" "$REUSE_ROOT"

old_anchor="$(anchor old "$OLD_ROOT")"
new_anchor="$(anchor new "$NEW_ROOT")"
reuse_anchor="$(anchor old "$REUSE_ROOT")"
rg -q 'schema=loom-witness-mesh-v1 .*quorum=4/4 ' <<< "$old_anchor" || \
  fail "old epoch did not independently anchor: $old_anchor"
rg -q 'schema=loom-witness-mesh-v1 .*quorum=4/4 ' <<< "$new_anchor" || \
  fail "new epoch did not independently anchor: $new_anchor"
rg -q 'schema=loom-witness-mesh-v1 .*quorum=4/4 recovered_checkpoints=1 ' \
  <<< "$reuse_anchor" || fail "same-membership control root did not recover: $reuse_anchor"
verify_mesh old "$OLD_ROOT" >/dev/null
verify_mesh new "$NEW_ROOT" >/dev/null

# Crash after the immutable handoff receipt is durable but before the active
# pointer rename. The retry must reuse the exact receipt and activate it.
EPOCH_STATE="$WORK/epoch-state-before-activation"
set +e
SOUNIO_LOOM_WITNESS_EPOCH_FAILPOINT=after-handoff-before-activation \
  handoff_command "$EPOCH_STATE" 1 2 \
    "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
    "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS" \
    >"$WORK/before-activation.out" 2>"$WORK/before-activation.err"
before_rc=$?
set -e
[[ "$before_rc" -eq 198 ]] || fail "before-activation failpoint returned rc=$before_rc"
rg -Fxq \
  'LOOM_WITNESS_EPOCH_FAILPOINT name=after-handoff-before-activation exit=198' \
  "$WORK/before-activation.err" || fail 'before-activation failpoint did not fire'
ACTIVE="$EPOCH_STATE/loom-witness-epochs/$WORLD/active-epoch.receipt"
[[ ! -e "$ACTIVE" ]] || fail 'before-activation crash published an active epoch'
HANDOFF="$(find "$EPOCH_STATE" -name 'handoff-*.receipt' -type f -print -quit)"
[[ -n "$HANDOFF" && -f "$HANDOFF" ]] || fail 'before-activation crash lost the handoff receipt'

recovered="$(handoff_command "$EPOCH_STATE" 1 2 \
  "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
  "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS")"
rg -q 'joint_quorum=3/4\+3/4 .*prepared=reused activated=yes idempotent=no native_frame=9015' \
  <<< "$recovered" || fail "prepared handoff did not recover: $recovered"
[[ -f "$ACTIVE" ]] || fail 'recovered handoff omitted the active pointer'
active_ok="$(verify_active "$EPOCH_STATE" "$NEW_ROOT" \
  "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS")"
rg -q 'epoch=2 .*remote_quorum=4/4 chain=VERIFIED native_frames=9014\+9015' \
  <<< "$active_ok" || fail "active epoch did not verify: $active_ok"

retry="$(handoff_command "$EPOCH_STATE" 1 2 \
  "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
  "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS")"
rg -q 'prepared=reused activated=yes idempotent=yes native_frame=9015' \
  <<< "$retry" || fail "activated retry was not idempotent: $retry"

# The second crash window happens after the atomic pointer rename. A cold
# retry must observe the activation and return the same proof as idempotent.
AFTER_STATE="$WORK/epoch-state-after-activation"
set +e
SOUNIO_LOOM_WITNESS_EPOCH_FAILPOINT=after-activation-before-return \
  handoff_command "$AFTER_STATE" 1 2 \
    "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
    "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS" \
    >"$WORK/after-activation.out" 2>"$WORK/after-activation.err"
after_rc=$?
set -e
[[ "$after_rc" -eq 198 ]] || fail "after-activation failpoint returned rc=$after_rc"
rg -Fxq \
  'LOOM_WITNESS_EPOCH_FAILPOINT name=after-activation-before-return exit=198' \
  "$WORK/after-activation.err" || fail 'after-activation failpoint did not fire'
[[ -f "$AFTER_STATE/loom-witness-epochs/$WORLD/active-epoch.receipt" ]] || \
  fail 'after-activation crash lost the active pointer'
after_retry="$(handoff_command "$AFTER_STATE" 1 2 \
  "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
  "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS")"
rg -q 'prepared=reused activated=yes idempotent=yes native_frame=9015' \
  <<< "$after_retry" || fail "post-activation retry was not idempotent: $after_retry"

# Each side is necessary. A current quorum from only the old or only the new
# authority cannot authorize a transition.
stop_witness new 3
stop_witness new 4
expect_refusal old-only-quorum \
  'witness-epoch-current-quorum-unavailable:valid=2:required=3' \
  handoff_command "$WORK/old-only-state" 1 2 \
    "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
    "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS"
restart_pair new

stop_witness old 3
stop_witness old 4
expect_refusal new-only-quorum \
  'witness-epoch-current-quorum-unavailable:valid=2:required=3' \
  handoff_command "$WORK/new-only-state" 1 2 \
    "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
    "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS"
restart_pair old

expect_refusal epoch-skip 'witness-epoch-not-adjacent:1:3' \
  handoff_command "$WORK/epoch-skip-state" 1 3 \
    "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
    "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS"
expect_refusal epoch-out-of-range 'witness-epoch-out-of-range:65:max=64' \
  handoff_command "$WORK/epoch-out-of-range-state" 65 66 \
    "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
    "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS"
expect_refusal root-reuse 'witness-epoch-state-root-reuse' \
  handoff_command "$WORK/root-reuse-state" 1 2 \
    "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
    "$OLD_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS"
expect_refusal membership-reuse 'witness-epoch-membership-reuse' \
  handoff_command "$WORK/membership-reuse-state" 1 2 \
    "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
    "$REUSE_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS"

# Content and pointer substitution both remain detectable after activation.
cp "$ACTIVE" "$WORK/active.good"
awk -F= 'BEGIN { OFS="=" }
  $1 == "handoff_sha256" {
    first=substr($2,1,1); replacement=(first == "0" ? "1" : "0");
    $2=replacement substr($2,2)
  }
  { print }
' "$WORK/active.good" > "$ACTIVE"
expect_refusal active-pointer-tamper 'witness-active-epoch-handoff-mismatch' \
  verify_active "$EPOCH_STATE" "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS"
cp "$WORK/active.good" "$ACTIVE"

cp "$HANDOFF" "$WORK/handoff.good"
awk -F= 'BEGIN { OFS="=" }
  $1 == "old_state_root_sha256" {
    first=substr($2,1,1); replacement=(first == "0" ? "1" : "0");
    $2=replacement substr($2,2)
  }
  { print }
' "$WORK/handoff.good" > "$HANDOFF"
expect_refusal handoff-content-tamper 'witness-epoch-state-root-digest-mismatch' \
  verify_active "$EPOCH_STATE" "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS"
cp "$WORK/handoff.good" "$HANDOFF"

cp "$NEW_MEMBERSHIP" "$WORK/new-membership.good"
printf '#retained-config-tamper\n' >> "$NEW_MEMBERSHIP"
expect_refusal membership-file-tamper \
  'witness-epoch-membership-file-digest-mismatch' \
  verify_active "$EPOCH_STATE" "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS"
cp "$WORK/new-membership.good" "$NEW_MEMBERSHIP"
verify_active "$EPOCH_STATE" "$NEW_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS" \
  >/dev/null

# Drift and rollback are exercised last because the external witnesses retain
# their new monotonic high-water marks.
DRIFT_ROOT="$WORK/drift-root"
cp -a "$NEW_ROOT" "$DRIFT_ROOT"
observe "$DRIFT_ROOT" drift divergent
drift_anchor="$(anchor new "$DRIFT_ROOT")"
rg -q 'sequence=2 .*quorum=4/4 ' <<< "$drift_anchor" || \
  fail "drift control did not advance the new epoch: $drift_anchor"
expect_refusal checkpoint-drift 'witness-epoch-checkpoint-drift' \
  handoff_command "$WORK/checkpoint-drift-state" 1 2 \
    "$OLD_ROOT" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
    "$DRIFT_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS"

OLD_ROLLBACK="$WORK/old-rollback"
OLD_ADVANCED="$WORK/old-advanced"
cp -a "$OLD_ROOT" "$OLD_ROLLBACK"
cp -a "$OLD_ROOT" "$OLD_ADVANCED"
observe "$OLD_ADVANCED" later advanced
advanced_anchor="$(anchor old "$OLD_ADVANCED")"
rg -q 'sequence=2 .*quorum=4/4 ' <<< "$advanced_anchor" || \
  fail "rollback control did not advance the old witnesses: $advanced_anchor"
expect_refusal old-root-rollback 'witness-rollback-detected:old' \
  handoff_command "$WORK/rollback-state" 1 2 \
    "$OLD_ROLLBACK" "$OLD_MEMBERSHIP" "$OLD_ENDPOINTS" \
    "$DRIFT_ROOT" "$NEW_MEMBERSHIP" "$NEW_ENDPOINTS"

printf 'SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_GATE_PASS=true schema=loom-witness-epoch-handoff-v0 frame=9015 old_witnesses=4 new_witnesses=4 joint_quorum=3/4+3/4 independent_current_quorums=PASS before_activation_crash=RECOVERED after_activation_crash=RECOVERED exact_retry=IDEMPOTENT old_only=REFUSED new_only=REFUSED epoch_skip=REFUSED epoch_out_of_range=REFUSED root_reuse=REFUSED membership_reuse=REFUSED checkpoint_drift=REFUSED old_root_rollback=REFUSED active_pointer_tamper=REFUSED handoff_content_tamper=REFUSED retained_membership_tamper=REFUSED transition=ATOMIC_POINTER_RENAME max_transitions=64 custody_assumption=EPOCH_CONTROL_DIR_OUTSIDE_ROLLBACK_AUTHORITY dynamic_consensus_claim=NONE liveness_claim=NONE state_transfer_claim=NONE runtime=OCaml+Sounio\n'
