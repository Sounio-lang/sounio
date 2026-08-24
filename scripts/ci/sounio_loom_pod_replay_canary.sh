#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="${SOUNIO_CANARY_SOURCE_ROOT:-$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)}"
CANARY_ROOT="${SOUNIO_LOOM_POD_CANARY_ROOT:-/state/loom-pod-replay}"
COORD_DIR="$CANARY_ROOT/coord"
LOOM_DIR="$CANARY_ROOT/loom"
STATE_FILE="$CANARY_ROOT/phase.env"
RESULT_FILE="$CANARY_ROOT/result.txt"
HARNESS="$CANARY_ROOT/beagle-pod-canary"
LOOM="$ROOT_DIR/bin/sounio-loom"
RUNTIME="$ROOT_DIR/scripts/dev/sounio_coord_runtime.sh"
PANE_ID="${SOUNIO_LOOM_POD_CANARY_PANE_ID:-loom-pod-replay:terminal}"
AGENT=beagle-workbench
LANE="pane-$(printf '%s' "$PANE_ID" | od -An -tx1 | tr -d ' \n')"
SENDER_AGENT="${SOUNIO_LOOM_POD_CANARY_SENDER_AGENT:-loom-pod-sender}"
SENDER_LANE="${SOUNIO_LOOM_POD_CANARY_SENDER_LANE:-pending-inbox-replay}"
SESSION_ID="${SOUNIO_LOOM_POD_CANARY_SESSION_ID:-loom-pod-replay-v1}"
POD_UID="${POD_UID:-}"
POD_NAME="${POD_NAME:-}"

fail() {
  printf 'sounio-loom-pod-replay-canary: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local name="$1" value="$2"
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<< "$value"
}

coord() {
  (
    cd "$ROOT_DIR"
    SOUNIO_COORD_WORKTREE="$ROOT_DIR" SOUNIO_COORD_DIR="$COORD_DIR" \
      SOUNIO_COORD_RUNTIME_MODE=local "$RUNTIME" "$@"
  )
}

coord_retry() {
  local output='' attempt
  for attempt in $(seq 1 120); do
    if output="$(coord "$@" 2>&1)"; then
      printf '%s\n' "$output"
      return 0
    fi
    [[ "$output" == *'coordination state is being changed'* ]] || {
      printf '%s\n' "$output" >&2
      return 1
    }
    sleep 0.05
  done
  fail "coordination lock did not clear: $output"
}

loom_status() {
  SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" status --state-dir "$LOOM_DIR" \
    --cwd "$ROOT_DIR" --agent "$AGENT" --lane "$LANE"
}

wait_loom_status() {
  local output='' attempt
  for attempt in $(seq 1 120); do
    output="$(loom_status 2>/dev/null || true)"
    [[ "$output" == *'state=active'* && "$output" == *'instance_id='* ]] && {
      printf '%s\n' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "Loom generation did not become active: $output"
}

wait_endpoint() {
  local generation="$1" output='' attempt
  for attempt in $(seq 1 120); do
    output="$(coord endpoint-status --agent "$AGENT" --lane "$LANE" 2>/dev/null || true)"
    [[ "$output" == *'state=active'* && "$output" == *'transport=loom'* && \
       "$output" == *"instance_id=$generation"* ]] && {
      printf '%s\n' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "generation $generation did not publish a verified Loom endpoint: $output"
}

wait_snapshot() {
  local message_id="$1" output='' attempt
  for attempt in $(seq 1 120); do
    output="$(SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" snapshot \
      --state-dir "$LOOM_DIR" --cwd "$ROOT_DIR" --agent "$AGENT" \
      --lane "$LANE" --cursor 0 2>/dev/null || true)"
    [[ "$output" == *"$message_id"* ]] && {
      printf '%s\n' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "generation snapshot omitted message $message_id"
}

hash_file() {
  local path="$1"
  [[ -f "$path" ]] || fail "receipt artifact is missing: $path"
  sha256sum "$path" | awk '{print $1}'
}

continuity_receipt_digest() {
  local generation="$1" api_digest matches=()
  mapfile -t matches < <(
    find "$LOOM_DIR" -path "*/generations/$generation/sounio-continuity.receipt" \
      -type f -print
  )
  [[ "${#matches[@]}" -eq 1 ]] || \
    fail "generation $generation has ${#matches[@]} native Sounio continuity receipts"
  api_digest="$(json_string_value "$CANARY_ROOT/spawn-$POD_UID.json" sounioPolicyReceipt)"
  [[ "${#api_digest}" -eq 64 && "$(hash_file "${matches[0]}")" == "$api_digest" ]] || \
    fail "generation $generation API receipt does not match its native Sounio artifact"
  printf '%s\n' "$api_digest"
}

json_string_value() {
  local path="$1" key="$2"
  sed -n "s/.*\"${key}\":\"\([^\"]*\)\".*/\1/p" "$path" | head -1
}

start_bridge() {
  local log="$CANARY_ROOT/bridge-$POD_UID.log"
  local pid_file="$CANARY_ROOT/bridge-$POD_UID.pid"
  local bridge_pid port attempt
  : > "$log"
  SOUNIO_COORD_COMMAND="$RUNTIME" SOUNIO_COORD_DIR="$COORD_DIR" \
  SOUNIO_COORD_RUNTIME_MODE=local \
    nohup "$LOOM" beagle-serve --state-dir "$LOOM_DIR" --bind 127.0.0.1 \
      --port 0 > "$log" 2>&1 < /dev/null 9>&- &
  bridge_pid=$!
  printf '%s\n' "$bridge_pid" > "$pid_file"
  for attempt in $(seq 1 120); do
    port="$(sed -n 's#.*url=http://127\.0\.0\.1:\([0-9][0-9]*\).*#\1#p' "$log" | head -1)"
    [[ -n "$port" ]] && {
      printf 'http://127.0.0.1:%s\n' "$port"
      return 0
    }
    kill -0 "$bridge_pid" 2>/dev/null || \
      fail "Beagle bridge exited before readiness: $(tail -40 "$log")"
    sleep 0.05
  done
  fail "Beagle bridge did not publish a loopback endpoint: $(tail -40 "$log")"
}

stop_bridge() {
  local pid_file="$CANARY_ROOT/bridge-$POD_UID.pid" bridge_pid=''
  [[ -f "$pid_file" ]] || return 0
  bridge_pid="$(cat "$pid_file")"
  [[ "$bridge_pid" =~ ^[1-9][0-9]*$ ]] || fail 'Beagle bridge PID artifact is invalid'
  kill "$bridge_pid" 2>/dev/null || true
}

prepare_runtime() {
  mkdir -p "$CANARY_ROOT" "$COORD_DIR" "$LOOM_DIR"
  [[ -n "$POD_UID" && -n "$POD_NAME" ]] || \
    fail 'POD_UID and POD_NAME must come from the Kubernetes downward API'
  [[ -x "$LOOM" && -x "$RUNTIME" ]] || fail "incomplete source checkout: $ROOT_DIR"
  command -v curl >/dev/null || fail 'curl is required inside the canary Pod'
  if ! "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >"$CANARY_ROOT/build.log" 2>&1; then
    tail -120 "$CANARY_ROOT/build.log" >&2 || true
    fail 'could not build the OCaml Loom plus native Sounio adapter in the canary Pod'
  fi
  if [[ ! -x "$HARNESS" ]]; then
    cat > "$HARNESS" <<'HARNESS'
#!/bin/sh
stty -echo
printf 'POD_CANARY_READY\n'
while IFS= read -r line; do
  printf 'POD_CANARY_ECHO:%s\n' "$line"
done
HARNESS
    chmod 700 "$HARNESS"
  fi
}

start_generation() {
  local base_url spawn_file spawn_body status generation policy_runtime adapter_digest
  base_url="$(start_bridge)"
  spawn_file="$CANARY_ROOT/spawn-$POD_UID.json"
  spawn_body="{\"sessionId\":\"$SESSION_ID\",\"paneId\":\"$PANE_ID\",\"cwd\":\"$ROOT_DIR\",\"shell\":\"$HARNESS\",\"cols\":100,\"rows\":30}"
  if ! curl --fail --silent --show-error --request POST \
    --header 'content-type: application/json' --data "$spawn_body" \
    "$base_url/v1/spawn" > "$spawn_file"; then
    fail "Beagle spawn refused the Pod generation: $(cat "$spawn_file" 2>/dev/null || true)"
  fi
  grep -q '"sounioPolicyVerified":true' "$spawn_file" || \
    fail 'Beagle spawn did not receive native Sounio policy authority'
  policy_runtime="$(json_string_value "$spawn_file" sounioPolicyRuntimeDigest)"
  adapter_digest="$(hash_file "$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime")"
  [[ "${#policy_runtime}" -eq 64 && "$policy_runtime" == "$adapter_digest" ]] || \
    fail 'Beagle policy runtime digest does not match the native Sounio adapter'
  generation="$(json_string_value "$spawn_file" loomInstanceId)"
  [[ -n "$generation" ]] || fail "Beagle spawn omitted generation identity: $(cat "$spawn_file")"
  local status
  status="$(wait_loom_status)"
  [[ "$(field instance_id "$status")" == "$generation" ]] || \
    fail 'Beagle API and Loom status disagree on the generation identity'
  wait_endpoint "$generation" >/dev/null
  printf '%s\n' "$generation"
}

write_state() {
  local value="$1" temporary="$STATE_FILE.$$"
  printf '%s\n' "$value" > "$temporary"
  mv "$temporary" "$STATE_FILE"
}

load_state() {
  [[ -f "$STATE_FILE" ]] || fail 'canary phase state is missing'
  # The file contains only canary-generated alphanumeric identities.
  source "$STATE_FILE"
}

phase_one() {
  [[ ! -e "$STATE_FILE" ]] || fail 'phase one refuses an existing canary state'
  local generation send_output message_id message_status first_receipt retry snapshot
  local receipt_one adapter_digest
  generation="$(start_generation)"
  receipt_one="$(continuity_receipt_digest "$generation")"
  adapter_digest="$(json_string_value "$CANARY_ROOT/spawn-$POD_UID.json" sounioPolicyRuntimeDigest)"
  coord_retry scope --agent "$SENDER_AGENT" --lane "$SENDER_LANE" \
    --intent 'separate-Pod pending inbox replay canary' >/dev/null
  send_output="$(coord_retry send --agent "$SENDER_AGENT" --lane "$SENDER_LANE" \
    --to-agent "$AGENT" --to-lane "$LANE" --kind request \
    --message 'separate Pod replay canary: ACK only in successor generation')"
  message_id="$(sed -n 's/.*message_id=\([^ ]*\).*/\1/p' <<< "$send_output" | head -1)"
  [[ -n "$message_id" ]] || fail 'send omitted the durable message identity'
  snapshot="$(wait_snapshot "$message_id")"
  [[ "$snapshot" == *'Sounio coordination wake:'* ]] || \
    fail 'first Pod did not receive the wake metadata'
  message_status="$(coord_retry message-status --agent "$SENDER_AGENT" \
    --lane "$SENDER_LANE" --message "$message_id")"
  first_receipt="$(grep "^WAKE_RECEIPT message_id=$message_id " <<< "$message_status")"
  [[ "$(field generation "$first_receipt")" == "$generation" ]] || \
    fail 'first wake receipt was not bound to the first Pod generation'
  retry="$(coord_retry wake --agent "$SENDER_AGENT" --lane "$SENDER_LANE" \
    --message "$message_id")"
  [[ "$retry" == *'WAKE_SKIPPED'* && "$retry" == *'reason=already-delivered'* && \
     "$retry" == *"generation=$generation"* ]] || \
    fail "first Pod did not deduplicate its retry: $retry"
  write_state "phase=one
message_id=$message_id
pod_name_one=$POD_NAME
pod_uid_one=$POD_UID
generation_one=$generation"
  printf 'receipt_one=%s\nadapter_digest=%s\n' "$receipt_one" "$adapter_digest" >> "$STATE_FILE"
  printf 'CANARY_PHASE_ONE pod=%s pod_uid=%s generation=%s message_id=%s wake=delivered retry=deduplicated ack=absent\n' \
    "$POD_NAME" "$POD_UID" "$generation" "$message_id"
}

phase_two() {
  load_state
  [[ "${phase:-}" == one ]] || fail "phase two requires phase=one, got ${phase:-missing}"
  [[ "$POD_UID" != "$pod_uid_one" ]] || fail 'phase two is still running in the first Pod UID'
  local generation wake snapshot retry message_status receipt_count receipt_two current_adapter_digest
  generation="$(start_generation)"
  [[ "$generation" != "$generation_one" ]] || \
    fail 'successor Pod retained the predecessor Loom generation'
  receipt_two="$(continuity_receipt_digest "$generation")"
  [[ "$receipt_two" != "$receipt_one" ]] || \
    fail 'successor Pod reused the predecessor native Sounio continuity receipt'
  current_adapter_digest="$(json_string_value "$CANARY_ROOT/spawn-$POD_UID.json" sounioPolicyRuntimeDigest)"
  [[ "$current_adapter_digest" == "$adapter_digest" ]] || \
    fail 'native Sounio policy runtime changed between Pod generations'
  wake="$(coord_retry wake --agent "$SENDER_AGENT" --lane "$SENDER_LANE" \
    --message "$message_id")"
  [[ "$wake" == *'WAKE_DELIVERED'* && "$wake" == *"generation=$generation"* ]] || \
    fail "unacknowledged message did not replay into the successor Pod: $wake"
  snapshot="$(wait_snapshot "$message_id")"
  [[ "$snapshot" == *"$message_id"* ]] || fail 'successor Pod snapshot omitted the replay'
  retry="$(coord_retry wake --agent "$SENDER_AGENT" --lane "$SENDER_LANE" \
    --message "$message_id")"
  [[ "$retry" == *'WAKE_SKIPPED'* && "$retry" == *'reason=already-delivered'* && \
     "$retry" == *"generation=$generation"* ]] || \
    fail "successor Pod did not deduplicate its retry: $retry"
  message_status="$(coord_retry message-status --agent "$SENDER_AGENT" \
    --lane "$SENDER_LANE" --message "$message_id")"
  receipt_count="$(grep -c "^WAKE_RECEIPT message_id=$message_id " <<< "$message_status")"
  [[ "$message_status" == *'wakes=2'* && "$receipt_count" -eq 2 ]] || \
    fail "successor Pod did not retain two generation receipts: $message_status"
  grep -q "generation=$generation_one$" <<< "$message_status" || \
    fail 'successor Pod lost the first generation receipt'
  grep -q "generation=$generation$" <<< "$message_status" || \
    fail 'successor Pod omitted its own generation receipt'
  coord_retry ack --agent "$AGENT" --lane "$LANE" --message "$message_id" >/dev/null
  write_state "phase=two
message_id=$message_id
pod_name_one=$pod_name_one
pod_uid_one=$pod_uid_one
generation_one=$generation_one
pod_name_two=$POD_NAME
pod_uid_two=$POD_UID
generation_two=$generation"
  printf 'receipt_one=%s\nreceipt_two=%s\nadapter_digest=%s\n' \
    "$receipt_one" "$receipt_two" "$adapter_digest" >> "$STATE_FILE"
  printf 'CANARY_PHASE_TWO pod=%s pod_uid=%s generation=%s predecessor_generation=%s message_id=%s wake=replayed retry=deduplicated ack=durable receipts=2\n' \
    "$POD_NAME" "$POD_UID" "$generation" "$generation_one" "$message_id"
}

phase_three() {
  load_state
  [[ "${phase:-}" == two ]] || fail "phase three requires phase=two, got ${phase:-missing}"
  [[ "$POD_UID" != "$pod_uid_one" && "$POD_UID" != "$pod_uid_two" ]] || \
    fail 'phase three did not enter a distinct Pod UID'
  local generation retry snapshot message_status receipt_count result
  local receipt_three current_adapter_digest message_status_digest
  generation="$(start_generation)"
  [[ "$generation" != "$generation_one" && "$generation" != "$generation_two" ]] || \
    fail 'third Pod retained a predecessor Loom generation'
  receipt_three="$(continuity_receipt_digest "$generation")"
  [[ "$receipt_three" != "$receipt_one" && "$receipt_three" != "$receipt_two" ]] || \
    fail 'third Pod reused a predecessor native Sounio continuity receipt'
  current_adapter_digest="$(json_string_value "$CANARY_ROOT/spawn-$POD_UID.json" sounioPolicyRuntimeDigest)"
  [[ "$current_adapter_digest" == "$adapter_digest" ]] || \
    fail 'native Sounio policy runtime changed before the ACK control generation'
  retry="$(coord_retry wake --agent "$SENDER_AGENT" --lane "$SENDER_LANE" \
    --message "$message_id")"
  [[ "$retry" == *'WAKE_SKIPPED'* && "$retry" == *'reason=acknowledged'* ]] || \
    fail "durable ACK did not suppress third-Pod replay: $retry"
  sleep 0.2
  snapshot="$(SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" snapshot \
    --state-dir "$LOOM_DIR" --cwd "$ROOT_DIR" --agent "$AGENT" \
    --lane "$LANE" --cursor 0 2>/dev/null || true)"
  [[ "$snapshot" != *"$message_id"* ]] || \
    fail 'acknowledged message was injected into the third Pod'
  message_status="$(coord_retry message-status --agent "$SENDER_AGENT" \
    --lane "$SENDER_LANE" --message "$message_id")"
  receipt_count="$(grep -c "^WAKE_RECEIPT message_id=$message_id " <<< "$message_status")"
  [[ "$message_status" == *'acknowledged=1'* && "$message_status" == *'wakes=2'* && \
     "$receipt_count" -eq 2 ]] || \
    fail "ACK control changed the durable receipts: $message_status"
  message_status_digest="$(printf '%s\n' "$message_status" | sha256sum | awk '{print $1}')"
  result="SOUNIO_LOOM_SEPARATE_POD_REPLAY_PASS=true
schema=sounio-loom-separate-pod-replay-v1
source_commit=$(git -C "$ROOT_DIR" rev-parse HEAD)
message_id=$message_id
pod_name_one=$pod_name_one
pod_uid_one=$pod_uid_one
generation_one=$generation_one
pod_name_two=$pod_name_two
pod_uid_two=$pod_uid_two
generation_two=$generation_two
pod_name_three=$POD_NAME
pod_uid_three=$POD_UID
generation_three=$generation
native_sounio_policy_runtime_sha256=$adapter_digest
native_sounio_receipt_one_sha256=$receipt_one
native_sounio_receipt_two_sha256=$receipt_two
native_sounio_receipt_three_sha256=$receipt_three
message_status_sha256=$message_status_digest
wake_receipts=2
unacked_successor_replay=delivered
same_generation_retries=deduplicated
durable_ack=recorded
acked_third_generation_replay=suppressed"
  printf '%s\n' "$result" > "$RESULT_FILE"
  write_state "phase=complete
message_id=$message_id
pod_uid_one=$pod_uid_one
generation_one=$generation_one
pod_uid_two=$pod_uid_two
generation_two=$generation_two
pod_uid_three=$POD_UID
generation_three=$generation"
  SOUNIO_COORD_RUNTIME_MODE=local "$LOOM" stop --state-dir "$LOOM_DIR" \
    --cwd "$ROOT_DIR" --agent "$AGENT" --lane "$LANE" >/dev/null
  stop_bridge
  printf 'CANARY_PHASE_THREE pod=%s pod_uid=%s generation=%s message_id=%s wake=ack-suppressed receipts=2 result=%s\n' \
    "$POD_NAME" "$POD_UID" "$generation" "$message_id" "$RESULT_FILE"
}

command="${1:-}"
mkdir -p "$CANARY_ROOT"
exec 9>"$CANARY_ROOT/phase.lock"
flock -n 9 || fail 'another canary phase owns the persistent lock'

case "$command" in
  phase-one) prepare_runtime; phase_one ;;
  phase-two) prepare_runtime; phase_two ;;
  phase-three) prepare_runtime; phase_three ;;
  report) [[ -f "$RESULT_FILE" ]] || fail 'final result is missing'; cat "$RESULT_FILE" ;;
  *) fail 'usage: sounio_loom_pod_replay_canary.sh phase-one|phase-two|phase-three|report' ;;
esac
