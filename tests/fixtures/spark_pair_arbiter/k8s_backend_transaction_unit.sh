#!/usr/bin/env bash

set -euo pipefail
umask 077

[[ $# -eq 2 ]] || {
  printf 'usage: %s BACKEND WORK_DIR\n' "$0" >&2
  exit 64
}

BACKEND="$1"
WORK_DIR="$2"
mkdir -p "$WORK_DIR"
export SOUNIO_SPARK_PAIR_BACKEND_LIBRARY_MODE=1

# shellcheck source=/dev/null
source "$BACKEND"

POLICY="$WORK_DIR/policy"
FREEZE="$WORK_DIR/freeze"
receipt="$WORK_DIR/decision.receipt"
: > "$POLICY"
: > "$FREEZE"
: > "$receipt"

hex64() {
  local value="$1" index
  for ((index = 0; index < 64; index++)); do
    printf '%s' "$value"
  done
  printf '\n'
}
source_sha="$(hex64 a)"
freeze_sha="$(hex64 b)"
decision_sha="$(hex64 d)"
prepare0="$(hex64 e)"
prepare1="$(hex64 f)"
lease_uid=22222222-2222-4222-8222-222222222222

policy_value() {
  case "$2" in
    authority_sha256) printf '%s\n' "$source_sha" ;;
    *) printf 'unit-value\n' ;;
  esac
}
sha256_file() {
  if [[ "$1" == "$FREEZE" ]]; then printf '%s\n' "$freeze_sha"; else printf '%s\n' "$decision_sha"; fi
}
guard_mutation() { return 0; }
require_lease_context() { return 0; }
node0() { printf 'spark-3c59\n'; }
node1() { printf 'spark-8e54\n'; }
lease_json() {
  printf '{"metadata":{"uid":"%s","resourceVersion":"101"}}\n' "$lease_uid"
}
bind_host_pair_intent() {
  printf 'intent\n' >> "$TRACE"
  printf '102\n'
}
host_state_path() { printf '%s/%s.state\n' "$WORK_DIR" "$1"; }
write_host_state() {
  printf 'grant_mode=%s\ngrant_valid=%s\n' "$2" "$3" > "$(host_state_path "$1")"
}
host_fence_report() {
  local node="$1" state mode valid
  state="$(host_state_path "$node")"
  mode="$(sed -n 's/^grant_mode=//p' "$state")"
  valid="$(sed -n 's/^grant_valid=//p' "$state")"
  printf 'PIREUS_HOST_FACTS node=%s grant_mode=%s grant_valid=%s\n' \
    "$node" "$mode" "$valid"
}
bind_host_commit_receipts() {
  printf 'final-cas\n' >> "$TRACE"
  [[ "$SCENARIO" != cas-conflict ]]
}
host_fence_exec() {
  local node="$1" command="$2"
  printf '%s:%s\n' "$command" "$node" >> "$TRACE"
  case "$command:$node" in
    prepare:spark-3c59)
      printf 'PIREUS_HOST_PREPARED node=spark-3c59 prepare_receipt_sha256=%s\n' "$prepare0"
      ;;
    prepare:spark-8e54)
      printf 'PIREUS_HOST_PREPARED node=spark-8e54 prepare_receipt_sha256=%s\n' "$prepare1"
      ;;
    commit:spark-3c59)
      write_host_state spark-3c59 K8S 1
      [[ "$SCENARIO" != kill-after-commit-1 ]] || kill -KILL "$BASHPID"
      ;;
    commit:spark-8e54)
      write_host_state spark-8e54 K8S 1
      [[ "$SCENARIO" != kill-after-commit-2 ]] || kill -KILL "$BASHPID"
      ;;
    fence:spark-3c59|fence:spark-8e54)
      write_host_state "$node" FENCED 0
      ;;
  esac
}

run_scenario() {
  SCENARIO="$1"
  TRACE="$WORK_DIR/$SCENARIO.trace"
  export SCENARIO TRACE
  : > "$TRACE"
  write_host_state spark-3c59 FENCED 0
  write_host_state spark-8e54 FENCED 0
  set +e
  (grant_host_pair K8S --holder holder --epoch 7 --receipt "$receipt") \
    >/dev/null 2>&1
  status=$?
  set -e
  [[ $status -ne 0 ]] || {
    printf 'backend-transaction-unit: %s unexpectedly succeeded\n' "$SCENARIO" >&2
    exit 1
  }
  intent_line="$(grep -n '^intent$' "$TRACE" | cut -d: -f1)"
  commit0_line="$(grep -n '^commit:spark-3c59$' "$TRACE" | cut -d: -f1)"
  [[ -n "$intent_line" && -n "$commit0_line" && $intent_line -lt $commit0_line ]] || {
    printf 'backend-transaction-unit: %s authorized before durable intent\n' "$SCENARIO" >&2
    exit 1
  }
  case "$SCENARIO" in
    kill-after-commit-1)
      ! grep -Fq 'commit:spark-8e54' "$TRACE"
      ! grep -Fq 'final-cas' "$TRACE"
      [[ "$(sed -n 's/^grant_mode=//p' "$(host_state_path spark-3c59)")" == K8S ]]
      [[ "$(sed -n 's/^grant_mode=//p' "$(host_state_path spark-8e54)")" == FENCED ]]
      refence_host_pair_transaction holder 7 "$source_sha" "$freeze_sha" \
        "$(hex64 c)" "$lease_uid" 102 "$decision_sha"
      ;;
    kill-after-commit-2)
      grep -Fq 'commit:spark-8e54' "$TRACE"
      ! grep -Fq 'final-cas' "$TRACE"
      [[ "$(sed -n 's/^grant_mode=//p' "$(host_state_path spark-3c59)")" == K8S ]]
      [[ "$(sed -n 's/^grant_mode=//p' "$(host_state_path spark-8e54)")" == K8S ]]
      refence_host_pair_transaction holder 7 "$source_sha" "$freeze_sha" \
        "$(hex64 c)" "$lease_uid" 102 "$decision_sha"
      ;;
    cas-conflict)
      grep -Fq 'final-cas' "$TRACE"
      grep -Fq 'fence:spark-3c59' "$TRACE"
      grep -Fq 'fence:spark-8e54' "$TRACE"
      ;;
  esac
  [[ "$(host_fence_report spark-3c59)" == *'grant_mode=FENCED grant_valid=0' ]]
  [[ "$(host_fence_report spark-8e54)" == *'grant_mode=FENCED grant_valid=0' ]]
}

run_scenario kill-after-commit-1
run_scenario kill-after-commit-2
run_scenario cas-conflict

printf 'K8S_BACKEND_TRANSACTION_UNIT_PASS kill_after_commit_1=REFENCED kill_after_commit_2=REFENCED cas_conflict=REFENCED persisted_grants=PROVEN\n'
