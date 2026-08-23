#!/usr/bin/env bash

set -euo pipefail
umask 077

SOUNIO_COORD_PROTOCOL_VERSION=3
SOUNIO_COORD_RUNTIME_VERSION=2026.08.23.3

usage() {
  cat <<'USAGE'
Usage: bin/sounio-coord <command> [options]

Small shared coordination bus for Sounio worktrees. Claims live outside Git
so every worktree attached to the same repository can see the same leases.

Commands:
  runtime-version                 show the runtime protocol and implementation version
  brief                          show the startup-sized coordination summary
  status                         show claims, conflicts, and relevant worktrees
  status --all-worktrees         include a full (slower) worktree scan
  check                          status, then fail when active ownership conflicts exist
  claim   --agent ID --lane ID --intent TEXT
          [--resources RESOURCE ...] [--files PATH ...]
                                 reserve file and semantic resource sets for one lane
  scope   --agent ID --lane ID --intent TEXT
          [--resources RESOURCE ...] [--files PATH ...]
                                 create or extend a session-aware lease
  heartbeat --agent ID --lane ID
                                 refresh an existing claim
  release --agent ID --lane ID --reason TEXT
                                 release a claim and record the handoff event
  authorize --agent ID [--lane ID] [--resources RESOURCE ...] [--files PATH ...]
                                 verify that a local active claim covers the requested ownership
  endpoint-register --agent ID --lane ID --harness claude|codex
          --transport tmux --address PANE --socket PATH [--ttl-seconds N]
                                 register an expiring, verified delivery endpoint
  endpoint-unregister --agent ID --lane ID
                                 remove the lane's delivery endpoint
  endpoint-status --agent ID --lane ID
                                 inspect one delivery endpoint
  wake    --agent ID --lane ID --message MESSAGE_ID
                                 retry immediate delivery for a visible directed message
  handoff --agent ID --lane ID --to-agent ID --to-lane ID --message TEXT
          --commit SHA --gate NAME=PASS [--gate NAME=PASS ...]
          --evidence PATH [--evidence PATH ...] [--reply-to MESSAGE_ID]
                                 publish proof metadata, then release the owned claim
  send    --agent ID --lane ID [--to-agent ID] [--to-lane ID]
          [--thread ID] [--reply-to MESSAGE_ID] --kind KIND --message TEXT
                                 send a message to another lane or broadcast
  inbox   --agent ID --lane ID [--all] [--directed-only] [--newest-first]
          [--limit N] [--from-agent ID] [--from-lane ID] [--kind KIND]
          [--thread ID] [--since-epoch N]
                                 show unread messages for one lane
  injected --agent ID --lane ID --messages ID [ID ...]
                                 record that the hook surfaced messages to a lane
  ack     --agent ID --lane ID --message ID
                                 mark a message as read by one lane
  message-status --agent ID --lane ID --message ID
                                 show injection, acknowledgement, and request state
  wait    --agent ID --lane ID --message ID [--timeout-seconds N]
          [--poll-seconds N]
                                 wait for a reply, blocker, or handoff in the thread
  prune                          remove expired claims, messages, and delivery endpoints

Environment:
  SOUNIO_AGENT_ID                default agent identifier for claim commands
  SOUNIO_COORD_TTL_SECONDS       claim lease duration (default: 14400)
  SOUNIO_COORD_ENDPOINT_TTL_SECONDS
                                 delivery endpoint duration (default: 1800)
  SOUNIO_COORD_DIR               shared state directory override

Examples:
  bin/sounio-coord status
  bin/sounio-coord claim --agent codex-1 --lane parser-fix \
    --intent "isolate parser regression" --resources concept:parser diagnostic:E231 \
    --files 'self-hosted/parser/**' tests/compile-fail/foo.sio
  bin/sounio-coord scope --agent codex --lane session-abc123 \
    --intent "active Codex session" --files self-hosted/parser/ast.sio
  bin/sounio-coord heartbeat --agent codex-1 --lane parser-fix
  bin/sounio-coord endpoint-register --agent codex-1 --lane parser-fix \
    --harness codex --transport tmux --address "$TMUX_PANE" --socket "${TMUX%%,*}"
  bin/sounio-coord handoff --agent codex-1 --lane parser-fix \
    --to-agent claude --to-lane integration --message "parser fix ready" \
    --commit HEAD --gate parser-selftest=PASS --evidence artifacts/parser-gate.txt
  bin/sounio-coord send --agent codex-1 --lane parser-fix \
    --to-agent claude --kind request --message "Can you review the parser boundary?"
  bin/sounio-coord inbox --agent claude --lane integration
  bin/sounio-coord release --agent codex-1 --lane parser-fix --reason "handed to shepherd"
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_arg() {
  local option="$1"
  (($# >= 2)) || die "$option requires a value"
  [[ -n "$2" ]] || die "$option requires a non-empty value"
}

WORKTREE="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[[ -n "$WORKTREE" ]] || die "run this command from a Sounio worktree"
WORKTREE="$(cd "$WORKTREE" && pwd -P)"

GIT_COMMON_DIR="$(git -C "$WORKTREE" rev-parse --git-common-dir 2>/dev/null || true)"
[[ -n "$GIT_COMMON_DIR" ]] || die "cannot resolve the shared Git directory"
case "$GIT_COMMON_DIR" in
  /*) ;;
  *) GIT_COMMON_DIR="$(cd "$WORKTREE/$GIT_COMMON_DIR" && pwd -P)" ;;
esac

REPO_KEY="$(printf '%s' "$GIT_COMMON_DIR" | cksum | awk '{print $1}')"
STATE_DIR="${SOUNIO_COORD_DIR:-${TMPDIR:-/tmp}/sounio-coord/$REPO_KEY}"
CLAIMS_DIR="$STATE_DIR/claims"
MESSAGES_DIR="$STATE_DIR/messages"
ACKS_DIR="$STATE_DIR/message-acks"
INJECTIONS_DIR="$STATE_DIR/message-injections"
ENDPOINTS_DIR="$STATE_DIR/delivery-endpoints"
WAKES_DIR="$STATE_DIR/message-wakes"
EVENT_LOG="$STATE_DIR/events.log"
mkdir -p "$CLAIMS_DIR" "$MESSAGES_DIR" "$ACKS_DIR" "$INJECTIONS_DIR" \
  "$ENDPOINTS_DIR" "$WAKES_DIR"

NOW_EPOCH="$(date +%s)"
NOW_TICK="$(date +%s%N)"
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
shopt -s nullglob
LOCK_TO_CLEAN=''
LOCK_FD=''

cleanup_lock() {
  if [[ -n "$LOCK_FD" ]]; then
    flock -u "$LOCK_FD" 2>/dev/null || true
  fi
  if [[ -n "$LOCK_TO_CLEAN" ]]; then
    rmdir "$LOCK_TO_CLEAN" 2>/dev/null || true
    LOCK_TO_CLEAN=''
  fi
}

trap cleanup_lock EXIT

acquire_state_lock() {
  local action="$1" lock_dir lock_epoch
  if command -v flock >/dev/null 2>&1; then
    exec {LOCK_FD}>"$STATE_DIR/.claims.lock"
    flock -n "$LOCK_FD" || die "coordination state is being changed; retry $action"
    return 0
  fi

  lock_dir="$STATE_DIR/.claims.lock.d"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    lock_epoch="$(stat -c %Y "$lock_dir" 2>/dev/null || printf 0)"
    if [[ "$lock_epoch" =~ ^[0-9]+$ ]] && ((NOW_EPOCH - lock_epoch > 30)); then
      rmdir "$lock_dir" 2>/dev/null || true
    fi
    mkdir "$lock_dir" 2>/dev/null || die "coordination state is being changed; retry $action"
  fi
  LOCK_TO_CLEAN="$lock_dir"
}

current_branch() {
  local branch
  branch="$(git -C "$WORKTREE" branch --show-current 2>/dev/null || true)"
  if [[ -n "$branch" ]]; then
    printf '%s' "$branch"
  else
    printf 'detached@%s' "$(git -C "$WORKTREE" rev-parse --short=10 HEAD 2>/dev/null || printf unknown)"
  fi
}

current_sha() {
  git -C "$WORKTREE" rev-parse --short=12 HEAD 2>/dev/null || printf unknown
}

slug() {
  local value
  value="$(printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-80)"
  [[ -n "$value" ]] || value=unnamed
  printf '%s' "$value"
}

claim_id_for() {
  printf '%s--%s' "$(slug "$1")" "$(slug "$2")"
}

normalize_path() {
  local path="$1"
  case "$path" in
    "$WORKTREE"/*) printf '%s' "${path#"$WORKTREE"/}" ;;
    /*) printf '%s' "$path" ;;
    ./*) printf '%s' "${path#./}" ;;
    *) printf '%s' "$path" ;;
  esac
}

validate_value() {
  local label="$1" value="$2"
  case "$value" in
    *$'\n'*|*$'\r'*|*$'\t'*) die "$label cannot contain tabs or newlines" ;;
  esac
}

join_files() {
  local output='' file
  for file in "${C_FILES[@]}"; do
    if [[ -n "$output" ]]; then
      output="$output,$file"
    else
      output="$file"
    fi
  done
  printf '%s' "$output"
}

join_resources() {
  local output='' resource
  for resource in "${C_RESOURCES[@]}"; do
    if [[ -n "$output" ]]; then
      output="$output,$resource"
    else
      output="$resource"
    fi
  done
  printf '%s' "$output"
}

join_values() {
  local output='' value
  for value in "$@"; do
    if [[ -n "$output" ]]; then
      output="$output,$value"
    else
      output="$value"
    fi
  done
  printf '%s' "$output"
}

normalize_resource() {
  local resource="$1" kind value
  validate_value resource "$resource"
  [[ "$resource" == *:* ]] || \
    die "resource must use KIND:VALUE syntax: $resource"
  kind="${resource%%:*}"
  value="${resource#*:}"
  [[ "$kind" =~ ^(concept|diagnostic|gate|api)$ ]] || \
    die "resource kind must be concept, diagnostic, gate, or api: $resource"
  [[ -n "$value" && "$value" =~ ^[A-Za-z0-9._:/@+-]+(/\*\*)?$ ]] || \
    die "resource value contains unsupported characters: $resource"
  case "$value" in
    *\* ) [[ "$value" == */\*\* ]] || die "resource wildcard is only supported as a trailing /**: $resource" ;;
  esac
  printf '%s:%s' "$kind" "$value"
}

resource_kind() {
  printf '%s' "${1%%:*}"
}

resource_value() {
  printf '%s' "${1#*:}"
}

resource_is_wildcard() {
  [[ "$(resource_value "$1")" == */\*\* ]]
}

resource_scope() {
  local value
  value="$(resource_value "$1")"
  case "$value" in
    */\*\*) value="${value%/\*\*}" ;;
  esac
  while [[ "$value" == */ ]]; do value="${value%/}"; done
  printf '%s' "$value"
}

resources_overlap() {
  local left="$1" right="$2" left_scope right_scope
  [[ "$(resource_kind "$left")" == "$(resource_kind "$right")" ]] || return 1
  [[ "$left" == "$right" ]] && return 0
  left_scope="$(resource_scope "$left")"
  right_scope="$(resource_scope "$right")"
  if resource_is_wildcard "$left" && \
    [[ "$right_scope" == "$left_scope" || "$right_scope" == "$left_scope"/* ]]; then
    return 0
  fi
  if resource_is_wildcard "$right" && \
    [[ "$left_scope" == "$right_scope" || "$left_scope" == "$right_scope"/* ]]; then
    return 0
  fi
  return 1
}

resource_covers() {
  local claimed="$1" requested="$2" claimed_scope requested_scope
  [[ "$(resource_kind "$claimed")" == "$(resource_kind "$requested")" ]] || return 1
  [[ "$claimed" == "$requested" ]] && return 0
  resource_is_wildcard "$claimed" || return 1
  claimed_scope="$(resource_scope "$claimed")"
  requested_scope="$(resource_scope "$requested")"
  [[ "$requested_scope" == "$claimed_scope" || "$requested_scope" == "$claimed_scope"/* ]]
}

load_claim() {
  local claim_file="$1" line
  C_ID=''
  C_AGENT=''
  C_LANE=''
  C_WORKTREE=''
  C_BRANCH=''
  C_SHA=''
  C_CREATED_UTC=''
  C_LAST_UTC=''
  C_LAST_EPOCH=0
  C_TTL=0
  C_INTENT=''
  C_FILES=()
  C_RESOURCES=()

  # Unreadable claim file (e.g. written by a root session): treat as empty so
  # callers skip it via claim_expired instead of dying on the read.
  [[ -r "$claim_file" ]] || return 0

  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      claim_id=*) C_ID="${line#claim_id=}" ;;
      agent=*) C_AGENT="${line#agent=}" ;;
      lane=*) C_LANE="${line#lane=}" ;;
      worktree=*) C_WORKTREE="${line#worktree=}" ;;
      branch=*) C_BRANCH="${line#branch=}" ;;
      sha=*) C_SHA="${line#sha=}" ;;
      created_utc=*) C_CREATED_UTC="${line#created_utc=}" ;;
      last_seen_utc=*) C_LAST_UTC="${line#last_seen_utc=}" ;;
      last_seen_epoch=*) C_LAST_EPOCH="${line#last_seen_epoch=}" ;;
      ttl_seconds=*) C_TTL="${line#ttl_seconds=}" ;;
      intent=*) C_INTENT="${line#intent=}" ;;
      file=*) C_FILES+=("${line#file=}") ;;
      resource=*) C_RESOURCES+=("${line#resource=}") ;;
    esac
  done < "$claim_file"
}

claim_expired() {
  local ttl="${C_TTL:-0}" last="${C_LAST_EPOCH:-0}"
  [[ "$ttl" =~ ^[0-9]+$ && "$last" =~ ^[0-9]+$ ]] || return 0
  if (( NOW_EPOCH > last + ttl )); then
    return 0
  fi
  return 1
}

path_scope() {
  local path="$1"
  case "$path" in
    */\*\*) path="${path%/**}" ;;
  esac
  while [[ "$path" == */ ]]; do path="${path%/}"; done
  printf '%s' "$path"
}

paths_overlap() {
  local left right
  left="$(path_scope "$1")"
  right="$(path_scope "$2")"
  [[ "$left" == "$right" ]] && return 0
  [[ "$left" == "$right"/* ]] && return 0
  [[ "$right" == "$left"/* ]] && return 0
  return 1
}

path_covers() {
  local claimed requested
  claimed="$(path_scope "$1")"
  requested="$(path_scope "$2")"
  [[ -n "$claimed" && -n "$requested" ]] || return 1
  [[ "$claimed" == "$requested" ]] && return 0
  [[ "$requested" == "$claimed"/* ]] && return 0
  return 1
}

claim_files_clean() {
  local file scope status
  for file in "${C_FILES[@]}"; do
    case "$file" in
      "$WORKTREE"/*) scope="${file#"$WORKTREE"/}" ;;
      /*) continue ;;
      *) scope="$file" ;;
    esac
    scope="$(path_scope "$scope")"
    [[ -n "$scope" ]] || return 1
    status="$(git -C "$WORKTREE" status --porcelain=v1 --untracked-files=all -- "$scope")" || \
      return 1
    [[ -z "$status" ]] || return 1
  done
  return 0
}

write_claim() {
  local claim_file="$1" tmp_file file
  tmp_file="$(mktemp "$CLAIMS_DIR/.claim-write.XXXXXX")"
  {
    printf 'claim_id=%s\n' "$C_ID"
    printf 'agent=%s\n' "$C_AGENT"
    printf 'lane=%s\n' "$C_LANE"
    printf 'worktree=%s\n' "$C_WORKTREE"
    printf 'branch=%s\n' "$C_BRANCH"
    printf 'sha=%s\n' "$C_SHA"
    printf 'created_utc=%s\n' "$C_CREATED_UTC"
    printf 'last_seen_utc=%s\n' "$C_LAST_UTC"
    printf 'last_seen_epoch=%s\n' "$C_LAST_EPOCH"
    printf 'ttl_seconds=%s\n' "$C_TTL"
    printf 'intent=%s\n' "$C_INTENT"
    for file in "${C_FILES[@]}"; do
      [[ -n "$file" ]] && printf 'file=%s\n' "$file"
    done
    for resource in "${C_RESOURCES[@]}"; do
      [[ -n "$resource" ]] && printf 'resource=%s\n' "$resource"
    done
  } > "$tmp_file"
  mv "$tmp_file" "$claim_file"
}

array_contains() {
  local needle="$1" item
  shift
  for item in "$@"; do
    [[ "$item" == "$needle" ]] && return 0
  done
  return 1
}

append_event() {
  local event="$1" reason="${2:-}"
  printf 'utc=%s event=%s agent=%s lane=%s worktree=%s branch=%s sha=%s files=%s resources=%s intent=%s reason=%s\n' \
    "$NOW_UTC" "$event" "$C_AGENT" "$C_LANE" "$C_WORKTREE" "$C_BRANCH" \
    "$C_SHA" "$(join_files)" "$(join_resources)" "$C_INTENT" "$reason" >> "$EVENT_LOG"
}

load_message() {
  local message_file="$1" line
  M_ID=''
  M_CREATED_UTC=''
  M_CREATED_EPOCH=0
  M_TTL=0
  M_FROM_AGENT=''
  M_FROM_LANE=''
  M_FROM_WORKTREE=''
  M_FROM_BRANCH=''
  M_TO_AGENT=''
  M_TO_LANE=''
  M_KIND=''
  M_TEXT=''
  M_THREAD_ID=''
  M_REPLY_TO=''
  M_COMMIT_SHA=''
  M_GATES=()
  M_EVIDENCE=()
  M_CLAIM_FILES=()
  M_CLAIM_RESOURCES=()

  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      message_id=*) M_ID="${line#message_id=}" ;;
      created_utc=*) M_CREATED_UTC="${line#created_utc=}" ;;
      created_epoch=*) M_CREATED_EPOCH="${line#created_epoch=}" ;;
      ttl_seconds=*) M_TTL="${line#ttl_seconds=}" ;;
      from_agent=*) M_FROM_AGENT="${line#from_agent=}" ;;
      from_lane=*) M_FROM_LANE="${line#from_lane=}" ;;
      from_worktree=*) M_FROM_WORKTREE="${line#from_worktree=}" ;;
      from_branch=*) M_FROM_BRANCH="${line#from_branch=}" ;;
      to_agent=*) M_TO_AGENT="${line#to_agent=}" ;;
      to_lane=*) M_TO_LANE="${line#to_lane=}" ;;
      kind=*) M_KIND="${line#kind=}" ;;
      text=*) M_TEXT="${line#text=}" ;;
      thread_id=*) M_THREAD_ID="${line#thread_id=}" ;;
      reply_to=*) M_REPLY_TO="${line#reply_to=}" ;;
      commit_sha=*) M_COMMIT_SHA="${line#commit_sha=}" ;;
      gate=*) M_GATES+=("${line#gate=}") ;;
      evidence=*) M_EVIDENCE+=("${line#evidence=}") ;;
      claim_file=*) M_CLAIM_FILES+=("${line#claim_file=}") ;;
      claim_resource=*) M_CLAIM_RESOURCES+=("${line#claim_resource=}") ;;
    esac
  done < "$message_file"
  [[ -n "$M_THREAD_ID" ]] || M_THREAD_ID="$M_ID"
}

message_expired() {
  [[ "$M_TTL" =~ ^[0-9]+$ && "$M_CREATED_EPOCH" =~ ^[0-9]+$ ]] || return 0
  ((NOW_EPOCH > M_CREATED_EPOCH + M_TTL))
}

message_ack_path() {
  printf '%s/%s--%s--%s.ack' "$ACKS_DIR" "$(slug "$1")" "$(slug "$2")" "$(slug "$3")"
}

message_injection_path() {
  printf '%s/%s--%s--%s.injected' \
    "$INJECTIONS_DIR" "$(slug "$1")" "$(slug "$2")" "$(slug "$3")"
}

endpoint_path() {
  printf '%s/%s.endpoint' "$ENDPOINTS_DIR" "$(claim_id_for "$1" "$2")"
}

wake_receipt_path() {
  printf '%s/%s--%s.wake' "$WAKES_DIR" "$(slug "$1")" "$(slug "$2")"
}

load_endpoint() {
  local endpoint_file="$1" line
  E_ID=''
  E_AGENT=''
  E_LANE=''
  E_WORKTREE=''
  E_HARNESS=''
  E_TRANSPORT=''
  E_ADDRESS=''
  E_SOCKET=''
  E_PANE_PID=''
  E_COMMAND=''
  E_CREATED_UTC=''
  E_LAST_UTC=''
  E_LAST_EPOCH=0
  E_TTL=0
  [[ -r "$endpoint_file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      endpoint_id=*) E_ID="${line#endpoint_id=}" ;;
      agent=*) E_AGENT="${line#agent=}" ;;
      lane=*) E_LANE="${line#lane=}" ;;
      worktree=*) E_WORKTREE="${line#worktree=}" ;;
      harness=*) E_HARNESS="${line#harness=}" ;;
      transport=*) E_TRANSPORT="${line#transport=}" ;;
      address=*) E_ADDRESS="${line#address=}" ;;
      socket=*) E_SOCKET="${line#socket=}" ;;
      pane_pid=*) E_PANE_PID="${line#pane_pid=}" ;;
      command=*) E_COMMAND="${line#command=}" ;;
      created_utc=*) E_CREATED_UTC="${line#created_utc=}" ;;
      last_seen_utc=*) E_LAST_UTC="${line#last_seen_utc=}" ;;
      last_seen_epoch=*) E_LAST_EPOCH="${line#last_seen_epoch=}" ;;
      ttl_seconds=*) E_TTL="${line#ttl_seconds=}" ;;
    esac
  done < "$endpoint_file"
}

endpoint_expired() {
  local ttl="${E_TTL:-0}" last="${E_LAST_EPOCH:-0}"
  [[ "$ttl" =~ ^[0-9]+$ && "$last" =~ ^[0-9]+$ ]] || return 0
  ((NOW_EPOCH > last + ttl))
}

write_endpoint() {
  local endpoint_file="$1" tmp_file
  tmp_file="$(mktemp "$ENDPOINTS_DIR/.endpoint-write.XXXXXX")"
  {
    printf 'endpoint_id=%s\n' "$E_ID"
    printf 'agent=%s\n' "$E_AGENT"
    printf 'lane=%s\n' "$E_LANE"
    printf 'worktree=%s\n' "$E_WORKTREE"
    printf 'harness=%s\n' "$E_HARNESS"
    printf 'transport=%s\n' "$E_TRANSPORT"
    printf 'address=%s\n' "$E_ADDRESS"
    printf 'socket=%s\n' "$E_SOCKET"
    printf 'pane_pid=%s\n' "$E_PANE_PID"
    printf 'command=%s\n' "$E_COMMAND"
    printf 'created_utc=%s\n' "$E_CREATED_UTC"
    printf 'last_seen_utc=%s\n' "$E_LAST_UTC"
    printf 'last_seen_epoch=%s\n' "$E_LAST_EPOCH"
    printf 'ttl_seconds=%s\n' "$E_TTL"
  } > "$tmp_file"
  mv "$tmp_file" "$endpoint_file"
}

tmux_endpoint_snapshot() {
  local socket="$1" address="$2" pane_line pane_id pane_pid pane_command pane_path pane_root
  pane_line="$(tmux -S "$socket" display-message -p -t "$address" \
    '#{pane_id}|#{pane_pid}|#{pane_current_command}|#{pane_current_path}' 2>/dev/null || true)"
  IFS='|' read -r pane_id pane_pid pane_command pane_path <<< "$pane_line"
  [[ -n "$pane_id" && "$pane_pid" =~ ^[1-9][0-9]*$ && -n "$pane_command" && -n "$pane_path" ]] || \
    return 1
  pane_root="$(git -C "$pane_path" rev-parse --show-toplevel 2>/dev/null || true)"
  [[ -n "$pane_root" ]] || return 1
  pane_root="$(cd "$pane_root" && pwd -P)"
  [[ "$pane_root" == "$WORKTREE" ]] || return 1
  T_PANE_ID="$pane_id"
  T_PANE_PID="$pane_pid"
  T_COMMAND="$pane_command"
  T_PATH="$pane_path"
}

harness_command_matches() {
  local harness="$1" command="$2"
  case "$harness" in
    claude) [[ "$command" == claude* ]] ;;
    codex) [[ "$command" == codex* || "$command" == node ]] ;;
    *) return 1 ;;
  esac
}

ENDPOINT_STATE='unavailable'
endpoint_state() {
  local current_pane current_pid current_command current_path current_root
  ENDPOINT_STATE='unavailable'
  if endpoint_expired; then
    ENDPOINT_STATE='stale'
    return 1
  fi
  [[ "$E_TRANSPORT" == tmux && -S "$E_SOCKET" ]] || return 1
  current_pane="$(tmux -S "$E_SOCKET" display-message -p -t "$E_ADDRESS" '#{pane_id}' 2>/dev/null || true)"
  current_pid="$(tmux -S "$E_SOCKET" display-message -p -t "$E_ADDRESS" '#{pane_pid}' 2>/dev/null || true)"
  current_command="$(tmux -S "$E_SOCKET" display-message -p -t "$E_ADDRESS" '#{pane_current_command}' 2>/dev/null || true)"
  current_path="$(tmux -S "$E_SOCKET" display-message -p -t "$E_ADDRESS" '#{pane_current_path}' 2>/dev/null || true)"
  if [[ "$current_pane" != "$E_ADDRESS" || "$current_pid" != "$E_PANE_PID" || \
    "$current_command" != "$E_COMMAND" || -z "$current_path" ]] || \
    ! harness_command_matches "$E_HARNESS" "$current_command"; then
    ENDPOINT_STATE='drifted'
    return 1
  fi
  current_root="$(git -C "$current_path" rev-parse --show-toplevel 2>/dev/null || true)"
  [[ -n "$current_root" ]] && current_root="$(cd "$current_root" && pwd -P)"
  if [[ "$current_root" != "$E_WORKTREE" ]]; then
    ENDPOINT_STATE='drifted'
    return 1
  fi
  ENDPOINT_STATE='active'
}

remove_endpoint_for_lane() {
  local agent="$1" lane="$2" worktree="$3" reason="$4" endpoint_file
  endpoint_file="$(endpoint_path "$agent" "$lane")"
  [[ -f "$endpoint_file" ]] || return 0
  load_endpoint "$endpoint_file"
  [[ "$E_AGENT" == "$agent" && "$E_LANE" == "$lane" ]] || die "endpoint owner mismatch"
  [[ "$E_WORKTREE" == "$worktree" ]] || die "endpoint belongs to worktree $E_WORKTREE"
  unlink "$endpoint_file"
  printf 'utc=%s event=ENDPOINT_UNREGISTERED endpoint_id=%s agent=%s lane=%s worktree=%s reason=%s\n' \
    "$NOW_UTC" "$E_ID" "$E_AGENT" "$E_LANE" "$E_WORKTREE" "$reason" >> "$EVENT_LOG"
}

WAKE_STATUS='unavailable'
attempt_message_wake() {
  local message_id="$1" message_file endpoint_file receipt_file prompt tmp_file
  WAKE_STATUS='unavailable'
  message_file="$MESSAGES_DIR/$(slug "$message_id").message"
  [[ -f "$message_file" ]] || return 1
  load_message "$message_file"
  [[ -n "$M_TO_AGENT" && -n "$M_TO_LANE" ]] || return 1
  endpoint_file="$(endpoint_path "$M_TO_AGENT" "$M_TO_LANE")"
  [[ -f "$endpoint_file" ]] || return 1
  load_endpoint "$endpoint_file"
  if ! endpoint_state; then
    WAKE_STATUS="$ENDPOINT_STATE"
    if [[ "$ENDPOINT_STATE" == drifted ]]; then
      printf 'WAKE_REFUSED message_id=%s endpoint_id=%s reason=endpoint-drift\n' "$M_ID" "$E_ID" >&2
      WAKE_STATUS='failed-closed'
    fi
    return 1
  fi
  receipt_file="$(wake_receipt_path "$M_ID" "$E_ID")"
  if [[ -f "$receipt_file" ]]; then
    WAKE_STATUS='deduplicated'
    printf 'WAKE_SKIPPED message_id=%s endpoint_id=%s reason=already-delivered\n' "$M_ID" "$E_ID"
    return 0
  fi

  prompt="Sounio coordination wake: $M_KIND $M_ID from $(slug "$M_FROM_AGENT")/$(slug "$M_FROM_LANE") is waiting. Run bin/sounio-coord inbox --agent $E_AGENT --lane $E_LANE --directed-only --newest-first, then reply or ack $M_ID."
  if ! tmux -S "$E_SOCKET" send-keys -t "$E_ADDRESS" -l "$prompt" 2>/dev/null || \
    ! tmux -S "$E_SOCKET" send-keys -t "$E_ADDRESS" Enter 2>/dev/null; then
    WAKE_STATUS='failed'
    printf 'WAKE_FAILED message_id=%s endpoint_id=%s transport=tmux\n' "$M_ID" "$E_ID" >&2
    return 1
  fi

  tmp_file="$(mktemp "$WAKES_DIR/.wake-write.XXXXXX")"
  printf 'utc=%s message_id=%s endpoint_id=%s transport=%s address=%s\n' \
    "$NOW_UTC" "$M_ID" "$E_ID" "$E_TRANSPORT" "$E_ADDRESS" > "$tmp_file"
  mv "$tmp_file" "$receipt_file"
  printf 'utc=%s event=WAKE_DELIVERED message_id=%s endpoint_id=%s agent=%s lane=%s transport=%s address=%s\n' \
    "$NOW_UTC" "$M_ID" "$E_ID" "$E_AGENT" "$E_LANE" "$E_TRANSPORT" "$E_ADDRESS" >> "$EVENT_LOG"
  WAKE_STATUS='delivered'
  printf 'WAKE_DELIVERED message_id=%s endpoint_id=%s transport=%s address=%s\n' \
    "$M_ID" "$E_ID" "$E_TRANSPORT" "$E_ADDRESS"
}

print_message_line() {
  printf 'MESSAGE id=%s utc=%s from_agent=%s from_lane=%s kind=%s text=%s thread=%s reply_to=%s' \
    "$M_ID" "$M_CREATED_UTC" "$M_FROM_AGENT" "$M_FROM_LANE" "$M_KIND" "$M_TEXT" \
    "$M_THREAD_ID" "${M_REPLY_TO:--}"
  if [[ "$M_KIND" == handoff && -n "$M_COMMIT_SHA" ]]; then
    printf ' commit=%s gates=%s evidence=%s files=%s resources=%s' \
      "$M_COMMIT_SHA" "$(join_values "${M_GATES[@]}")" \
      "$(join_values "${M_EVIDENCE[@]}")" "$(join_values "${M_CLAIM_FILES[@]}")" \
      "$(join_values "${M_CLAIM_RESOURCES[@]}")"
  fi
  printf '\n'
}

claim_paths=()
refresh_claim_paths() {
  claim_paths=("$CLAIMS_DIR"/*.claim)
}

worktree_has_claim() {
  local candidate="$1" claim_file
  refresh_claim_paths
  for claim_file in "${claim_paths[@]}"; do
    [[ -f "$claim_file" ]] || continue
    load_claim "$claim_file"
    claim_expired && continue
    [[ "$C_WORKTREE" == "$candidate" ]] && return 0
  done
  return 1
}

worktree_in_list() {
  local candidate="$1" item
  for item in "${inspect_worktrees[@]}"; do
    [[ "$item" == "$candidate" ]] && return 0
  done
  return 1
}

claim_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' intent='' ttl="${SOUNIO_COORD_TTL_SECONDS:-14400}"
  local file resource claim_file existing old_file new_file old_resource new_resource
  local file_conflict=0 resource_conflict=0
  local files=() resources=()

  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --intent) require_arg "$1" "$2"; intent="$2"; shift 2 ;;
      --ttl-seconds) require_arg "$1" "$2"; ttl="$2"; shift 2 ;;
      --resources)
        shift
        (($#)) || die "--resources requires at least one resource"
        while (($#)) && [[ "$1" != --files ]]; do
          [[ "$1" != --* ]] || die "put command options before --resources and --files"
          resources+=("$(normalize_resource "$1")")
          shift
        done
        ;;
      --files)
        shift
        while (($#)); do files+=("$(normalize_path "$1")"); shift; done
        ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown claim option: $1 (put paths after --files)" ;;
    esac
  done

  [[ -n "$agent" ]] || die "claim requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "claim requires --lane"
  [[ -n "$intent" ]] || die "claim requires --intent"
  ((${#files[@]} + ${#resources[@]} > 0)) || \
    die "claim requires at least one resource or file"
  [[ "$ttl" =~ ^[1-9][0-9]*$ ]] || die "--ttl-seconds must be a positive integer"
  validate_value agent "$agent"
  validate_value lane "$lane"
  validate_value intent "$intent"
  for file in "${files[@]}"; do validate_value file "$file"; done
  for resource in "${resources[@]}"; do validate_value resource "$resource"; done

  C_ID="$(claim_id_for "$agent" "$lane")"
  claim_file="$CLAIMS_DIR/$C_ID.claim"
  acquire_state_lock "the claim"
  [[ ! -e "$claim_file" ]] || die "claim already exists: $C_ID (use heartbeat or release it first)"

  refresh_claim_paths
  for existing in "${claim_paths[@]}"; do
    [[ -f "$existing" ]] || continue
    load_claim "$existing"
    claim_expired && continue
    [[ "$C_ID" == "$(claim_id_for "$agent" "$lane")" ]] && continue
    for new_file in "${files[@]}"; do
      for old_file in "${C_FILES[@]}"; do
        if paths_overlap "$new_file" "$old_file"; then
          printf 'CONFLICT existing_claim=%s agent=%s lane=%s path=%s requested_by=%s requested_lane=%s\n' \
            "$C_ID" "$C_AGENT" "$C_LANE" "$old_file" "$agent" "$lane" >&2
          file_conflict=1
        fi
      done
    done
    for new_resource in "${resources[@]}"; do
      for old_resource in "${C_RESOURCES[@]}"; do
        if resources_overlap "$new_resource" "$old_resource"; then
          printf 'CONFLICT existing_claim=%s agent=%s lane=%s resource=%s requested_resource=%s requested_by=%s requested_lane=%s\n' \
            "$C_ID" "$C_AGENT" "$C_LANE" "$old_resource" "$new_resource" "$agent" "$lane" >&2
          resource_conflict=1
        fi
      done
    done
  done
  ((file_conflict == 0)) || die "requested file set overlaps an active claim"
  ((resource_conflict == 0)) || die "requested semantic resource set overlaps an active claim"

  C_AGENT="$agent"
  C_LANE="$lane"
  C_WORKTREE="$WORKTREE"
  C_BRANCH="$(current_branch)"
  C_SHA="$(current_sha)"
  C_CREATED_UTC="$NOW_UTC"
  C_LAST_UTC="$NOW_UTC"
  C_LAST_EPOCH="$NOW_EPOCH"
  C_TTL="$ttl"
  C_INTENT="$intent"
  C_FILES=("${files[@]}")
  C_RESOURCES=("${resources[@]}")
  C_ID="$(claim_id_for "$agent" "$lane")"
  write_claim "$claim_file"
  append_event CLAIM
  printf 'CLAIMED claim_id=%s\n' "$C_ID"
  printf 'agent=%s lane=%s worktree=%s branch=%s sha=%s\n' "$agent" "$lane" "$WORKTREE" "$C_BRANCH" "$C_SHA"
  printf 'files=%s\n' "$(join_files)"
  printf 'resources=%s\n' "$(join_resources)"
  printf 'state_dir=%s\n' "$STATE_DIR"
  printf 'next=bin/sounio-coord heartbeat --agent %s --lane %s\n' "$agent" "$lane"
}

scope_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' intent='' ttl="${SOUNIO_COORD_TTL_SECONDS:-14400}"
  local file resource claim_file existing old_file new_file old_resource new_resource
  local requested_id created_utc event='SCOPE' file_conflict=0 resource_conflict=0
  local files=() merged_files=() existing_files=()
  local resources=() merged_resources=() existing_resources=()

  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --intent) require_arg "$1" "$2"; intent="$2"; shift 2 ;;
      --ttl-seconds) require_arg "$1" "$2"; ttl="$2"; shift 2 ;;
      --resources)
        shift
        (($#)) || die "--resources requires at least one resource"
        while (($#)) && [[ "$1" != --files ]]; do
          [[ "$1" != --* ]] || die "put command options before --resources and --files"
          resources+=("$(normalize_resource "$1")")
          shift
        done
        ;;
      --files)
        shift
        while (($#)); do files+=("$(normalize_path "$1")"); shift; done
        ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown scope option: $1 (put paths after --files)" ;;
    esac
  done

  [[ -n "$agent" ]] || die "scope requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "scope requires --lane"
  [[ -n "$intent" ]] || die "scope requires --intent"
  [[ "$ttl" =~ ^[1-9][0-9]*$ ]] || die "--ttl-seconds must be a positive integer"
  validate_value agent "$agent"
  validate_value lane "$lane"
  validate_value intent "$intent"
  for file in "${files[@]}"; do validate_value file "$file"; done
  for resource in "${resources[@]}"; do validate_value resource "$resource"; done

  requested_id="$(claim_id_for "$agent" "$lane")"
  claim_file="$CLAIMS_DIR/$requested_id.claim"
  acquire_state_lock "the scope update"

  if [[ -f "$claim_file" ]]; then
    load_claim "$claim_file"
    if claim_expired; then
      append_event EXPIRED "lease expired before scope refresh"
      unlink "$claim_file"
      created_utc="$NOW_UTC"
      event='CLAIM'
    else
      [[ "$C_AGENT" == "$agent" ]] || die "claim owner mismatch: $C_AGENT"
      [[ "$C_WORKTREE" == "$WORKTREE" ]] || die "claim belongs to worktree $C_WORKTREE"
      [[ "$C_BRANCH" == "$(current_branch)" ]] || die "branch changed from $C_BRANCH; release and scope again"
      existing_files=("${C_FILES[@]}")
      existing_resources=("${C_RESOURCES[@]}")
      created_utc="${C_CREATED_UTC:-$NOW_UTC}"
    fi
  else
    created_utc="$NOW_UTC"
    event='CLAIM'
  fi

  merged_files=("${existing_files[@]}")
  for file in "${files[@]}"; do
    [[ -n "$file" ]] || continue
    if ! array_contains "$file" "${merged_files[@]}"; then
      merged_files+=("$file")
    fi
  done

  merged_resources=("${existing_resources[@]}")
  for resource in "${resources[@]}"; do
    [[ -n "$resource" ]] || continue
    if ! array_contains "$resource" "${merged_resources[@]}"; then
      merged_resources+=("$resource")
    fi
  done

  refresh_claim_paths
  for existing in "${claim_paths[@]}"; do
    [[ -f "$existing" && "$existing" != "$claim_file" ]] || continue
    load_claim "$existing"
    claim_expired && continue
    for new_file in "${merged_files[@]}"; do
      [[ -n "$new_file" ]] || continue
      for old_file in "${C_FILES[@]}"; do
        [[ -n "$old_file" ]] || continue
        if paths_overlap "$new_file" "$old_file"; then
          printf 'CONFLICT existing_claim=%s agent=%s lane=%s path=%s requested_by=%s requested_lane=%s\n' \
            "$C_ID" "$C_AGENT" "$C_LANE" "$old_file" "$agent" "$lane" >&2
          file_conflict=1
        fi
      done
    done
    for new_resource in "${merged_resources[@]}"; do
      [[ -n "$new_resource" ]] || continue
      for old_resource in "${C_RESOURCES[@]}"; do
        [[ -n "$old_resource" ]] || continue
        if resources_overlap "$new_resource" "$old_resource"; then
          printf 'CONFLICT existing_claim=%s agent=%s lane=%s resource=%s requested_resource=%s requested_by=%s requested_lane=%s\n' \
            "$C_ID" "$C_AGENT" "$C_LANE" "$old_resource" "$new_resource" "$agent" "$lane" >&2
          resource_conflict=1
        fi
      done
    done
  done
  ((file_conflict == 0)) || die "requested file set overlaps an active claim"
  ((resource_conflict == 0)) || die "requested semantic resource set overlaps an active claim"

  C_ID="$requested_id"
  C_AGENT="$agent"
  C_LANE="$lane"
  C_WORKTREE="$WORKTREE"
  C_BRANCH="$(current_branch)"
  C_SHA="$(current_sha)"
  C_CREATED_UTC="$created_utc"
  C_LAST_UTC="$NOW_UTC"
  C_LAST_EPOCH="$NOW_EPOCH"
  C_TTL="$ttl"
  C_INTENT="$intent"
  C_FILES=("${merged_files[@]}")
  C_RESOURCES=("${merged_resources[@]}")
  write_claim "$claim_file"
  append_event "$event"
  printf 'SCOPED claim_id=%s files=%s resources=%s last_seen=%s\n' \
    "$C_ID" "$(join_files)" "$(join_resources)" "$C_LAST_UTC"
}

heartbeat_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' claim_file
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown heartbeat option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "heartbeat requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "heartbeat requires --lane"
  claim_file="$CLAIMS_DIR/$(claim_id_for "$agent" "$lane").claim"
  [[ -f "$claim_file" ]] || die "claim not found: $(claim_id_for "$agent" "$lane")"
  load_claim "$claim_file"
  [[ "$C_AGENT" == "$agent" ]] || die "claim owner mismatch: $C_AGENT"
  [[ "$C_WORKTREE" == "$WORKTREE" ]] || die "claim belongs to worktree $C_WORKTREE"
  [[ "$C_BRANCH" == "$(current_branch)" ]] || die "branch changed from $C_BRANCH; release and claim again"

  acquire_state_lock "the heartbeat"
  C_LAST_UTC="$NOW_UTC"
  C_LAST_EPOCH="$NOW_EPOCH"
  C_SHA="$(current_sha)"
  write_claim "$claim_file"
  append_event HEARTBEAT
  printf 'HEARTBEAT claim_id=%s last_seen=%s sha=%s\n' "$C_ID" "$C_LAST_UTC" "$C_SHA"
}

release_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' reason='' claim_file
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --reason) require_arg "$1" "$2"; reason="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown release option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "release requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "release requires --lane"
  [[ -n "$reason" ]] || die "release requires --reason"
  validate_value reason "$reason"
  claim_file="$CLAIMS_DIR/$(claim_id_for "$agent" "$lane").claim"
  [[ -f "$claim_file" ]] || die "claim not found: $(claim_id_for "$agent" "$lane")"
  load_claim "$claim_file"
  [[ "$C_AGENT" == "$agent" ]] || die "claim owner mismatch: $C_AGENT"
  [[ "$C_WORKTREE" == "$WORKTREE" ]] || die "claim belongs to worktree $C_WORKTREE"
  acquire_state_lock "the release"
  C_BRANCH="$(current_branch)"
  C_SHA="$(current_sha)"
  append_event RELEASE "$reason"
  unlink "$claim_file"
  remove_endpoint_for_lane "$agent" "$lane" "$WORKTREE" release
  printf 'RELEASED claim_id=%s reason=%s\n' "$C_ID" "${reason:-unspecified}"
}

authorize_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' requested claim_file claimed_file claimed_resource covered
  local file_conflict=0 resource_conflict=0
  local files=() resources=()

  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --resources)
        shift
        (($#)) || die "--resources requires at least one resource"
        while (($#)) && [[ "$1" != --files ]]; do
          [[ "$1" != --* ]] || die "put command options before --resources and --files"
          resources+=("$(normalize_resource "$1")")
          shift
        done
        ;;
      --files)
        shift
        while (($#)); do files+=("$(normalize_path "$1")"); shift; done
        ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown authorize option: $1 (put paths after --files)" ;;
    esac
  done

  [[ -n "$agent" ]] || die "authorize requires --agent or SOUNIO_AGENT_ID"
  ((${#files[@]} + ${#resources[@]} > 0)) || \
    die "authorize requires at least one resource or file"
  validate_value agent "$agent"
  validate_value lane "$lane"
  for requested in "${files[@]}"; do validate_value file "$requested"; done
  for requested in "${resources[@]}"; do validate_value resource "$requested"; done

  refresh_claim_paths
  for claim_file in "${claim_paths[@]}"; do
    [[ -f "$claim_file" ]] || continue
    load_claim "$claim_file"
    claim_expired && continue
    [[ "$C_AGENT" == "$agent" ]] || continue
    [[ -z "$lane" || "$C_LANE" == "$lane" ]] || continue
    [[ "$C_WORKTREE" == "$WORKTREE" ]] || continue
    [[ "$C_BRANCH" == "$(current_branch)" ]] || continue

    covered=1
    for requested in "${files[@]}"; do
      for claimed_file in "${C_FILES[@]}"; do
        if path_covers "$claimed_file" "$requested"; then
          continue 2
        fi
      done
      covered=0
      break
    done
    if ((covered)); then
      for requested in "${resources[@]}"; do
        for claimed_resource in "${C_RESOURCES[@]}"; do
          if resource_covers "$claimed_resource" "$requested"; then
            continue 2
          fi
        done
        covered=0
        break
      done
    fi
    if ((covered)); then
      printf 'AUTHORIZED claim_id=%s agent=%s lane=%s worktree=%s branch=%s\n' \
        "$C_ID" "$C_AGENT" "$C_LANE" "$C_WORKTREE" "$C_BRANCH"
      return 0
    fi
  done

  for claim_file in "${claim_paths[@]}"; do
    [[ -f "$claim_file" ]] || continue
    load_claim "$claim_file"
    claim_expired && continue
    for requested in "${files[@]}"; do
      for claimed_file in "${C_FILES[@]}"; do
        if paths_overlap "$requested" "$claimed_file"; then
          printf 'CONFLICT existing_claim=%s agent=%s lane=%s path=%s requested_by=%s requested_lane=%s\n' \
            "$C_ID" "$C_AGENT" "$C_LANE" "$claimed_file" "$agent" "${lane:-any}" >&2
          file_conflict=1
        fi
      done
    done
    for requested in "${resources[@]}"; do
      for claimed_resource in "${C_RESOURCES[@]}"; do
        if resources_overlap "$requested" "$claimed_resource"; then
          printf 'CONFLICT existing_claim=%s agent=%s lane=%s resource=%s requested_resource=%s requested_by=%s requested_lane=%s\n' \
            "$C_ID" "$C_AGENT" "$C_LANE" "$claimed_resource" "$requested" "$agent" "${lane:-any}" >&2
          resource_conflict=1
        fi
      done
    done
  done
  ((file_conflict == 0)) || die "requested file set overlaps an active claim but is not authorized"
  ((resource_conflict == 0)) || \
    die "requested semantic resource set overlaps an active claim but is not authorized"
  die "no active claim in worktree $WORKTREE covers the requested ownership for agent $agent"
}

endpoint_register_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' harness='' transport='' address='' socket=''
  local ttl="${SOUNIO_COORD_ENDPOINT_TTL_SECONDS:-1800}" claim_file endpoint_file created_utc
  local existing_endpoint
  local -a endpoint_paths=()
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --harness) require_arg "$1" "$2"; harness="$2"; shift 2 ;;
      --transport) require_arg "$1" "$2"; transport="$2"; shift 2 ;;
      --address) require_arg "$1" "$2"; address="$2"; shift 2 ;;
      --socket) require_arg "$1" "$2"; socket="$2"; shift 2 ;;
      --ttl-seconds) require_arg "$1" "$2"; ttl="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown endpoint-register option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "endpoint-register requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "endpoint-register requires --lane"
  [[ "$harness" =~ ^(claude|codex)$ ]] || die "--harness must be claude or codex"
  [[ "$transport" == tmux ]] || die "--transport currently supports only tmux"
  [[ -n "$address" ]] || die "endpoint-register requires --address"
  [[ -n "$socket" ]] || die "endpoint-register requires --socket"
  [[ "$ttl" =~ ^[1-9][0-9]*$ ]] || die "--ttl-seconds must be a positive integer"
  validate_value agent "$agent"
  validate_value lane "$lane"
  [[ "$agent" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "endpoint agent contains unsupported characters: $agent"
  [[ "$lane" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "endpoint lane contains unsupported characters: $lane"
  validate_value address "$address"
  validate_value socket "$socket"
  socket="$(readlink -f "$socket" 2>/dev/null || true)"
  [[ -n "$socket" && -S "$socket" ]] || die "tmux socket is not available: ${socket:-missing}"
  tmux_endpoint_snapshot "$socket" "$address" || \
    die "tmux endpoint does not resolve to the current worktree"
  harness_command_matches "$harness" "$T_COMMAND" || \
    die "tmux pane command $T_COMMAND does not match harness $harness"

  claim_file="$CLAIMS_DIR/$(claim_id_for "$agent" "$lane").claim"
  endpoint_file="$(endpoint_path "$agent" "$lane")"
  acquire_state_lock "the endpoint registration"
  [[ -f "$claim_file" ]] || die "claim not found: $(claim_id_for "$agent" "$lane")"
  load_claim "$claim_file"
  claim_expired && die "claim expired before endpoint registration: $C_ID"
  [[ "$C_AGENT" == "$agent" && "$C_LANE" == "$lane" ]] || die "claim owner mismatch"
  [[ "$C_WORKTREE" == "$WORKTREE" ]] || die "claim belongs to worktree $C_WORKTREE"
  endpoint_paths=("$ENDPOINTS_DIR"/*.endpoint)
  for existing_endpoint in "${endpoint_paths[@]}"; do
    [[ -f "$existing_endpoint" && "$existing_endpoint" != "$endpoint_file" ]] || continue
    load_endpoint "$existing_endpoint"
    endpoint_expired && continue
    if [[ "$E_SOCKET" == "$socket" && "$E_ADDRESS" == "$T_PANE_ID" ]]; then
      die "tmux endpoint is already owned by $E_AGENT/$E_LANE"
    fi
  done
  created_utc="$NOW_UTC"
  if [[ -f "$endpoint_file" ]]; then
    load_endpoint "$endpoint_file"
    [[ "$E_AGENT" == "$agent" && "$E_LANE" == "$lane" ]] || die "endpoint owner mismatch"
    [[ "$E_WORKTREE" == "$WORKTREE" ]] || die "endpoint belongs to worktree $E_WORKTREE"
    created_utc="${E_CREATED_UTC:-$NOW_UTC}"
  fi
  E_ID="$(claim_id_for "$agent" "$lane")"
  E_AGENT="$agent"
  E_LANE="$lane"
  E_WORKTREE="$WORKTREE"
  E_HARNESS="$harness"
  E_TRANSPORT="$transport"
  E_ADDRESS="$T_PANE_ID"
  E_SOCKET="$socket"
  E_PANE_PID="$T_PANE_PID"
  E_COMMAND="$T_COMMAND"
  E_CREATED_UTC="$created_utc"
  E_LAST_UTC="$NOW_UTC"
  E_LAST_EPOCH="$NOW_EPOCH"
  E_TTL="$ttl"
  write_endpoint "$endpoint_file"
  printf 'utc=%s event=ENDPOINT_REGISTERED endpoint_id=%s agent=%s lane=%s worktree=%s harness=%s transport=%s address=%s\n' \
    "$NOW_UTC" "$E_ID" "$E_AGENT" "$E_LANE" "$E_WORKTREE" "$E_HARNESS" \
    "$E_TRANSPORT" "$E_ADDRESS" >> "$EVENT_LOG"
  printf 'ENDPOINT_REGISTERED endpoint_id=%s harness=%s transport=%s address=%s pane_pid=%s command=%s expires_in=%s\n' \
    "$E_ID" "$E_HARNESS" "$E_TRANSPORT" "$E_ADDRESS" "$E_PANE_PID" "$E_COMMAND" "$E_TTL"
}

endpoint_unregister_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' endpoint_file
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown endpoint-unregister option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "endpoint-unregister requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "endpoint-unregister requires --lane"
  endpoint_file="$(endpoint_path "$agent" "$lane")"
  acquire_state_lock "the endpoint removal"
  if [[ ! -f "$endpoint_file" ]]; then
    printf 'ENDPOINT_ABSENT endpoint_id=%s\n' "$(claim_id_for "$agent" "$lane")"
    return 0
  fi
  load_endpoint "$endpoint_file"
  [[ "$E_AGENT" == "$agent" && "$E_LANE" == "$lane" ]] || die "endpoint owner mismatch"
  [[ "$E_WORKTREE" == "$WORKTREE" ]] || die "endpoint belongs to worktree $E_WORKTREE"
  unlink "$endpoint_file"
  printf 'utc=%s event=ENDPOINT_UNREGISTERED endpoint_id=%s agent=%s lane=%s worktree=%s\n' \
    "$NOW_UTC" "$E_ID" "$E_AGENT" "$E_LANE" "$E_WORKTREE" >> "$EVENT_LOG"
  printf 'ENDPOINT_UNREGISTERED endpoint_id=%s\n' "$E_ID"
}

endpoint_status_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' endpoint_file state
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown endpoint-status option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "endpoint-status requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "endpoint-status requires --lane"
  endpoint_file="$(endpoint_path "$agent" "$lane")"
  [[ -f "$endpoint_file" ]] || die "endpoint not found: $(claim_id_for "$agent" "$lane")"
  load_endpoint "$endpoint_file"
  endpoint_state || true
  state="$ENDPOINT_STATE"
  printf 'ENDPOINT_STATUS endpoint_id=%s state=%s agent=%s lane=%s worktree=%s harness=%s transport=%s address=%s pane_pid=%s command=%s last_seen=%s\n' \
    "$E_ID" "$state" "$E_AGENT" "$E_LANE" "$E_WORKTREE" "$E_HARNESS" \
    "$E_TRANSPORT" "$E_ADDRESS" "$E_PANE_PID" "$E_COMMAND" "$E_LAST_UTC"
}

wake_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' message_id='' message_file
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --message) require_arg "$1" "$2"; message_id="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown wake option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "wake requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "wake requires --lane"
  [[ -n "$message_id" ]] || die "wake requires --message"
  message_file="$MESSAGES_DIR/$(slug "$message_id").message"
  [[ -f "$message_file" ]] || die "message not found: $message_id"
  load_message "$message_file"
  if [[ "$M_FROM_AGENT" != "$agent" || "$M_FROM_LANE" != "$lane" ]]; then
    [[ "$M_TO_AGENT" == "$agent" && "$M_TO_LANE" == "$lane" ]] || \
      die "message wake is not visible to $agent/$lane"
  fi
  acquire_state_lock "the message wake"
  if attempt_message_wake "$message_id"; then
    return 0
  fi
  printf 'WAKE_UNAVAILABLE message_id=%s status=%s\n' "$message_id" "$WAKE_STATUS" >&2
  return 3
}

send_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' to_agent='' to_lane=''
  local kind='info' message='' ttl="${SOUNIO_COORD_MESSAGE_TTL_SECONDS:-604800}"
  local thread_id='' reply_to='' message_id message_file tmp_file reply_file

  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --to-agent) require_arg "$1" "$2"; to_agent="$2"; shift 2 ;;
      --to-lane) require_arg "$1" "$2"; to_lane="$2"; shift 2 ;;
      --thread) require_arg "$1" "$2"; thread_id="$2"; shift 2 ;;
      --reply-to) require_arg "$1" "$2"; reply_to="$2"; shift 2 ;;
      --kind) require_arg "$1" "$2"; kind="$2"; shift 2 ;;
      --message) require_arg "$1" "$2"; message="$2"; shift 2 ;;
      --ttl-seconds) require_arg "$1" "$2"; ttl="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown send option: $1" ;;
    esac
  done

  [[ -n "$agent" ]] || die "send requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "send requires --lane"
  [[ -n "$message" ]] || die "send requires --message"
  [[ "$kind" =~ ^(info|request|reply|blocker|handoff)$ ]] || \
    die "--kind must be info, request, reply, blocker, or handoff"
  [[ "$kind" != reply || -n "$reply_to" ]] || \
    die "--kind reply requires --reply-to so the request thread remains observable"
  [[ "$ttl" =~ ^[1-9][0-9]*$ ]] || die "--ttl-seconds must be a positive integer"
  validate_value agent "$agent"
  validate_value lane "$lane"
  validate_value to-agent "$to_agent"
  validate_value to-lane "$to_lane"
  validate_value thread "$thread_id"
  validate_value reply-to "$reply_to"
  validate_value message "$message"

  message_id="msg-${NOW_TICK}-$$-${RANDOM}"
  message_file="$MESSAGES_DIR/$message_id.message"
  acquire_state_lock "the message"
  if [[ -n "$reply_to" ]]; then
    reply_file="$MESSAGES_DIR/$(slug "$reply_to").message"
    [[ -f "$reply_file" ]] || die "reply message not found: $reply_to"
    load_message "$reply_file"
    [[ "$M_ID" == "$reply_to" ]] || die "reply message not found: $reply_to"
    [[ -n "$thread_id" ]] || thread_id="$M_THREAD_ID"
    [[ -n "$to_agent" ]] || to_agent="$M_FROM_AGENT"
    [[ -n "$to_lane" ]] || to_lane="$M_FROM_LANE"
  fi
  [[ -n "$thread_id" ]] || thread_id="$message_id"
  validate_value to-agent "$to_agent"
  validate_value to-lane "$to_lane"
  tmp_file="$(mktemp "$MESSAGES_DIR/.message-write.XXXXXX")"
  {
    printf 'message_id=%s\n' "$message_id"
    printf 'created_utc=%s\n' "$NOW_UTC"
    printf 'created_epoch=%s\n' "$NOW_EPOCH"
    printf 'ttl_seconds=%s\n' "$ttl"
    printf 'from_agent=%s\n' "$agent"
    printf 'from_lane=%s\n' "$lane"
    printf 'from_worktree=%s\n' "$WORKTREE"
    printf 'from_branch=%s\n' "$(current_branch)"
    printf 'to_agent=%s\n' "$to_agent"
    printf 'to_lane=%s\n' "$to_lane"
    printf 'kind=%s\n' "$kind"
    printf 'text=%s\n' "$message"
    printf 'thread_id=%s\n' "$thread_id"
    printf 'reply_to=%s\n' "$reply_to"
  } > "$tmp_file"
  mv "$tmp_file" "$message_file"
  printf 'utc=%s event=MESSAGE message_id=%s from_agent=%s from_lane=%s to_agent=%s to_lane=%s kind=%s\n' \
    "$NOW_UTC" "$message_id" "$agent" "$lane" "$to_agent" "$to_lane" "$kind" >> "$EVENT_LOG"
  printf 'SENT message_id=%s to_agent=%s to_lane=%s kind=%s thread_id=%s reply_to=%s\n' \
    "$message_id" "${to_agent:-*}" "${to_lane:-*}" "$kind" "$thread_id" "${reply_to:--}"
  if [[ -n "$to_agent" && -n "$to_lane" ]] && ! attempt_message_wake "$message_id"; then
    printf 'WAKE_UNAVAILABLE message_id=%s status=%s\n' "$message_id" "$WAKE_STATUS"
  fi
}

handoff_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' to_agent='' to_lane='' message=''
  local commit='' commit_sha='' head_sha='' reply_to='' thread_id=''
  local ttl="${SOUNIO_COORD_MESSAGE_TTL_SECONDS:-604800}"
  local gate evidence evidence_path claim_file reply_file message_id message_file tmp_file
  local -a gates=() evidence_paths=()

  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --to-agent) require_arg "$1" "$2"; to_agent="$2"; shift 2 ;;
      --to-lane) require_arg "$1" "$2"; to_lane="$2"; shift 2 ;;
      --message) require_arg "$1" "$2"; message="$2"; shift 2 ;;
      --commit) require_arg "$1" "$2"; commit="$2"; shift 2 ;;
      --gate) require_arg "$1" "$2"; gates+=("$2"); shift 2 ;;
      --evidence) require_arg "$1" "$2"; evidence_paths+=("$(normalize_path "$2")"); shift 2 ;;
      --reply-to) require_arg "$1" "$2"; reply_to="$2"; shift 2 ;;
      --ttl-seconds) require_arg "$1" "$2"; ttl="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown handoff option: $1" ;;
    esac
  done

  [[ -n "$agent" ]] || die "handoff requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "handoff requires --lane"
  [[ -n "$to_agent" ]] || die "handoff requires --to-agent"
  [[ -n "$to_lane" ]] || die "handoff requires --to-lane"
  [[ -n "$message" ]] || die "handoff requires --message"
  [[ -n "$commit" ]] || die "handoff requires --commit"
  ((${#gates[@]} > 0)) || die "handoff requires at least one --gate NAME=PASS"
  ((${#evidence_paths[@]} > 0)) || die "handoff requires at least one --evidence PATH"
  [[ "$ttl" =~ ^[1-9][0-9]*$ ]] || die "--ttl-seconds must be a positive integer"
  validate_value agent "$agent"
  validate_value lane "$lane"
  validate_value to-agent "$to_agent"
  validate_value to-lane "$to_lane"
  validate_value message "$message"
  validate_value reply-to "$reply_to"
  for gate in "${gates[@]}"; do
    validate_value gate "$gate"
    [[ "$gate" =~ ^[A-Za-z0-9._:/+-]+=PASS$ ]] || \
      die "handoff gates must use NAME=PASS: $gate"
  done
  for evidence in "${evidence_paths[@]}"; do
    validate_value evidence "$evidence"
    case "$evidence" in
      /*) evidence_path="$evidence" ;;
      *) evidence_path="$WORKTREE/$evidence" ;;
    esac
    [[ -e "$evidence_path" ]] || die "handoff evidence does not exist: $evidence"
  done

  commit_sha="$(git -C "$WORKTREE" rev-parse --verify "$commit^{commit}" 2>/dev/null || true)"
  [[ -n "$commit_sha" ]] || die "handoff commit does not resolve to a commit: $commit"
  head_sha="$(git -C "$WORKTREE" rev-parse HEAD 2>/dev/null || true)"
  [[ "$commit_sha" == "$head_sha" ]] || \
    die "handoff commit must equal current HEAD: commit=$commit_sha head=$head_sha"

  claim_file="$CLAIMS_DIR/$(claim_id_for "$agent" "$lane").claim"
  message_id="msg-${NOW_TICK}-$$-${RANDOM}"
  message_file="$MESSAGES_DIR/$message_id.message"
  acquire_state_lock "the proof-carrying handoff"
  [[ -f "$claim_file" ]] || die "claim not found: $(claim_id_for "$agent" "$lane")"
  load_claim "$claim_file"
  claim_expired && die "claim expired before handoff: $C_ID"
  [[ "$C_AGENT" == "$agent" ]] || die "claim owner mismatch: $C_AGENT"
  [[ "$C_LANE" == "$lane" ]] || die "claim lane mismatch: $C_LANE"
  [[ "$C_WORKTREE" == "$WORKTREE" ]] || die "claim belongs to worktree $C_WORKTREE"
  [[ "$C_BRANCH" == "$(current_branch)" ]] || \
    die "branch changed from $C_BRANCH; release and scope again"
  [[ "$commit_sha" == "$(git -C "$WORKTREE" rev-parse HEAD 2>/dev/null || true)" ]] || \
    die "HEAD changed while preparing the handoff; retry with the new commit"
  claim_files_clean || die "handoff claim files differ from commit $commit_sha"

  if [[ -n "$reply_to" ]]; then
    reply_file="$MESSAGES_DIR/$(slug "$reply_to").message"
    [[ -f "$reply_file" ]] || die "reply message not found: $reply_to"
    load_message "$reply_file"
    [[ "$M_ID" == "$reply_to" ]] || die "reply message not found: $reply_to"
    [[ "$M_KIND" == request ]] || die "handoff --reply-to must reference a request"
    [[ -z "$M_TO_AGENT" || "$M_TO_AGENT" == "$agent" ]] || \
      die "handoff request is addressed to agent $M_TO_AGENT"
    [[ -z "$M_TO_LANE" || "$M_TO_LANE" == "$lane" ]] || \
      die "handoff request is addressed to lane $M_TO_LANE"
    [[ "$M_FROM_AGENT" == "$to_agent" && "$M_FROM_LANE" == "$to_lane" ]] || \
      die "handoff destination must match the requesting lane"
    thread_id="$M_THREAD_ID"
  fi
  [[ -n "$thread_id" ]] || thread_id="$message_id"

  tmp_file="$(mktemp "$MESSAGES_DIR/.message-write.XXXXXX")"
  {
    printf 'message_id=%s\n' "$message_id"
    printf 'created_utc=%s\n' "$NOW_UTC"
    printf 'created_epoch=%s\n' "$NOW_EPOCH"
    printf 'ttl_seconds=%s\n' "$ttl"
    printf 'from_agent=%s\n' "$agent"
    printf 'from_lane=%s\n' "$lane"
    printf 'from_worktree=%s\n' "$WORKTREE"
    printf 'from_branch=%s\n' "$(current_branch)"
    printf 'to_agent=%s\n' "$to_agent"
    printf 'to_lane=%s\n' "$to_lane"
    printf 'kind=handoff\n'
    printf 'text=%s\n' "$message"
    printf 'thread_id=%s\n' "$thread_id"
    printf 'reply_to=%s\n' "$reply_to"
    printf 'commit_sha=%s\n' "$commit_sha"
    for gate in "${gates[@]}"; do printf 'gate=%s\n' "$gate"; done
    for evidence in "${evidence_paths[@]}"; do printf 'evidence=%s\n' "$evidence"; done
    for evidence in "${C_FILES[@]}"; do printf 'claim_file=%s\n' "$evidence"; done
    for evidence in "${C_RESOURCES[@]}"; do printf 'claim_resource=%s\n' "$evidence"; done
  } > "$tmp_file"
  mv "$tmp_file" "$message_file"

  C_BRANCH="$(current_branch)"
  C_SHA="${commit_sha:0:12}"
  append_event HANDOFF "message_id=$message_id commit=$commit_sha"
  unlink "$claim_file"
  remove_endpoint_for_lane "$agent" "$lane" "$WORKTREE" handoff
  printf 'HANDED_OFF claim_id=%s message_id=%s commit=%s to_agent=%s to_lane=%s gates=%s evidence=%s\n' \
    "$C_ID" "$message_id" "$commit_sha" "$to_agent" "$to_lane" \
    "$(join_values "${gates[@]}")" "$(join_values "${evidence_paths[@]}")"
  if ! attempt_message_wake "$message_id"; then
    printf 'WAKE_UNAVAILABLE message_id=%s status=%s\n' "$message_id" "$WAKE_STATUS"
  fi
}

inbox_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' show_all=0 directed_only=0 newest_first=0 limit_set=0
  local limit=0 from_agent='' from_lane='' kind='' thread_id='' since_epoch=0
  local message_file ack_file shown=0 matching=0 omitted=0 index
  local -a message_paths=() matching_paths=() ordered_paths=()
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --all) show_all=1; shift ;;
      --directed-only) directed_only=1; shift ;;
      --newest-first) newest_first=1; shift ;;
      --limit) require_arg "$1" "$2"; limit="$2"; limit_set=1; shift 2 ;;
      --from-agent) require_arg "$1" "$2"; from_agent="$2"; shift 2 ;;
      --from-lane) require_arg "$1" "$2"; from_lane="$2"; shift 2 ;;
      --kind) require_arg "$1" "$2"; kind="$2"; shift 2 ;;
      --thread) require_arg "$1" "$2"; thread_id="$2"; shift 2 ;;
      --since-epoch) require_arg "$1" "$2"; since_epoch="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown inbox option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "inbox requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "inbox requires --lane"
  ((limit_set == 0)) || [[ "$limit" =~ ^[1-9][0-9]*$ ]] || \
    die "--limit must be a positive integer"
  [[ "$since_epoch" =~ ^[0-9]+$ ]] || die "--since-epoch must be a non-negative integer"
  [[ -z "$kind" || "$kind" =~ ^(info|request|reply|blocker|handoff)$ ]] || \
    die "--kind must be info, request, reply, blocker, or handoff"
  validate_value from-agent "$from_agent"
  validate_value from-lane "$from_lane"
  validate_value thread "$thread_id"

  message_paths=("$MESSAGES_DIR"/*.message)
  for message_file in "${message_paths[@]}"; do
    [[ -f "$message_file" ]] || continue
    load_message "$message_file"
    message_expired && continue
    [[ -z "$M_TO_AGENT" || "$M_TO_AGENT" == "$agent" ]] || continue
    [[ -z "$M_TO_LANE" || "$M_TO_LANE" == "$lane" ]] || continue
    [[ "$M_FROM_AGENT" != "$agent" || "$M_FROM_LANE" != "$lane" ]] || continue
    ((directed_only == 0)) || [[ -n "$M_TO_AGENT" || -n "$M_TO_LANE" ]] || continue
    [[ -z "$from_agent" || "$M_FROM_AGENT" == "$from_agent" ]] || continue
    [[ -z "$from_lane" || "$M_FROM_LANE" == "$from_lane" ]] || continue
    [[ -z "$kind" || "$M_KIND" == "$kind" ]] || continue
    [[ -z "$thread_id" || "$M_THREAD_ID" == "$thread_id" ]] || continue
    ((M_CREATED_EPOCH >= since_epoch)) || continue
    ack_file="$(message_ack_path "$M_ID" "$agent" "$lane")"
    ((show_all)) || [[ ! -f "$ack_file" ]] || continue
    matching_paths+=("$message_file")
  done

  matching="${#matching_paths[@]}"
  if ((newest_first)); then
    for ((index = matching - 1; index >= 0; index--)); do
      ordered_paths+=("${matching_paths[index]}")
    done
  else
    ordered_paths=("${matching_paths[@]}")
  fi

  for message_file in "${ordered_paths[@]}"; do
    ((limit == 0 || shown < limit)) || break
    load_message "$message_file"
    print_message_line
    shown=$((shown + 1))
  done
  omitted=$((matching - shown))
  printf 'inbox_messages=%s\n' "$shown"
  printf 'inbox_matching=%s\n' "$matching"
  printf 'inbox_omitted=%s\n' "$omitted"
}

injected_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' message_id message_file injection_file index
  local -a message_ids=() injection_files=()
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --messages)
        shift
        (($#)) || die "--messages requires at least one message id"
        message_ids=("$@")
        set --
        ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown injected option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "injected requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "injected requires --lane"
  ((${#message_ids[@]} > 0)) || die "injected requires --messages"
  validate_value agent "$agent"
  validate_value lane "$lane"

  for message_id in "${message_ids[@]}"; do
    validate_value message "$message_id"
    message_file="$MESSAGES_DIR/$(slug "$message_id").message"
    [[ -f "$message_file" ]] || die "message not found: $message_id"
    load_message "$message_file"
    [[ "$M_ID" == "$message_id" ]] || die "message not found: $message_id"
    [[ -z "$M_TO_AGENT" || "$M_TO_AGENT" == "$agent" ]] || \
      die "message is addressed to agent $M_TO_AGENT"
    [[ -z "$M_TO_LANE" || "$M_TO_LANE" == "$lane" ]] || \
      die "message is addressed to lane $M_TO_LANE"
    injection_files+=("$(message_injection_path "$message_id" "$agent" "$lane")")
  done

  acquire_state_lock "the injection receipt"
  for ((index = 0; index < ${#message_ids[@]}; index++)); do
    message_id="${message_ids[index]}"
    injection_file="${injection_files[index]}"
    if [[ ! -f "$injection_file" ]]; then
      printf 'utc=%s agent=%s lane=%s\n' "$NOW_UTC" "$agent" "$lane" > "$injection_file"
    fi
    printf 'INJECTED message_id=%s agent=%s lane=%s\n' "$message_id" "$agent" "$lane"
  done
}

message_status_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' message_id='' message_file receipt_file
  local original_from_agent original_from_lane original_to_agent original_to_lane
  local original_kind original_thread original_epoch request_state latest_response='-'
  local latest_kind='' latest_epoch=0 latest_file='' responses=0 injected=0 acknowledged=0 wakes=0
  local receipt_utc receipt_agent receipt_lane token_utc token_agent token_lane
  local token_message token_endpoint token_transport token_address
  local -a message_paths=() injection_paths=() ack_paths=() wake_paths=()
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --message) require_arg "$1" "$2"; message_id="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown message-status option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "message-status requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "message-status requires --lane"
  [[ -n "$message_id" ]] || die "message-status requires --message"
  message_file="$MESSAGES_DIR/$(slug "$message_id").message"
  [[ -f "$message_file" ]] || die "message not found: $message_id"
  load_message "$message_file"
  [[ "$M_ID" == "$message_id" ]] || die "message not found: $message_id"
  if [[ "$M_FROM_AGENT" != "$agent" || "$M_FROM_LANE" != "$lane" ]]; then
    [[ -z "$M_TO_AGENT" || "$M_TO_AGENT" == "$agent" ]] || \
      die "message status is not visible to agent $agent"
    [[ -z "$M_TO_LANE" || "$M_TO_LANE" == "$lane" ]] || \
      die "message status is not visible to lane $lane"
  fi

  original_from_agent="$M_FROM_AGENT"
  original_from_lane="$M_FROM_LANE"
  original_to_agent="$M_TO_AGENT"
  original_to_lane="$M_TO_LANE"
  original_kind="$M_KIND"
  original_thread="$M_THREAD_ID"
  original_epoch="$M_CREATED_EPOCH"
  request_state='not-request'
  [[ "$original_kind" != request ]] || request_state='open'

  message_paths=("$MESSAGES_DIR"/*.message)
  for message_file in "${message_paths[@]}"; do
    [[ -f "$message_file" ]] || continue
    load_message "$message_file"
    message_expired && continue
    [[ "$M_ID" != "$message_id" && "$M_THREAD_ID" == "$original_thread" ]] || continue
    ((M_CREATED_EPOCH >= original_epoch)) || continue
    [[ "$M_FROM_AGENT" != "$original_from_agent" || "$M_FROM_LANE" != "$original_from_lane" ]] || continue
    [[ -z "$original_to_agent" || "$M_FROM_AGENT" == "$original_to_agent" ]] || continue
    [[ -z "$original_to_lane" || "$M_FROM_LANE" == "$original_to_lane" ]] || continue
    [[ -z "$M_TO_AGENT" || "$M_TO_AGENT" == "$original_from_agent" ]] || continue
    [[ -z "$M_TO_LANE" || "$M_TO_LANE" == "$original_from_lane" ]] || continue
    [[ "$M_KIND" =~ ^(reply|blocker|handoff)$ ]] || continue
    responses=$((responses + 1))
    if ((M_CREATED_EPOCH > latest_epoch)) || \
      { ((M_CREATED_EPOCH == latest_epoch)) && [[ "$message_file" > "$latest_file" ]]; }; then
      latest_epoch="$M_CREATED_EPOCH"
      latest_file="$message_file"
      latest_response="$M_ID"
      latest_kind="$M_KIND"
    fi
  done
  if [[ "$original_kind" == request && -n "$latest_kind" ]]; then
    if [[ "$latest_kind" == blocker ]]; then
      request_state='blocked'
    else
      request_state='answered'
    fi
  fi

  injection_paths=("$INJECTIONS_DIR/$(slug "$message_id")"--*.injected)
  ack_paths=("$ACKS_DIR/$(slug "$message_id")"--*.ack)
  wake_paths=("$WAKES_DIR/$(slug "$message_id")"--*.wake)
  injected="${#injection_paths[@]}"
  acknowledged="${#ack_paths[@]}"
  wakes="${#wake_paths[@]}"
  printf 'MESSAGE_STATUS id=%s kind=%s thread=%s request_state=%s injected=%s acknowledged=%s responses=%s latest_response=%s wakes=%s\n' \
    "$message_id" "$original_kind" "$original_thread" "$request_state" "$injected" \
    "$acknowledged" "$responses" "$latest_response" "$wakes"
  for receipt_file in "${injection_paths[@]}"; do
    [[ -f "$receipt_file" ]] || continue
    read -r token_utc token_agent token_lane < "$receipt_file" || true
    receipt_utc="${token_utc#utc=}"
    receipt_agent="${token_agent#agent=}"
    receipt_lane="${token_lane#lane=}"
    printf 'INJECTION message_id=%s utc=%s agent=%s lane=%s\n' \
      "$message_id" "$receipt_utc" "$receipt_agent" "$receipt_lane"
  done
  for receipt_file in "${ack_paths[@]}"; do
    [[ -f "$receipt_file" ]] || continue
    read -r token_utc token_agent token_lane < "$receipt_file" || true
    receipt_utc="${token_utc#utc=}"
    receipt_agent="${token_agent#agent=}"
    receipt_lane="${token_lane#lane=}"
    printf 'ACKNOWLEDGEMENT message_id=%s utc=%s agent=%s lane=%s\n' \
      "$message_id" "$receipt_utc" "$receipt_agent" "$receipt_lane"
  done
  for receipt_file in "${wake_paths[@]}"; do
    [[ -f "$receipt_file" ]] || continue
    read -r token_utc token_message token_endpoint token_transport token_address < "$receipt_file" || true
    printf 'WAKE_RECEIPT message_id=%s utc=%s endpoint_id=%s transport=%s address=%s\n' \
      "$message_id" "${token_utc#utc=}" "${token_endpoint#endpoint_id=}" \
      "${token_transport#transport=}" "${token_address#address=}"
  done
}

wait_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' message_id='' timeout_seconds=300 poll_seconds=1
  local message_file original_from_agent original_from_lane original_to_agent original_to_lane
  local original_thread original_epoch response_file='' response_state deadline current_epoch
  local -a message_paths=()
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --message) require_arg "$1" "$2"; message_id="$2"; shift 2 ;;
      --timeout-seconds) require_arg "$1" "$2"; timeout_seconds="$2"; shift 2 ;;
      --poll-seconds) require_arg "$1" "$2"; poll_seconds="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown wait option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "wait requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "wait requires --lane"
  [[ -n "$message_id" ]] || die "wait requires --message"
  [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || die "--timeout-seconds must be a non-negative integer"
  [[ "$poll_seconds" =~ ^[1-9][0-9]*$ ]] || die "--poll-seconds must be a positive integer"
  message_file="$MESSAGES_DIR/$(slug "$message_id").message"
  [[ -f "$message_file" ]] || die "message not found: $message_id"
  load_message "$message_file"
  [[ "$M_ID" == "$message_id" ]] || die "message not found: $message_id"
  [[ "$M_FROM_AGENT" == "$agent" && "$M_FROM_LANE" == "$lane" ]] || \
    die "only the sending lane can wait for this message"
  original_from_agent="$M_FROM_AGENT"
  original_from_lane="$M_FROM_LANE"
  original_to_agent="$M_TO_AGENT"
  original_to_lane="$M_TO_LANE"
  original_thread="$M_THREAD_ID"
  original_epoch="$M_CREATED_EPOCH"
  deadline=$(( $(date +%s) + timeout_seconds ))

  while true; do
    response_file=''
    message_paths=("$MESSAGES_DIR"/*.message)
    for message_file in "${message_paths[@]}"; do
      [[ -f "$message_file" ]] || continue
      load_message "$message_file"
      message_expired && continue
      [[ "$M_ID" != "$message_id" && "$M_THREAD_ID" == "$original_thread" ]] || continue
      ((M_CREATED_EPOCH >= original_epoch)) || continue
      [[ "$M_FROM_AGENT" != "$original_from_agent" || "$M_FROM_LANE" != "$original_from_lane" ]] || continue
      [[ -z "$original_to_agent" || "$M_FROM_AGENT" == "$original_to_agent" ]] || continue
      [[ -z "$original_to_lane" || "$M_FROM_LANE" == "$original_to_lane" ]] || continue
      [[ -z "$M_TO_AGENT" || "$M_TO_AGENT" == "$original_from_agent" ]] || continue
      [[ -z "$M_TO_LANE" || "$M_TO_LANE" == "$original_from_lane" ]] || continue
      [[ "$M_KIND" =~ ^(reply|blocker|handoff)$ ]] || continue
      response_file="$message_file"
    done
    if [[ -n "$response_file" ]]; then
      load_message "$response_file"
      response_state='answered'
      [[ "$M_KIND" != blocker ]] || response_state='blocked'
      printf 'WAIT_RESPONSE request_id=%s request_state=%s\n' "$message_id" "$response_state"
      print_message_line
      return 0
    fi
    current_epoch="$(date +%s)"
    if ((current_epoch >= deadline)); then
      printf 'WAIT_TIMEOUT message_id=%s timeout_seconds=%s\n' "$message_id" "$timeout_seconds"
      return 3
    fi
    sleep "$poll_seconds"
  done
}

ack_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' message_id='' message_file ack_file
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --message) require_arg "$1" "$2"; message_id="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown ack option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "ack requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "ack requires --lane"
  [[ -n "$message_id" ]] || die "ack requires --message"
  message_file="$MESSAGES_DIR/$(slug "$message_id").message"
  [[ -f "$message_file" ]] || die "message not found: $message_id"
  load_message "$message_file"
  [[ -z "$M_TO_AGENT" || "$M_TO_AGENT" == "$agent" ]] || die "message is addressed to agent $M_TO_AGENT"
  [[ -z "$M_TO_LANE" || "$M_TO_LANE" == "$lane" ]] || die "message is addressed to lane $M_TO_LANE"
  acquire_state_lock "the acknowledgement"
  ack_file="$(message_ack_path "$M_ID" "$agent" "$lane")"
  printf 'utc=%s agent=%s lane=%s\n' "$NOW_UTC" "$agent" "$lane" > "$ack_file"
  printf 'ACKED message_id=%s agent=%s lane=%s\n' "$M_ID" "$agent" "$lane"
}

prune_command() {
  local removed=0 messages_removed=0 endpoints_removed=0
  local claim_file message_file ack_file injection_file endpoint_file wake_file
  local -a message_paths=() ack_paths=() injection_paths=() endpoint_paths=() wake_paths=()
  acquire_state_lock "prune"
  refresh_claim_paths
  for claim_file in "${claim_paths[@]}"; do
    [[ -f "$claim_file" ]] || continue
    load_claim "$claim_file"
    if claim_expired; then
      append_event EXPIRED "lease expired"
      unlink "$claim_file"
      remove_endpoint_for_lane "$C_AGENT" "$C_LANE" "$C_WORKTREE" claim-expired
      removed=$((removed + 1))
      printf 'PRUNED claim_id=%s agent=%s lane=%s\n' "$C_ID" "$C_AGENT" "$C_LANE"
    fi
  done
  message_paths=("$MESSAGES_DIR"/*.message)
  for message_file in "${message_paths[@]}"; do
    [[ -f "$message_file" ]] || continue
    load_message "$message_file"
    if message_expired; then
      unlink "$message_file"
      ack_paths=("$ACKS_DIR/$(slug "$M_ID")"--*.ack)
      for ack_file in "${ack_paths[@]}"; do
        [[ -f "$ack_file" ]] && unlink "$ack_file"
      done
      injection_paths=("$INJECTIONS_DIR/$(slug "$M_ID")"--*.injected)
      for injection_file in "${injection_paths[@]}"; do
        [[ -f "$injection_file" ]] && unlink "$injection_file"
      done
      wake_paths=("$WAKES_DIR/$(slug "$M_ID")"--*.wake)
      for wake_file in "${wake_paths[@]}"; do
        [[ -f "$wake_file" ]] && unlink "$wake_file"
      done
      messages_removed=$((messages_removed + 1))
      printf 'PRUNED_MESSAGE message_id=%s\n' "$M_ID"
    fi
  done
  endpoint_paths=("$ENDPOINTS_DIR"/*.endpoint)
  for endpoint_file in "${endpoint_paths[@]}"; do
    [[ -f "$endpoint_file" ]] || continue
    load_endpoint "$endpoint_file"
    if endpoint_expired; then
      unlink "$endpoint_file"
      endpoints_removed=$((endpoints_removed + 1))
      printf 'PRUNED_ENDPOINT endpoint_id=%s agent=%s lane=%s\n' "$E_ID" "$E_AGENT" "$E_LANE"
    fi
  done
  printf 'pruned=%s pruned_messages=%s\n' "$removed" "$messages_removed"
  printf 'pruned_endpoints=%s\n' "$endpoints_removed"
}

status_command() {
  local claim_file endpoint_file existing other_file wt branch dirty head marker context_branch file resource local_conflict
  local active=0 stale=0 conflicts=0 total_wt=0 dirty_wt=0 claimed_wt=0 shown_wt=0 relevant_wt=0 max_rows
  local active_endpoints=0 stale_endpoints=0 drifted_endpoints=0 unavailable_endpoints=0 endpoint_marker
  local brief=0 inspect_all=0 worktree_scan='current-and-claimed'
  local -a worktrees=() inspect_worktrees=() first_files=() second_files=() endpoint_paths=()
  local -a first_resources=() second_resources=()
  max_rows="${SOUNIO_COORD_MAX_WORKTREE_ROWS:-40}"

  while (($#)); do
    case "$1" in
      --all-worktrees)
        inspect_all=1
        worktree_scan='all'
        shift
        ;;
      --brief)
        brief=1
        shift
        ;;
      --max-rows)
        require_arg "$1" "$2"
        max_rows="$2"
        shift 2
        ;;
      -h|--help)
        usage
        return 0
        ;;
      *) die "unknown status option: $1" ;;
    esac
  done
  [[ "$max_rows" =~ ^[1-9][0-9]*$ ]] || die "--max-rows must be a positive integer"

  printf 'Sounio coordination status\n'
  printf 'snapshot_utc=%s\n' "$NOW_UTC"
  printf 'repo_root=%s\n' "$WORKTREE"
  printf 'git_common_dir=%s\n' "$GIT_COMMON_DIR"
  printf 'state_dir=%s\n' "$STATE_DIR"
  printf 'current_worktree=%s\n' "$WORKTREE"
  printf 'current_branch=%s\n' "$(current_branch)"
  printf 'current_sha=%s\n' "$(current_sha)"

  context_branch=''
  if [[ -f "$WORKTREE/.beagle/context/current-context-packet.json" ]]; then
    context_branch="$(sed -n 's|.*\"branch\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*|\1|p' "$WORKTREE/.beagle/context/current-context-packet.json" | head -1)"
  fi
  if [[ -n "$context_branch" ]]; then
    printf 'beagle_context_branch=%s\n' "$context_branch"
    if [[ "$context_branch" != "$(current_branch)" ]]; then
      printf 'WARNING context_branch_mismatch=git:%s beagle:%s\n' "$(current_branch)" "$context_branch"
    fi
  fi

  printf '\n== Claims ==\n'
  refresh_claim_paths
  for claim_file in "${claim_paths[@]}"; do
    [[ -f "$claim_file" ]] || continue
    load_claim "$claim_file"
    if claim_expired; then
      stale=$((stale + 1))
      printf 'STALE claim_id=%s agent=%s lane=%s last_seen=%s worktree=%s files=%s resources=%s\n' \
        "$C_ID" "$C_AGENT" "$C_LANE" "$C_LAST_UTC" "$C_WORKTREE" \
        "$(join_files)" "$(join_resources)"
    else
      active=$((active + 1))
      printf 'ACTIVE claim_id=%s agent=%s lane=%s last_seen=%s branch=%s sha=%s worktree=%s files=%s resources=%s\n' \
        "$C_ID" "$C_AGENT" "$C_LANE" "$C_LAST_UTC" "$C_BRANCH" "$C_SHA" \
        "$C_WORKTREE" "$(join_files)" "$(join_resources)"
    fi
  done
  ((active + stale > 0)) || printf 'NONE\n'

  printf '\n== Delivery endpoints ==\n'
  endpoint_paths=("$ENDPOINTS_DIR"/*.endpoint)
  for endpoint_file in "${endpoint_paths[@]}"; do
    [[ -f "$endpoint_file" ]] || continue
    load_endpoint "$endpoint_file"
    endpoint_state || true
    case "$ENDPOINT_STATE" in
      active) active_endpoints=$((active_endpoints + 1)) ;;
      stale) stale_endpoints=$((stale_endpoints + 1)) ;;
      drifted) drifted_endpoints=$((drifted_endpoints + 1)) ;;
      *) unavailable_endpoints=$((unavailable_endpoints + 1)) ;;
    esac
    endpoint_marker="${ENDPOINT_STATE^^}_ENDPOINT"
    printf '%s endpoint_id=%s agent=%s lane=%s worktree=%s harness=%s transport=%s address=%s last_seen=%s\n' \
      "$endpoint_marker" "$E_ID" "$E_AGENT" "$E_LANE" "$E_WORKTREE" "$E_HARNESS" \
      "$E_TRANSPORT" "$E_ADDRESS" "$E_LAST_UTC"
  done
  ((active_endpoints + stale_endpoints + drifted_endpoints + unavailable_endpoints > 0)) || printf 'NONE\n'

  printf '\n== Active conflicts ==\n'
  for existing in "${claim_paths[@]}"; do
    [[ -f "$existing" ]] || continue
    load_claim "$existing"
    claim_expired && continue
    for other_file in "${claim_paths[@]}"; do
      [[ -f "$other_file" && "$other_file" > "$existing" ]] || continue
      load_claim "$other_file"
      claim_expired && continue
      second_files=("${C_FILES[@]}")
      second_resources=("${C_RESOURCES[@]}")
      load_claim "$existing"
      first_files=("${C_FILES[@]}")
      first_resources=("${C_RESOURCES[@]}")
      local_conflict=0
      for file in "${first_files[@]}"; do
        for wt in "${second_files[@]}"; do
          if paths_overlap "$file" "$wt"; then local_conflict=1; fi
        done
      done
      for resource in "${first_resources[@]}"; do
        for wt in "${second_resources[@]}"; do
          if resources_overlap "$resource" "$wt"; then local_conflict=1; fi
        done
      done
      if ((local_conflict)); then
        load_claim "$existing"
        printf 'CONFLICT claim_id=%s agent=%s lane=%s with_claim=%s\n' \
          "$C_ID" "$C_AGENT" "$C_LANE" "$(basename "$other_file" .claim)"
        conflicts=$((conflicts + 1))
      fi
    done
  done
  ((conflicts > 0)) || printf 'NONE\n'

  printf '\n== Worktrees ==\n'
  mapfile -t worktrees < <(git -C "$WORKTREE" worktree list --porcelain | sed -n 's/^worktree //p')
  total_wt="${#worktrees[@]}"
  inspect_worktrees=("$WORKTREE")
  if ((inspect_all)); then
    inspect_worktrees=("${worktrees[@]}")
  else
    for claim_file in "${claim_paths[@]}"; do
      [[ -f "$claim_file" ]] || continue
      load_claim "$claim_file"
      claim_expired && continue
      if ! worktree_in_list "$C_WORKTREE"; then
        inspect_worktrees+=("$C_WORKTREE")
      fi
    done
  fi
  printf 'worktree_scan=%s\n' "$worktree_scan"
  for wt in "${inspect_worktrees[@]}"; do
    [[ -d "$wt" ]] || continue
    dirty="$(git -C "$wt" status --porcelain=v1 --untracked-files=all 2>/dev/null | wc -l | tr -d ' ')"
    ((dirty > 0)) && dirty_wt=$((dirty_wt + 1))
    marker=''
    [[ "$wt" == "$WORKTREE" ]] && marker=' CURRENT'
    if worktree_has_claim "$wt"; then
      marker="$marker CLAIMED"
      claimed_wt=$((claimed_wt + 1))
    fi
    if ((dirty > 0)) || [[ "$wt" == "$WORKTREE" ]] || [[ "$marker" == *CLAIMED* ]]; then
      relevant_wt=$((relevant_wt + 1))
      if ((shown_wt < max_rows)); then
        branch="$(git -C "$wt" branch --show-current 2>/dev/null || true)"
        [[ -n "$branch" ]] || branch="detached@$(git -C "$wt" rev-parse --short=10 HEAD 2>/dev/null || printf unknown)"
        head="$(git -C "$wt" rev-parse --short=10 HEAD 2>/dev/null || printf unknown)"
        printf 'WORKTREE%s path=%s branch=%s head=%s dirty=%s\n' "$marker" "$wt" "$branch" "$head" "$dirty"
        if ((dirty > 0 && brief == 0)); then
          git -C "$wt" status --porcelain=v1 --untracked-files=all 2>/dev/null | sed -n '1,6s/^/  change=/'
        fi
        shown_wt=$((shown_wt + 1))
      fi
    fi
  done
  if ((relevant_wt > shown_wt)); then
    printf 'WORKTREE_OUTPUT_TRUNCATED max_rows=%s\n' "$max_rows"
  fi
  printf 'worktrees_total=%s worktrees_inspected=%s worktrees_dirty=%s worktrees_claimed=%s\n' \
    "$total_wt" "${#inspect_worktrees[@]}" "$dirty_wt" "$claimed_wt"

  if ((brief == 0)); then
    printf '\n== Recent events ==\n'
    if [[ -s "$EVENT_LOG" ]]; then
      tail -12 "$EVENT_LOG"
    else
      printf 'NONE\n'
    fi
  fi
  printf '\nsummary=active_claims:%s stale_claims:%s conflicts:%s\n' "$active" "$stale" "$conflicts"
  printf 'delivery_summary=active_endpoints:%s stale_endpoints:%s drifted_endpoints:%s unavailable_endpoints:%s\n' \
    "$active_endpoints" "$stale_endpoints" "$drifted_endpoints" "$unavailable_endpoints"
  STATUS_CONFLICTS="$conflicts"
}

STATUS_CONFLICTS=0
command="${1:-status}"
if (($#)); then shift; fi

case "$command" in
  runtime-version)
    (($# == 0)) || die "runtime-version does not accept arguments"
    printf 'protocol_version=%s\n' "$SOUNIO_COORD_PROTOCOL_VERSION"
    printf 'runtime_version=%s\n' "$SOUNIO_COORD_RUNTIME_VERSION"
    printf 'implementation_path=%s\n' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
    ;;
  brief)
    status_command --brief "$@"
    ;;
  status|list)
    status_command "$@"
    ;;
  check)
    status_command "$@"
    if ((STATUS_CONFLICTS > 0)); then
      printf 'COORDINATION_CHECK=FAIL conflicts=%s\n' "$STATUS_CONFLICTS" >&2
      exit 3
    fi
    printf 'COORDINATION_CHECK=PASS\n'
    ;;
  claim) claim_command "$@" ;;
  scope) scope_command "$@" ;;
  heartbeat) heartbeat_command "$@" ;;
  release) release_command "$@" ;;
  authorize) authorize_command "$@" ;;
  endpoint-register) endpoint_register_command "$@" ;;
  endpoint-unregister) endpoint_unregister_command "$@" ;;
  endpoint-status) endpoint_status_command "$@" ;;
  wake) wake_command "$@" ;;
  handoff) handoff_command "$@" ;;
  send) send_command "$@" ;;
  inbox) inbox_command "$@" ;;
  injected) injected_command "$@" ;;
  ack) ack_command "$@" ;;
  message-status) message_status_command "$@" ;;
  wait) wait_command "$@" ;;
  prune)
    (($# == 0)) || die "prune does not accept arguments"
    prune_command
    ;;
  -h|--help|help) usage ;;
  *) die "unknown command: $command (try runtime-version, brief, status, check, claim, scope, heartbeat, release, authorize, endpoint-register, endpoint-unregister, endpoint-status, wake, handoff, send, inbox, injected, ack, message-status, wait, or prune)" ;;
esac
