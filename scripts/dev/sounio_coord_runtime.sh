#!/usr/bin/env bash

set -euo pipefail
umask 077

SOUNIO_COORD_PROTOCOL_VERSION=3
SOUNIO_COORD_RUNTIME_VERSION=2026.08.31.0

usage() {
  cat <<'USAGE'
Usage: bin/sounio-coord <command> [options]

Small shared coordination bus for Sounio worktrees. Claims live outside Git
so every worktree attached to the same repository can see the same leases.

Commands:
  runtime-version                 show the runtime protocol and implementation version
  brief                          show the startup-sized coordination summary
  cockpit-snapshot               emit the lightweight machine fleet snapshot
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
  endpoint-register --agent ID --lane ID --harness claude|codex|grok|cursor|kimi|beagle
          --transport tmux|agentd|loom --address ADDRESS --socket PATH
          [--token-file PATH] [--ttl-seconds N]
                                 register an expiring, verified delivery endpoint
  endpoint-unregister --agent ID --lane ID
                                 remove the lane's delivery endpoint
  endpoint-status --agent ID --lane ID
                                 inspect one delivery endpoint
  presence-register --agent ID --lane ID --harness NAME --session-id ID
          --pid PID --pid-start TICK --boot-id ID --pid-namespace ID
          --host HOST [--ttl-seconds N]
                                 bind a lane to a tmux-independent process identity
  presence-unregister --agent ID --lane ID
                                 remove the lane's process identity on a clean exit
  hook-capability-register --agent ID --lane ID --session-id ID
                                 attest a native OCaml hook generation
  hook-capability-unregister --agent ID --lane ID
                                 retire a native hook attestation
  hook-capability-status --agent ID --lane ID
                                 inspect native hook eligibility
  hook-caller-attest --agent ID
                                 attest the native hook's exact provider caller
  recover [--agent ID --lane ID] [--all]
                                 reconstruct one lane or audit the fleet after a crash
  obligation-open --agent ID --lane ID --message ID
                                 materialize one directed request as durable work
  obligation-consume --agent ID --lane ID --message ID [--ttl-seconds N]
                                 bind request consumption to the live process generation
  obligation-claim --agent ID --lane ID --message ID [--claim ID] [--ttl-seconds N]
  obligation-renew --agent ID --lane ID --message ID --claim ID [--ttl-seconds N]
  obligation-interrupt --agent ID --lane ID --message ID [--claim ID] [--reason TEXT]
  obligation-recover --agent ID --lane ID --message ID
  obligation-complete --agent ID --lane ID --message ID --claim ID
          --outcome PATH --evidence PATH
                                 drive generation-fenced obligation transitions
  obligation-status --agent ID --lane ID --message ID [--json]
  obligation-list [--json]
  obligation-tui
  obligation-serve [--bind ADDRESS] [--port N] [--allow-remote]
  obligation-reconcile           open any directed request missed by a crash window
  obligation-supervise [--once] [--interval-seconds N]
  obligation-supervisor-status
                                 run or inspect the tmux-independent replay supervisor
  obligation-supervisor-ensure [--interval-seconds N] [--timeout-seconds N]
  obligation-supervisor-stop [--timeout-seconds N]
                                 idempotently start or stop the detached control service
  wake    --agent ID --lane ID --message MESSAGE_ID
                                 retry immediate delivery for a visible directed message
  wake-reconcile                 retry pending submissions without reinserting prompts
  experiment-open --agent ID --lane ID --receipt PATH --statement TEXT
          --falsifier TEXT --intervention TEXT --treatment-predicate TEXT
          --control-predicate TEXT --resource RESOURCE [--resource RESOURCE ...]
                                 preregister a falsifiable, versioned experiment
  experiment-close --agent ID --lane ID --prereg PATH --outcome PATH
          --verdict supported|falsified|inconclusive --treatment NAME=PASS|FAIL
          --control NAME=PASS|FAIL --treatment-evidence PATH
          --control-evidence PATH
                                 record a Git-bound treatment and sabotage outcome
  experiment-status --prereg PATH [--outcome PATH]
                                 verify and inspect a causal experiment chain
  handoff --agent ID --lane ID --to-agent ID --to-lane ID --message TEXT
          --commit SHA --gate NAME=PASS [--gate NAME=PASS ...]
          --evidence PATH [--evidence PATH ...] [--reply-to MESSAGE_ID]
          [--experiment-prereg PATH --experiment-outcome PATH]
                                 publish proof metadata, then release the owned claim
  send    --agent ID --lane ID [--to-agent ID] [--to-lane ID]
          [--thread ID] [--reply-to MESSAGE_ID] --kind KIND --message TEXT
                                 send a message to another lane or broadcast
  reply  --agent ID --lane ID --reply-to MESSAGE_ID --message TEXT
                                 reply to the original sender and preserve its thread
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
  SOUNIO_COORD_PRESENCE_TTL_SECONDS
                                 process heartbeat duration (default: 1800)
  SOUNIO_COORD_DURABLE_OBLIGATIONS
                                 set to 0 only for the explicit legacy message path
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

coord_durable_obligations_enabled() {
  case "${SOUNIO_COORD_DURABLE_OBLIGATIONS:-1}" in
    1) return 0 ;;
    0) return 1 ;;
    *) die "SOUNIO_COORD_DURABLE_OBLIGATIONS must be 0 or 1" ;;
  esac
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
LEGACY_STATE_DIR="${TMPDIR:-/tmp}/sounio-coord/$REPO_KEY"
DURABLE_STATE_DIR="$GIT_COMMON_DIR/sounio-coord-state"
OBLIGATION_ACTIVATION_FILE="$DURABLE_STATE_DIR/loom-obligation-activation.v1"

migrate_legacy_state() {
  local lock_file="$GIT_COMMON_DIR/.sounio-coord-state-migration.lock"
  exec 8>"$lock_file"
  flock 8

  if [[ -e "$DURABLE_STATE_DIR" && -e "$LEGACY_STATE_DIR" && \
    "$(readlink -f "$DURABLE_STATE_DIR")" != "$(readlink -f "$LEGACY_STATE_DIR")" ]]; then
    die "split coordination state detected: durable=$DURABLE_STATE_DIR legacy=$LEGACY_STATE_DIR"
  fi

  if [[ ! -e "$DURABLE_STATE_DIR" && -d "$LEGACY_STATE_DIR" ]]; then
    mv "$LEGACY_STATE_DIR" "$DURABLE_STATE_DIR"
  fi
  mkdir -p "$DURABLE_STATE_DIR"

  # Older runtimes still look under TMPDIR. Keep that path as an alias so an
  # in-flight pre-upgrade command cannot create a second coordination world.
  if [[ ! -e "$LEGACY_STATE_DIR" && ! -L "$LEGACY_STATE_DIR" ]]; then
    mkdir -p "$(dirname "$LEGACY_STATE_DIR")"
    ln -s "$DURABLE_STATE_DIR" "$LEGACY_STATE_DIR"
  fi
  flock -u 8
}

if [[ -n "${SOUNIO_COORD_DIR:-}" ]]; then
  STATE_DIR="$SOUNIO_COORD_DIR"
else
  migrate_legacy_state
  STATE_DIR="$DURABLE_STATE_DIR"
fi
CLAIMS_DIR="$STATE_DIR/claims"
MESSAGES_DIR="$STATE_DIR/messages"
ACKS_DIR="$STATE_DIR/message-acks"
INJECTIONS_DIR="$STATE_DIR/message-injections"
ENDPOINTS_DIR="$STATE_DIR/delivery-endpoints"
PRESENCES_DIR="$STATE_DIR/process-presences"
HOOK_CAPABILITIES_DIR="$STATE_DIR/hook-capabilities"
WAKES_DIR="$STATE_DIR/message-wakes"
WAKE_SUBMISSIONS_DIR="$STATE_DIR/message-wake-submissions"
EVENT_LOG="$STATE_DIR/events.log"
mkdir -p "$CLAIMS_DIR" "$MESSAGES_DIR" "$ACKS_DIR" "$INJECTIONS_DIR" \
  "$ENDPOINTS_DIR" "$PRESENCES_DIR" "$HOOK_CAPABILITIES_DIR" "$WAKES_DIR" \
  "$WAKE_SUBMISSIONS_DIR"

NOW_EPOCH="$(date +%s)"
NOW_TICK="$(date +%s%N)"
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
shopt -s nullglob
LOCK_TO_CLEAN=''
LOCK_FD=''

cleanup_lock() {
  if [[ -n "$LOCK_FD" ]]; then
    flock -u "$LOCK_FD" 2>/dev/null || true
    eval "exec ${LOCK_FD}>&-" 2>/dev/null || true
    LOCK_FD=''
  fi
  if [[ -n "$LOCK_TO_CLEAN" ]]; then
    rmdir "$LOCK_TO_CLEAN" 2>/dev/null || true
    LOCK_TO_CLEAN=''
  fi
}

trap cleanup_lock EXIT

acquire_state_lock() {
  local action="$1" lock_dir lock_epoch lock_wait
  if command -v flock >/dev/null 2>&1; then
    lock_wait="${SOUNIO_COORD_LOCK_WAIT_SECONDS:-2}"
    [[ "$lock_wait" =~ ^[0-9]+([.][0-9]+)?$ ]] || \
      die "SOUNIO_COORD_LOCK_WAIT_SECONDS must be a non-negative number"
    exec {LOCK_FD}>"$STATE_DIR/.claims.lock"
    flock -w "$lock_wait" "$LOCK_FD" || \
      die "coordination state is being changed; retry $action"
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

causal_runtime_command() {
  local script_dir helper
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  if [[ -x "$script_dir/sounio-coord-causal-runtime" ]]; then
    helper="$script_dir/sounio-coord-causal-runtime"
  elif [[ -x "$script_dir/sounio_coord_causal_runtime.py" ]]; then
    helper="$script_dir/sounio_coord_causal_runtime.py"
  else
    die "causal coordination runtime is not installed beside ${BASH_SOURCE[0]}"
  fi
  SOUNIO_COORD_WORKTREE="$WORKTREE" SOUNIO_COORD_STATE_DIR="$STATE_DIR" \
    "$helper" "$@"
}

agentd_runtime_command() {
  local script_dir helper
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  if [[ -x "$script_dir/sounio-agentd-runtime" ]]; then
    helper="$script_dir/sounio-agentd-runtime"
  elif [[ -x "$script_dir/sounio_coord_agentd.py" ]]; then
    helper="$script_dir/sounio_coord_agentd.py"
  else
    die "agent supervisor runtime is not installed beside ${BASH_SOURCE[0]}"
  fi
  "$helper" "$@"
}

loom_runtime_command() {
  local script_dir helper
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  if [[ -x "$script_dir/sounio-loom-runtime" ]]; then
    helper="$script_dir/sounio-loom-runtime"
  elif [[ -x "$script_dir/../../bin/sounio-loom" ]]; then
    helper="$script_dir/../../bin/sounio-loom"
  else
    die "Loom runtime is not installed beside ${BASH_SOURCE[0]}"
  fi
  (cd "$WORKTREE" && "$helper" "$@")
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
  M_OBLIGATION_SCHEMA=''
  M_OBLIGATION_OPT_OUT=''
  M_COMMIT_SHA=''
  M_EXPERIMENT_ID=''
  M_EXPERIMENT_PREREG=''
  M_EXPERIMENT_OUTCOME=''
  M_EXPERIMENT_PREREG_SHA256=''
  M_EXPERIMENT_OUTCOME_SHA256=''
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
      obligation_schema=*) M_OBLIGATION_SCHEMA="${line#obligation_schema=}" ;;
      obligation_opt_out=*) M_OBLIGATION_OPT_OUT="${line#obligation_opt_out=}" ;;
      commit_sha=*) M_COMMIT_SHA="${line#commit_sha=}" ;;
      experiment_id=*) M_EXPERIMENT_ID="${line#experiment_id=}" ;;
      experiment_prereg=*) M_EXPERIMENT_PREREG="${line#experiment_prereg=}" ;;
      experiment_outcome=*) M_EXPERIMENT_OUTCOME="${line#experiment_outcome=}" ;;
      experiment_prereg_sha256=*) M_EXPERIMENT_PREREG_SHA256="${line#experiment_prereg_sha256=}" ;;
      experiment_outcome_sha256=*) M_EXPERIMENT_OUTCOME_SHA256="${line#experiment_outcome_sha256=}" ;;
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

presence_path() {
  printf '%s/%s.presence' "$PRESENCES_DIR" "$(claim_id_for "$1" "$2")"
}

wake_receipt_path() {
  local generation="${3:-}"
  if [[ -n "$generation" ]]; then
    printf '%s/%s--%s--%s.wake' "$WAKES_DIR" "$(slug "$1")" \
      "$(slug "$2")" "$(slug "$generation")"
  else
    printf '%s/%s--%s.wake' "$WAKES_DIR" "$(slug "$1")" "$(slug "$2")"
  fi
}

wake_submission_path() {
  printf '%s/%s--%s--%s.submitted' "$WAKE_SUBMISSIONS_DIR" "$(slug "$1")" \
    "$(slug "$2")" "$(slug "$3")"
}

load_wake_submission() {
  local submission_file="$1" line
  S_SCHEMA=''
  S_STATE=''
  S_MESSAGE_ID=''
  S_ENDPOINT_ID=''
  S_AGENT=''
  S_LANE=''
  S_HARNESS=''
  S_WORKTREE=''
  S_TRANSPORT=''
  S_ADDRESS=''
  S_SOCKET=''
  S_GENERATION=''
  S_DISCOVERY=''
  S_CREATED_UTC=''
  S_INSERTION_STATE=''
  S_INSERTED_UTC=''
  S_SUBMITTED_UTC=''
  S_LAST_ATTEMPT_EPOCH=0
  S_ATTEMPTS=0
  [[ -r "$submission_file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      schema=*) S_SCHEMA="${line#schema=}" ;;
      state=*) S_STATE="${line#state=}" ;;
      message_id=*) S_MESSAGE_ID="${line#message_id=}" ;;
      endpoint_id=*) S_ENDPOINT_ID="${line#endpoint_id=}" ;;
      agent=*) S_AGENT="${line#agent=}" ;;
      lane=*) S_LANE="${line#lane=}" ;;
      harness=*) S_HARNESS="${line#harness=}" ;;
      worktree=*) S_WORKTREE="${line#worktree=}" ;;
      transport=*) S_TRANSPORT="${line#transport=}" ;;
      address=*) S_ADDRESS="${line#address=}" ;;
      socket=*) S_SOCKET="${line#socket=}" ;;
      generation=*) S_GENERATION="${line#generation=}" ;;
      discovery=*) S_DISCOVERY="${line#discovery=}" ;;
      created_utc=*) S_CREATED_UTC="${line#created_utc=}" ;;
      insertion_state=*) S_INSERTION_STATE="${line#insertion_state=}" ;;
      inserted_utc=*) S_INSERTED_UTC="${line#inserted_utc=}" ;;
      submitted_utc=*) S_SUBMITTED_UTC="${line#submitted_utc=}" ;;
      last_attempt_epoch=*) S_LAST_ATTEMPT_EPOCH="${line#last_attempt_epoch=}" ;;
      attempts=*) S_ATTEMPTS="${line#attempts=}" ;;
    esac
  done < "$submission_file"
}

write_wake_submission() {
  local submission_file="$1" tmp_file
  tmp_file="$(mktemp "$WAKE_SUBMISSIONS_DIR/.submission-write.XXXXXX")"
  {
    printf 'schema=loom-wake-submission-v1\n'
    printf 'state=%s\n' "$S_STATE"
    printf 'message_id=%s\n' "$S_MESSAGE_ID"
    printf 'endpoint_id=%s\n' "$S_ENDPOINT_ID"
    printf 'agent=%s\n' "$S_AGENT"
    printf 'lane=%s\n' "$S_LANE"
    printf 'harness=%s\n' "$S_HARNESS"
    printf 'worktree=%s\n' "$S_WORKTREE"
    printf 'transport=%s\n' "$S_TRANSPORT"
    printf 'address=%s\n' "$S_ADDRESS"
    printf 'socket=%s\n' "$S_SOCKET"
    printf 'generation=%s\n' "$S_GENERATION"
    printf 'discovery=%s\n' "$S_DISCOVERY"
    printf 'created_utc=%s\n' "$S_CREATED_UTC"
    printf 'insertion_state=%s\n' "$S_INSERTION_STATE"
    printf 'inserted_utc=%s\n' "$S_INSERTED_UTC"
    printf 'submitted_utc=%s\n' "$S_SUBMITTED_UTC"
    printf 'last_attempt_epoch=%s\n' "$S_LAST_ATTEMPT_EPOCH"
    printf 'attempts=%s\n' "$S_ATTEMPTS"
  } > "$tmp_file"
  mv "$tmp_file" "$submission_file"
}

process_presence_delivery_generation() {
  local agent="$1" lane="$2" worktree="$3" harness="$4" presence_file
  presence_file="$(presence_path "$agent" "$lane")"
  [[ -f "$presence_file" ]] || return 1
  load_presence "$presence_file"
  presence_state || return 1
  [[ "$PRESENCE_STATE" == live && "$P_AGENT" == "$agent" && "$P_LANE" == "$lane" && \
    "$P_WORKTREE" == "$worktree" && "$P_HARNESS" == "$harness" && \
    -n "$P_SESSION_ID" && "$P_GENERATION" =~ ^[1-9][0-9]*$ ]] || return 1
  printf 'process-%s-g%s-%s-%s' "$P_SESSION_ID" "$P_GENERATION" "$P_PID" "$P_PID_START"
}

registered_delivery_generation() {
  case "$E_TRANSPORT" in
    agentd|loom)
      [[ -n "$E_INSTANCE_ID" ]] || return 1
      printf '%s' "$E_INSTANCE_ID"
      ;;
    tmux)
      process_presence_delivery_generation \
        "$E_AGENT" "$E_LANE" "$E_WORKTREE" "$E_HARNESS" && return 0
      [[ -n "$E_ADDRESS" && "$E_PANE_PID" =~ ^[1-9][0-9]*$ ]] || return 1
      printf 'tmux-%s-%s' "$E_ADDRESS" "$E_PANE_PID"
      ;;
    *) return 1 ;;
  esac
}

discovered_delivery_generation() {
  process_presence_delivery_generation \
    "$D_AGENT" "$D_LANE" "$D_WORKTREE" "$D_HARNESS" && return 0
  [[ -n "$D_ADDRESS" && "$D_PANE_PID" =~ ^[1-9][0-9]*$ ]] || return 1
  printf 'tmux-%s-%s' "$D_ADDRESS" "$D_PANE_PID"
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
  E_TOKEN_FILE=''
  E_PANE_PID=''
  E_INSTANCE_ID=''
  E_SESSION_ID=''
  E_HARNESS_PID=''
  E_HARNESS_PID_START=''
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
      token_file=*) E_TOKEN_FILE="${line#token_file=}" ;;
      pane_pid=*) E_PANE_PID="${line#pane_pid=}" ;;
      instance_id=*) E_INSTANCE_ID="${line#instance_id=}" ;;
      session_id=*) E_SESSION_ID="${line#session_id=}" ;;
      harness_pid=*) E_HARNESS_PID="${line#harness_pid=}" ;;
      harness_pid_start=*) E_HARNESS_PID_START="${line#harness_pid_start=}" ;;
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
    printf 'token_file=%s\n' "$E_TOKEN_FILE"
    printf 'pane_pid=%s\n' "$E_PANE_PID"
    printf 'instance_id=%s\n' "$E_INSTANCE_ID"
    printf 'session_id=%s\n' "$E_SESSION_ID"
    printf 'harness_pid=%s\n' "$E_HARNESS_PID"
    printf 'harness_pid_start=%s\n' "$E_HARNESS_PID_START"
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

A_STATE=''
A_AGENT=''
A_LANE=''
A_SESSION_ID=''
A_WORKTREE=''
A_INSTANCE_ID=''
A_DAEMON_PID=''
A_DAEMON_PID_START=''
A_HARNESS_PID=''
A_HARNESS_PID_START=''
A_COMMAND=''
agentd_endpoint_snapshot() {
  local agent="$1" lane="$2" socket="$3" token_file="$4" output line
  A_STATE=''
  A_AGENT=''
  A_LANE=''
  A_SESSION_ID=''
  A_WORKTREE=''
  A_INSTANCE_ID=''
  A_DAEMON_PID=''
  A_DAEMON_PID_START=''
  A_HARNESS_PID=''
  A_HARNESS_PID_START=''
  A_COMMAND=''
  output="$(agentd_runtime_command status --agent "$agent" --lane "$lane" \
    --socket "$socket" --token-file "$token_file" 2>/dev/null)" || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      state=*) A_STATE="${line#state=}" ;;
      agent=*) A_AGENT="${line#agent=}" ;;
      lane=*) A_LANE="${line#lane=}" ;;
      session_id=*) A_SESSION_ID="${line#session_id=}" ;;
      worktree=*) A_WORKTREE="${line#worktree=}" ;;
      instance_id=*) A_INSTANCE_ID="${line#instance_id=}" ;;
      daemon_pid=*) A_DAEMON_PID="${line#daemon_pid=}" ;;
      daemon_pid_start=*) A_DAEMON_PID_START="${line#daemon_pid_start=}" ;;
      harness_pid=*) A_HARNESS_PID="${line#harness_pid=}" ;;
      harness_pid_start=*) A_HARNESS_PID_START="${line#harness_pid_start=}" ;;
      command=*) A_COMMAND="${line#command=}" ;;
    esac
  done <<< "$output"
  [[ "$A_STATE" == active && "$A_AGENT" == "$agent" && "$A_LANE" == "$lane" && \
    -n "$A_SESSION_ID" && -n "$A_WORKTREE" && -n "$A_INSTANCE_ID" && \
    "$A_DAEMON_PID" =~ ^[1-9][0-9]*$ && "$A_DAEMON_PID_START" =~ ^[1-9][0-9]*$ && \
    "$A_HARNESS_PID" =~ ^[1-9][0-9]*$ && "$A_HARNESS_PID_START" =~ ^[1-9][0-9]*$ && \
    -n "$A_COMMAND" ]]
}

loom_endpoint_snapshot() {
  local agent="$1" lane="$2" socket="$3" token_file="$4" output line
  A_STATE=''
  A_AGENT=''
  A_LANE=''
  A_SESSION_ID=''
  A_WORKTREE=''
  A_INSTANCE_ID=''
  A_DAEMON_PID=''
  A_DAEMON_PID_START=''
  A_HARNESS_PID=''
  A_HARNESS_PID_START=''
  A_COMMAND=''
  output="$(loom_runtime_command status --machine --agent "$agent" --lane "$lane" \
    --cwd "$WORKTREE" --socket "$socket" --token-file "$token_file" 2>/dev/null)" || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      state=*) A_STATE="${line#state=}" ;;
      agent=*) A_AGENT="${line#agent=}" ;;
      lane=*) A_LANE="${line#lane=}" ;;
      session_id=*) A_SESSION_ID="${line#session_id=}" ;;
      worktree=*) A_WORKTREE="${line#worktree=}" ;;
      instance_id=*) A_INSTANCE_ID="${line#instance_id=}" ;;
      daemon_pid=*) A_DAEMON_PID="${line#daemon_pid=}" ;;
      daemon_pid_start=*) A_DAEMON_PID_START="${line#daemon_pid_start=}" ;;
      harness_pid=*) A_HARNESS_PID="${line#harness_pid=}" ;;
      harness_pid_start=*) A_HARNESS_PID_START="${line#harness_pid_start=}" ;;
      command=*) A_COMMAND="${line#command=}" ;;
    esac
  done <<< "$output"
  [[ "$A_STATE" == active && "$A_AGENT" == "$agent" && "$A_LANE" == "$lane" && \
    -n "$A_SESSION_ID" && -n "$A_WORKTREE" && -n "$A_INSTANCE_ID" && \
    "$A_DAEMON_PID" =~ ^[1-9][0-9]*$ && "$A_DAEMON_PID_START" =~ ^[1-9][0-9]*$ && \
    "$A_HARNESS_PID" =~ ^[1-9][0-9]*$ && "$A_HARNESS_PID_START" =~ ^[1-9][0-9]*$ && \
    -n "$A_COMMAND" ]]
}

harness_command_matches() {
  local harness="$1" command="$2"
  case "$harness" in
    claude) [[ "$command" == claude* ]] ;;
    codex) [[ "$command" == codex* || "$command" == node ]] ;;
    grok) [[ "$command" == grok* ]] ;;
    cursor) [[ "$command" == cursor-agent* || "$command" == cursor* ]] ;;
    kimi) [[ "$command" == kimi-code* || "$command" == kimi* ]] ;;
    beagle) [[ "$command" == bash || "$command" == sh || "$command" == zsh || \
      "$command" == fish || "$command" == beagle-* ]] ;;
    *) return 1 ;;
  esac
}

harness_for_agent() {
  case "$1" in
    claude|claude-*) printf 'claude' ;;
    codex|codex-*) printf 'codex' ;;
    grok|grok-*) printf 'grok' ;;
    cursor|cursor-*) printf 'cursor' ;;
    kimi|kimi-*) printf 'kimi' ;;
    beagle|beagle-*) printf 'beagle' ;;
    *) return 1 ;;
  esac
}

RECOVERY_HARNESS=''
RECOVERY_SESSION_ID=''
RECOVERY_WORKTREE=''
RECOVERY_HISTORY_FILE=''
discover_resume_identity() {
  local agent="$1" lane="$2" prefix candidate session_id history_cwd candidate_root candidate_common identity_key
  local passwd_home history_home
  local -a history_homes=() matches=() valid=() valid_keys=()
  RECOVERY_HARNESS=''
  RECOVERY_SESSION_ID=''
  RECOVERY_WORKTREE=''
  RECOVERY_HISTORY_FILE=''
  [[ "$lane" == session-* ]] || return 1
  prefix="${lane#session-}"
  ((${#prefix} >= 8)) || return 1
  RECOVERY_HARNESS="$(harness_for_agent "$agent")" || return 1

  history_homes+=("${SOUNIO_COORD_HISTORY_HOME:-$HOME}")
  if [[ "$HOME" == */.agents/* ]]; then
    history_home="${HOME%%/.agents/*}"
    if ! array_contains "$history_home" "${history_homes[@]}"; then
      history_homes+=("$history_home")
    fi
  fi
  passwd_home="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6)"
  if [[ -n "$passwd_home" ]] && ! array_contains "$passwd_home" "${history_homes[@]}"; then
    history_homes+=("$passwd_home")
  fi

  case "$RECOVERY_HARNESS" in
    claude)
      for history_home in "${history_homes[@]}"; do
        [[ -d "$history_home/.claude/projects" ]] || continue
        for candidate in "$history_home/.claude/projects"/*/"$prefix"*.jsonl; do
          [[ -f "$candidate" ]] && matches+=("$candidate")
        done
      done
      ;;
    codex)
      for history_home in "${history_homes[@]}"; do
        if [[ "$history_home" == "$HOME" && -n "${CODEX_HOME:-}" ]]; then
          history_home="$CODEX_HOME"
        else
          history_home="$history_home/.codex"
        fi
        [[ -d "$history_home/sessions" ]] || continue
        for candidate in "$history_home/sessions"/*/*/*/*"$prefix"*.jsonl; do
          [[ -f "$candidate" ]] && matches+=("$candidate")
        done
      done
      ;;
    *) return 1 ;;
  esac

  if ((${#matches[@]})); then
    mapfile -t matches < <(printf '%s\n' "${matches[@]}" | sort -u)
  fi
  for candidate in "${matches[@]}"; do
    case "$RECOVERY_HARNESS" in
      claude) session_id="$(basename "$candidate" .jsonl)" ;;
      codex)
        session_id="$(basename "$candidate" | grep -oE \
          '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | tail -1)"
        ;;
    esac
    [[ "$session_id" == "$prefix"* ]] || continue
    history_cwd="$(grep -m1 -o '"cwd":"[^"]*"' "$candidate" 2>/dev/null | \
      sed 's/^"cwd":"//; s/"$//' || true)"
    [[ -n "$history_cwd" ]] || continue
    candidate_root="$(git -C "$history_cwd" rev-parse --show-toplevel 2>/dev/null || true)"
    [[ -n "$candidate_root" ]] || continue
    candidate_root="$(cd "$candidate_root" && pwd -P)"
    candidate_common="$(git -C "$candidate_root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
    [[ "$candidate_common" == "$GIT_COMMON_DIR" ]] || continue
    identity_key="$session_id|$candidate_root"
    if ! array_contains "$identity_key" "${valid_keys[@]}"; then
      valid_keys+=("$identity_key")
      valid+=("$identity_key|$candidate")
    fi
  done
  ((${#valid[@]} == 1)) || return 1
  IFS='|' read -r RECOVERY_SESSION_ID RECOVERY_WORKTREE RECOVERY_HISTORY_FILE <<< "${valid[0]}"
  return 0
}

coord_inbox_launcher() {
  local shared_runtime
  shared_runtime="$GIT_COMMON_DIR/sounio-coord-runtime/current/bin/sounio-coord-runtime"
  if [[ -x "$shared_runtime" ]]; then
    printf '%s' "$shared_runtime"
  else
    printf 'bin/sounio-coord'
  fi
}

discover_history_endpoint() {
  local target_agent="$1" target_lane="$2" harness='' best_file='' best_epoch=0
  local message_file candidate_worktree candidate_branch candidate_common socket
  local pane_id pane_pid pane_command pane_path pane_root pane_common pane_lines matches=0
  local history_branch_matches=0
  local -a message_paths=()

  D_ADDRESS=''
  D_PANE_PID=''
  D_WORKTREE=''
  D_HARNESS=''

  harness="$(harness_for_agent "$target_agent")" || return 1

  message_paths=("$MESSAGES_DIR"/*.message)
  for message_file in "${message_paths[@]}"; do
    [[ -f "$message_file" ]] || continue
    load_message "$message_file"
    message_expired && continue
    [[ "$M_FROM_AGENT" == "$target_agent" && "$M_FROM_LANE" == "$target_lane" ]] || continue
    [[ -n "$M_FROM_WORKTREE" && -n "$M_FROM_BRANCH" ]] || continue
    if ((M_CREATED_EPOCH > best_epoch)) || \
      { ((M_CREATED_EPOCH == best_epoch)) && [[ "$message_file" > "$best_file" ]]; }; then
      best_epoch="$M_CREATED_EPOCH"
      best_file="$message_file"
    fi
  done
  [[ -n "$best_file" ]] || return 1

  load_message "$best_file"
  candidate_worktree="$(git -C "$M_FROM_WORKTREE" rev-parse --show-toplevel 2>/dev/null || true)"
  [[ -n "$candidate_worktree" ]] || return 1
  candidate_worktree="$(cd "$candidate_worktree" && pwd -P)"
  candidate_common="$(git -C "$candidate_worktree" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  [[ "$candidate_common" == "$GIT_COMMON_DIR" ]] || return 1
  candidate_branch="$(git -C "$candidate_worktree" branch --show-current 2>/dev/null || true)"
  if [[ -z "$candidate_branch" ]]; then
    candidate_branch="detached@$(git -C "$candidate_worktree" rev-parse --short=12 HEAD 2>/dev/null || true)"
  fi
  if [[ "$candidate_branch" == "$M_FROM_BRANCH" ]]; then
    history_branch_matches=1
  fi

  socket="${SOUNIO_COORD_DISCOVERY_SOCKET:-${TMUX%%,*}}"
  [[ -n "$socket" && -S "$socket" ]] || return 1
  pane_lines="$(tmux -S "$socket" list-panes -a -F \
    '#{pane_id}|#{pane_pid}|#{pane_current_command}|#{pane_current_path}' 2>/dev/null || true)"
  if ((history_branch_matches)); then
    while IFS='|' read -r pane_id pane_pid pane_command pane_path; do
      [[ -n "$pane_id" && "$pane_pid" =~ ^[1-9][0-9]*$ ]] || continue
      harness_command_matches "$harness" "$pane_command" || continue
      pane_root="$(git -C "$pane_path" rev-parse --show-toplevel 2>/dev/null || true)"
      [[ -n "$pane_root" ]] || continue
      pane_root="$(cd "$pane_root" && pwd -P)"
      [[ "$pane_root" == "$candidate_worktree" ]] || continue
      matches=$((matches + 1))
      D_ADDRESS="$pane_id"
      D_PANE_PID="$pane_pid"
      D_WORKTREE="$pane_root"
    done <<< "$pane_lines"
  fi
  ((matches <= 1)) || return 1
  if ((matches == 1)); then
    D_DISCOVERY='history'
  else
    while IFS='|' read -r pane_id pane_pid pane_command pane_path; do
      [[ -n "$pane_id" && "$pane_pid" =~ ^[1-9][0-9]*$ ]] || continue
      harness_command_matches "$harness" "$pane_command" || continue
      pane_root="$(git -C "$pane_path" rev-parse --show-toplevel 2>/dev/null || true)"
      [[ -n "$pane_root" ]] || continue
      pane_root="$(cd "$pane_root" && pwd -P)"
      [[ "${pane_root##*/}" == "$target_agent" ]] || continue
      pane_common="$(git -C "$pane_root" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
      [[ "$pane_common" == "$GIT_COMMON_DIR" ]] || continue
      matches=$((matches + 1))
      D_ADDRESS="$pane_id"
      D_PANE_PID="$pane_pid"
      D_WORKTREE="$pane_root"
    done <<< "$pane_lines"
    if [[ "$matches" == 1 ]]; then
      D_DISCOVERY='identity-root'
    else
      matches=0
      discover_resume_identity "$target_agent" "$target_lane" || return 1
      while IFS='|' read -r pane_id pane_pid pane_command pane_path; do
        [[ -n "$pane_id" && "$pane_pid" =~ ^[1-9][0-9]*$ ]] || continue
        harness_command_matches "$harness" "$pane_command" || continue
        pane_root="$(git -C "$pane_path" rev-parse --show-toplevel 2>/dev/null || true)"
        [[ -n "$pane_root" ]] || continue
        pane_root="$(cd "$pane_root" && pwd -P)"
        [[ "$pane_root" == "$RECOVERY_WORKTREE" ]] || continue
        matches=$((matches + 1))
        D_ADDRESS="$pane_id"
        D_PANE_PID="$pane_pid"
        D_WORKTREE="$pane_root"
      done <<< "$pane_lines"
      [[ "$matches" == 1 ]] || return 1
      D_DISCOVERY='session-history'
    fi
  fi

  D_ENDPOINT_ID="$D_DISCOVERY-$(claim_id_for "$target_agent" "$target_lane")"
  D_AGENT="$target_agent"
  D_LANE="$target_lane"
  D_HARNESS="$harness"
  D_SOCKET="$socket"
}

ENDPOINT_STATE='unavailable'
endpoint_state() {
  local current_pane current_pid current_command current_path current_root
  ENDPOINT_STATE='unavailable'
  if endpoint_expired; then
    ENDPOINT_STATE='stale'
    return 1
  fi
  case "$E_TRANSPORT" in
    tmux)
      [[ -S "$E_SOCKET" ]] || return 1
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
      ;;
    agentd|loom)
      [[ -S "$E_SOCKET" && -f "$E_TOKEN_FILE" ]] || return 1
      if [[ "$E_TRANSPORT" == agentd ]]; then
        agentd_endpoint_snapshot "$E_AGENT" "$E_LANE" "$E_SOCKET" "$E_TOKEN_FILE" || {
          ENDPOINT_STATE='drifted'
          return 1
        }
      elif ! loom_endpoint_snapshot "$E_AGENT" "$E_LANE" "$E_SOCKET" "$E_TOKEN_FILE"; then
        ENDPOINT_STATE='drifted'
        return 1
      fi
      if [[ "$A_WORKTREE" != "$E_WORKTREE" || "$A_INSTANCE_ID" != "$E_INSTANCE_ID" || \
        "$A_SESSION_ID" != "$E_SESSION_ID" || "$A_HARNESS_PID" != "$E_HARNESS_PID" || \
        "$A_HARNESS_PID_START" != "$E_HARNESS_PID_START" || "$A_COMMAND" != "$E_COMMAND" ]] || \
        ! harness_command_matches "$E_HARNESS" "$A_COMMAND"; then
        ENDPOINT_STATE='drifted'
        return 1
      fi
      ;;
    *) return 1 ;;
  esac
  ENDPOINT_STATE='active'
}

deliver_registered_endpoint() {
  local prompt="$1" message_id="$2"
  case "$E_TRANSPORT" in
    tmux)
      tmux -S "$E_SOCKET" send-keys -t "$E_ADDRESS" -l "$prompt" 2>/dev/null && \
        tmux -S "$E_SOCKET" send-keys -t "$E_ADDRESS" Enter 2>/dev/null
      ;;
    agentd)
      agentd_runtime_command wake --agent "$E_AGENT" --lane "$E_LANE" \
        --session-id "$E_SESSION_ID" --message-id "$message_id" --prompt "$prompt" \
        --socket "$E_SOCKET" --token-file "$E_TOKEN_FILE" >/dev/null 2>&1
      ;;
    loom)
      loom_runtime_command wake --agent "$E_AGENT" --lane "$E_LANE" \
        --session-id "$E_SESSION_ID" --message-id "$message_id" --prompt "$prompt" \
        --cwd "$WORKTREE" --socket "$E_SOCKET" --token-file "$E_TOKEN_FILE" >/dev/null 2>&1
      ;;
    *) return 1 ;;
  esac
}

remove_endpoint_for_lane() {
  local agent="$1" lane="$2" worktree="$3" reason="$4" endpoint_file
  endpoint_file="$(endpoint_path "$agent" "$lane")"
  [[ -f "$endpoint_file" ]] || return 0
  load_endpoint "$endpoint_file"
  [[ "$E_AGENT" == "$agent" && "$E_LANE" == "$lane" ]] || die "endpoint owner mismatch"
  unlink "$endpoint_file"
  printf 'utc=%s event=ENDPOINT_UNREGISTERED endpoint_id=%s agent=%s lane=%s worktree=%s reason=%s\n' \
    "$NOW_UTC" "$E_ID" "$E_AGENT" "$E_LANE" "$E_WORKTREE" "$reason" >> "$EVENT_LOG"
}

load_presence() {
  local presence_file="$1" line
  P_ID=''
  P_AGENT=''
  P_LANE=''
  P_WORKTREE=''
  P_HARNESS=''
  P_SESSION_ID=''
  P_HOST=''
  P_BOOT_ID=''
  P_PID_NAMESPACE=''
  P_PID=0
  P_PID_START=0
  P_GENERATION=0
  P_CREATED_UTC=''
  P_LAST_UTC=''
  P_LAST_EPOCH=0
  P_TTL=0
  [[ -r "$presence_file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      presence_id=*) P_ID="${line#presence_id=}" ;;
      agent=*) P_AGENT="${line#agent=}" ;;
      lane=*) P_LANE="${line#lane=}" ;;
      worktree=*) P_WORKTREE="${line#worktree=}" ;;
      harness=*) P_HARNESS="${line#harness=}" ;;
      session_id=*) P_SESSION_ID="${line#session_id=}" ;;
      host=*) P_HOST="${line#host=}" ;;
      boot_id=*) P_BOOT_ID="${line#boot_id=}" ;;
      pid_namespace=*) P_PID_NAMESPACE="${line#pid_namespace=}" ;;
      pid=*) P_PID="${line#pid=}" ;;
      pid_start=*) P_PID_START="${line#pid_start=}" ;;
      generation=*) P_GENERATION="${line#generation=}" ;;
      created_utc=*) P_CREATED_UTC="${line#created_utc=}" ;;
      last_seen_utc=*) P_LAST_UTC="${line#last_seen_utc=}" ;;
      last_seen_epoch=*) P_LAST_EPOCH="${line#last_seen_epoch=}" ;;
      ttl_seconds=*) P_TTL="${line#ttl_seconds=}" ;;
    esac
  done < "$presence_file"
}

write_presence() {
  local presence_file="$1" tmp_file
  tmp_file="$(mktemp "$PRESENCES_DIR/.presence-write.XXXXXX")"
  {
    printf 'presence_id=%s\n' "$P_ID"
    printf 'agent=%s\n' "$P_AGENT"
    printf 'lane=%s\n' "$P_LANE"
    printf 'worktree=%s\n' "$P_WORKTREE"
    printf 'harness=%s\n' "$P_HARNESS"
    printf 'session_id=%s\n' "$P_SESSION_ID"
    printf 'host=%s\n' "$P_HOST"
    printf 'boot_id=%s\n' "$P_BOOT_ID"
    printf 'pid_namespace=%s\n' "$P_PID_NAMESPACE"
    printf 'pid=%s\n' "$P_PID"
    printf 'pid_start=%s\n' "$P_PID_START"
    printf 'generation=%s\n' "$P_GENERATION"
    printf 'created_utc=%s\n' "$P_CREATED_UTC"
    printf 'last_seen_utc=%s\n' "$P_LAST_UTC"
    printf 'last_seen_epoch=%s\n' "$P_LAST_EPOCH"
    printf 'ttl_seconds=%s\n' "$P_TTL"
  } > "$tmp_file"
  mv "$tmp_file" "$presence_file"
}

PRESENCE_STATE='unbound'
PRESENCE_REASON='no-presence-record'
presence_state() {
  local current_boot current_namespace current_start proc_tail
  PRESENCE_STATE='orphaned'
  PRESENCE_REASON='invalid-record'
  [[ "$P_PID" =~ ^[1-9][0-9]*$ && "$P_PID_START" =~ ^[1-9][0-9]*$ && \
    "$P_LAST_EPOCH" =~ ^[0-9]+$ && "$P_TTL" =~ ^[1-9][0-9]*$ ]] || return 1

  current_boot="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || true)"
  current_namespace="$(readlink /proc/self/ns/pid 2>/dev/null || true)"
  if [[ -z "$current_boot" || "$current_boot" != "$P_BOOT_ID" ]]; then
    PRESENCE_REASON='boot-changed'
    return 1
  fi
  if [[ -z "$current_namespace" || "$current_namespace" != "$P_PID_NAMESPACE" ]]; then
    PRESENCE_REASON='pid-namespace-changed'
    return 1
  fi
  if [[ ! -r "/proc/$P_PID/stat" ]] || ! kill -0 "$P_PID" 2>/dev/null; then
    PRESENCE_REASON='process-missing'
    return 1
  fi
  proc_tail="$(sed 's/^[^)]*) //' "/proc/$P_PID/stat" 2>/dev/null || true)"
  current_start="$(awk '{print $20}' <<< "$proc_tail")"
  if [[ -z "$current_start" || "$current_start" != "$P_PID_START" ]]; then
    PRESENCE_REASON='pid-reused'
    return 1
  fi
  if ((NOW_EPOCH > P_LAST_EPOCH + P_TTL)); then
    PRESENCE_STATE='unresponsive'
    PRESENCE_REASON='heartbeat-expired'
    return 1
  fi
  PRESENCE_STATE='live'
  PRESENCE_REASON='process-verified'
  return 0
}

append_presence_event() {
  local event="$1" reason="${2:-}"
  printf 'utc=%s event=%s presence_id=%s agent=%s lane=%s worktree=%s harness=%s session_id=%s host=%s pid=%s pid_start=%s generation=%s reason=%s\n' \
    "$NOW_UTC" "$event" "$P_ID" "$P_AGENT" "$P_LANE" "$P_WORKTREE" \
    "$P_HARNESS" "$P_SESSION_ID" "$P_HOST" "$P_PID" "$P_PID_START" \
    "$P_GENERATION" "$reason" >> "$EVENT_LOG"
}

remove_presence_for_lane() {
  local agent="$1" lane="$2" worktree="$3" reason="$4" presence_file capability_file
  presence_file="$(presence_path "$agent" "$lane")"
  [[ -f "$presence_file" ]] || return 0
  load_presence "$presence_file"
  [[ "$P_AGENT" == "$agent" && "$P_LANE" == "$lane" ]] || die "presence owner mismatch"
  unlink "$presence_file"
  capability_file="$(hook_capability_path "$agent" "$lane")"
  [[ ! -f "$capability_file" ]] || unlink "$capability_file"
  append_presence_event PRESENCE_UNREGISTERED "$reason"
}

hook_capability_path() {
  printf '%s/%s.capability' "$HOOK_CAPABILITIES_DIR" "$(claim_id_for "$1" "$2")"
}

load_hook_capability() {
  local capability_file="$1" line
  HC_SCHEMA=''
  HC_STATE=''
  HC_AGENT=''
  HC_LANE=''
  HC_SESSION_ID=''
  HC_GENERATION=''
  HC_WORKTREE=''
  HC_HARNESS=''
  HC_PRESENCE_PID=0
  HC_PRESENCE_PID_START=0
  HC_PRESENCE_BOOT_ID=''
  HC_PRESENCE_PID_NAMESPACE=''
  HC_PRODUCER_EXECUTABLE=''
  HC_PRODUCER_SHA256=''
  HC_COORD_EXECUTABLE=''
  HC_COORD_SHA256=''
  HC_CALLER_PID=0
  HC_CALLER_PID_START=0
  HC_CALLER_BOOT_ID=''
  HC_CALLER_PID_NAMESPACE=''
  HC_CALLER_EXECUTABLE=''
  HC_CALLER_SHA256=''
  HC_WAKE_ELIGIBLE=0
  HC_RUNTIME_ID=''
  HC_SOURCE_SHA=''
  HC_CREATED_UTC=''
  HC_CREATED_EPOCH=0
  HC_EXPIRES_EPOCH=0
  [[ -r "$capability_file" ]] || return 0
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      schema=*) HC_SCHEMA="${line#schema=}" ;;
      state=*) HC_STATE="${line#state=}" ;;
      agent=*) HC_AGENT="${line#agent=}" ;;
      lane=*) HC_LANE="${line#lane=}" ;;
      session_id=*) HC_SESSION_ID="${line#session_id=}" ;;
      generation=*) HC_GENERATION="${line#generation=}" ;;
      worktree=*) HC_WORKTREE="${line#worktree=}" ;;
      harness=*) HC_HARNESS="${line#harness=}" ;;
      presence_pid=*) HC_PRESENCE_PID="${line#presence_pid=}" ;;
      presence_pid_start=*) HC_PRESENCE_PID_START="${line#presence_pid_start=}" ;;
      presence_boot_id=*) HC_PRESENCE_BOOT_ID="${line#presence_boot_id=}" ;;
      presence_pid_namespace=*) HC_PRESENCE_PID_NAMESPACE="${line#presence_pid_namespace=}" ;;
      producer_executable=*) HC_PRODUCER_EXECUTABLE="${line#producer_executable=}" ;;
      producer_sha256=*) HC_PRODUCER_SHA256="${line#producer_sha256=}" ;;
      coord_executable=*) HC_COORD_EXECUTABLE="${line#coord_executable=}" ;;
      coord_sha256=*) HC_COORD_SHA256="${line#coord_sha256=}" ;;
      caller_pid=*) HC_CALLER_PID="${line#caller_pid=}" ;;
      caller_pid_start=*) HC_CALLER_PID_START="${line#caller_pid_start=}" ;;
      caller_boot_id=*) HC_CALLER_BOOT_ID="${line#caller_boot_id=}" ;;
      caller_pid_namespace=*) HC_CALLER_PID_NAMESPACE="${line#caller_pid_namespace=}" ;;
      caller_executable=*) HC_CALLER_EXECUTABLE="${line#caller_executable=}" ;;
      caller_sha256=*) HC_CALLER_SHA256="${line#caller_sha256=}" ;;
      wake_eligible=*) HC_WAKE_ELIGIBLE="${line#wake_eligible=}" ;;
      runtime_id=*) HC_RUNTIME_ID="${line#runtime_id=}" ;;
      source_sha=*) HC_SOURCE_SHA="${line#source_sha=}" ;;
      created_utc=*) HC_CREATED_UTC="${line#created_utc=}" ;;
      created_epoch=*) HC_CREATED_EPOCH="${line#created_epoch=}" ;;
      expires_epoch=*) HC_EXPIRES_EPOCH="${line#expires_epoch=}" ;;
    esac
  done < "$capability_file"
}

manifest_field() {
  local manifest="$1" key="$2"
  sed -n "s/^${key}=//p" "$manifest" | head -n 1
}

active_coord_runtime_root() {
  local runtime_root
  runtime_root="${SOUNIO_COORD_RUNTIME_DIR:-$GIT_COMMON_DIR/sounio-coord-runtime}"
  runtime_root="$(readlink -f "$runtime_root" 2>/dev/null || true)"
  [[ -n "$runtime_root" ]] || return 1
  printf '%s\n' "$runtime_root"
}

native_hook_parent_identity() {
  local parent_pid="$PPID" runtime_self local_runtime local_loom parent_tail caller_tail
  local runtime_root parent_bundle runtime_bundle current_bundle manifest runtime_version expected_parent_sha expected_coord_sha
  NATIVE_HOOK_PARENT_EXECUTABLE="$(readlink -f "/proc/$parent_pid/exe" 2>/dev/null || true)"
  [[ -n "$NATIVE_HOOK_PARENT_EXECUTABLE" ]] || return 1
  runtime_self="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || true)"
  [[ -n "$runtime_self" ]] || return 1
  NATIVE_HOOK_COORD_EXECUTABLE="$runtime_self"
  NATIVE_HOOK_PARENT_SHA256="$(sha256sum "$NATIVE_HOOK_PARENT_EXECUTABLE" | awk '{print $1}')"
  NATIVE_HOOK_COORD_SHA256="$(sha256sum "$runtime_self" | awk '{print $1}')"
  [[ "$NATIVE_HOOK_PARENT_SHA256" =~ ^[0-9a-f]{64}$ && \
    "$NATIVE_HOOK_COORD_SHA256" =~ ^[0-9a-f]{64}$ ]] || return 1

  parent_tail="$(sed 's/^[^)]*) //' "/proc/$parent_pid/stat" 2>/dev/null || true)"
  NATIVE_HOOK_CALLER_PID="$(awk '{print $2}' <<< "$parent_tail")"
  [[ "$NATIVE_HOOK_CALLER_PID" =~ ^[1-9][0-9]*$ ]] || return 1
  caller_tail="$(sed 's/^[^)]*) //' "/proc/$NATIVE_HOOK_CALLER_PID/stat" 2>/dev/null || true)"
  NATIVE_HOOK_CALLER_PID_START="$(awk '{print $20}' <<< "$caller_tail")"
  NATIVE_HOOK_CALLER_EXECUTABLE="$(readlink -f "/proc/$NATIVE_HOOK_CALLER_PID/exe" 2>/dev/null || true)"
  NATIVE_HOOK_CALLER_COMMAND="$(basename "$NATIVE_HOOK_CALLER_EXECUTABLE")"
  NATIVE_HOOK_CALLER_SHA256="$(sha256sum "$NATIVE_HOOK_CALLER_EXECUTABLE" 2>/dev/null | awk '{print $1}')"
  NATIVE_HOOK_CALLER_CMDLINE="$(tr '\0' ' ' < "/proc/$NATIVE_HOOK_CALLER_PID/cmdline" 2>/dev/null || true)"
  NATIVE_HOOK_CALLER_BOOT_ID="$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || true)"
  NATIVE_HOOK_CALLER_PID_NAMESPACE="$(readlink "/proc/$NATIVE_HOOK_CALLER_PID/ns/pid" 2>/dev/null || true)"
  [[ "$NATIVE_HOOK_CALLER_PID_START" =~ ^[1-9][0-9]*$ && \
    -n "$NATIVE_HOOK_CALLER_EXECUTABLE" && \
    "$NATIVE_HOOK_CALLER_SHA256" =~ ^[0-9a-f]{64}$ && \
    -n "$NATIVE_HOOK_CALLER_BOOT_ID" && \
    -n "$NATIVE_HOOK_CALLER_PID_NAMESPACE" ]] || return 1

  local_runtime="$(readlink -f "$WORKTREE/scripts/dev/sounio_coord_runtime.sh" 2>/dev/null || true)"
  local_loom="$(readlink -f "$WORKTREE/tools/loom/_build/default/src/loom.exe" 2>/dev/null || true)"
  if [[ -n "$local_runtime" && -n "$local_loom" && \
    "$runtime_self" == "$local_runtime" && \
    "$NATIVE_HOOK_PARENT_EXECUTABLE" == "$local_loom" ]]; then
    [[ "${SOUNIO_COORD_RUNTIME_MODE:-}" == local && \
      "${SOUNIO_COORD_NATIVE_HOOK_SELFTEST:-0}" == 1 ]] || return 1
    case "$STATE_DIR" in
      "${TMPDIR:-/tmp}"/sounio-loom-native-hook.*/coord | \
        "${TMPDIR:-/tmp}"/sounio-loom-exec-capability.*/coord | \
        "${TMPDIR:-/tmp}"/sounio-loom-custody.*/coord | \
        "${TMPDIR:-/tmp}"/sounio-coord-crash-selftest.*/repo/.git/sounio-coord-state | \
        "${TMPDIR:-/tmp}"/sounio-coord-agentd-selftest.*/coord-state) ;;
      *) return 1 ;;
    esac
    NATIVE_HOOK_RUNTIME_ID="local-${SOUNIO_COORD_RUNTIME_VERSION}"
    NATIVE_HOOK_SOURCE_SHA="$(current_sha)"
    NATIVE_HOOK_WAKE_ELIGIBLE=0
  else
    runtime_root="$(active_coord_runtime_root)" || return 1
    parent_bundle="$(readlink -f "$(dirname "$NATIVE_HOOK_PARENT_EXECUTABLE")/.." 2>/dev/null || true)"
    runtime_bundle="$(readlink -f "$(dirname "$runtime_self")/.." 2>/dev/null || true)"
    [[ -n "$parent_bundle" && "$parent_bundle" == "$runtime_bundle" ]] || return 1
    case "$parent_bundle" in
      "$runtime_root"/versions/*) ;;
      *) return 1 ;;
    esac
    current_bundle="$(readlink -f "$runtime_root/current" 2>/dev/null || true)"
    [[ -n "$current_bundle" && "$parent_bundle" == "$current_bundle" ]] || return 1
    [[ "$NATIVE_HOOK_PARENT_EXECUTABLE" == "$parent_bundle/bin/sounio-loom-runtime" ]] || return 1
    manifest="$parent_bundle/manifest"
    [[ -r "$manifest" ]] || return 1
    runtime_version="$(manifest_field "$manifest" runtime_version)"
    [[ "$runtime_version" == "$SOUNIO_COORD_RUNTIME_VERSION" ]] || return 1
    expected_parent_sha="$(manifest_field "$manifest" loom_runtime_sha256)"
    expected_coord_sha="$(manifest_field "$manifest" coord_runtime_sha256)"
    [[ "$expected_parent_sha" =~ ^[0-9a-f]{64}$ && \
      "$expected_coord_sha" =~ ^[0-9a-f]{64}$ && \
      "$NATIVE_HOOK_PARENT_SHA256" == "$expected_parent_sha" && \
      "$NATIVE_HOOK_COORD_SHA256" == "$expected_coord_sha" ]] || return 1
    NATIVE_HOOK_RUNTIME_ID="$(manifest_field "$manifest" runtime_id)"
    NATIVE_HOOK_SOURCE_SHA="$(manifest_field "$manifest" source_sha)"
    [[ -n "$NATIVE_HOOK_RUNTIME_ID" && -n "$NATIVE_HOOK_SOURCE_SHA" ]] || return 1
    NATIVE_HOOK_WAKE_ELIGIBLE=1
    if [[ "${SOUNIO_COORD_RUNTIME_MODE:-}" == installed-selftest &&
      "${SOUNIO_COORD_NATIVE_HOOK_SELFTEST:-0}" == 1 ]]; then
      case "$STATE_DIR" in
        "${TMPDIR:-/tmp}"/sounio-loom-native-hook.*/coord)
          NATIVE_HOOK_WAKE_ELIGIBLE=0
          ;;
        *) return 1 ;;
      esac
    fi
  fi
}

native_hook_caller_is_exact_harness() {
  local harness="$1"
  case "$harness" in
    codex) [[ "$NATIVE_HOOK_CALLER_COMMAND" == codex ]] ;;
    claude)
      [[ "$NATIVE_HOOK_CALLER_COMMAND" == claude ||
        "$NATIVE_HOOK_CALLER_COMMAND" == claude.exe ]] ||
        [[ "$NATIVE_HOOK_CALLER_COMMAND" == node &&
          "$NATIVE_HOOK_CALLER_CMDLINE" == *'/@anthropic-ai/claude-code/'* &&
          "$NATIVE_HOOK_CALLER_CMDLINE" == *'cli.js'* ]]
      ;;
    cursor)
      [[ "$NATIVE_HOOK_CALLER_COMMAND" == cursor-agent ||
        "$NATIVE_HOOK_CALLER_COMMAND" == cursor ]] ||
        [[ "$NATIVE_HOOK_CALLER_COMMAND" == node &&
          "$NATIVE_HOOK_CALLER_CMDLINE" == *'/bin/cursor-agent '* &&
          "$NATIVE_HOOK_CALLER_CMDLINE" == *'/cursor-agent/versions/'*'/index.js'* ]]
      ;;
    grok)
      [[ "$NATIVE_HOOK_CALLER_COMMAND" == grok ]] ||
        [[ "$NATIVE_HOOK_CALLER_COMMAND" == grok-*-linux-x86_64 &&
          ( "$NATIVE_HOOK_CALLER_CMDLINE" == grok\ * ||
            "$NATIVE_HOOK_CALLER_CMDLINE" == */bin/grok\ * ) ]]
      ;;
    *) return 1 ;;
  esac
}

native_hook_caller_matches_presence() {
  [[ "$NATIVE_HOOK_CALLER_PID" == "$P_PID" && \
    "$NATIVE_HOOK_CALLER_PID_START" == "$P_PID_START" && \
    "$NATIVE_HOOK_CALLER_BOOT_ID" == "$P_BOOT_ID" && \
    "$NATIVE_HOOK_CALLER_PID_NAMESPACE" == "$P_PID_NAMESPACE" ]] || return 1
  if ((NATIVE_HOOK_WAKE_ELIGIBLE)); then
    native_hook_caller_is_exact_harness "$P_HARNESS"
  else
    [[ "${SOUNIO_COORD_NATIVE_HOOK_SELFTEST:-0}" == 1 ]]
  fi
}

hook_caller_attest_command() {
  local agent="${SOUNIO_AGENT_ID:-}" harness
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      *) die "unknown hook-caller-attest option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "hook-caller-attest requires --agent or SOUNIO_AGENT_ID"
  validate_value agent "$agent"
  harness="$(
    case "$agent" in
      codex*) printf codex ;;
      claude*) printf claude ;;
      cursor*) printf cursor ;;
      grok*) printf grok ;;
      *) die "unsupported native hook caller: $agent" ;;
    esac
  )"
  native_hook_parent_identity ||
    die "native hook caller attestation requires the matching OCaml runtime parent"
  if ((NATIVE_HOOK_WAKE_ELIGIBLE)); then
    native_hook_caller_is_exact_harness "$harness" ||
      die "native hook caller does not match provider $harness"
  else
    [[ "${SOUNIO_COORD_NATIVE_HOOK_SELFTEST:-0}" == 1 ]] ||
      die "native hook caller selftest attestation is disabled"
  fi
  printf 'HOOK_CALLER_ATTESTED agent=%s harness=%s caller_pid=%s caller_pid_start=%s caller_sha256=%s runtime_id=%s source_sha=%s\n' \
    "$agent" "$harness" "$NATIVE_HOOK_CALLER_PID" \
    "$NATIVE_HOOK_CALLER_PID_START" "$NATIVE_HOOK_CALLER_SHA256" \
    "$NATIVE_HOOK_RUNTIME_ID" "$NATIVE_HOOK_SOURCE_SHA"
}

native_hook_wake_selftest_fixture() {
  [[ "${SOUNIO_COORD_RUNTIME_MODE:-}" == local && \
    "${SOUNIO_COORD_NATIVE_HOOK_WAKE_SELFTEST:-0}" == 1 ]] || return 1
  case "$STATE_DIR" in
    "${TMPDIR:-/tmp}"/sounio-coord-wake-selftest.*/state) return 0 ;;
    "${TMPDIR:-/tmp}"/sounio-loom-native-hook.*/coord)
      [[ "${SOUNIO_COORD_NATIVE_HOOK_SELFTEST:-0}" == 1 ]]
      ;;
    *) return 1 ;;
  esac
}

HOOK_CAPABILITY_REASON='absent'
hook_capability_binding_is_current() {
  local agent="$1" lane="$2" generation="$3" capability_file presence_file
  local current_generation current_sha256 current_coord_sha256 current_caller_sha256
  local manifest runtime_root bundle runtime_self active_bundle
  HOOK_CAPABILITY_REASON='absent'
  capability_file="$(hook_capability_path "$agent" "$lane")"
  [[ -f "$capability_file" ]] || return 1
  load_hook_capability "$capability_file"
  [[ "$HC_SCHEMA" == loom-native-hook-capability-v1 && \
    "$HC_STATE" == NATIVE_HOOK_ATTESTED && "$HC_AGENT" == "$agent" && \
    "$HC_LANE" == "$lane" && "$HC_GENERATION" == "$generation" && \
    "$HC_WAKE_ELIGIBLE" =~ ^[01]$ && \
    "$HC_CREATED_EPOCH" =~ ^[0-9]+$ && "$HC_EXPIRES_EPOCH" =~ ^[0-9]+$ ]] || \
    { HOOK_CAPABILITY_REASON='invalid-record'; return 1; }
  ((NOW_EPOCH <= HC_EXPIRES_EPOCH)) || \
    { HOOK_CAPABILITY_REASON='expired'; return 1; }
  if ((! HC_WAKE_ELIGIBLE)); then
    [[ "$HC_SOURCE_SHA" == "$(current_sha)" ]] || \
      { HOOK_CAPABILITY_REASON='source-binding-drift'; return 1; }
  fi
  presence_file="$(presence_path "$agent" "$lane")"
  [[ -f "$presence_file" ]] || { HOOK_CAPABILITY_REASON='presence-absent'; return 1; }
  load_presence "$presence_file"
  presence_state || { HOOK_CAPABILITY_REASON="presence-${PRESENCE_REASON}"; return 1; }
  current_generation="$(process_presence_delivery_generation \
    "$agent" "$lane" "$HC_WORKTREE" "$HC_HARNESS" 2>/dev/null || true)"
  [[ -n "$current_generation" && "$current_generation" == "$generation" && \
    "$P_SESSION_ID" == "$HC_SESSION_ID" && "$P_PID" == "$HC_PRESENCE_PID" && \
    "$P_PID_START" == "$HC_PRESENCE_PID_START" && \
    "$P_BOOT_ID" == "$HC_PRESENCE_BOOT_ID" && \
    "$P_PID_NAMESPACE" == "$HC_PRESENCE_PID_NAMESPACE" ]] || \
    { HOOK_CAPABILITY_REASON='presence-generation-drift'; return 1; }
  [[ "$HC_CALLER_PID" == "$P_PID" && \
    "$HC_CALLER_PID_START" == "$P_PID_START" && \
    "$HC_CALLER_BOOT_ID" == "$P_BOOT_ID" && \
    "$HC_CALLER_PID_NAMESPACE" == "$P_PID_NAMESPACE" ]] || \
    { HOOK_CAPABILITY_REASON='caller-presence-drift'; return 1; }
  [[ -x "$HC_CALLER_EXECUTABLE" ]] || \
    { HOOK_CAPABILITY_REASON='caller-executable-absent'; return 1; }
  [[ -x "$HC_PRODUCER_EXECUTABLE" ]] || \
    { HOOK_CAPABILITY_REASON='producer-absent'; return 1; }
  [[ -x "$HC_COORD_EXECUTABLE" ]] || \
    { HOOK_CAPABILITY_REASON='coord-runtime-absent'; return 1; }
  current_sha256="$(sha256sum "$HC_PRODUCER_EXECUTABLE" | awk '{print $1}')"
  current_coord_sha256="$(sha256sum "$HC_COORD_EXECUTABLE" | awk '{print $1}')"
  current_caller_sha256="$(sha256sum "$HC_CALLER_EXECUTABLE" | awk '{print $1}')"
  [[ "$current_sha256" == "$HC_PRODUCER_SHA256" ]] || \
    { HOOK_CAPABILITY_REASON='producer-drift'; return 1; }
  [[ "$current_coord_sha256" == "$HC_COORD_SHA256" ]] || \
    { HOOK_CAPABILITY_REASON='coord-runtime-drift'; return 1; }
  [[ "$current_caller_sha256" == "$HC_CALLER_SHA256" && \
    "$(readlink -f "/proc/$HC_CALLER_PID/exe" 2>/dev/null || true)" == \
      "$HC_CALLER_EXECUTABLE" ]] || \
    { HOOK_CAPABILITY_REASON='caller-executable-drift'; return 1; }
  if ((HC_WAKE_ELIGIBLE)); then
    runtime_root="$(active_coord_runtime_root)" || \
      { HOOK_CAPABILITY_REASON='runtime-root-absent'; return 1; }
    runtime_self="$(readlink -f "${BASH_SOURCE[0]}" 2>/dev/null || true)"
    bundle="$(readlink -f "$(dirname "$HC_PRODUCER_EXECUTABLE")/.." 2>/dev/null || true)"
    active_bundle="$(readlink -f "$runtime_root/current" 2>/dev/null || true)"
    manifest="$bundle/manifest"
    [[ "$runtime_self" == "$HC_COORD_EXECUTABLE" && \
      "$bundle" == "$active_bundle" && -r "$manifest" && \
      "$(manifest_field "$manifest" runtime_id)" == "$HC_RUNTIME_ID" && \
      "$(manifest_field "$manifest" source_sha)" == "$HC_SOURCE_SHA" && \
      "$(manifest_field "$manifest" loom_runtime_sha256)" == "$HC_PRODUCER_SHA256" && \
      "$(manifest_field "$manifest" coord_runtime_sha256)" == "$HC_COORD_SHA256" ]] || \
      { HOOK_CAPABILITY_REASON='manifest-binding-drift'; return 1; }
  fi
  HOOK_CAPABILITY_REASON='native-generation-attested'
  return 0
}

hook_capability_is_current() {
  local agent="$1" lane="$2" generation="$3"
  if native_hook_wake_selftest_fixture; then
    HOOK_CAPABILITY_REASON='explicit-selftest-fixture'
    return 0
  fi
  hook_capability_binding_is_current "$agent" "$lane" "$generation" || return 1
  [[ "$HC_WAKE_ELIGIBLE" == 1 ]] || \
    { HOOK_CAPABILITY_REASON='selftest-only'; return 1; }
  return 0
}

hook_capability_register_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' session_id='' ttl capability_file
  local presence_file generation tmp_file
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --session-id) require_arg "$1" "$2"; session_id="$2"; shift 2 ;;
      *) die "unknown hook-capability-register option: $1" ;;
    esac
  done
  [[ -n "$agent" && -n "$lane" && -n "$session_id" ]] || \
    die "hook-capability-register requires --agent, --lane, and --session-id"
  validate_value agent "$agent"
  validate_value lane "$lane"
  validate_value session-id "$session_id"
  native_hook_parent_identity || \
    die "native hook capability requires the matching OCaml runtime parent"
  ttl="${SOUNIO_COORD_HOOK_TTL_SECONDS:-1800}"
  [[ "$ttl" =~ ^[1-9][0-9]*$ ]] || die "hook capability ttl must be positive"
  acquire_state_lock "the native hook capability registration"
  presence_file="$(presence_path "$agent" "$lane")"
  [[ -f "$presence_file" ]] || die "native hook capability requires process presence"
  load_presence "$presence_file"
  presence_state || die "native hook capability presence is not live: $PRESENCE_REASON"
  [[ "$P_AGENT" == "$agent" && "$P_LANE" == "$lane" && \
    "$P_SESSION_ID" == "$session_id" && "$P_WORKTREE" == "$WORKTREE" ]] || \
    die "native hook capability does not match process presence"
  native_hook_caller_matches_presence || \
    die "native hook caller does not match the existing process presence"
  generation="$(process_presence_delivery_generation \
    "$agent" "$lane" "$P_WORKTREE" "$P_HARNESS")" || \
    die "native hook capability has no process generation"
  capability_file="$(hook_capability_path "$agent" "$lane")"
  tmp_file="$(mktemp "$HOOK_CAPABILITIES_DIR/.hook-capability-write.XXXXXX")"
  {
    printf 'schema=loom-native-hook-capability-v1\n'
    printf 'state=NATIVE_HOOK_ATTESTED\n'
    printf 'agent=%s\n' "$agent"
    printf 'lane=%s\n' "$lane"
    printf 'session_id=%s\n' "$session_id"
    printf 'generation=%s\n' "$generation"
    printf 'worktree=%s\n' "$P_WORKTREE"
    printf 'harness=%s\n' "$P_HARNESS"
    printf 'presence_pid=%s\n' "$P_PID"
    printf 'presence_pid_start=%s\n' "$P_PID_START"
    printf 'presence_boot_id=%s\n' "$P_BOOT_ID"
    printf 'presence_pid_namespace=%s\n' "$P_PID_NAMESPACE"
    printf 'producer_executable=%s\n' "$NATIVE_HOOK_PARENT_EXECUTABLE"
    printf 'producer_sha256=%s\n' "$NATIVE_HOOK_PARENT_SHA256"
    printf 'coord_executable=%s\n' "$NATIVE_HOOK_COORD_EXECUTABLE"
    printf 'coord_sha256=%s\n' "$NATIVE_HOOK_COORD_SHA256"
    printf 'caller_pid=%s\n' "$NATIVE_HOOK_CALLER_PID"
    printf 'caller_pid_start=%s\n' "$NATIVE_HOOK_CALLER_PID_START"
    printf 'caller_boot_id=%s\n' "$NATIVE_HOOK_CALLER_BOOT_ID"
    printf 'caller_pid_namespace=%s\n' "$NATIVE_HOOK_CALLER_PID_NAMESPACE"
    printf 'caller_executable=%s\n' "$NATIVE_HOOK_CALLER_EXECUTABLE"
    printf 'caller_sha256=%s\n' "$NATIVE_HOOK_CALLER_SHA256"
    printf 'wake_eligible=%s\n' "$NATIVE_HOOK_WAKE_ELIGIBLE"
    printf 'runtime_id=%s\n' "$NATIVE_HOOK_RUNTIME_ID"
    printf 'source_sha=%s\n' "$NATIVE_HOOK_SOURCE_SHA"
    printf 'created_utc=%s\n' "$NOW_UTC"
    printf 'created_epoch=%s\n' "$NOW_EPOCH"
    printf 'expires_epoch=%s\n' "$((NOW_EPOCH + ttl))"
  } > "$tmp_file"
  mv "$tmp_file" "$capability_file"
  printf 'utc=%s event=HOOK_CAPABILITY_REGISTERED agent=%s lane=%s session_id=%s generation=%s runtime_id=%s source_sha=%s state=NATIVE_HOOK_ATTESTED wake_eligible=%s\n' \
    "$NOW_UTC" "$agent" "$lane" "$session_id" "$generation" \
    "$NATIVE_HOOK_RUNTIME_ID" "$NATIVE_HOOK_SOURCE_SHA" \
    "$NATIVE_HOOK_WAKE_ELIGIBLE" >> "$EVENT_LOG"
  printf 'HOOK_CAPABILITY_REGISTERED agent=%s lane=%s session_id=%s generation=%s runtime_id=%s source_sha=%s state=NATIVE_HOOK_ATTESTED wake_eligible=%s\n' \
    "$agent" "$lane" "$session_id" "$generation" "$NATIVE_HOOK_RUNTIME_ID" \
    "$NATIVE_HOOK_SOURCE_SHA" "$NATIVE_HOOK_WAKE_ELIGIBLE"
}

hook_capability_unregister_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' session_id='' capability_file presence_file
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --session-id) require_arg "$1" "$2"; session_id="$2"; shift 2 ;;
      *) die "unknown hook-capability-unregister option: $1" ;;
    esac
  done
  [[ -n "$agent" && -n "$lane" && -n "$session_id" ]] || \
    die "hook-capability-unregister requires --agent, --lane, and --session-id"
  native_hook_parent_identity || \
    die "native hook capability removal requires the matching OCaml runtime parent"
  acquire_state_lock "the native hook capability removal"
  presence_file="$(presence_path "$agent" "$lane")"
  [[ -f "$presence_file" ]] || die "native hook capability removal requires process presence"
  load_presence "$presence_file"
  [[ "$P_SESSION_ID" == "$session_id" ]] || die "native hook capability session mismatch"
  native_hook_caller_matches_presence || \
    die "native hook removal caller does not match process presence"
  capability_file="$(hook_capability_path "$agent" "$lane")"
  [[ -f "$capability_file" ]] && unlink "$capability_file"
  printf 'utc=%s event=HOOK_CAPABILITY_UNREGISTERED agent=%s lane=%s\n' \
    "$NOW_UTC" "$agent" "$lane" >> "$EVENT_LOG"
  printf 'HOOK_CAPABILITY_UNREGISTERED agent=%s lane=%s\n' "$agent" "$lane"
}

hook_capability_status_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' capability_file
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      *) die "unknown hook-capability-status option: $1" ;;
    esac
  done
  [[ -n "$agent" && -n "$lane" ]] || \
    die "hook-capability-status requires --agent and --lane"
  capability_file="$(hook_capability_path "$agent" "$lane")"
  [[ -f "$capability_file" ]] || die "native hook capability not found: $agent/$lane"
  load_hook_capability "$capability_file"
  if hook_capability_binding_is_current "$agent" "$lane" "$HC_GENERATION"; then
    printf 'HOOK_CAPABILITY_STATUS agent=%s lane=%s session_id=%s generation=%s runtime_id=%s source_sha=%s state=NATIVE_HOOK_ATTESTED wake_eligible=%s reason=%s\n' \
      "$agent" "$lane" "$HC_SESSION_ID" "$HC_GENERATION" "$HC_RUNTIME_ID" \
      "$HC_SOURCE_SHA" "$HC_WAKE_ELIGIBLE" "$HOOK_CAPABILITY_REASON"
  else
    printf 'HOOK_CAPABILITY_STATUS agent=%s lane=%s session_id=%s generation=%s runtime_id=%s source_sha=%s state=INELIGIBLE reason=%s\n' \
      "$agent" "$lane" "$HC_SESSION_ID" "$HC_GENERATION" "$HC_RUNTIME_ID" \
      "$HC_SOURCE_SHA" "$HOOK_CAPABILITY_REASON"
    return 1
  fi
}

wake_submission_generation_is_current() {
  local current_pane current_pid current_command current_path current_root current_generation
  case "$S_TRANSPORT" in
    tmux)
      [[ -S "$S_SOCKET" ]] || return 1
      current_pane="$(tmux -S "$S_SOCKET" display-message -p -t "$S_ADDRESS" '#{pane_id}' 2>/dev/null || true)"
      current_pid="$(tmux -S "$S_SOCKET" display-message -p -t "$S_ADDRESS" '#{pane_pid}' 2>/dev/null || true)"
      current_command="$(tmux -S "$S_SOCKET" display-message -p -t "$S_ADDRESS" '#{pane_current_command}' 2>/dev/null || true)"
      current_path="$(tmux -S "$S_SOCKET" display-message -p -t "$S_ADDRESS" '#{pane_current_path}' 2>/dev/null || true)"
      [[ "$current_pane" == "$S_ADDRESS" && "$current_pid" =~ ^[1-9][0-9]*$ && \
        -n "$current_path" ]] || return 1
      harness_command_matches "$S_HARNESS" "$current_command" || return 1
      current_root="$(git -C "$current_path" rev-parse --show-toplevel 2>/dev/null || true)"
      [[ -n "$current_root" ]] || return 1
      current_root="$(cd "$current_root" && pwd -P)"
      [[ "$current_root" == "$S_WORKTREE" ]] || return 1
      if [[ "$S_GENERATION" == process-* ]]; then
        current_generation="$(process_presence_delivery_generation \
          "$S_AGENT" "$S_LANE" "$S_WORKTREE" "$S_HARNESS" 2>/dev/null || true)"
      else
        current_generation="tmux-$S_ADDRESS-$current_pid"
      fi
      [[ -n "$current_generation" && "$current_generation" == "$S_GENERATION" ]]
      ;;
    agentd|loom)
      local endpoint_file
      endpoint_file="$(endpoint_path "$S_AGENT" "$S_LANE")"
      [[ -f "$endpoint_file" ]] || return 1
      load_endpoint "$endpoint_file"
      endpoint_state || return 1
      [[ "$E_ID" == "$S_ENDPOINT_ID" ]] || return 1
      current_generation="$(registered_delivery_generation 2>/dev/null || true)"
      [[ -n "$current_generation" && "$current_generation" == "$S_GENERATION" ]]
      ;;
    *) return 1 ;;
  esac
}

wait_for_wake_start() {
  local receipt_file="$1" timeout_millis checks attempt
  timeout_millis="${SOUNIO_COORD_WAKE_START_TIMEOUT_MILLIS:-1500}"
  [[ "$timeout_millis" =~ ^[0-9]+$ ]] || \
    die "SOUNIO_COORD_WAKE_START_TIMEOUT_MILLIS must be a non-negative integer"
  checks=$((timeout_millis / 50))
  for ((attempt = 0; attempt <= checks; attempt++)); do
    [[ -f "$receipt_file" ]] && return 0
    ((attempt == checks)) || sleep 0.05
  done
  return 1
}

tmux_wake_prompt_is_visible() {
  local socket="$1" address="$2" message_id="$3"
  tmux -S "$socket" capture-pane -p -J -t "$address" 2>/dev/null | \
    grep -Fq -- "$message_id"
}

attempt_tmux_wake_submission() {
  local message_id="$1" endpoint_id="$2" target_agent="$3" target_lane="$4"
  local harness="$5" target_worktree="$6" socket="$7" address="$8"
  local generation="$9" discovery="${10}" prompt="${11}"
  local receipt_file submission_file current_utc current_epoch needs_insert=1
  local insertion_uncertain=0

  receipt_file="$(wake_receipt_path "$message_id" "$endpoint_id" "$generation")"
  submission_file="$(wake_submission_path "$message_id" "$endpoint_id" "$generation")"
  if [[ -f "$receipt_file" ]]; then
    WAKE_STATUS='deduplicated'
    printf 'WAKE_SKIPPED message_id=%s endpoint_id=%s generation=%s reason=already-started discovery=%s\n' \
      "$message_id" "$endpoint_id" "$generation" "$discovery"
    return 0
  fi

  current_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  current_epoch="$(date +%s)"
  if [[ -f "$submission_file" ]]; then
    load_wake_submission "$submission_file"
    [[ "$S_SCHEMA" == loom-wake-submission-v1 && "$S_MESSAGE_ID" == "$message_id" && \
      "$S_ENDPOINT_ID" == "$endpoint_id" && "$S_AGENT" == "$target_agent" && \
      "$S_LANE" == "$target_lane" && "$S_GENERATION" == "$generation" ]] || \
      die "wake submission identity mismatch: $submission_file"
    case "$S_STATE" in
      prepared|submit-uncertain|submitted) ;;
      *) die "wake submission has invalid transport state: $submission_file" ;;
    esac
    case "$S_INSERTION_STATE" in
      not-attempted) needs_insert=1 ;;
      confirmed) needs_insert=0 ;;
      uncertain)
        needs_insert=0
        if tmux_wake_prompt_is_visible "$socket" "$address" "$message_id"; then
          S_INSERTION_STATE='confirmed'
          [[ -n "$S_INSERTED_UTC" ]] || S_INSERTED_UTC="$current_utc"
        else
          insertion_uncertain=1
        fi
        ;;
      *) die "wake submission has invalid insertion state: $submission_file" ;;
    esac
  else
    S_SCHEMA='loom-wake-submission-v1'
    S_STATE='prepared'
    S_MESSAGE_ID="$message_id"
    S_ENDPOINT_ID="$endpoint_id"
    S_AGENT="$target_agent"
    S_LANE="$target_lane"
    S_HARNESS="$harness"
    S_WORKTREE="$target_worktree"
    S_TRANSPORT='tmux'
    S_ADDRESS="$address"
    S_SOCKET="$socket"
    S_GENERATION="$generation"
    S_DISCOVERY="$discovery"
    S_CREATED_UTC="$current_utc"
    S_INSERTION_STATE='not-attempted'
    S_INSERTED_UTC=''
    S_SUBMITTED_UTC=''
    S_LAST_ATTEMPT_EPOCH=0
    S_ATTEMPTS=0
  fi
  if ! hook_capability_is_current "$target_agent" "$target_lane" "$generation"; then
    # A legacy lane may retain a durable obligation, but it must not look like
    # an attempted terminal write. A later native generation can start here.
    write_wake_submission "$submission_file"
    cleanup_lock
    WAKE_STATUS='pending-native-hook'
    printf 'utc=%s event=WAKE_DEFERRED message_id=%s endpoint_id=%s agent=%s lane=%s transport=tmux address=%s generation=%s reason=hook-capability-%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$message_id" "$endpoint_id" \
      "$target_agent" "$target_lane" "$address" "$generation" \
      "$HOOK_CAPABILITY_REASON" >> "$EVENT_LOG"
    printf 'WAKE_PENDING message_id=%s endpoint_id=%s transport=tmux address=%s generation=%s state=awaiting-native-hook reason=%s discovery=%s\n' \
      "$message_id" "$endpoint_id" "$address" "$generation" \
      "$HOOK_CAPABILITY_REASON" "$discovery"
    return 1
  fi

  S_LAST_ATTEMPT_EPOCH="$current_epoch"
  S_ATTEMPTS=$((S_ATTEMPTS + 1))
  if ((needs_insert)); then
    # Persist uncertainty immediately before the external write. A crash after
    # send-keys can recover by observing this exact message id, but may never
    # blindly reinsert.
    S_INSERTION_STATE='uncertain'
  fi
  write_wake_submission "$submission_file"
  cleanup_lock

  if ((insertion_uncertain)); then
    WAKE_STATUS='pending-insertion-uncertain'
    printf 'WAKE_PENDING message_id=%s endpoint_id=%s transport=tmux address=%s generation=%s state=insertion-uncertain discovery=%s\n' \
      "$message_id" "$endpoint_id" "$address" "$generation" "$discovery"
    return 1
  fi

  if [[ -f "$receipt_file" ]]; then
    WAKE_STATUS='started'
    printf 'WAKE_STARTED message_id=%s endpoint_id=%s transport=tmux address=%s generation=%s discovery=%s\n' \
      "$message_id" "$endpoint_id" "$address" "$generation" "$discovery"
    return 0
  fi

  if ((needs_insert)); then
    if ! tmux -S "$socket" send-keys -t "$address" -l "$prompt" 2>/dev/null; then
      WAKE_STATUS='pending-insertion-uncertain'
      printf 'WAKE_PENDING message_id=%s endpoint_id=%s transport=tmux address=%s generation=%s state=insertion-uncertain discovery=%s\n' \
        "$message_id" "$endpoint_id" "$address" "$generation" "$discovery"
      return 1
    fi
    if [[ "${SOUNIO_COORD_TEST_FAIL_AFTER_WAKE_INSERT:-0}" == 1 ]]; then
      WAKE_STATUS='pending-insertion-uncertain'
      printf 'WAKE_PENDING message_id=%s endpoint_id=%s transport=tmux address=%s generation=%s state=insertion-uncertain discovery=%s sabotage=after-external-insert\n' \
        "$message_id" "$endpoint_id" "$address" "$generation" "$discovery"
      return 1
    fi
    acquire_state_lock "the wake insertion receipt"
    if [[ -f "$submission_file" && ! -f "$receipt_file" ]]; then
      load_wake_submission "$submission_file"
      S_INSERTION_STATE='confirmed'
      S_INSERTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      write_wake_submission "$submission_file"
    fi
    cleanup_lock
    printf 'utc=%s event=WAKE_INSERTED message_id=%s endpoint_id=%s agent=%s lane=%s transport=tmux address=%s generation=%s discovery=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$message_id" "$endpoint_id" \
      "$target_agent" "$target_lane" "$address" "$generation" "$discovery" >> "$EVENT_LOG"
  fi

  acquire_state_lock "the wake submit preparation"
  if [[ -f "$submission_file" && ! -f "$receipt_file" ]]; then
    load_wake_submission "$submission_file"
    case "$S_STATE" in
      prepared) S_STATE='submit-uncertain' ;;
      submit-uncertain|submitted) ;;
      *) die "wake submission has invalid pre-submit state: $submission_file" ;;
    esac
    write_wake_submission "$submission_file"
  fi
  cleanup_lock

  if [[ ! -f "$receipt_file" ]] && \
    ! tmux -S "$socket" send-keys -t "$address" Enter 2>/dev/null; then
    WAKE_STATUS='pending-submit'
    printf 'WAKE_PENDING message_id=%s endpoint_id=%s transport=tmux address=%s generation=%s state=submit-pending discovery=%s\n' \
      "$message_id" "$endpoint_id" "$address" "$generation" "$discovery"
    return 1
  fi
  acquire_state_lock "the wake submit receipt"
  if [[ -f "$submission_file" && ! -f "$receipt_file" ]]; then
    load_wake_submission "$submission_file"
    S_STATE='submitted'
    [[ -n "$S_SUBMITTED_UTC" ]] || \
      S_SUBMITTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    write_wake_submission "$submission_file"
  fi
  cleanup_lock
  printf 'utc=%s event=WAKE_SUBMITTED message_id=%s endpoint_id=%s agent=%s lane=%s transport=tmux address=%s generation=%s discovery=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$message_id" "$endpoint_id" \
    "$target_agent" "$target_lane" "$address" "$generation" "$discovery" >> "$EVENT_LOG"

  if wait_for_wake_start "$receipt_file"; then
    WAKE_STATUS='started'
    printf 'WAKE_STARTED message_id=%s endpoint_id=%s transport=tmux address=%s generation=%s discovery=%s\n' \
      "$message_id" "$endpoint_id" "$address" "$generation" "$discovery"
    return 0
  fi
  WAKE_STATUS='pending-start'
  printf 'WAKE_PENDING message_id=%s endpoint_id=%s transport=tmux address=%s generation=%s state=awaiting-start discovery=%s\n' \
    "$message_id" "$endpoint_id" "$address" "$generation" "$discovery"
  return 1
}

WAKE_STATUS='unavailable'
attempt_message_wake() {
  local message_id="$1" message_file endpoint_file receipt_file prompt tmp_file ack_file
  local target_agent target_lane delivery_generation discovered=0 launcher
  WAKE_STATUS='unavailable'
  message_file="$MESSAGES_DIR/$(slug "$message_id").message"
  [[ -f "$message_file" ]] || return 1
  load_message "$message_file"
  [[ -n "$M_TO_AGENT" && -n "$M_TO_LANE" ]] || return 1
  target_agent="$M_TO_AGENT"
  target_lane="$M_TO_LANE"
  ack_file="$(message_ack_path "$M_ID" "$target_agent" "$target_lane")"
  if [[ -f "$ack_file" ]]; then
    WAKE_STATUS='acknowledged'
    printf 'WAKE_SKIPPED message_id=%s reason=acknowledged\n' "$M_ID"
    return 0
  fi
  endpoint_file="$(endpoint_path "$M_TO_AGENT" "$M_TO_LANE")"
  if [[ -f "$endpoint_file" ]]; then
    load_endpoint "$endpoint_file"
    if ! endpoint_state; then
      WAKE_STATUS="$ENDPOINT_STATE"
      if [[ "$ENDPOINT_STATE" == drifted ]]; then
        printf 'WAKE_REFUSED message_id=%s endpoint_id=%s reason=endpoint-drift\n' "$M_ID" "$E_ID" >&2
        WAKE_STATUS='failed-closed'
        return 1
      fi
      endpoint_file=''
    fi
  fi
  if [[ -z "$endpoint_file" || ! -f "$endpoint_file" ]]; then
    if discover_history_endpoint "$target_agent" "$target_lane"; then
      discovered=1
    fi
    load_message "$message_file"
    ((discovered)) || return 1
    delivery_generation="$(discovered_delivery_generation)" || return 1
    receipt_file="$(wake_receipt_path "$M_ID" "$D_ENDPOINT_ID" "$delivery_generation")"
    if [[ -f "$receipt_file" ]]; then
      WAKE_STATUS='deduplicated'
      printf 'WAKE_SKIPPED message_id=%s endpoint_id=%s generation=%s reason=already-started discovery=%s\n' \
        "$M_ID" "$D_ENDPOINT_ID" "$delivery_generation" "$D_DISCOVERY"
      return 0
    fi
    launcher="$(coord_inbox_launcher)"
    prompt="Sounio coordination wake: $M_KIND $M_ID from $(slug "$M_FROM_AGENT")/$(slug "$M_FROM_LANE") is waiting. Run $launcher inbox --agent $D_AGENT --lane $D_LANE --directed-only --newest-first, then run $launcher reply --agent $D_AGENT --lane $D_LANE --reply-to $M_ID --message \"<response>\" or $launcher ack --agent $D_AGENT --lane $D_LANE --message $M_ID."
    attempt_tmux_wake_submission "$M_ID" "$D_ENDPOINT_ID" "$D_AGENT" "$D_LANE" \
      "$D_HARNESS" "$D_WORKTREE" "$D_SOCKET" "$D_ADDRESS" "$delivery_generation" \
      "$D_DISCOVERY" "$prompt"
    return $?
  fi
  load_endpoint "$endpoint_file"
  delivery_generation="$(registered_delivery_generation)" || return 1
  receipt_file="$(wake_receipt_path "$M_ID" "$E_ID" "$delivery_generation")"
  if [[ -f "$receipt_file" ]]; then
    WAKE_STATUS='deduplicated'
    if [[ "$E_TRANSPORT" == tmux ]]; then
      printf 'WAKE_SKIPPED message_id=%s endpoint_id=%s generation=%s reason=already-started\n' \
        "$M_ID" "$E_ID" "$delivery_generation"
    else
      printf 'WAKE_SKIPPED message_id=%s endpoint_id=%s generation=%s reason=already-delivered\n' \
        "$M_ID" "$E_ID" "$delivery_generation"
    fi
    return 0
  fi

  launcher="$(coord_inbox_launcher)"
  prompt="Sounio coordination wake: $M_KIND $M_ID from $(slug "$M_FROM_AGENT")/$(slug "$M_FROM_LANE") is waiting. Run $launcher inbox --agent $E_AGENT --lane $E_LANE --directed-only --newest-first, then run $launcher reply --agent $E_AGENT --lane $E_LANE --reply-to $M_ID --message \"<response>\" or $launcher ack --agent $E_AGENT --lane $E_LANE --message $M_ID."
  if [[ "$E_TRANSPORT" == tmux ]]; then
    attempt_tmux_wake_submission "$M_ID" "$E_ID" "$E_AGENT" "$E_LANE" "$E_HARNESS" \
      "$E_WORKTREE" "$E_SOCKET" "$E_ADDRESS" "$delivery_generation" registered "$prompt"
    return $?
  fi
  if ! deliver_registered_endpoint "$prompt" "$M_ID"; then
    WAKE_STATUS='failed'
    printf 'WAKE_FAILED message_id=%s endpoint_id=%s transport=%s\n' \
      "$M_ID" "$E_ID" "$E_TRANSPORT" >&2
    return 1
  fi

  tmp_file="$(mktemp "$WAKES_DIR/.wake-write.XXXXXX")"
  printf 'utc=%s message_id=%s endpoint_id=%s transport=%s address=%s generation=%s\n' \
    "$NOW_UTC" "$M_ID" "$E_ID" "$E_TRANSPORT" "$E_ADDRESS" \
    "$delivery_generation" > "$tmp_file"
  mv "$tmp_file" "$receipt_file"
  printf 'utc=%s event=WAKE_DELIVERED message_id=%s endpoint_id=%s agent=%s lane=%s transport=%s address=%s generation=%s\n' \
    "$NOW_UTC" "$M_ID" "$E_ID" "$E_AGENT" "$E_LANE" "$E_TRANSPORT" \
    "$E_ADDRESS" "$delivery_generation" >> "$EVENT_LOG"
  WAKE_STATUS='delivered'
  printf 'WAKE_DELIVERED message_id=%s endpoint_id=%s transport=%s address=%s generation=%s\n' \
    "$M_ID" "$E_ID" "$E_TRANSPORT" "$E_ADDRESS" "$delivery_generation"
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
    if [[ -n "$M_EXPERIMENT_ID" ]]; then
      printf ' experiment=%s experiment_prereg=%s experiment_outcome=%s prereg_sha256=%s outcome_sha256=%s' \
        "$M_EXPERIMENT_ID" "$M_EXPERIMENT_PREREG" "$M_EXPERIMENT_OUTCOME" \
        "$M_EXPERIMENT_PREREG_SHA256" "$M_EXPERIMENT_OUTCOME_SHA256"
    fi
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
  remove_presence_for_lane "$agent" "$lane" "$WORKTREE" release
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
  local agent="${SOUNIO_AGENT_ID:-}" lane='' harness='' transport='' address='' socket='' token_file=''
  local ttl="${SOUNIO_COORD_ENDPOINT_TTL_SECONDS:-1800}" claim_file endpoint_file presence_file created_utc
  local existing_endpoint registered_address registered_pid registered_command token_mode token_owner
  local -a endpoint_paths=()
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --harness) require_arg "$1" "$2"; harness="$2"; shift 2 ;;
      --transport) require_arg "$1" "$2"; transport="$2"; shift 2 ;;
      --address) require_arg "$1" "$2"; address="$2"; shift 2 ;;
      --socket) require_arg "$1" "$2"; socket="$2"; shift 2 ;;
      --token-file) require_arg "$1" "$2"; token_file="$2"; shift 2 ;;
      --ttl-seconds) require_arg "$1" "$2"; ttl="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown endpoint-register option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "endpoint-register requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "endpoint-register requires --lane"
  [[ "$harness" =~ ^(claude|codex|grok|cursor|kimi|beagle)$ ]] || \
    die "--harness must be claude, codex, grok, cursor, kimi, or beagle"
  [[ "$transport" =~ ^(tmux|agentd|loom)$ ]] || die "--transport must be tmux, agentd, or loom"
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
  [[ -z "$token_file" ]] || validate_value token-file "$token_file"
  socket="$(readlink -f "$socket" 2>/dev/null || true)"
  [[ -n "$socket" && -S "$socket" ]] || die "$transport socket is not available: ${socket:-missing}"
  case "$transport" in
    tmux)
      [[ -z "$token_file" ]] || die "tmux endpoints do not accept --token-file"
      tmux_endpoint_snapshot "$socket" "$address" || \
        die "tmux endpoint does not resolve to the current worktree"
      harness_command_matches "$harness" "$T_COMMAND" || \
        die "tmux pane command $T_COMMAND does not match harness $harness"
      registered_address="$T_PANE_ID"
      registered_pid="$T_PANE_PID"
      registered_command="$T_COMMAND"
      ;;
    agentd|loom)
      [[ -n "$token_file" ]] || die "$transport endpoints require --token-file"
      token_file="$(readlink -f "$token_file" 2>/dev/null || true)"
      [[ -n "$token_file" && -f "$token_file" ]] || \
        die "$transport capability file is not available: ${token_file:-missing}"
      token_owner="$(stat -c %u "$token_file" 2>/dev/null || true)"
      token_mode="$(stat -c %a "$token_file" 2>/dev/null || true)"
      [[ "$token_owner" == "$(id -u)" && "$token_mode" == 600 ]] || \
        die "$transport capability file must be owned by the current uid with mode 600"
      [[ "$(readlink -f "$address" 2>/dev/null || true)" == "$socket" ]] || \
        die "$transport address must name its control socket"
      if [[ "$transport" == agentd ]]; then
        agentd_endpoint_snapshot "$agent" "$lane" "$socket" "$token_file" || \
          die "agentd endpoint did not return a verified live identity"
      else
        loom_endpoint_snapshot "$agent" "$lane" "$socket" "$token_file" || \
          die "Loom endpoint did not return a verified live identity"
      fi
      [[ "$A_WORKTREE" == "$WORKTREE" ]] || \
        die "$transport endpoint belongs to worktree $A_WORKTREE"
      harness_command_matches "$harness" "$A_COMMAND" || \
        die "$transport command $A_COMMAND does not match harness $harness"
      registered_address="$socket"
      registered_pid="$A_DAEMON_PID"
      registered_command="$A_COMMAND"
      ;;
  esac

  claim_file="$CLAIMS_DIR/$(claim_id_for "$agent" "$lane").claim"
  endpoint_file="$(endpoint_path "$agent" "$lane")"
  acquire_state_lock "the endpoint registration"
  [[ -f "$claim_file" ]] || die "claim not found: $(claim_id_for "$agent" "$lane")"
  load_claim "$claim_file"
  claim_expired && die "claim expired before endpoint registration: $C_ID"
  [[ "$C_AGENT" == "$agent" && "$C_LANE" == "$lane" ]] || die "claim owner mismatch"
  if [[ "$transport" == agentd || "$transport" == loom || "$C_WORKTREE" != "$WORKTREE" ]]; then
    presence_file="$(presence_path "$agent" "$lane")"
    [[ -f "$presence_file" ]] || \
      die "$transport endpoint requires verified process presence"
    load_presence "$presence_file"
    presence_state || die "$transport process presence is $PRESENCE_STATE: $PRESENCE_REASON"
    [[ "$P_WORKTREE" == "$WORKTREE" ]] || \
      die "process presence belongs to worktree $P_WORKTREE"
    if [[ "$transport" == agentd || "$transport" == loom ]]; then
      [[ "$PRESENCE_STATE" == live && "$P_HARNESS" == "$harness" && \
        "$P_SESSION_ID" == "$A_SESSION_ID" && "$P_PID" == "$A_HARNESS_PID" && \
        "$P_PID_START" == "$A_HARNESS_PID_START" ]] || \
        die "$transport identity does not match the live process-presence generation"
    fi
  fi
  endpoint_paths=("$ENDPOINTS_DIR"/*.endpoint)
  for existing_endpoint in "${endpoint_paths[@]}"; do
    [[ -f "$existing_endpoint" && "$existing_endpoint" != "$endpoint_file" ]] || continue
    load_endpoint "$existing_endpoint"
    endpoint_expired && continue
    if [[ "$E_SOCKET" == "$socket" && "$E_ADDRESS" == "$registered_address" ]]; then
      die "$transport endpoint is already owned by $E_AGENT/$E_LANE"
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
  E_ADDRESS="$registered_address"
  E_SOCKET="$socket"
  E_TOKEN_FILE="$token_file"
  E_PANE_PID="$registered_pid"
  E_INSTANCE_ID="${A_INSTANCE_ID:-}"
  E_SESSION_ID="${A_SESSION_ID:-}"
  E_HARNESS_PID="${A_HARNESS_PID:-}"
  E_HARNESS_PID_START="${A_HARNESS_PID_START:-}"
  E_COMMAND="$registered_command"
  E_CREATED_UTC="$created_utc"
  E_LAST_UTC="$NOW_UTC"
  E_LAST_EPOCH="$NOW_EPOCH"
  E_TTL="$ttl"
  write_endpoint "$endpoint_file"
  printf 'utc=%s event=ENDPOINT_REGISTERED endpoint_id=%s agent=%s lane=%s worktree=%s harness=%s transport=%s address=%s\n' \
    "$NOW_UTC" "$E_ID" "$E_AGENT" "$E_LANE" "$E_WORKTREE" "$E_HARNESS" \
    "$E_TRANSPORT" "$E_ADDRESS" >> "$EVENT_LOG"
  printf 'ENDPOINT_REGISTERED endpoint_id=%s harness=%s transport=%s address=%s endpoint_pid=%s command=%s expires_in=%s\n' \
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
  printf 'ENDPOINT_STATUS endpoint_id=%s state=%s agent=%s lane=%s worktree=%s harness=%s transport=%s address=%s endpoint_pid=%s command=%s session_id=%s instance_id=%s last_seen=%s\n' \
    "$E_ID" "$state" "$E_AGENT" "$E_LANE" "$E_WORKTREE" "$E_HARNESS" \
    "$E_TRANSPORT" "$E_ADDRESS" "$E_PANE_PID" "$E_COMMAND" \
    "${E_SESSION_ID:--}" "${E_INSTANCE_ID:--}" "$E_LAST_UTC"
}

presence_register_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' harness='' session_id='' host=''
  local boot_id='' pid_namespace='' pid='' pid_start=''
  local ttl="${SOUNIO_COORD_PRESENCE_TTL_SECONDS:-1800}"
  local claim_file presence_file claim_common created_utc event='PRESENCE_REGISTERED' generation=1
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --harness) require_arg "$1" "$2"; harness="$2"; shift 2 ;;
      --session-id) require_arg "$1" "$2"; session_id="$2"; shift 2 ;;
      --host) require_arg "$1" "$2"; host="$2"; shift 2 ;;
      --boot-id) require_arg "$1" "$2"; boot_id="$2"; shift 2 ;;
      --pid-namespace) require_arg "$1" "$2"; pid_namespace="$2"; shift 2 ;;
      --pid) require_arg "$1" "$2"; pid="$2"; shift 2 ;;
      --pid-start) require_arg "$1" "$2"; pid_start="$2"; shift 2 ;;
      --ttl-seconds) require_arg "$1" "$2"; ttl="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown presence-register option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "presence-register requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "presence-register requires --lane"
  [[ "$harness" =~ ^(claude|codex|grok|cursor|kimi|beagle)$ ]] || \
    die "--harness must be claude, codex, grok, cursor, kimi, or beagle"
  [[ -n "$session_id" ]] || die "presence-register requires --session-id"
  [[ -n "$host" ]] || die "presence-register requires --host"
  [[ -n "$boot_id" ]] || die "presence-register requires --boot-id"
  [[ -n "$pid_namespace" ]] || die "presence-register requires --pid-namespace"
  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || die "--pid must be a positive integer"
  [[ "$pid_start" =~ ^[1-9][0-9]*$ ]] || die "--pid-start must be a positive integer"
  [[ "$ttl" =~ ^[1-9][0-9]*$ ]] || die "--ttl-seconds must be a positive integer"
  validate_value agent "$agent"
  validate_value lane "$lane"
  validate_value session-id "$session_id"
  validate_value host "$host"
  validate_value boot-id "$boot_id"
  validate_value pid-namespace "$pid_namespace"

  claim_file="$CLAIMS_DIR/$(claim_id_for "$agent" "$lane").claim"
  presence_file="$(presence_path "$agent" "$lane")"
  acquire_state_lock "the process-presence registration"
  [[ -f "$claim_file" ]] || die "claim not found: $(claim_id_for "$agent" "$lane")"
  load_claim "$claim_file"
  claim_expired && die "claim expired before presence registration: $C_ID"
  [[ "$C_AGENT" == "$agent" && "$C_LANE" == "$lane" ]] || die "claim owner mismatch"
  claim_common="$(git -C "$C_WORKTREE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  [[ "$claim_common" == "$GIT_COMMON_DIR" ]] || \
    die "claim worktree is no longer attached to this repository: $C_WORKTREE"

  created_utc="$NOW_UTC"
  if [[ -f "$presence_file" ]]; then
    load_presence "$presence_file"
    [[ "$P_AGENT" == "$agent" && "$P_LANE" == "$lane" ]] || die "presence owner mismatch"
    [[ "$P_WORKTREE" == "$WORKTREE" ]] || die "presence belongs to worktree $P_WORKTREE"
    created_utc="${P_CREATED_UTC:-$NOW_UTC}"
    generation="${P_GENERATION:-1}"
    if [[ "$P_BOOT_ID" == "$boot_id" && "$P_PID_NAMESPACE" == "$pid_namespace" && \
      "$P_PID" == "$pid" && "$P_PID_START" == "$pid_start" ]]; then
      event='PRESENCE_REFRESHED'
    else
      presence_state || true
      case "$PRESENCE_STATE" in
        live|unresponsive)
          die "lane is still bound to generation $P_GENERATION pid $P_PID ($PRESENCE_STATE)"
          ;;
        orphaned)
          generation=$((P_GENERATION + 1))
          event='PRESENCE_RECOVERED'
          ;;
        *)
          generation=$((P_GENERATION + 1))
          event='PRESENCE_RECOVERED'
          ;;
      esac
    fi
  fi

  P_ID="$(claim_id_for "$agent" "$lane")"
  P_AGENT="$agent"
  P_LANE="$lane"
  P_WORKTREE="$WORKTREE"
  P_HARNESS="$harness"
  P_SESSION_ID="$session_id"
  P_HOST="$host"
  P_BOOT_ID="$boot_id"
  P_PID_NAMESPACE="$pid_namespace"
  P_PID="$pid"
  P_PID_START="$pid_start"
  P_GENERATION="$generation"
  P_CREATED_UTC="$created_utc"
  P_LAST_UTC="$NOW_UTC"
  P_LAST_EPOCH="$NOW_EPOCH"
  P_TTL="$ttl"
  presence_state || die "refusing unverifiable process presence: $PRESENCE_REASON"
  write_presence "$presence_file"
  append_presence_event "$event" "$PRESENCE_REASON"
  printf '%s presence_id=%s generation=%s pid=%s session_id=%s last_seen=%s\n' \
    "$event" "$P_ID" "$P_GENERATION" "$P_PID" "$P_SESSION_ID" "$P_LAST_UTC"
}

presence_unregister_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' presence_file
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown presence-unregister option: $1" ;;
    esac
  done
  [[ -n "$agent" ]] || die "presence-unregister requires --agent or SOUNIO_AGENT_ID"
  [[ -n "$lane" ]] || die "presence-unregister requires --lane"
  presence_file="$(presence_path "$agent" "$lane")"
  acquire_state_lock "the process-presence removal"
  if [[ ! -f "$presence_file" ]]; then
    printf 'PRESENCE_ABSENT presence_id=%s\n' "$(claim_id_for "$agent" "$lane")"
    return 0
  fi
  remove_presence_for_lane "$agent" "$lane" "$WORKTREE" clean-exit
  printf 'PRESENCE_UNREGISTERED presence_id=%s\n' "$(claim_id_for "$agent" "$lane")"
}

pending_directed_count() {
  local agent="$1" lane="$2" message_file ack_file count=0
  local -a message_paths=()
  message_paths=("$MESSAGES_DIR"/*.message)
  for message_file in "${message_paths[@]}"; do
    [[ -f "$message_file" ]] || continue
    load_message "$message_file"
    message_expired && continue
    [[ "$M_TO_AGENT" == "$agent" && "$M_TO_LANE" == "$lane" ]] || continue
    [[ "$M_FROM_AGENT" != "$agent" || "$M_FROM_LANE" != "$lane" ]] || continue
    ack_file="$(message_ack_path "$M_ID" "$agent" "$lane")"
    [[ -f "$ack_file" ]] || count=$((count + 1))
  done
  printf '%s' "$count"
}

recover_one() {
  local agent="$1" lane="$2" compact="$3"
  local claim_file presence_file endpoint_file claim_state='missing' presence='unbound'
  local presence_reason='no-presence-record' delivery='unavailable' lane_state='missing'
  local worktree='' session_worktree='' branch='' sha='' intent='' files='' resources='' last_seen=''
  local harness='' session_id='' resume_source='none' generation=0 pid=0 pending=0 actual_branch='' actual_sha='' dirty='missing'

  claim_file="$CLAIMS_DIR/$(claim_id_for "$agent" "$lane").claim"
  if [[ -f "$claim_file" ]]; then
    load_claim "$claim_file"
    if claim_expired; then claim_state='stale'; else claim_state='active'; fi
    worktree="$C_WORKTREE"
    branch="$C_BRANCH"
    sha="$C_SHA"
    intent="$C_INTENT"
    files="$(join_files)"
    resources="$(join_resources)"
    last_seen="$C_LAST_UTC"
  fi

  presence_file="$(presence_path "$agent" "$lane")"
  if [[ -f "$presence_file" ]]; then
    load_presence "$presence_file"
    presence_state || true
    presence="$PRESENCE_STATE"
    presence_reason="$PRESENCE_REASON"
    harness="$P_HARNESS"
    session_id="$P_SESSION_ID"
    resume_source='process-presence'
    generation="$P_GENERATION"
    pid="$P_PID"
    session_worktree="$P_WORKTREE"
    [[ -n "$worktree" ]] || worktree="$P_WORKTREE"
  elif discover_resume_identity "$agent" "$lane"; then
    harness="$RECOVERY_HARNESS"
    session_id="$RECOVERY_SESSION_ID"
    resume_source='session-history'
    presence_reason='session-history-verified'
    session_worktree="$RECOVERY_WORKTREE"
    [[ -n "$worktree" ]] || worktree="$RECOVERY_WORKTREE"
  fi

  endpoint_file="$(endpoint_path "$agent" "$lane")"
  if [[ -f "$endpoint_file" ]]; then
    load_endpoint "$endpoint_file"
    endpoint_state || true
    delivery="$ENDPOINT_STATE"
  fi
  pending="$(pending_directed_count "$agent" "$lane")"

  case "$presence" in
    live) lane_state='live' ;;
    unresponsive) lane_state='unresponsive' ;;
    orphaned) lane_state='orphaned' ;;
    unbound)
      if [[ "$resume_source" == session-history ]]; then lane_state='legacy-recoverable';
      elif [[ "$claim_state" == active ]]; then lane_state='legacy-unbound';
      elif [[ "$claim_state" == stale ]]; then lane_state='stale-unbound'; fi
      ;;
  esac

  if ((compact)); then
    printf 'LANE_RECOVERY agent=%s lane=%s state=%s claim=%s presence=%s reason=%s delivery=%s pending=%s generation=%s worktree=%s\n' \
      "$agent" "$lane" "$lane_state" "$claim_state" "$presence" "$presence_reason" \
      "$delivery" "$pending" "$generation" "${worktree:--}"
    return 0
  fi

  if [[ -d "$worktree" ]]; then
    actual_branch="$(git -C "$worktree" branch --show-current 2>/dev/null || true)"
    [[ -n "$actual_branch" ]] || actual_branch="detached@$(git -C "$worktree" rev-parse --short=10 HEAD 2>/dev/null || printf unknown)"
    actual_sha="$(git -C "$worktree" rev-parse --short=12 HEAD 2>/dev/null || printf unknown)"
    dirty="$(git -C "$worktree" status --porcelain=v1 --untracked-files=all 2>/dev/null | wc -l | tr -d ' ')"
  fi
  printf 'Sounio lane recovery\n'
  printf 'snapshot_utc=%s\n' "$NOW_UTC"
  printf 'state_dir=%s\n' "$STATE_DIR"
  printf 'durability=git-common-dir\n'
  printf 'agent=%s\n' "$agent"
  printf 'lane=%s\n' "$lane"
  printf 'lane_state=%s\n' "$lane_state"
  printf 'claim_state=%s\n' "$claim_state"
  printf 'claim_last_seen=%s\n' "${last_seen:--}"
  printf 'presence_state=%s\n' "$presence"
  printf 'presence_reason=%s\n' "$presence_reason"
  printf 'presence_generation=%s\n' "$generation"
  printf 'process_pid=%s\n' "$pid"
  printf 'delivery_state=%s\n' "$delivery"
  printf 'harness=%s\n' "${harness:--}"
  printf 'resume_session_id=%s\n' "${session_id:--}"
  printf 'resume_source=%s\n' "$resume_source"
  printf 'session_worktree=%s\n' "${session_worktree:--}"
  printf 'worktree=%s\n' "${worktree:--}"
  printf 'recorded_branch=%s\n' "${branch:--}"
  printf 'actual_branch=%s\n' "${actual_branch:--}"
  printf 'recorded_sha=%s\n' "${sha:--}"
  printf 'actual_sha=%s\n' "${actual_sha:--}"
  printf 'dirty_paths=%s\n' "$dirty"
  printf 'intent=%s\n' "${intent:--}"
  printf 'files=%s\n' "${files:--}"
  printf 'resources=%s\n' "${resources:--}"
  printf 'pending_directed=%s\n' "$pending"
  printf 'inbox_next=bin/sounio-coord inbox --agent %s --lane %s --directed-only --newest-first\n' "$agent" "$lane"
  if [[ "$lane_state" == orphaned ]]; then
    printf 'recovery_next=resume harness=%s session_id=%s in worktree=%s; the next hook boundary will fence generation %s\n' \
      "${harness:--}" "${session_id:--}" "${session_worktree:-${worktree:--}}" "$((generation + 1))"
  fi
}

recover_command() {
  local agent='' lane='' all=0 claim_file presence_file
  local -a ids=() seen=()
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --all) all=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown recover option: $1" ;;
    esac
  done
  if ((all)); then
    [[ -z "$agent" && -z "$lane" ]] || die "recover --all does not accept --agent or --lane"
    printf 'Sounio fleet recovery snapshot_utc=%s state_dir=%s durability=git-common-dir\n' "$NOW_UTC" "$STATE_DIR"
    for claim_file in "$CLAIMS_DIR"/*.claim; do
      [[ -f "$claim_file" ]] || continue
      load_claim "$claim_file"
      ids+=("$C_AGENT|$C_LANE")
    done
    for presence_file in "$PRESENCES_DIR"/*.presence; do
      [[ -f "$presence_file" ]] || continue
      load_presence "$presence_file"
      ids+=("$P_AGENT|$P_LANE")
    done
    if ((${#ids[@]})); then
      mapfile -t seen < <(printf '%s\n' "${ids[@]}" | sort -u)
    fi
    for claim_file in "${seen[@]}"; do
      recover_one "${claim_file%%|*}" "${claim_file#*|}" 1
    done
    printf 'fleet_lanes=%s\n' "${#seen[@]}"
    return 0
  fi
  [[ -n "$agent" ]] || die "recover requires --agent and --lane, or --all"
  [[ -n "$lane" ]] || die "recover requires --agent and --lane, or --all"
  recover_one "$agent" "$lane" 0
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

coord_loom_obligation_runtime() {
  local script_dir sibling local_binary
  if [[ -n "${SOUNIO_COORD_LOOM_RUNTIME:-}" ]]; then
    [[ -x "$SOUNIO_COORD_LOOM_RUNTIME" ]] || \
      die "configured Loom runtime is not executable: $SOUNIO_COORD_LOOM_RUNTIME"
    printf '%s' "$SOUNIO_COORD_LOOM_RUNTIME"
    return 0
  fi
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
  sibling="$script_dir/sounio-loom-runtime"
  local_binary="$WORKTREE/tools/loom/_build/default/src/loom.exe"
  if [[ -x "$sibling" ]]; then
    printf '%s' "$sibling"
  elif [[ -x "$local_binary" ]]; then
    printf '%s' "$local_binary"
  else
    die "Sounio Loom runtime is unavailable; run scripts/dev/build_sounio_loom.sh"
  fi
}

coord_obligation_invoke() {
  local loom
  loom="$(coord_loom_obligation_runtime)"
  "$loom" "$@" --state-dir "$STATE_DIR"
}

coord_obligation_exec() {
  local loom
  loom="$(coord_loom_obligation_runtime)"
  exec "$loom" "$@" --state-dir "$STATE_DIR"
}

coord_load_obligation_activation() {
  local line
  COORD_OBLIGATION_ACTIVATION_SCHEMA=''
  COORD_OBLIGATION_ACTIVATION_EPOCH=''
  COORD_OBLIGATION_ACTIVATION_RUNTIME=''
  [[ -f "$OBLIGATION_ACTIVATION_FILE" ]] || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      schema=*) COORD_OBLIGATION_ACTIVATION_SCHEMA="${line#schema=}" ;;
      activated_epoch=*) COORD_OBLIGATION_ACTIVATION_EPOCH="${line#activated_epoch=}" ;;
      runtime_id=*) COORD_OBLIGATION_ACTIVATION_RUNTIME="${line#runtime_id=}" ;;
    esac
  done < "$OBLIGATION_ACTIVATION_FILE"
  [[ "$COORD_OBLIGATION_ACTIVATION_SCHEMA" == loom-obligation-activation-v1 &&
    "$COORD_OBLIGATION_ACTIVATION_EPOCH" =~ ^[1-9][0-9]*$ &&
    -n "$COORD_OBLIGATION_ACTIVATION_RUNTIME" ]] ||
    die "invalid durable obligation activation watermark: $OBLIGATION_ACTIVATION_FILE"
}

coord_obligation_contract_loaded_message() {
  COORD_OBLIGATION_CONTRACT_SOURCE=''
  if [[ -n "$M_OBLIGATION_SCHEMA" && -n "$M_OBLIGATION_OPT_OUT" ]]; then
    return 2
  fi
  if [[ -n "$M_OBLIGATION_SCHEMA" ]]; then
    [[ "$M_OBLIGATION_SCHEMA" == loom-durable-obligation-v1 ]] || return 2
    COORD_OBLIGATION_CONTRACT_SOURCE=marked
    return 0
  fi
  if [[ -n "$M_OBLIGATION_OPT_OUT" ]]; then
    [[ "$M_OBLIGATION_OPT_OUT" == 1 ]] || return 2
    return 1
  fi
  coord_load_obligation_activation || return 1
  [[ "$M_CREATED_EPOCH" =~ ^[0-9]+$ ]] || return 2
  if ((M_CREATED_EPOCH >= COORD_OBLIGATION_ACTIVATION_EPOCH)); then
    COORD_OBLIGATION_CONTRACT_SOURCE=legacy
    return 0
  fi
  return 1
}

coord_obligation_message() {
  local message_id="$1" message_file
  message_file="$MESSAGES_DIR/$(slug "$message_id").message"
  [[ -f "$message_file" ]] || die "message not found: $message_id"
  load_message "$message_file"
  [[ "$M_ID" == "$message_id" ]] || die "message not found: $message_id"
  [[ "$M_KIND" == request ]] || die "message is not a request: $message_id"
  [[ -n "$M_TO_AGENT" && -n "$M_TO_LANE" ]] || \
    die "durable obligations require an exactly directed request"
  if [[ -z "$M_OBLIGATION_SCHEMA" && "$M_OBLIGATION_OPT_OUT" == 1 ]]; then
    die "request explicitly opted out of durable obligations: $message_id"
  fi
  local contract_status
  if coord_obligation_contract_loaded_message; then
    contract_status=0
  else
    contract_status=$?
  fi
  case "$contract_status" in
    0) ;;
    1) die "request is outside the durable obligation activation boundary: $message_id" ;;
    *) die "request has an invalid durable obligation contract: $message_id" ;;
  esac
  COORD_OBLIGATION_MESSAGE_FILE="$message_file"
}

coord_obligation_open_loaded_request() {
  local message_file="$1" digest
  digest="$(sha256sum "$message_file" | awk '{print $1}')"
  coord_obligation_invoke obligation-open --message "$M_ID" \
    --message-digest "$digest" --from-agent "$M_FROM_AGENT" \
    --from-lane "$M_FROM_LANE" --to-agent "$M_TO_AGENT" --to-lane "$M_TO_LANE"
}

coord_obligation_live_generation() {
  local agent="$1" lane="$2" presence_file
  presence_file="$(presence_path "$agent" "$lane")"
  [[ -f "$presence_file" ]] || die "live process presence is required for $agent/$lane"
  load_presence "$presence_file"
  presence_state || true
  [[ "$PRESENCE_STATE" == live ]] || \
    die "process presence for $agent/$lane is $PRESENCE_STATE: $PRESENCE_REASON"
  [[ "$P_AGENT" == "$agent" && "$P_LANE" == "$lane" ]] || \
    die "process presence identity mismatch"
  [[ "$P_WORKTREE" == "$WORKTREE" ]] || \
    die "process presence belongs to worktree $P_WORKTREE"
  printf 'process-%s-g%s-%s-%s' "$P_SESSION_ID" "$P_GENERATION" "$P_PID" "$P_PID_START"
}

coord_obligation_open_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' message_id='' message_file
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --message) require_arg "$1" "$2"; message_id="$2"; shift 2 ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown obligation-open option: $1" ;;
    esac
  done
  [[ -n "$agent" && -n "$lane" && -n "$message_id" ]] || \
    die "obligation-open requires --agent, --lane, and --message"
  coord_obligation_message "$message_id"
  message_file="$COORD_OBLIGATION_MESSAGE_FILE"
  [[ "$M_FROM_AGENT" == "$agent" && "$M_FROM_LANE" == "$lane" ]] || \
    die "only the request sender may open its obligation"
  coord_obligation_open_loaded_request "$message_file"
}

coord_obligation_recipient_command() {
  local action="$1"; shift
  local agent="${SOUNIO_AGENT_ID:-}" lane='' message_id='' claim='' ttl=''
  local reason='' outcome='' evidence='' json=0 message_file generation
  while (($#)); do
    case "$1" in
      --agent) require_arg "$1" "$2"; agent="$2"; shift 2 ;;
      --lane) require_arg "$1" "$2"; lane="$2"; shift 2 ;;
      --message) require_arg "$1" "$2"; message_id="$2"; shift 2 ;;
      --claim) require_arg "$1" "$2"; claim="$2"; shift 2 ;;
      --ttl-seconds) require_arg "$1" "$2"; ttl="$2"; shift 2 ;;
      --reason) require_arg "$1" "$2"; reason="$2"; shift 2 ;;
      --outcome) require_arg "$1" "$2"; outcome="$2"; shift 2 ;;
      --evidence) require_arg "$1" "$2"; evidence="$2"; shift 2 ;;
      --json) json=1; shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown obligation-$action option: $1" ;;
    esac
  done
  [[ -n "$agent" && -n "$lane" && -n "$message_id" ]] || \
    die "obligation-$action requires --agent, --lane, and --message"
  coord_obligation_message "$message_id"
  message_file="$COORD_OBLIGATION_MESSAGE_FILE"
  if [[ "$action" == status ]]; then
    if [[ "$M_TO_AGENT" != "$agent" || "$M_TO_LANE" != "$lane" ]]; then
      [[ "$M_FROM_AGENT" == "$agent" && "$M_FROM_LANE" == "$lane" ]] || \
        die "obligation status is visible only to request sender or recipient"
    fi
    local -a status_args=(obligation-status --message "$message_id")
    ((json == 0)) || status_args+=(--json)
    coord_obligation_invoke "${status_args[@]}"
    return 0
  fi
  [[ "$M_TO_AGENT" == "$agent" && "$M_TO_LANE" == "$lane" ]] || \
    die "request is addressed to $M_TO_AGENT/$M_TO_LANE"
  generation="$(coord_obligation_live_generation "$agent" "$lane")"
  local -a args=("obligation-$action" --message "$message_id" --actor "$agent" \
    --lane "$lane" --generation "$generation")
  case "$action" in
    consume)
      [[ -z "$ttl" ]] || args+=(--ttl-seconds "$ttl")
      ;;
    claim)
      [[ -z "$claim" ]] || args+=(--claim "$claim")
      [[ -z "$ttl" ]] || args+=(--ttl-seconds "$ttl")
      ;;
    renew)
      [[ -n "$claim" ]] || die "obligation-renew requires --claim"
      args+=(--claim "$claim")
      [[ -z "$ttl" ]] || args+=(--ttl-seconds "$ttl")
      ;;
    interrupt)
      [[ -z "$claim" ]] || args+=(--claim "$claim")
      [[ -z "$reason" ]] || args+=(--reason "$reason")
      ;;
    recover) ;;
    complete)
      [[ -n "$claim" && -n "$outcome" && -n "$evidence" ]] || \
        die "obligation-complete requires --claim, --outcome, and --evidence"
      args+=(--claim "$claim" --outcome "$outcome" --evidence "$evidence")
      ;;
    *) die "unknown obligation recipient action: $action" ;;
  esac
  coord_obligation_invoke "${args[@]}"
}

coord_obligation_list_command() {
  local -a args=(obligation-list)
  while (($#)); do
    case "$1" in
      --json) args+=(--json); shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown obligation-list option: $1" ;;
    esac
  done
  coord_obligation_invoke "${args[@]}"
}

coord_obligation_projection_command() {
  local action="$1"; shift
  local -a args=("obligation-$action")
  while (($#)); do
    case "$1" in
      --bind) require_arg "$1" "$2"; args+=(--bind "$2"); shift 2 ;;
      --port) require_arg "$1" "$2"; args+=(--port "$2"); shift 2 ;;
      --allow-remote) args+=(--allow-remote); shift ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown obligation-$action option: $1" ;;
    esac
  done
  coord_obligation_invoke "${args[@]}"
}

coord_obligation_reconcile_command() {
  (($# == 0)) || die "obligation-reconcile does not accept arguments"
  local message_file opened=0 marked=0 legacy=0 ignored=0 contract_status
  for message_file in "$MESSAGES_DIR"/*.message; do
    [[ -f "$message_file" ]] || continue
    load_message "$message_file"
    [[ "$M_KIND" == request && -n "$M_TO_AGENT" && -n "$M_TO_LANE" ]] || continue
    if coord_obligation_contract_loaded_message; then
      contract_status=0
    else
      contract_status=$?
    fi
    case "$contract_status" in
      0)
        case "$COORD_OBLIGATION_CONTRACT_SOURCE" in
          marked) marked=$((marked + 1)) ;;
          legacy) legacy=$((legacy + 1)) ;;
          *) die "internal durable obligation contract classification failure" ;;
        esac
        ;;
      1) ignored=$((ignored + 1)); continue ;;
      *) die "request has an invalid durable obligation contract: $M_ID" ;;
    esac
    coord_obligation_open_loaded_request "$message_file" >/dev/null
    opened=$((opened + 1))
  done
  printf 'LOOM_OBLIGATION_RECONCILE requests=%s marked=%s legacy=%s ignored=%s state=PASS\n' \
    "$opened" "$marked" "$legacy" "$ignored"
}

coord_wake_reconcile_command() {
  (($# == 0)) || die "wake-reconcile does not accept arguments"
  local submission_file message_file runtime_self now_epoch retry_interval
  local attempted=0 started=0 pending=0 skipped=0 sender_agent sender_lane output
  local -a submission_paths=()
  retry_interval="${SOUNIO_COORD_WAKE_RETRY_INTERVAL_SECONDS:-1}"
  [[ "$retry_interval" =~ ^[1-9][0-9]*$ ]] || \
    die "SOUNIO_COORD_WAKE_RETRY_INTERVAL_SECONDS must be a positive integer"
  now_epoch="$(date +%s)"
  runtime_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
  submission_paths=("$WAKE_SUBMISSIONS_DIR"/*.submitted)
  for submission_file in "${submission_paths[@]}"; do
    [[ -f "$submission_file" ]] || continue
    load_wake_submission "$submission_file"
    [[ "$S_SCHEMA" == loom-wake-submission-v1 && \
      "$S_LAST_ATTEMPT_EPOCH" =~ ^[0-9]+$ ]] || \
      { skipped=$((skipped + 1)); continue; }
    case "$S_STATE" in
      prepared|submit-uncertain|submitted) ;;
      *) skipped=$((skipped + 1)); continue ;;
    esac
    ((now_epoch >= S_LAST_ATTEMPT_EPOCH + retry_interval)) || { skipped=$((skipped + 1)); continue; }
    message_file="$MESSAGES_DIR/$(slug "$S_MESSAGE_ID").message"
    [[ -f "$message_file" ]] || { skipped=$((skipped + 1)); continue; }
    load_message "$message_file"
    sender_agent="$M_FROM_AGENT"
    sender_lane="$M_FROM_LANE"
    attempted=$((attempted + 1))
    if output="$(env SOUNIO_COORD_DIR="$STATE_DIR" \
      SOUNIO_COORD_WAKE_START_TIMEOUT_MILLIS="${SOUNIO_COORD_WAKE_RETRY_WAIT_MILLIS:-250}" \
      "$runtime_self" wake --agent "$sender_agent" --lane "$sender_lane" \
      --message "$S_MESSAGE_ID" 2>&1)"; then
      started=$((started + 1))
    else
      pending=$((pending + 1))
    fi
    printf '%s\n' "$output"
  done
  printf 'WAKE_RECONCILE attempted=%s started=%s pending=%s skipped=%s\n' \
    "$attempted" "$started" "$pending" "$skipped"
}

coord_obligation_supervisor_stop_children() {
  local child
  for child in $(jobs -pr); do
    kill "$child" 2>/dev/null || true
  done
}

coord_obligation_supervisor_state() {
  local state_file="$STATE_DIR/obligation-supervisor.state" line proc_tail current_start
  COORD_OBLIGATION_SUPERVISOR_SCHEMA=''
  COORD_OBLIGATION_SUPERVISOR_PID=''
  COORD_OBLIGATION_SUPERVISOR_PID_START=''
  COORD_OBLIGATION_SUPERVISOR_REPLAYED_UTC=''
  [[ -f "$state_file" ]] || return 1
  while IFS= read -r line || [[ -n "$line" ]]; do
    case "$line" in
      schema=loom-obligation-supervisor-v1)
        COORD_OBLIGATION_SUPERVISOR_SCHEMA=loom-obligation-supervisor-v1
        ;;
      schema=*) die "invalid obligation supervisor state schema: $state_file" ;;
      pid=*) COORD_OBLIGATION_SUPERVISOR_PID="${line#pid=}" ;;
      pid_start=*) COORD_OBLIGATION_SUPERVISOR_PID_START="${line#pid_start=}" ;;
      replayed_utc=*) COORD_OBLIGATION_SUPERVISOR_REPLAYED_UTC="${line#replayed_utc=}" ;;
    esac
  done < "$state_file"
  [[ "$COORD_OBLIGATION_SUPERVISOR_SCHEMA" == loom-obligation-supervisor-v1 &&
    "$COORD_OBLIGATION_SUPERVISOR_PID" =~ ^[1-9][0-9]*$ &&
    "$COORD_OBLIGATION_SUPERVISOR_PID_START" =~ ^[1-9][0-9]*$ ]] ||
    die "invalid obligation supervisor process identity: $state_file"
  [[ -r "/proc/$COORD_OBLIGATION_SUPERVISOR_PID/stat" ]] || return 1
  kill -0 "$COORD_OBLIGATION_SUPERVISOR_PID" 2>/dev/null || return 1
  proc_tail="$(sed 's/^[^)]*) //' "/proc/$COORD_OBLIGATION_SUPERVISOR_PID/stat" 2>/dev/null || true)"
  current_start="$(awk '{print $20}' <<< "$proc_tail")"
  [[ "$current_start" == "$COORD_OBLIGATION_SUPERVISOR_PID_START" ]] || return 1
}

coord_obligation_supervisor_owned_executable() {
  local executable="$1" runtime_root local_loom
  runtime_root="${SOUNIO_COORD_RUNTIME_DIR:-$GIT_COMMON_DIR/sounio-coord-runtime}"
  runtime_root="$(readlink -f "$runtime_root" 2>/dev/null || true)"
  local_loom="$(readlink -f "$WORKTREE/tools/loom/_build/default/src/loom.exe" 2>/dev/null || true)"
  if [[ -n "$runtime_root" ]]; then
    case "$executable" in
      "$runtime_root"/versions/*/bin/sounio-loom-runtime) return 0 ;;
    esac
  fi
  [[ -n "$local_loom" && "$executable" == "$local_loom" ]]
}

coord_obligation_supervisor_owned_pids() {
  local proc pid owner ppid value index observed_state_dir script_path runtime_root local_runtime
  local expected_state_dir env_value
  local -a argv=()
  expected_state_dir="$(readlink -f "$STATE_DIR" 2>/dev/null || true)"
  [[ -n "$expected_state_dir" ]] || return 0
  runtime_root="${SOUNIO_COORD_RUNTIME_DIR:-$GIT_COMMON_DIR/sounio-coord-runtime}"
  runtime_root="$(readlink -f "$runtime_root" 2>/dev/null || true)"
  local_runtime="$(readlink -f "$WORKTREE/scripts/dev/sounio_coord_runtime.sh" 2>/dev/null || true)"
  for proc in /proc/[1-9]*; do
    [[ -d "$proc" ]] || continue
    pid="${proc##*/}"
    owner="$(stat -c %u "$proc" 2>/dev/null || true)"
    [[ "$owner" == "$(id -u)" ]] || continue
    ppid="$(sed -n 's/^PPid:[[:space:]]*//p' "$proc/status" 2>/dev/null || true)"
    [[ "$ppid" == 1 ]] || continue
    argv=()
    while IFS= read -r -d '' value; do
      argv+=("$value")
    done < "$proc/cmdline"
    [[ "${argv[2]:-}" == obligation-supervise ]] || continue
    script_path="$(readlink -f "${argv[1]:-}" 2>/dev/null || true)"
    [[ -n "$script_path" ]] || continue
    if [[ -n "$runtime_root" ]]; then
      case "$script_path" in
        "$runtime_root"/versions/*/bin/sounio-coord-runtime) ;;
        *) [[ -n "$local_runtime" && "$script_path" == "$local_runtime" ]] || continue ;;
      esac
    elif [[ -z "$local_runtime" || "$script_path" != "$local_runtime" ]]; then
      continue
    fi
    observed_state_dir=''
    for ((index = 3; index + 1 < ${#argv[@]}; index++)); do
      if [[ "${argv[$index]}" == --state-dir ]]; then
        observed_state_dir="$(readlink -f "${argv[$((index + 1))]}" 2>/dev/null || true)"
        break
      fi
    done
    if [[ -z "$observed_state_dir" && -r "$proc/environ" ]]; then
      while IFS= read -r -d '' env_value; do
        case "$env_value" in
          SOUNIO_COORD_DIR=*)
            observed_state_dir="$(readlink -f "${env_value#SOUNIO_COORD_DIR=}" 2>/dev/null || true)"
            break
            ;;
        esac
      done < "$proc/environ"
    fi
    if [[ -z "$observed_state_dir" && -n "$runtime_root" ]]; then
      case "$script_path" in
        "$runtime_root"/versions/*/bin/sounio-coord-runtime)
          observed_state_dir="$(readlink -f "$GIT_COMMON_DIR/sounio-coord-state" 2>/dev/null || true)"
          ;;
      esac
    fi
    [[ "$observed_state_dir" == "$expected_state_dir" ]] || continue
    printf '%s\n' "$pid"
  done
  return 0
}

coord_obligation_supervisor_service_command() {
  local action="$1"; shift
  local interval=2 timeout=120 lock_file="$STATE_DIR/.obligation-supervisor-bootstrap.lock"
  local leader_lock="$STATE_DIR/.obligation-supervisor-leader.lock"
  local runtime_self log_file attempt previous_pid='' previous_start=''
  local expected_loom='' actual_loom='' ensured_state=started state_live=0 pid
  local supervisor_wrapper_pid=''
  local -a owned_pids=() remaining_pids=()
  while (($#)); do
    case "$1" in
      --interval-seconds)
        [[ "$action" == ensure ]] || die "$1 is only valid for obligation-supervisor-ensure"
        require_arg "$1" "$2"; interval="$2"; shift 2
        ;;
      --timeout-seconds)
        require_arg "$1" "$2"; timeout="$2"; shift 2
        ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown obligation-supervisor-$action option: $1" ;;
    esac
  done
  [[ "$interval" =~ ^[1-9][0-9]*$ ]] && ((interval <= 60)) ||
    die "obligation supervisor interval must be between 1 and 60 seconds"
  [[ "$timeout" =~ ^[1-9][0-9]*$ ]] && ((timeout <= 300)) ||
    die "obligation supervisor timeout must be between 1 and 300 seconds"
  mkdir -p "$STATE_DIR"
  exec 6>"$lock_file"
  flock 6
  if coord_obligation_supervisor_state; then
    state_live=1
    previous_pid="$COORD_OBLIGATION_SUPERVISOR_PID"
    previous_start="$COORD_OBLIGATION_SUPERVISOR_PID_START"
  fi
  mapfile -t owned_pids < <(coord_obligation_supervisor_owned_pids)
  if ((!state_live)) && [[ "$action" == ensure ]] && ((${#owned_pids[@]} == 1)); then
    for ((attempt = 0; attempt < timeout * 10; attempt++)); do
      if coord_obligation_supervisor_state; then
        state_live=1
        previous_pid="$COORD_OBLIGATION_SUPERVISOR_PID"
        previous_start="$COORD_OBLIGATION_SUPERVISOR_PID_START"
        break
      fi
      sleep 0.1
    done
  fi
  if ((state_live)); then
    actual_loom="$(readlink -f "/proc/$previous_pid/exe" 2>/dev/null || true)"
    coord_obligation_supervisor_owned_executable "$actual_loom" ||
      die "refusing to signal unowned obligation supervisor pid=$previous_pid executable=${actual_loom:-unknown}"
    supervisor_wrapper_pid="$(sed -n 's/^PPid:[[:space:]]*//p' "/proc/$previous_pid/status" 2>/dev/null || true)"
    array_contains "$supervisor_wrapper_pid" "${owned_pids[@]}" ||
      die "refusing to signal obligation supervisor outside the selected state directory: pid=$previous_pid wrapper=${supervisor_wrapper_pid:-unknown}"
    if [[ "$action" == ensure ]]; then
      expected_loom="$(readlink -f "$(coord_loom_obligation_runtime)")"
      if [[ "$actual_loom" == "$expected_loom" && ${#owned_pids[@]} == 1 ]]; then
        printf 'LOOM_OBLIGATION_SUPERVISOR_ENSURED state=already-running pid=%s pid_start=%s replayed_utc=%s\n' \
          "$previous_pid" "$previous_start" "$COORD_OBLIGATION_SUPERVISOR_REPLAYED_UTC"
        flock -u 6
        return 0
      fi
      ensured_state=restarted
    fi
  elif ((${#owned_pids[@]})) && [[ "$action" == ensure ]]; then
    ensured_state=restarted
  fi
  if ((${#owned_pids[@]})); then
    for pid in "${owned_pids[@]}"; do
      kill -TERM "$pid" 2>/dev/null || true
    done
    for ((attempt = 0; attempt < timeout * 10; attempt++)); do
      remaining_pids=()
      for pid in "${owned_pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && remaining_pids+=("$pid")
      done
      ((${#remaining_pids[@]} == 0)) && break
      sleep 0.1
    done
    ((${#remaining_pids[@]} == 0)) ||
      die "obligation supervisors did not stop within ${timeout}s: pids=$(IFS=,; printf '%s' "${remaining_pids[*]}")"
    flock -w "$timeout" "$leader_lock" -c true ||
      die "obligation supervisor leader lock did not release within ${timeout}s: lock=$leader_lock"
    if [[ "$action" == stop ]]; then
      printf 'LOOM_OBLIGATION_SUPERVISOR_STOPPED state=stopped pid=%s pid_start=%s retired=%s\n' \
        "${previous_pid:--}" "${previous_start:--}" "${#owned_pids[@]}"
      flock -u 6
      return 0
    fi
  fi
  if [[ "$action" == stop ]]; then
    printf 'LOOM_OBLIGATION_SUPERVISOR_STOPPED state=already-stopped pid=- pid_start=-\n'
    flock -u 6
    return 0
  fi
  [[ -x /usr/bin/setsid ]] || die "obligation supervisor ensure requires /usr/bin/setsid"
  runtime_self="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")"
  log_file="$STATE_DIR/obligation-supervisor.log"
  touch "$log_file"
  chmod 0600 "$log_file"
  SOUNIO_COORD_DIR="$STATE_DIR" /usr/bin/setsid -f "$runtime_self" obligation-supervise \
    --interval-seconds "$interval" >> "$log_file" 2>&1 </dev/null 6>&-
  for ((attempt = 0; attempt < timeout * 10; attempt++)); do
    if coord_obligation_supervisor_state; then
      printf 'LOOM_OBLIGATION_SUPERVISOR_ENSURED state=%s pid=%s pid_start=%s replayed_utc=%s log=%s\n' \
        "$ensured_state" "$COORD_OBLIGATION_SUPERVISOR_PID" "$COORD_OBLIGATION_SUPERVISOR_PID_START" \
        "$COORD_OBLIGATION_SUPERVISOR_REPLAYED_UTC" "$log_file"
      flock -u 6
      return 0
    fi
    sleep 0.1
  done
  die "obligation supervisor did not become live within ${timeout}s; log=$log_file"
}

coord_obligation_supervisor_command() {
  local action="$1"; shift
  local once=0 interval=2 bridge_pid='' loom_pid='' supervisor_status
  local leader_lock="$STATE_DIR/.obligation-supervisor-leader.lock"
  local -a args=("obligation-$action")
  while (($#)); do
    case "$1" in
      --once) once=1; args+=(--once); shift ;;
      --interval-seconds)
        require_arg "$1" "$2"
        interval="$2"
        args+=(--interval-seconds "$2")
        shift 2
        ;;
      -h|--help) usage; return 0 ;;
      *) die "unknown obligation-$action option: $1" ;;
    esac
  done
  if [[ "$action" != supervise ]]; then
    coord_obligation_invoke "${args[@]}"
    return 0
  fi
  [[ "$interval" =~ ^[1-9][0-9]*$ ]] && ((interval <= 60)) ||
    die "obligation supervisor interval must be between 1 and 60 seconds"
  if ((!once)); then
    mkdir -p "$STATE_DIR"
    exec 7>"$leader_lock"
    if ! flock -n 7; then
      printf 'LOOM_OBLIGATION_SUPERVISOR_REFUSED state=duplicate-leader lock=%s\n' \
        "$leader_lock" >&2
      return 73
    fi
  fi
  coord_obligation_reconcile_command >/dev/null
  coord_wake_reconcile_command >/dev/null
  if ((once)); then
    coord_obligation_invoke "${args[@]}"
    return 0
  fi
  trap 'coord_obligation_supervisor_stop_children' EXIT
  trap 'coord_obligation_supervisor_stop_children; exit 130' INT
  trap 'coord_obligation_supervisor_stop_children; exit 143' TERM
  (
    while sleep "$interval"; do
      if ! (coord_obligation_reconcile_command >/dev/null && \
        coord_wake_reconcile_command >/dev/null); then
        kill -TERM "$$" 2>/dev/null || true
        exit 1
      fi
    done
  ) &
  bridge_pid=$!
  coord_obligation_exec "${args[@]}" &
  loom_pid=$!
  if wait "$loom_pid"; then
    supervisor_status=0
  else
    supervisor_status=$?
  fi
  kill "$loom_pid" 2>/dev/null || true
  wait "$loom_pid" 2>/dev/null || true
  kill "$bridge_pid" 2>/dev/null || true
  wait "$bridge_pid" 2>/dev/null || true
  trap - EXIT INT TERM
  return "$supervisor_status"
}

send_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' to_agent='' to_lane=''
  local kind='info' message='' ttl="${SOUNIO_COORD_MESSAGE_TTL_SECONDS:-604800}"
  local thread_id='' reply_to='' message_id message_file tmp_file reply_file
  local durable_obligation=0 explicit_obligation_opt_out=0

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
  if [[ "$kind" == request && -n "$to_agent" && -n "$to_lane" ]]; then
    if coord_durable_obligations_enabled; then
      durable_obligation=1
    else
      explicit_obligation_opt_out=1
    fi
  fi
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
    ((durable_obligation == 0)) || \
      printf 'obligation_schema=loom-durable-obligation-v1\n'
    ((explicit_obligation_opt_out == 0)) || printf 'obligation_opt_out=1\n'
  } > "$tmp_file"
  mv "$tmp_file" "$message_file"
  if ((durable_obligation)); then
    load_message "$message_file"
    coord_obligation_open_loaded_request "$message_file"
  fi
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
  local experiment_prereg='' experiment_outcome='' experiment_id=''
  local experiment_prereg_sha256='' experiment_outcome_sha256='' causal_output=''
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
      --experiment-prereg) require_arg "$1" "$2"; experiment_prereg="$(normalize_path "$2")"; shift 2 ;;
      --experiment-outcome) require_arg "$1" "$2"; experiment_outcome="$(normalize_path "$2")"; shift 2 ;;
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
  validate_value experiment-prereg "$experiment_prereg"
  validate_value experiment-outcome "$experiment_outcome"
  if [[ -n "$experiment_prereg" || -n "$experiment_outcome" ]]; then
    [[ -n "$experiment_prereg" && -n "$experiment_outcome" ]] || \
      die "causal handoff requires both --experiment-prereg and --experiment-outcome"
  fi
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

  if [[ -n "$experiment_prereg" ]]; then
    if ! causal_output="$(causal_runtime_command verify \
      --prereg "$experiment_prereg" --outcome "$experiment_outcome" \
      --agent "$agent" --lane "$lane" --head "$commit_sha" --require-supported)"; then
      die "causal experiment chain refused the handoff"
    fi
    experiment_id="$(sed -n 's/.*experiment_id=\([^ ]*\).*/\1/p' <<< "$causal_output")"
    experiment_prereg_sha256="$(sed -n 's/.*prereg_sha256=\([^ ]*\).*/\1/p' <<< "$causal_output")"
    experiment_outcome_sha256="$(sed -n 's/.*outcome_sha256=\([^ ]*\).*/\1/p' <<< "$causal_output")"
    [[ -n "$experiment_id" && "$experiment_prereg_sha256" =~ ^[0-9a-f]{64}$ && \
      "$experiment_outcome_sha256" =~ ^[0-9a-f]{64}$ ]] || \
      die "causal verifier returned an invalid receipt"
  fi

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
    if [[ -n "$experiment_id" ]]; then
      printf 'experiment_id=%s\n' "$experiment_id"
      printf 'experiment_prereg=%s\n' "$experiment_prereg"
      printf 'experiment_outcome=%s\n' "$experiment_outcome"
      printf 'experiment_prereg_sha256=%s\n' "$experiment_prereg_sha256"
      printf 'experiment_outcome_sha256=%s\n' "$experiment_outcome_sha256"
    fi
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
  remove_presence_for_lane "$agent" "$lane" "$WORKTREE" handoff
  printf 'HANDED_OFF claim_id=%s message_id=%s commit=%s to_agent=%s to_lane=%s gates=%s evidence=%s experiment=%s\n' \
    "$C_ID" "$message_id" "$commit_sha" "$to_agent" "$to_lane" \
    "$(join_values "${gates[@]}")" "$(join_values "${evidence_paths[@]}")" \
    "${experiment_id:--}"
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

  coord_obligation_reconcile_command >/dev/null

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

promote_wake_submissions_for_injection() {
  local message_id="$1" agent="$2" lane="$3" submission_file receipt_file tmp_file
  local promoted=0
  local -a submission_paths=()
  submission_paths=("$WAKE_SUBMISSIONS_DIR/$(slug "$message_id")"--*.submitted)
  for submission_file in "${submission_paths[@]}"; do
    [[ -f "$submission_file" ]] || continue
    load_wake_submission "$submission_file"
    [[ "$S_SCHEMA" == loom-wake-submission-v1 && \
      "$S_MESSAGE_ID" == "$message_id" && "$S_AGENT" == "$agent" && \
      "$S_LANE" == "$lane" && "$S_INSERTION_STATE" == confirmed && \
      -n "$S_INSERTED_UTC" ]] || continue
    case "$S_STATE" in
      submit-uncertain) ;;
      submitted) [[ -n "$S_SUBMITTED_UTC" ]] || continue ;;
      *) continue ;;
    esac
    if ! hook_capability_is_current "$agent" "$lane" "$S_GENERATION"; then
      printf 'utc=%s event=WAKE_START_REFUSED message_id=%s endpoint_id=%s agent=%s lane=%s generation=%s reason=hook-capability-%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$message_id" "$S_ENDPOINT_ID" \
        "$agent" "$lane" "$S_GENERATION" "$HOOK_CAPABILITY_REASON" >> "$EVENT_LOG"
      continue
    fi
    if ! wake_submission_generation_is_current; then
      printf 'utc=%s event=WAKE_START_REFUSED message_id=%s endpoint_id=%s agent=%s lane=%s generation=%s reason=generation-drift\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$message_id" "$S_ENDPOINT_ID" \
        "$agent" "$lane" "$S_GENERATION" >> "$EVENT_LOG"
      continue
    fi
    if [[ "$S_STATE" == submit-uncertain ]]; then
      S_STATE='submitted'
      S_SUBMITTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      write_wake_submission "$submission_file"
    fi
    receipt_file="$(wake_receipt_path "$message_id" "$S_ENDPOINT_ID" "$S_GENERATION")"
    if [[ ! -f "$receipt_file" ]]; then
      tmp_file="$(mktemp "$WAKES_DIR/.wake-start-write.XXXXXX")"
      printf 'utc=%s message_id=%s endpoint_id=%s transport=%s address=%s generation=%s discovery=%s state=started insertion_state=%s inserted_utc=%s submitted_utc=%s attempts=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$message_id" "$S_ENDPOINT_ID" \
        "$S_TRANSPORT" "$S_ADDRESS" "$S_GENERATION" "$S_DISCOVERY" \
        "$S_INSERTION_STATE" "${S_INSERTED_UTC:--}" "$S_SUBMITTED_UTC" \
        "$S_ATTEMPTS" > "$tmp_file"
      mv "$tmp_file" "$receipt_file"
      printf 'utc=%s event=WAKE_STARTED message_id=%s endpoint_id=%s agent=%s lane=%s transport=%s address=%s generation=%s discovery=%s attempts=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$message_id" "$S_ENDPOINT_ID" \
        "$agent" "$lane" "$S_TRANSPORT" "$S_ADDRESS" "$S_GENERATION" \
        "$S_DISCOVERY" "$S_ATTEMPTS" >> "$EVENT_LOG"
    fi
    unlink "$submission_file"
    promoted=$((promoted + 1))
    printf 'WAKE_STARTED message_id=%s endpoint_id=%s transport=%s address=%s generation=%s discovery=%s\n' \
      "$message_id" "$S_ENDPOINT_ID" "$S_TRANSPORT" "$S_ADDRESS" \
      "$S_GENERATION" "$S_DISCOVERY"
  done
  if ((promoted)); then
    submission_paths=("$WAKE_SUBMISSIONS_DIR/$(slug "$message_id")"--*.submitted)
    for submission_file in "${submission_paths[@]}"; do
      [[ -f "$submission_file" ]] || continue
      load_wake_submission "$submission_file"
      if [[ "$S_SCHEMA" == loom-wake-submission-v1 && \
        "$S_MESSAGE_ID" == "$message_id" && "$S_AGENT" == "$agent" && \
        "$S_LANE" == "$lane" ]]; then
        unlink "$submission_file"
      fi
    done
  fi
  return 0
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
    promote_wake_submissions_for_injection "$message_id" "$agent" "$lane"
  done
}

message_status_command() {
  local agent="${SOUNIO_AGENT_ID:-}" lane='' message_id='' message_file receipt_file
  local original_from_agent original_from_lane original_to_agent original_to_lane
  local original_kind original_thread original_epoch request_state latest_response='-'
  local latest_kind='' latest_epoch=0 latest_file='' responses=0 injected=0 acknowledged=0 wakes=0 wake_pending=0
  local receipt_utc receipt_agent receipt_lane token_utc token_agent token_lane
  local token token_message token_endpoint token_transport token_address token_generation token_state
  local receipt_generation
  local -a message_paths=() injection_paths=() ack_paths=() wake_paths=() submission_paths=() receipt_tokens=()
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
    [[ "$M_REPLY_TO" == "$message_id" || "$M_KIND" =~ ^(reply|blocker|handoff)$ ]] || continue
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
  submission_paths=("$WAKE_SUBMISSIONS_DIR/$(slug "$message_id")"--*.submitted)
  injected="${#injection_paths[@]}"
  acknowledged="${#ack_paths[@]}"
  wakes="${#wake_paths[@]}"
  wake_pending="${#submission_paths[@]}"
  printf 'MESSAGE_STATUS id=%s kind=%s thread=%s request_state=%s injected=%s acknowledged=%s responses=%s latest_response=%s wakes=%s wake_pending=%s\n' \
    "$message_id" "$original_kind" "$original_thread" "$request_state" "$injected" \
    "$acknowledged" "$responses" "$latest_response" "$wakes" "$wake_pending"
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
    token_utc=''
    token_message=''
    token_endpoint=''
    token_transport=''
    token_address=''
    token_generation=''
    token_state=''
    receipt_tokens=()
    read -r -a receipt_tokens < "$receipt_file" || true
    for token in "${receipt_tokens[@]}"; do
      case "$token" in
        utc=*) token_utc="$token" ;;
        message_id=*) token_message="$token" ;;
        endpoint_id=*) token_endpoint="$token" ;;
        transport=*) token_transport="$token" ;;
        address=*) token_address="$token" ;;
        generation=*) token_generation="$token" ;;
        state=*) token_state="$token" ;;
      esac
    done
    receipt_generation="${token_generation#generation=}"
    printf 'WAKE_RECEIPT message_id=%s utc=%s endpoint_id=%s transport=%s address=%s' \
      "$message_id" "${token_utc#utc=}" "${token_endpoint#endpoint_id=}" \
      "${token_transport#transport=}" "${token_address#address=}"
    [[ -z "$token_state" ]] || printf ' state=%s' "${token_state#state=}"
    printf ' generation=%s\n' "${receipt_generation:--}"
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
      [[ "$M_REPLY_TO" == "$message_id" || "$M_KIND" =~ ^(reply|blocker|handoff)$ ]] || continue
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
  local agent="${SOUNIO_AGENT_ID:-}" lane='' message_id='' message_file ack_file submission_file
  local cancelled=0
  local -a submission_paths=()
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
  submission_paths=("$WAKE_SUBMISSIONS_DIR/$(slug "$M_ID")"--*.submitted)
  for submission_file in "${submission_paths[@]}"; do
    [[ -f "$submission_file" ]] || continue
    load_wake_submission "$submission_file"
    unlink "$submission_file"
    cancelled=$((cancelled + 1))
    printf 'utc=%s event=WAKE_CANCELLED message_id=%s endpoint_id=%s agent=%s lane=%s generation=%s reason=acknowledged\n' \
      "$NOW_UTC" "$M_ID" "$S_ENDPOINT_ID" "$agent" "$lane" \
      "$S_GENERATION" >> "$EVENT_LOG"
  done
  printf 'ACKED message_id=%s agent=%s lane=%s wake_cancelled=%s\n' \
    "$M_ID" "$agent" "$lane" "$cancelled"
}

prune_command() {
  local removed=0 messages_removed=0 endpoints_removed=0 presences_removed=0
  local recovery_retention="${SOUNIO_COORD_RECOVERY_RETENTION_SECONDS:-604800}"
  local claim_file message_file ack_file injection_file endpoint_file presence_file wake_file submission_file
  local -a message_paths=() ack_paths=() injection_paths=() endpoint_paths=() presence_paths=() wake_paths=() submission_paths=()
  [[ "$recovery_retention" =~ ^[1-9][0-9]*$ ]] || \
    die "SOUNIO_COORD_RECOVERY_RETENTION_SECONDS must be a positive integer"
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
      submission_paths=("$WAKE_SUBMISSIONS_DIR/$(slug "$M_ID")"--*.submitted)
      for submission_file in "${submission_paths[@]}"; do
        [[ -f "$submission_file" ]] && unlink "$submission_file"
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
  presence_paths=("$PRESENCES_DIR"/*.presence)
  for presence_file in "${presence_paths[@]}"; do
    [[ -f "$presence_file" ]] || continue
    load_presence "$presence_file"
    presence_state || true
    if [[ "$PRESENCE_STATE" == orphaned ]] && \
      ((NOW_EPOCH > P_LAST_EPOCH + recovery_retention)); then
      unlink "$presence_file"
      presences_removed=$((presences_removed + 1))
      printf 'PRUNED_PRESENCE presence_id=%s agent=%s lane=%s\n' "$P_ID" "$P_AGENT" "$P_LANE"
    fi
  done
  printf 'pruned=%s pruned_messages=%s\n' "$removed" "$messages_removed"
  printf 'pruned_endpoints=%s\n' "$endpoints_removed"
  printf 'pruned_presences=%s\n' "$presences_removed"
}

status_command() {
  local claim_file endpoint_file presence_file existing other_file wt branch dirty head marker context_branch file resource local_conflict
  local active=0 stale=0 conflicts=0 total_wt=0 dirty_wt=0 claimed_wt=0 shown_wt=0 relevant_wt=0 max_rows
  local active_endpoints=0 stale_endpoints=0 drifted_endpoints=0 unavailable_endpoints=0 endpoint_marker
  local live_presences=0 unresponsive_presences=0 orphaned_presences=0 presence_marker
  local brief=0 inspect_all=0 worktree_scan='current-and-claimed'
  local -a worktrees=() inspect_worktrees=() first_files=() second_files=() endpoint_paths=() presence_paths=()
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

  printf '\n== Process presence ==\n'
  presence_paths=("$PRESENCES_DIR"/*.presence)
  for presence_file in "${presence_paths[@]}"; do
    [[ -f "$presence_file" ]] || continue
    load_presence "$presence_file"
    presence_state || true
    case "$PRESENCE_STATE" in
      live) live_presences=$((live_presences + 1)) ;;
      unresponsive) unresponsive_presences=$((unresponsive_presences + 1)) ;;
      orphaned) orphaned_presences=$((orphaned_presences + 1)) ;;
    esac
    presence_marker="${PRESENCE_STATE^^}_PRESENCE"
    printf '%s presence_id=%s agent=%s lane=%s generation=%s harness=%s session_id=%s host=%s pid=%s reason=%s last_seen=%s worktree=%s\n' \
      "$presence_marker" "$P_ID" "$P_AGENT" "$P_LANE" "$P_GENERATION" \
      "$P_HARNESS" "$P_SESSION_ID" "$P_HOST" "$P_PID" "$PRESENCE_REASON" \
      "$P_LAST_UTC" "$P_WORKTREE"
  done
  ((live_presences + unresponsive_presences + orphaned_presences > 0)) || printf 'NONE\n'

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
  printf 'presence_summary=live:%s unresponsive:%s orphaned:%s\n' \
    "$live_presences" "$unresponsive_presences" "$orphaned_presences"
  STATUS_CONFLICTS="$conflicts"
}

cockpit_snapshot_command() {
  local claim_file endpoint_file presence_file state
  local active_claims=0 stale_claims=0
  local active_endpoints=0 stale_endpoints=0 drifted_endpoints=0 unavailable_endpoints=0
  local live_presences=0 unresponsive_presences=0 orphaned_presences=0
  local -a endpoint_paths=() presence_paths=()

  (($# == 0)) || die "cockpit-snapshot does not accept arguments"
  printf 'COCKPIT\tprotocol=1\tsnapshot_utc=%s\n' "$NOW_UTC"

  refresh_claim_paths
  for claim_file in "${claim_paths[@]}"; do
    [[ -f "$claim_file" ]] || continue
    load_claim "$claim_file"
    [[ -n "$C_ID" ]] || continue
    if claim_expired; then
      state=stale
      stale_claims=$((stale_claims + 1))
    else
      state=active
      active_claims=$((active_claims + 1))
    fi
    printf 'CLAIM\tstate=%s\tagent=%s\tlane=%s\tlast_seen=%s\tworktree=%s\tbranch=%s\tsha=%s\n' \
      "$state" "$C_AGENT" "$C_LANE" "$C_LAST_UTC" "$C_WORKTREE" "$C_BRANCH" "$C_SHA"
  done

  endpoint_paths=("$ENDPOINTS_DIR"/*.endpoint)
  for endpoint_file in "${endpoint_paths[@]}"; do
    [[ -f "$endpoint_file" ]] || continue
    load_endpoint "$endpoint_file"
    [[ -n "$E_ID" ]] || continue
    endpoint_state || true
    case "$ENDPOINT_STATE" in
      active) active_endpoints=$((active_endpoints + 1)) ;;
      stale) stale_endpoints=$((stale_endpoints + 1)) ;;
      drifted) drifted_endpoints=$((drifted_endpoints + 1)) ;;
      *) unavailable_endpoints=$((unavailable_endpoints + 1)) ;;
    esac
    printf 'ENDPOINT\tstate=%s\tagent=%s\tlane=%s\tharness=%s\ttransport=%s\tinstance_id=%s\tlast_seen=%s\tworktree=%s\n' \
      "$ENDPOINT_STATE" "$E_AGENT" "$E_LANE" "$E_HARNESS" "$E_TRANSPORT" \
      "$E_INSTANCE_ID" "$E_LAST_UTC" "$E_WORKTREE"
  done

  presence_paths=("$PRESENCES_DIR"/*.presence)
  for presence_file in "${presence_paths[@]}"; do
    [[ -f "$presence_file" ]] || continue
    load_presence "$presence_file"
    [[ -n "$P_ID" ]] || continue
    presence_state || true
    case "$PRESENCE_STATE" in
      live) live_presences=$((live_presences + 1)) ;;
      unresponsive) unresponsive_presences=$((unresponsive_presences + 1)) ;;
      orphaned) orphaned_presences=$((orphaned_presences + 1)) ;;
    esac
    printf 'PRESENCE\tstate=%s\treason=%s\tagent=%s\tlane=%s\tharness=%s\tsession_id=%s\tgeneration=%s\tpid=%s\tlast_seen=%s\tworktree=%s\n' \
      "$PRESENCE_STATE" "$PRESENCE_REASON" "$P_AGENT" "$P_LANE" "$P_HARNESS" \
      "$P_SESSION_ID" "$P_GENERATION" "$P_PID" "$P_LAST_UTC" "$P_WORKTREE"
  done

  printf 'SUMMARY\tactive_claims=%s\tstale_claims=%s\tactive_endpoints=%s\tstale_endpoints=%s\tdrifted_endpoints=%s\tunavailable_endpoints=%s\tlive_presences=%s\tunresponsive_presences=%s\torphaned_presences=%s\n' \
    "$active_claims" "$stale_claims" "$active_endpoints" "$stale_endpoints" \
    "$drifted_endpoints" "$unavailable_endpoints" "$live_presences" \
    "$unresponsive_presences" "$orphaned_presences"
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
  cockpit-snapshot) cockpit_snapshot_command "$@" ;;
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
  presence-register) presence_register_command "$@" ;;
  presence-unregister) presence_unregister_command "$@" ;;
  hook-capability-register) hook_capability_register_command "$@" ;;
  hook-capability-unregister) hook_capability_unregister_command "$@" ;;
  hook-capability-status) hook_capability_status_command "$@" ;;
  hook-caller-attest) hook_caller_attest_command "$@" ;;
  recover) recover_command "$@" ;;
  obligation-open) coord_obligation_open_command "$@" ;;
  obligation-consume) coord_obligation_recipient_command consume "$@" ;;
  obligation-claim) coord_obligation_recipient_command claim "$@" ;;
  obligation-renew) coord_obligation_recipient_command renew "$@" ;;
  obligation-interrupt) coord_obligation_recipient_command interrupt "$@" ;;
  obligation-recover) coord_obligation_recipient_command recover "$@" ;;
  obligation-complete) coord_obligation_recipient_command complete "$@" ;;
  obligation-status) coord_obligation_recipient_command status "$@" ;;
  obligation-list) coord_obligation_list_command "$@" ;;
  obligation-tui) coord_obligation_projection_command tui "$@" ;;
  obligation-serve) coord_obligation_projection_command serve "$@" ;;
  obligation-reconcile) coord_obligation_reconcile_command "$@" ;;
  obligation-supervise) coord_obligation_supervisor_command supervise "$@" ;;
  obligation-supervisor-status) coord_obligation_supervisor_command supervisor-status "$@" ;;
  obligation-supervisor-ensure) coord_obligation_supervisor_service_command ensure "$@" ;;
  obligation-supervisor-stop) coord_obligation_supervisor_service_command stop "$@" ;;
  wake) wake_command "$@" ;;
  wake-reconcile) coord_wake_reconcile_command "$@" ;;
  experiment-open) causal_runtime_command open "$@" ;;
  experiment-close) causal_runtime_command close "$@" ;;
  experiment-status) causal_runtime_command status "$@" ;;
  handoff) handoff_command "$@" ;;
  send) send_command "$@" ;;
  reply) send_command "$@" --kind reply ;;
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
  *) die "unknown command: $command (try runtime-version, brief, status, check, claim, scope, heartbeat, release, authorize, endpoint-register, endpoint-unregister, endpoint-status, presence-register, presence-unregister, hook-capability-register, hook-capability-unregister, hook-capability-status, hook-caller-attest, recover, obligation-open, obligation-consume, obligation-claim, obligation-renew, obligation-interrupt, obligation-recover, obligation-complete, obligation-status, obligation-list, obligation-reconcile, obligation-supervise, obligation-supervisor-ensure, obligation-supervisor-stop, wake, wake-reconcile, experiment-open, experiment-close, experiment-status, handoff, send, reply, inbox, injected, ack, message-status, wait, or prune)" ;;
esac
