#!/usr/bin/env bash

set -euo pipefail
umask 077

fail() {
  printf 'sounio-loom-host-durable-lane-supervisor-host-selftest: FAIL reason=%s same_physical_reattach=false transport_pod_deleted=false\n' "$*" >&2
  exit 1
}

unavailable() {
  printf 'sounio-loom-host-durable-lane-supervisor-host-selftest: HOST_GATE_UNAVAILABLE reason=%s same_physical_reattach=false transport_pod_deleted=false\n' "$*" >&2
  exit 77
}

usage() {
  printf 'usage: %s --phase prepare|measure|cleanup --root ABSOLUTE_PATH --run-id ID [--archive ABSOLUTE_PATH --archive-sha256 HEX --authority-runtime ABSOLUTE_PATH --authority-runtime-sha256 HEX] [--transport-a-uid UID --transport-b-uid UID]\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

record_value() {
  local path="$1" key="$2" line name value found=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$found" ]] || fail "duplicate field $key in $path"
      found="$value"
    fi
  done < "$path"
  [[ -n "$found" ]] || fail "missing field $key in $path"
  printf '%s\n' "$found"
}

status_value() {
  local text="$1" key="$2" value
  value="$(sed -n "s/^${key}=//p" <<< "$text")"
  [[ -n "$value" && "$value" != *$'\n'* ]] || fail "status field $key is absent or duplicated"
  printf '%s\n' "$value"
}

wait_machine_status() {
  local expected="$1" output='' attempt
  for attempt in $(seq 1 200); do
    output="$($LOOM status --machine --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
      --agent "$AGENT" --lane "$LANE" 2>/dev/null || true)"
    if [[ "$(sed -n 's/^state=//p' <<< "$output")" == "$expected" ]]; then
      printf '%s\n' "$output"
      return 0
    fi
    sleep 0.05
  done
  fail "lane did not reach state=$expected last=$(tr '\n' ',' <<< "$output")"
}

wait_guardian_bridge_zero() {
  local output='' attempt
  for attempt in $(seq 1 200); do
    output="$($LOOM guardian-status --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
      --agent "$AGENT" --lane "$LANE" 2>/dev/null || true)"
    [[ "$output" == *' bridge_clients=0 '* ]] && {
      printf '%s\n' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "Guardian did not release the dead kernel bridge last=$output"
}

wait_snapshot() {
  local marker="$1" cursor="$2" output='' attempt
  for attempt in $(seq 1 200); do
    output="$($LOOM snapshot --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
      --agent "$AGENT" --lane "$LANE" --cursor "$cursor" 2>/dev/null || true)"
    [[ "$output" == *"$marker"* ]] && {
      printf '%s\n' "$output"
      return 0
    }
    sleep 0.05
  done
  fail "durable output omitted marker=$marker"
}

verify_process_identity() {
  local pid="$1" start_tick="$2" unit="$3" actual_start cgroup
  [[ "$pid" =~ ^[1-9][0-9]*$ && "$start_tick" =~ ^[1-9][0-9]*$ ]] ||
    fail 'process identity is non-canonical'
  kill -0 "$pid" 2>/dev/null || fail "process is not live pid=$pid"
  actual_start="$(sed 's/.*) //' "/proc/$pid/stat" | cut -d ' ' -f 20)"
  [[ "$actual_start" == "$start_tick" ]] || fail "process start tick drifted pid=$pid"
  cgroup="$(tr '\n' ',' < "/proc/$pid/cgroup")"
  [[ "$cgroup" == *"/$unit"* ]] || fail "process is not host-unit-owned pid=$pid cgroup=$cgroup expected=$unit"
}

prefix_sha256() {
  local path="$1" bytes="$2"
  [[ "$bytes" =~ ^[1-9][0-9]*$ ]] || fail "prefix byte count is invalid: $bytes"
  head -c "$bytes" "$path" | sha256sum | cut -d ' ' -f 1
}

stop_units() {
  systemctl stop "$RECOVER_UNIT" "$START_UNIT" >/dev/null 2>&1 || true
  systemctl reset-failed "$RECOVER_UNIT" "$START_UNIT" >/dev/null 2>&1 || true
}

PHASE=''
ROOT=''
RUN_ID=''
ARCHIVE=''
ARCHIVE_SHA256=''
AUTHORITY_RUNTIME=''
AUTHORITY_RUNTIME_SHA256=''
TRANSPORT_A_UID=''
TRANSPORT_B_UID=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:-}"; shift 2 ;;
    --root) ROOT="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --archive) ARCHIVE="${2:-}"; shift 2 ;;
    --archive-sha256) ARCHIVE_SHA256="${2:-}"; shift 2 ;;
    --authority-runtime) AUTHORITY_RUNTIME="${2:-}"; shift 2 ;;
    --authority-runtime-sha256) AUTHORITY_RUNTIME_SHA256="${2:-}"; shift 2 ;;
    --transport-a-uid) TRANSPORT_A_UID="${2:-}"; shift 2 ;;
    --transport-b-uid) TRANSPORT_B_UID="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$PHASE" == prepare || "$PHASE" == measure || "$PHASE" == cleanup ]] || usage
[[ "$ROOT" =~ ^/var/tmp/sounio-loom-durable-[a-z0-9-]+$ && \
   "$RUN_ID" =~ ^[a-z0-9-]{8,48}$ ]] || usage
[[ "$(id -u):$(id -g)" == 0:0 ]] || unavailable 'root identity is absent'
[[ "$(tr -d '\n' < /proc/1/comm 2>/dev/null)" == systemd ]] ||
  unavailable 'PID 1 is not systemd'
for tool in systemctl systemd-run sha256sum stat tar find sed cut head grep \
  timeout readlink tee tr seq mkdir mv rm chmod cat; do
  command -v "$tool" >/dev/null 2>&1 || unavailable "required host tool is absent: $tool"
done

START_UNIT="sounio-loom-durable-${RUN_ID}-a.service"
RECOVER_UNIT="sounio-loom-durable-${RUN_ID}-b.service"
RELEASE="$ROOT/release"
STATE_DIR="$ROOT/state"
WORK_DIR="$ROOT/work"
PHASE_A="$ROOT/phase-a.v1"
HOST_RECEIPT="$ROOT/host-receipt.v1"
AGENT="durable-${RUN_ID}"
LANE='same-physical'
SESSION_ID="durable-${RUN_ID}"
LOOM="$RELEASE/bin/sounio-loom-runtime"
POLICY_ROOT="$RELEASE/authority-root"

if [[ "$PHASE" == cleanup ]]; then
  stop_units
  if [[ -d "$ROOT" && ! -L "$ROOT" ]]; then
    chmod -R u+rwX "$ROOT" 2>/dev/null || true
    rm -rf "$ROOT"
  fi
  printf 'sounio-loom-host-durable-lane-supervisor-host-selftest: CLEANUP_PASS run_id=%s\n' "$RUN_ID"
  exit 0
fi

[[ -d "$ROOT" && ! -L "$ROOT" && "$(stat -c '%u:%g' "$ROOT")" == 0:0 ]] ||
  fail 'host experiment root is absent, linked, or not root-owned'

if [[ "$PHASE" == prepare ]]; then
  [[ "$ARCHIVE" == "$ROOT"/* && -f "$ARCHIVE" && ! -L "$ARCHIVE" && \
     "$AUTHORITY_RUNTIME" == "$ROOT"/* && -f "$AUTHORITY_RUNTIME" && \
     ! -L "$AUTHORITY_RUNTIME" && "$ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ && \
     "$AUTHORITY_RUNTIME_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
  [[ "$(sha256_file "$ARCHIVE")" == "$ARCHIVE_SHA256" ]] || fail 'capsule transport hash drifted'
  [[ "$(sha256_file "$AUTHORITY_RUNTIME")" == "$AUTHORITY_RUNTIME_SHA256" ]] ||
    fail 'action-9032 runtime transport hash drifted'
  [[ ! -e "$RELEASE" && ! -e "$STATE_DIR" ]] || fail 'experiment state already exists'
  while IFS= read -r member; do
    [[ "$member" =~ ^[A-Za-z0-9._/-]+$ && \
       ( "$member" == capsule-v1 || "$member" == capsule-v1/* ) && \
       "/$member/" != *'/../'* && "/$member/" != *'/./'* ]] ||
      fail "unsafe capsule member: $member"
  done < <(tar -tf "$ARCHIVE")
  while IFS= read -r verbose; do
    [[ "${verbose:0:1}" == d || "${verbose:0:1}" == - ]] ||
      fail 'capsule contains a non-file archive entry'
  done < <(tar -tvf "$ARCHIVE")
  stage="$ROOT/extract"
  mkdir -m 0700 "$stage" "$STATE_DIR" "$WORK_DIR"
  tar --no-same-owner --same-permissions -xf "$ARCHIVE" -C "$stage"
  [[ -d "$stage/capsule-v1/release" && \
     -z "$(find "$stage/capsule-v1" -type l -print -quit)" ]] ||
    fail 'capsule release topology is incomplete or linked'
  mv "$stage/capsule-v1/release" "$RELEASE"
  chmod -R u+rwX "$stage"
  rm -rf "$stage"
  [[ -x "$LOOM" && -d "$POLICY_ROOT/.git" && \
     "$(sha256_file "$AUTHORITY_RUNTIME")" == "$AUTHORITY_RUNTIME_SHA256" ]] ||
    fail 'extracted runtime topology is incomplete'

  systemd-run --quiet --unit="$START_UNIT" \
    --property=Type=oneshot --property=RemainAfterExit=yes \
    --property=KillMode=control-group --property=TimeoutStartSec=120 \
    --setenv=SOUNIO_LOOM_HOOK_TEST_MODE=1 \
    --setenv="SOUNIO_LOOM_PRODUCT_ACTIVATION_ROOT=$POLICY_ROOT" \
    --setenv=SOUNIO_LOOM_DURABLE_LANE_CANARY=1 \
    "$LOOM" start --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
      --session-id "$SESSION_ID" --cwd "$WORK_DIR" -- \
      "$LOOM" _durable-lane-canary
  status_a="$(wait_machine_status active)"
  guardian_a="$(status_value "$status_a" guardian_pid)"
  guardian_start_a="$(status_value "$status_a" guardian_pid_start)"
  harness_a="$(status_value "$status_a" harness_pid)"
  harness_start_a="$(status_value "$status_a" harness_pid_start)"
  daemon_a="$(status_value "$status_a" daemon_pid)"
  daemon_start_a="$(status_value "$status_a" daemon_pid_start)"
  instance_a="$(status_value "$status_a" instance_id)"
  verify_process_identity "$guardian_a" "$guardian_start_a" "$START_UNIT"
  verify_process_identity "$harness_a" "$harness_start_a" "$START_UNIT"
  verify_process_identity "$daemon_a" "$daemon_start_a" "$START_UNIT"
  cursor_before="$(status_value "$status_a" output_cursor)"
  "$LOOM" wake --state-dir "$STATE_DIR" --cwd "$WORK_DIR" --agent "$AGENT" \
    --lane "$LANE" --session-id "$SESSION_ID" --message-id pod-a-before-delete \
    --prompt "POD_A_BEFORE_DELETE:$RUN_ID" >/dev/null
  wait_snapshot 'LOOM_DURABLE_LANE_CHILD ACK sequence=1 ' "$cursor_before" >/dev/null
  status_after_wake="$(wait_machine_status active)"
  output_cursor_a="$(status_value "$status_after_wake" output_cursor)"
  descriptor="$STATE_DIR/sessions/${AGENT}--${LANE}/session.state"
  journal_file="$(record_value "$descriptor" journal_file)"
  guardian_journal_file="$(record_value "$descriptor" guardian_journal_file)"
  output_file="$(record_value "$descriptor" output_file)"
  "$LOOM" verify-journal --journal "$journal_file" | grep -q 'phase=active' ||
    fail 'semantic journal did not verify before Pod deletion'
  "$LOOM" verify-guardian-journal --journal "$guardian_journal_file" | grep -q 'phase=active' ||
    fail 'Guardian journal did not verify before Pod deletion'
  cat > "$PHASE_A" <<EOF
schema=loom-host-durable-lane-phase-a-v1
run_id=$RUN_ID
state_root=$(readlink -f "$STATE_DIR")
boot_id=$(tr -d '\n' < /proc/sys/kernel/random/boot_id)
guardian_pid=$guardian_a
guardian_start_tick=$guardian_start_a
harness_pid=$harness_a
harness_start_tick=$harness_start_a
daemon_pid=$daemon_a
daemon_start_tick=$daemon_start_a
instance_id=$instance_a
command=$(status_value "$status_after_wake" command)
argv_digest=$(status_value "$status_after_wake" argv_digest)
output_file=$output_file
output_prefix_bytes=$output_cursor_a
output_prefix_sha256=$(prefix_sha256 "$output_file" "$output_cursor_a")
journal_file=$journal_file
journal_prefix_bytes=$(stat -c %s "$journal_file")
journal_prefix_sha256=$(sha256_file "$journal_file")
guardian_journal_file=$guardian_journal_file
guardian_journal_prefix_bytes=$(stat -c %s "$guardian_journal_file")
guardian_journal_prefix_sha256=$(sha256_file "$guardian_journal_file")
archive_sha256=$ARCHIVE_SHA256
authority_runtime_sha256=$AUTHORITY_RUNTIME_SHA256
start_unit=$START_UNIT
EOF
  chmod 0400 "$PHASE_A"
  printf 'sounio-loom-host-durable-lane-supervisor-host-selftest: PHASE_A_PASS run_id=%s guardian_pid=%s guardian_start_tick=%s harness_pid=%s harness_start_tick=%s daemon_pid=%s instance_id=%s start_unit=%s custody=host-systemd semantic_authority=Sounio action=9032 transport_pod_deleted=false same_physical_reattach=pending python_executed=false rust_executed=false\n' \
    "$RUN_ID" "$guardian_a" "$guardian_start_a" "$harness_a" \
    "$harness_start_a" "$daemon_a" "$instance_a" "$START_UNIT"
  exit 0
fi

[[ "$TRANSPORT_A_UID" =~ ^[0-9a-f-]{36}$ && \
   "$TRANSPORT_B_UID" =~ ^[0-9a-f-]{36}$ && \
   "$TRANSPORT_A_UID" != "$TRANSPORT_B_UID" ]] ||
  fail 'transport Pod identities are absent, malformed, or equal'
[[ -f "$PHASE_A" && ! -L "$PHASE_A" && "$(stat -c '%a' "$PHASE_A")" == 400 ]] ||
  fail 'phase-A receipt is absent, linked, or mutable'
[[ "$(record_value "$PHASE_A" run_id)" == "$RUN_ID" ]] || fail 'phase-A run identity drifted'
AUTHORITY_RUNTIME="$ROOT/input/action-9032"
AUTHORITY_RUNTIME_SHA256="$(record_value "$PHASE_A" authority_runtime_sha256)"
[[ -x "$LOOM" && -x "$AUTHORITY_RUNTIME" && \
   "$(sha256_file "$AUTHORITY_RUNTIME")" == "$AUTHORITY_RUNTIME_SHA256" ]] ||
  fail 'measured runtime or Sounio authority is absent or drifted'

status_b="$(wait_machine_status active)"
guardian_b="$(status_value "$status_b" guardian_pid)"
guardian_start_b="$(status_value "$status_b" guardian_pid_start)"
harness_b="$(status_value "$status_b" harness_pid)"
harness_start_b="$(status_value "$status_b" harness_pid_start)"
daemon_b_before="$(status_value "$status_b" daemon_pid)"
instance_b="$(status_value "$status_b" instance_id)"
[[ "$guardian_b" == "$(record_value "$PHASE_A" guardian_pid)" && \
   "$guardian_start_b" == "$(record_value "$PHASE_A" guardian_start_tick)" && \
   "$harness_b" == "$(record_value "$PHASE_A" harness_pid)" && \
   "$harness_start_b" == "$(record_value "$PHASE_A" harness_start_tick)" && \
   "$instance_b" == "$(record_value "$PHASE_A" instance_id)" && \
   "$(status_value "$status_b" command)" == "$(record_value "$PHASE_A" command)" && \
   "$(status_value "$status_b" argv_digest)" == "$(record_value "$PHASE_A" argv_digest)" && \
   "$(readlink -f "$STATE_DIR")" == "$(record_value "$PHASE_A" state_root)" && \
   "$(tr -d '\n' < /proc/sys/kernel/random/boot_id)" == "$(record_value "$PHASE_A" boot_id)" ]] ||
  fail 'same-physical identity changed after Pod-A deletion'
verify_process_identity "$guardian_b" "$guardian_start_b" "$START_UNIT"
verify_process_identity "$harness_b" "$harness_start_b" "$START_UNIT"

cursor_b="$(status_value "$status_b" output_cursor)"
"$LOOM" wake --state-dir "$STATE_DIR" --cwd "$WORK_DIR" --agent "$AGENT" \
  --lane "$LANE" --session-id "$SESSION_ID" --message-id pod-b-after-delete \
  --prompt "POD_B_AFTER_DELETE:$RUN_ID" >/dev/null
wait_snapshot 'LOOM_DURABLE_LANE_CHILD ACK sequence=2 ' "$cursor_b" >/dev/null

output_file="$(record_value "$PHASE_A" output_file)"
output_prefix_bytes="$(record_value "$PHASE_A" output_prefix_bytes)"
[[ "$(prefix_sha256 "$output_file" "$output_prefix_bytes")" == \
   "$(record_value "$PHASE_A" output_prefix_sha256)" ]] ||
  fail 'durable output prefix changed after Pod-A deletion'
journal_file="$(record_value "$PHASE_A" journal_file)"
guardian_journal_file="$(record_value "$PHASE_A" guardian_journal_file)"
[[ "$(prefix_sha256 "$journal_file" \
      "$(record_value "$PHASE_A" journal_prefix_bytes)")" == \
   "$(record_value "$PHASE_A" journal_prefix_sha256)" ]] ||
  fail 'semantic journal prefix changed after Pod-A deletion'
[[ "$(prefix_sha256 "$guardian_journal_file" \
      "$(record_value "$PHASE_A" guardian_journal_prefix_bytes)")" == \
   "$(record_value "$PHASE_A" guardian_journal_prefix_sha256)" ]] ||
  fail 'Guardian journal prefix changed after Pod-A deletion'
"$LOOM" verify-journal --journal "$journal_file" | grep -q 'phase=active' ||
  fail 'semantic journal did not verify after transport replacement'
"$LOOM" verify-guardian-journal --journal "$guardian_journal_file" | grep -q 'phase=active' ||
  fail 'Guardian journal did not verify after transport replacement'

"$LOOM" crash-kernel --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
  --agent "$AGENT" --lane "$LANE" --at now >/dev/null
wait_guardian_bridge_zero >/dev/null
kill -0 "$guardian_b" 2>/dev/null || fail 'Guardian died with the OCaml kernel'
kill -0 "$harness_b" 2>/dev/null || fail 'harness died with the OCaml kernel'

systemd-run --quiet --unit="$RECOVER_UNIT" \
  --property=Type=oneshot --property=RemainAfterExit=yes \
  --property=KillMode=control-group --property=TimeoutStartSec=120 \
  --setenv=SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  --setenv="SOUNIO_LOOM_PRODUCT_ACTIVATION_ROOT=$POLICY_ROOT" \
  --setenv=SOUNIO_LOOM_DURABLE_LANE_CANARY=1 \
  "$LOOM" recover --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
    --cwd "$WORK_DIR"
status_recovered="$(wait_machine_status active)"
daemon_recovered="$(status_value "$status_recovered" daemon_pid)"
daemon_recovered_start="$(status_value "$status_recovered" daemon_pid_start)"
[[ "$daemon_recovered" != "$daemon_b_before" && \
   "$(status_value "$status_recovered" guardian_pid)" == "$guardian_b" && \
   "$(status_value "$status_recovered" guardian_pid_start)" == "$guardian_start_b" && \
   "$(status_value "$status_recovered" harness_pid)" == "$harness_b" && \
   "$(status_value "$status_recovered" harness_pid_start)" == "$harness_start_b" && \
   "$(status_value "$status_recovered" instance_id)" == "$instance_b" ]] ||
  fail 'kernel recovery changed Guardian, harness, or instance identity'
verify_process_identity "$daemon_recovered" "$daemon_recovered_start" "$RECOVER_UNIT"
"$LOOM" verify-journal --journal "$journal_file" | grep -q 'phase=active' ||
  fail 'semantic journal did not verify after kernel recovery'
"$LOOM" verify-guardian-journal --journal "$guardian_journal_file" | grep -q 'phase=active' ||
  fail 'Guardian journal did not verify after kernel recovery'
grep -q $'\tKERNEL_RECOVERED\t' "$journal_file" ||
  fail 'semantic journal omitted KERNEL_RECOVERED'

positive_frame='9032 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1'
positive_output="$(printf '%s\n' "$positive_frame" | "$AUTHORITY_RUNTIME")"
[[ "$positive_output" == \
   'SOUNIO_HOST_DURABLE_LANE SAME_PHYSICAL_REATTACH semantic_authority=Sounio action=9032' ]] ||
  fail "Sounio refused the measured same-physical frame: $positive_output"
sabotage_frame='9032 3 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 1 1 1'
set +e
sabotage_output="$(printf '%s\n' "$sabotage_frame" | "$AUTHORITY_RUNTIME" 2>&1)"
sabotage_status=$?
set -e
[[ $sabotage_status -eq 42 && "$sabotage_output" == \
   'SOUNIO_HOST_DURABLE_LANE DENY526 semantic_authority=Sounio action=9032' ]] ||
  fail "guardian-start sabotage was not causally refused status=$sabotage_status output=$sabotage_output"

"$LOOM" stop --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
  --agent "$AGENT" --lane "$LANE" >/dev/null
for _ in $(seq 1 200); do
  [[ "$(record_value "$STATE_DIR/sessions/${AGENT}--${LANE}/session.state" state)" == exited ]] && break
  sleep 0.05
done
[[ "$(record_value "$STATE_DIR/sessions/${AGENT}--${LANE}/session.state" state)" == exited ]] ||
  fail 'lane did not reach a terminal state'
"$LOOM" verify-journal --journal "$journal_file" | grep -q 'phase=exited' ||
  fail 'semantic journal did not reach terminal state'
"$LOOM" verify-guardian-journal --journal "$guardian_journal_file" | grep -q 'phase=exited' ||
  fail 'Guardian journal did not reach terminal state'
stop_units
for pid in "$guardian_b" "$harness_b" "$daemon_recovered"; do
  kill -0 "$pid" 2>/dev/null && fail "experiment process survived cleanup pid=$pid"
done

receipt="sounio-loom-host-durable-lane-supervisor-host-selftest: HOST_MEASUREMENT_PASS semantic_authority=Sounio action=9032 operational_language=OCaml operational_role=EFFECT_PARITY material_platform=Linux+systemd transport_a_uid=$TRANSPORT_A_UID transport_b_uid=$TRANSPORT_B_UID transport_replaced=true predecessor_transport_extinct=true transport_pod_deleted=true state_root_equal=true guardian_pid=$guardian_b guardian_start_tick=$guardian_start_b guardian_pid_equal=true guardian_start_equal=true guardian_instance_equal=true harness_pid=$harness_b harness_start_tick=$harness_start_b harness_pid_equal=true harness_start_equal=true command_equal=true boot_id_equal=true output_prefix_preserved=true semantic_journal_verified=true guardian_journal_verified=true kernel_pid_before=$daemon_b_before kernel_pid_after=$daemon_recovered kernel_recovered=true same_physical_reattach=true sounio_decision=SAME_PHYSICAL_REATTACH sabotage_decision=DENY526 sabotage_process_created=false causal_sabotage=PASS full_extinction=true host_systemd_custody=true tmux_used=false python_executed=false rust_executed=false production_activation=false parity_open=false claim_ready=false"
printf '%s\n' "$receipt" | tee "$HOST_RECEIPT"
