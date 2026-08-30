#!/usr/bin/env bash

set -euo pipefail
umask 077

fail() {
  printf 'sounio-loom-hostd-systemd-host-selftest: FAIL reason=%s systemd_activation=false same_physical_recovery=false full_extinction=false\n' "$*" >&2
  exit 1
}

unavailable() {
  printf 'sounio-loom-hostd-systemd-host-selftest: HOST_GATE_UNAVAILABLE reason=%s systemd_activation=false same_physical_recovery=false full_extinction=false\n' "$*" >&2
  exit 77
}

usage() {
  printf 'usage: %s --phase prepare|measure|cleanup --root ABS --run-id ID [--archive ABS --archive-sha256 HEX] [--transport-a-uid UID --transport-b-uid UID]\n' "$0" >&2
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
  [[ -n "$value" && "$value" != *$'\n'* ]] ||
    fail "status field $key is absent or duplicated"
  printf '%s\n' "$value"
}

process_start_tick() {
  sed 's/.*) //' "/proc/$1/stat" | cut -d ' ' -f 20
}

verify_process_identity() {
  local pid="$1" start_tick="$2" expected_unit="${3:-}" cgroup
  [[ "$pid" =~ ^[1-9][0-9]*$ && "$start_tick" =~ ^[1-9][0-9]*$ ]] ||
    fail "non-canonical process identity pid=$pid start=$start_tick"
  kill -0 "$pid" 2>/dev/null || fail "process is absent pid=$pid"
  [[ "$(process_start_tick "$pid")" == "$start_tick" ]] ||
    fail "process start tick drifted pid=$pid"
  if [[ -n "$expected_unit" ]]; then
    cgroup="$(tr '\n' ',' < "/proc/$pid/cgroup")"
    [[ "$cgroup" == *"/$expected_unit"* ]] ||
      fail "process is outside expected unit pid=$pid expected=$expected_unit cgroup=$cgroup"
  fi
}

wait_machine_status() {
  local expected="$1" different_pid="${2:-}" output='' observed='' attempt
  for attempt in $(seq 1 300); do
    output="$($LOOM status --machine --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
      --agent "$AGENT" --lane "$LANE" 2>/dev/null || true)"
    if [[ "$(sed -n 's/^state=//p' <<< "$output")" == "$expected" ]]; then
      observed="$(sed -n 's/^daemon_pid=//p' <<< "$output")"
      if [[ -z "$different_pid" || "$observed" != "$different_pid" ]]; then
        printf '%s\n' "$output"
        return 0
      fi
    fi
    sleep 0.05
  done
  fail "lane did not reach state=$expected with a new kernel last=$(tr '\n' ',' <<< "$output")"
}

wait_guardian_bridge_zero() {
  local output='' attempt
  for attempt in $(seq 1 300); do
    output="$($LOOM guardian-status --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
      --agent "$AGENT" --lane "$LANE" 2>/dev/null || true)"
    [[ "$output" == *' bridge_clients=0 '* ]] && return 0
    sleep 0.05
  done
  fail "Guardian did not release the dead kernel last=$output"
}

wait_unit_pid() {
  local different_pid="${1:-}" pid='' attempt
  for attempt in $(seq 1 3000); do
    pid="$(systemctl show "$HOSTD_UNIT" --property MainPID --value 2>/dev/null || true)"
    if systemctl is-active --quiet "$HOSTD_UNIT" && [[ "$pid" =~ ^[1-9][0-9]*$ ]] &&
       [[ -z "$different_pid" || "$pid" != "$different_pid" ]]; then
      printf '%s\n' "$pid"
      return 0
    fi
    sleep 0.1
  done
  fail "hostd unit did not expose a live distinct MainPID last=$pid"
}

exec_cell_boot_receipt_count() {
  journalctl --unit "$HOSTD_UNIT" --no-pager --output=cat 2>/dev/null |
    grep -c '^loom-product-exec-cell-host: PASS ' || true
}

wait_exec_cell_boot_receipts() {
  local minimum="$1" count=0 latest='' attempt
  for attempt in $(seq 1 2400); do
    count="$(exec_cell_boot_receipt_count)"
    if [[ "$count" =~ ^[0-9]+$ ]] && (( count >= minimum )); then
      latest="$(journalctl --unit "$HOSTD_UNIT" --no-pager --output=cat \
        2>/dev/null | grep '^loom-product-exec-cell-host: PASS ' | tail -n 1)"
      [[ "$latest" == *'semantic_authority=Sounio action=9030 lane_action=9031 '* &&
         "$latest" == *'simultaneous_distinct_dynamic_users=true '* &&
         "$latest" == *'outcome=DONE extinction_complete=true '* &&
         "$latest" == *'command_mismatch=DENY492 causal_sabotage=PASS '* &&
         "$latest" == *'python_executed=false rust_executed=false '* &&
         "$latest" == *'test_only=true production_activation=false '* ]] ||
        fail "ExecCell boot receipt widened or diverged: $latest"
      printf '%s\n' "$count"
      return 0
    fi
    sleep 0.1
  done
  fail "ExecCell boot receipt did not reach minimum=$minimum last=$count"
}

wait_refusal() {
  local state="$STATE_DIR/hostd/supervisor.state" attempt
  for attempt in $(seq 1 200); do
    if [[ -f "$state" ]] && [[ "$(sed -n 's/^state=//p' "$state")" == refused ]] &&
       journalctl --unit "$HOSTD_UNIT" --no-pager -n 80 2>/dev/null | grep -q 'DENY545'; then
      return 0
    fi
    sleep 0.05
  done
  fail 'systemd supervisor did not publish the Sounio DENY545 refusal'
}

wait_process_absent() {
  local pid="$1" attempt
  for attempt in $(seq 1 200); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.05
  done
  fail "process survived final extinction pid=$pid"
}

prefix_sha256() {
  head -c "$2" "$1" | sha256sum | cut -d ' ' -f 1
}

stop_and_remove_units() {
  systemctl disable --now "$HOSTD_UNIT" >/dev/null 2>&1 || true
  systemctl stop "$START_UNIT" >/dev/null 2>&1 || true
  systemctl reset-failed "$HOSTD_UNIT" "$START_UNIT" >/dev/null 2>&1 || true
  rm -f "/etc/systemd/system/$HOSTD_UNIT"
  systemctl daemon-reload >/dev/null 2>&1 || true
}

PHASE=''
ROOT=''
RUN_ID=''
ARCHIVE=''
ARCHIVE_SHA256=''
TRANSPORT_A_UID=''
TRANSPORT_B_UID=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="${2:-}"; shift 2 ;;
    --root) ROOT="${2:-}"; shift 2 ;;
    --run-id) RUN_ID="${2:-}"; shift 2 ;;
    --archive) ARCHIVE="${2:-}"; shift 2 ;;
    --archive-sha256) ARCHIVE_SHA256="${2:-}"; shift 2 ;;
    --transport-a-uid) TRANSPORT_A_UID="${2:-}"; shift 2 ;;
    --transport-b-uid) TRANSPORT_B_UID="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$PHASE" == prepare || "$PHASE" == measure || "$PHASE" == cleanup ]] || usage
[[ "$ROOT" =~ ^/var/tmp/sounio-loom-hostd-systemd-[a-z0-9-]+$ &&
   "$RUN_ID" =~ ^[a-z0-9-]{8,48}$ ]] || usage
[[ "$(id -u):$(id -g)" == 0:0 ]] || unavailable 'root identity is absent'
[[ "$(tr -d '\n' < /proc/1/comm 2>/dev/null)" == systemd ]] ||
  unavailable 'PID 1 is not systemd'
for tool in systemctl systemd-run journalctl sha256sum tar stat find sed cut head \
  grep tr seq mkdir mv rm chmod cat cp bash hostname uname; do
  command -v "$tool" >/dev/null 2>&1 || unavailable "required host tool is absent: $tool"
done

HOSTD_UNIT="sounio-loom-hostd-canary-${RUN_ID}.service"
START_UNIT="sounio-loom-hostd-lane-${RUN_ID}.service"
PREFIX="/opt/sounio/loom-hostd-canary-$RUN_ID"
STATE_DIR="/var/lib/sounio/loom-canary-$RUN_ID"
WORK_DIR="$STATE_DIR/work"
BUNDLE_DIR="$ROOT/bundle-v1"
PHASE_A="$ROOT/phase-a.v1"
HOST_RECEIPT="$ROOT/host-receipt.v1"
LOOM="$PREFIX/bin/sounio-loom-runtime"
AGENT="hostd-${RUN_ID}"
LANE='systemd-canary'
SESSION_ID="hostd-${RUN_ID}"

if [[ "$PHASE" == cleanup ]]; then
  if [[ -x "$LOOM" ]]; then
    "$LOOM" stop --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
      --agent "$AGENT" --lane "$LANE" >/dev/null 2>&1 || true
  fi
  stop_and_remove_units
  if [[ -d "$ROOT" && ! -L "$ROOT" ]]; then
    chmod -R u+rwX "$ROOT" 2>/dev/null || true
    rm -rf "$ROOT"
  fi
  [[ "$PREFIX" == "/opt/sounio/loom-hostd-canary-$RUN_ID" ]] && rm -rf "$PREFIX"
  [[ "$STATE_DIR" == "/var/lib/sounio/loom-canary-$RUN_ID" ]] && rm -rf "$STATE_DIR"
  printf 'sounio-loom-hostd-systemd-host-selftest: CLEANUP_PASS run_id=%s\n' "$RUN_ID"
  exit 0
fi

[[ -d "$ROOT" && ! -L "$ROOT" && "$(stat -c '%u:%g' "$ROOT")" == 0:0 ]] ||
  fail 'host experiment root is absent, linked, or not root-owned'

if [[ "$PHASE" == prepare ]]; then
  [[ "$ARCHIVE" == "$ROOT"/* && -f "$ARCHIVE" && ! -L "$ARCHIVE" &&
     "$ARCHIVE_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
  [[ "$(sha256_file "$ARCHIVE")" == "$ARCHIVE_SHA256" ]] ||
    fail 'bundle transport hash drifted'
  [[ ! -e "$BUNDLE_DIR" && ! -e "$STATE_DIR" ]] ||
    fail 'host experiment state already exists'
  while IFS= read -r member; do
    [[ "$member" =~ ^[A-Za-z0-9._/-]+$ &&
       ( "$member" == bundle-v1 || "$member" == bundle-v1/* ) &&
       "/$member/" != *'/../'* && "/$member/" != *'/./'* ]] ||
      fail "unsafe bundle member: $member"
  done < <(tar -tf "$ARCHIVE")
  while IFS= read -r verbose; do
    [[ "${verbose:0:1}" == d || "${verbose:0:1}" == - ]] ||
      fail 'bundle contains a non-file archive entry'
  done < <(tar -tvf "$ARCHIVE")
  extract="$ROOT/extract"
  mkdir -m 0700 "$extract"
  tar --no-same-owner --same-permissions -xf "$ARCHIVE" -C "$extract"
  [[ -d "$extract/bundle-v1" && -z "$(find "$extract/bundle-v1" -type l -print -quit)" ]] ||
    fail 'bundle topology is incomplete or linked'
  mv "$extract/bundle-v1" "$BUNDLE_DIR"
  rm -rf "$extract"

  STAGED_PREFIX="$BUNDLE_DIR/stage/opt/sounio/loom-hostd"
  INSTALLER="$BUNDLE_DIR/install_loom_hostd.sh"
  EXEC_CELL_CAPSULE="$BUNDLE_DIR/exec-cell-capsule.tar"
  EXEC_CELL_CAPSULE_SHA256="$(record_value \
    "$BUNDLE_DIR/bundle-manifest.v1" exec_cell_capsule_sha256)"
  [[ -x "$INSTALLER" && -x "$STAGED_PREFIX/bin/sounio-loom-runtime" &&
     -x "$STAGED_PREFIX/bin/sounio-loom-host-boot-reconciler" &&
     -x "$STAGED_PREFIX/bin/sounio-loom-resident-membrane-runtime-v5" &&
     -d "$STAGED_PREFIX/policy/product-activation" &&
     -f "$EXEC_CELL_CAPSULE" && ! -L "$EXEC_CELL_CAPSULE" &&
     "$EXEC_CELL_CAPSULE_SHA256" =~ ^[0-9a-f]{64}$ &&
     "$(sha256_file "$EXEC_CELL_CAPSULE")" == \
       "$EXEC_CELL_CAPSULE_SHA256" ]] ||
    fail 'staged installed topology is incomplete'
  set +e
  bash "$INSTALLER" --prefix "$PREFIX" --state-dir "$STATE_DIR" \
    --unit-dir /etc/systemd/system --unit-name "$HOSTD_UNIT" --user root \
    --runtime "$STAGED_PREFIX/bin/sounio-loom-runtime" \
    --authority "$STAGED_PREFIX/bin/sounio-loom-host-boot-reconciler" \
    --resident "$STAGED_PREFIX/bin/sounio-loom-resident-membrane-runtime-v5" \
    --policy-root "$STAGED_PREFIX/policy/product-activation" \
    --exec-cell-capsule "$EXEC_CELL_CAPSULE" \
    --exec-cell-capsule-sha256 "$EXEC_CELL_CAPSULE_SHA256" --activate \
    > "$ROOT/install.out" 2> "$ROOT/install.err"
  install_status=$?
  set -e
  if [[ $install_status -ne 0 ]]; then
    unit_diagnostic="$({ systemctl status "$HOSTD_UNIT" --no-pager 2>&1 || true; } |
      tail -n 30 | tr '\n' ',')"
    journal_diagnostic="$(journalctl --unit "$HOSTD_UNIT" --no-pager \
      --output=cat -n 80 2>&1 | tr '\n' ',')"
    fail "activated installer refused status=$install_status stdout=$(tr '\n' ',' < "$ROOT/install.out") stderr=$(tr '\n' ',' < "$ROOT/install.err") unit=$unit_diagnostic journal=$journal_diagnostic"
  fi
  mkdir -m 0700 "$WORK_DIR"
  [[ "$(systemctl is-enabled "$HOSTD_UNIT")" == enabled ]] ||
    fail 'installer did not enable the canary unit'
  supervisor_a="$(wait_unit_pid)"
  supervisor_start_a="$(process_start_tick "$supervisor_a")"
  exec_cell_boot_receipts_a="$(wait_exec_cell_boot_receipts 1)"
  grep -Fxq 'KillMode=process' "/etc/systemd/system/$HOSTD_UNIT" ||
    fail 'activated unit lost KillMode=process'
  grep -Fxq 'PrivateTmp=false' "/etc/systemd/system/$HOSTD_UNIT" ||
    fail 'activated unit isolated the lane socket namespace'
  grep -Fxq 'Environment=XDG_RUNTIME_DIR=/tmp' "/etc/systemd/system/$HOSTD_UNIT" ||
    fail 'activated unit did not bind the managed socket namespace'
  grep -Fq 'ExecStartPre=' "/etc/systemd/system/$HOSTD_UNIT" ||
    fail 'activated unit omitted the ExecCell boot gate'
  grep -Fq ' --selftest-product-exec-cell-host ' \
    "/etc/systemd/system/$HOSTD_UNIT" ||
    fail 'activated unit ExecCell boot gate changed mode'
  grep -Fxq "ReadWritePaths=$STATE_DIR /tmp/sounio-loom-0" \
    "/etc/systemd/system/$HOSTD_UNIT" ||
    fail 'activated unit lost the exact socket write boundary'
  [[ -d /tmp/sounio-loom-0 && ! -L /tmp/sounio-loom-0 &&
     "$(stat -c '%u:%g:%a' /tmp/sounio-loom-0)" == 0:0:700 ]] ||
    fail 'managed host socket namespace is absent, linked, or weakly owned'
  grep -Fxq 'production_activation=true' "$PREFIX/manifest.v1" ||
    fail 'activated installer manifest remained dark'
  grep -Fxq 'service_uid=0' "$PREFIX/manifest.v1" ||
    fail 'activated manifest lost the service uid'
  grep -Fxq 'socket_root=/tmp/sounio-loom-0' "$PREFIX/manifest.v1" ||
    fail 'activated manifest lost the managed socket root'
  grep -Fxq 'exec_cell_bundle_present=true' "$PREFIX/manifest.v1" ||
    fail 'activated manifest lost the immutable ExecCell bundle'
  grep -Fxq 'exec_cell_boot_gate_configured=true' "$PREFIX/manifest.v1" ||
    fail 'activated manifest lost the ExecCell boot gate'
  grep -Fxq 'exec_cell_boot_gate_test_only=true' "$PREFIX/manifest.v1" ||
    fail 'activated manifest widened the ExecCell canary'
  grep -Fxq 'exec_attached=false' "$PREFIX/manifest.v1" ||
    fail 'activated manifest preclaimed general ExecCell attachment'
  grep -Fxq 'semantic_authority=Sounio' \
    "$PREFIX/share/product-activation-policy.v1" ||
    fail 'installed policy root lost Sounio authority'

  systemd-run --quiet --unit="$START_UNIT" --property=Type=oneshot \
    --property=RemainAfterExit=yes --property=KillMode=control-group \
    --property=TimeoutStartSec=120 --setenv=SOUNIO_LOOM_DURABLE_LANE_CANARY=1 \
    --setenv=XDG_RUNTIME_DIR=/tmp \
    "$LOOM" start --state-dir "$STATE_DIR" --agent "$AGENT" --lane "$LANE" \
      --session-id "$SESSION_ID" --cwd "$WORK_DIR" -- \
      "$LOOM" _durable-lane-canary
  status_a="$(wait_machine_status active)"
  "$LOOM" host-enroll --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
    --agent "$AGENT" --lane "$LANE" >/dev/null
  for _ in $(seq 1 400); do
    "$LOOM" host-verify --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
      --agent "$AGENT" --lane "$LANE" >/dev/null 2>&1 && break
    sleep 0.05
  done
  if ! "$LOOM" host-verify --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
    --agent "$AGENT" --lane "$LANE" >/dev/null 2>&1; then
    supervisor_state='absent'
    [[ -f "$STATE_DIR/hostd/supervisor.state" ]] &&
      supervisor_state="$(tr '\n' ',' < "$STATE_DIR/hostd/supervisor.state")"
    unit_state="$(systemctl show "$HOSTD_UNIT" --property ActiveState \
      --property SubState --property Result --property MainPID 2>/dev/null | tr '\n' ',')"
    journal_tail="$(journalctl --unit "$HOSTD_UNIT" --no-pager -n 12 2>/dev/null | tr '\n' ',')"
    fail "systemd supervisor omitted enrolled-lane receipt supervisor=$supervisor_state unit=$unit_state journal=$journal_tail"
  fi

  guardian_a="$(status_value "$status_a" guardian_pid)"
  guardian_start_a="$(status_value "$status_a" guardian_pid_start)"
  harness_a="$(status_value "$status_a" harness_pid)"
  harness_start_a="$(status_value "$status_a" harness_pid_start)"
  daemon_a="$(status_value "$status_a" daemon_pid)"
  instance_a="$(status_value "$status_a" instance_id)"
  verify_process_identity "$guardian_a" "$guardian_start_a" "$START_UNIT"
  verify_process_identity "$harness_a" "$harness_start_a" "$START_UNIT"
  descriptor="$STATE_DIR/sessions/${AGENT}--${LANE}/session.state"
  output_file="$(record_value "$descriptor" output_file)"
  journal_file="$(record_value "$descriptor" journal_file)"
  guardian_journal_file="$(record_value "$descriptor" guardian_journal_file)"
  output_prefix_bytes="$(stat -c %s "$output_file")"
  journal_prefix_bytes="$(stat -c %s "$journal_file")"
  guardian_journal_prefix_bytes="$(stat -c %s "$guardian_journal_file")"

  "$LOOM" crash-kernel --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
    --agent "$AGENT" --lane "$LANE" --at now >/dev/null
  wait_guardian_bridge_zero
  cat > "$PHASE_A" <<EOF
schema=loom-hostd-systemd-phase-a-v1
run_id=$RUN_ID
boot_id=$(tr -d '\n' < /proc/sys/kernel/random/boot_id)
archive_sha256=$ARCHIVE_SHA256
hostd_unit=$HOSTD_UNIT
start_unit=$START_UNIT
supervisor_pid=$supervisor_a
supervisor_start_tick=$supervisor_start_a
guardian_pid=$guardian_a
guardian_start_tick=$guardian_start_a
harness_pid=$harness_a
harness_start_tick=$harness_start_a
daemon_pid=$daemon_a
instance_id=$instance_a
command=$(status_value "$status_a" command)
argv_digest=$(status_value "$status_a" argv_digest)
output_file=$output_file
output_prefix_bytes=$output_prefix_bytes
output_prefix_sha256=$(prefix_sha256 "$output_file" "$output_prefix_bytes")
journal_file=$journal_file
journal_prefix_bytes=$journal_prefix_bytes
journal_prefix_sha256=$(prefix_sha256 "$journal_file" "$journal_prefix_bytes")
guardian_journal_file=$guardian_journal_file
guardian_journal_prefix_bytes=$guardian_journal_prefix_bytes
guardian_journal_prefix_sha256=$(prefix_sha256 "$guardian_journal_file" "$guardian_journal_prefix_bytes")
host_boot_semantics_sha256=0d5174cd87b8c18b5f3bbfa7ed44d0258795a96f146730c879c46167abdddf7d
host_boot_authority_sha256=99f5062729a171ac2d8c1b9b181497fbe1b8c9317859ee0fdc4d2cd4acaedb5b
exec_cell_boot_receipts=$exec_cell_boot_receipts_a
exec_cell_capsule_sha256=$EXEC_CELL_CAPSULE_SHA256
EOF
  chmod 0400 "$PHASE_A"
  printf 'sounio-loom-hostd-systemd-host-selftest: PHASE_A_PASS run_id=%s unit=%s enabled=true supervisor_pid=%s guardian_pid=%s harness_pid=%s kernel_pid_before=%s kernel_crashed=true transport_pod_deleted=false semantic_authority=Sounio actions=9030,9031,9041 exec_cell_boot_gate=true exec_cell_boot_receipts=%s exec_cell_boot_gate_test_only=true exec_attached=false python_executed=false rust_executed=false\n' \
    "$RUN_ID" "$HOSTD_UNIT" "$supervisor_a" "$guardian_a" "$harness_a" \
    "$daemon_a" "$exec_cell_boot_receipts_a"
  exit 0
fi

[[ "$TRANSPORT_A_UID" =~ ^[0-9a-f-]{36}$ &&
   "$TRANSPORT_B_UID" =~ ^[0-9a-f-]{36}$ &&
   "$TRANSPORT_A_UID" != "$TRANSPORT_B_UID" ]] ||
  fail 'transport Pod identities are absent, malformed, or equal'
[[ -f "$PHASE_A" && ! -L "$PHASE_A" && "$(stat -c '%a' "$PHASE_A")" == 400 ]] ||
  fail 'phase-A receipt is absent, linked, or mutable'
[[ "$(record_value "$PHASE_A" run_id)" == "$RUN_ID" &&
   "$(record_value "$PHASE_A" hostd_unit)" == "$HOSTD_UNIT" ]] ||
  fail 'phase-A identity drifted'
[[ "$(systemctl is-enabled "$HOSTD_UNIT")" == enabled ]] ||
  fail 'canary unit lost boot enablement after transport replacement'

supervisor_a="$(record_value "$PHASE_A" supervisor_pid)"
supervisor_start_a="$(record_value "$PHASE_A" supervisor_start_tick)"
verify_process_identity "$supervisor_a" "$supervisor_start_a" "$HOSTD_UNIT"
status_recovered_a="$(wait_machine_status active "$(record_value "$PHASE_A" daemon_pid)")"
daemon_recovered_a="$(status_value "$status_recovered_a" daemon_pid)"
daemon_recovered_a_start="$(status_value "$status_recovered_a" daemon_pid_start)"
guardian="$(record_value "$PHASE_A" guardian_pid)"
guardian_start="$(record_value "$PHASE_A" guardian_start_tick)"
harness="$(record_value "$PHASE_A" harness_pid)"
harness_start="$(record_value "$PHASE_A" harness_start_tick)"
instance="$(record_value "$PHASE_A" instance_id)"
[[ "$(status_value "$status_recovered_a" guardian_pid)" == "$guardian" &&
   "$(status_value "$status_recovered_a" guardian_pid_start)" == "$guardian_start" &&
   "$(status_value "$status_recovered_a" harness_pid)" == "$harness" &&
   "$(status_value "$status_recovered_a" harness_pid_start)" == "$harness_start" &&
   "$(status_value "$status_recovered_a" instance_id)" == "$instance" &&
   "$(status_value "$status_recovered_a" command)" == "$(record_value "$PHASE_A" command)" &&
   "$(status_value "$status_recovered_a" argv_digest)" == "$(record_value "$PHASE_A" argv_digest)" &&
   "$(tr -d '\n' < /proc/sys/kernel/random/boot_id)" == "$(record_value "$PHASE_A" boot_id)" ]] ||
  fail 'automatic recovery changed the same-physical lane identity'
verify_process_identity "$guardian" "$guardian_start" "$START_UNIT"
verify_process_identity "$harness" "$harness_start" "$START_UNIT"
verify_process_identity "$daemon_recovered_a" "$daemon_recovered_a_start" "$HOSTD_UNIT"

[[ "$(prefix_sha256 "$(record_value "$PHASE_A" output_file)" \
      "$(record_value "$PHASE_A" output_prefix_bytes)")" == \
   "$(record_value "$PHASE_A" output_prefix_sha256)" ]] ||
  fail 'output prefix changed across systemd recovery'
[[ "$(prefix_sha256 "$(record_value "$PHASE_A" journal_file)" \
      "$(record_value "$PHASE_A" journal_prefix_bytes)")" == \
   "$(record_value "$PHASE_A" journal_prefix_sha256)" ]] ||
  fail 'semantic journal prefix changed across systemd recovery'
[[ "$(prefix_sha256 "$(record_value "$PHASE_A" guardian_journal_file)" \
      "$(record_value "$PHASE_A" guardian_journal_prefix_bytes)")" == \
   "$(record_value "$PHASE_A" guardian_journal_prefix_sha256)" ]] ||
  fail 'Guardian journal prefix changed across systemd recovery'

systemctl stop "$HOSTD_UNIT"
systemctl is-active --quiet "$HOSTD_UNIT" && fail 'hostd unit remained active after stop'
verify_process_identity "$guardian" "$guardian_start" "$START_UNIT"
verify_process_identity "$harness" "$harness_start" "$START_UNIT"
verify_process_identity "$daemon_recovered_a" "$daemon_recovered_a_start" "$HOSTD_UNIT"
status_stopped="$(wait_machine_status active)"
[[ "$(status_value "$status_stopped" daemon_pid)" == "$daemon_recovered_a" ]] ||
  fail 'KillMode=process did not preserve the recovered kernel'

systemctl start "$HOSTD_UNIT"
supervisor_b="$(wait_unit_pid "$supervisor_a")"
supervisor_b_start="$(process_start_tick "$supervisor_b")"
verify_process_identity "$supervisor_b" "$supervisor_b_start" "$HOSTD_UNIT"
status_restarted="$(wait_machine_status active)"
[[ "$(status_value "$status_restarted" guardian_pid)" == "$guardian" &&
   "$(status_value "$status_restarted" harness_pid)" == "$harness" &&
   "$(status_value "$status_restarted" instance_id)" == "$instance" ]] ||
  fail 'supervisor restart changed lane identity'

"$LOOM" crash-kernel --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
  --agent "$AGENT" --lane "$LANE" --at now >/dev/null
wait_guardian_bridge_zero
status_recovered_b="$(wait_machine_status active "$daemon_recovered_a")"
daemon_recovered_b="$(status_value "$status_recovered_b" daemon_pid)"
daemon_recovered_b_start="$(status_value "$status_recovered_b" daemon_pid_start)"
verify_process_identity "$daemon_recovered_b" "$daemon_recovered_b_start" "$HOSTD_UNIT"
[[ "$(status_value "$status_recovered_b" guardian_pid)" == "$guardian" &&
   "$(status_value "$status_recovered_b" harness_pid)" == "$harness" &&
   "$(status_value "$status_recovered_b" instance_id)" == "$instance" ]] ||
  fail 'restarted supervisor did not preserve the physical lane'

systemctl stop "$HOSTD_UNIT"
"$LOOM" crash-kernel --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
  --agent "$AGENT" --lane "$LANE" --at now >/dev/null
wait_guardian_bridge_zero
descriptor="$STATE_DIR/sessions/${AGENT}--${LANE}/session.state"
cp "$descriptor" "$ROOT/descriptor.clean"
sed -i 's/^guardian_pid_start=.*/guardian_pid_start=1/' "$descriptor"
systemctl start "$HOSTD_UNIT" || true
wait_refusal
verify_process_identity "$guardian" "$guardian_start" "$START_UNIT"
verify_process_identity "$harness" "$harness_start" "$START_UNIT"
systemctl stop "$HOSTD_UNIT" >/dev/null 2>&1 || true
cp "$ROOT/descriptor.clean" "$descriptor"
systemctl reset-failed "$HOSTD_UNIT" >/dev/null 2>&1 || true
systemctl start "$HOSTD_UNIT"
supervisor_c="$(wait_unit_pid "$supervisor_b")"
status_recovered_c="$(wait_machine_status active "$daemon_recovered_b")"
daemon_recovered_c="$(status_value "$status_recovered_c" daemon_pid)"
daemon_recovered_c_start="$(status_value "$status_recovered_c" daemon_pid_start)"
verify_process_identity "$daemon_recovered_c" "$daemon_recovered_c_start" "$HOSTD_UNIT"
[[ "$(status_value "$status_recovered_c" guardian_pid)" == "$guardian" &&
   "$(status_value "$status_recovered_c" harness_pid)" == "$harness" &&
   "$(status_value "$status_recovered_c" instance_id)" == "$instance" ]] ||
  fail 'post-sabotage recovery changed the physical lane'

exec_cell_boot_minimum="$((
  $(record_value "$PHASE_A" exec_cell_boot_receipts) + 3
))"
exec_cell_boot_receipts_final="$(wait_exec_cell_boot_receipts \
  "$exec_cell_boot_minimum")"

verified="$($LOOM host-verify --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
  --agent "$AGENT" --lane "$LANE")"
[[ "$verified" == *'hash_chain=PASS semantic_authority=Sounio action=9041'* ]] ||
  fail "hostd receipt chain did not verify: $verified"
receipt_count="$(sed -n 's/.* receipts=\([0-9][0-9]*\) .*/\1/p' <<< "$verified")"
[[ "$receipt_count" =~ ^[1-9][0-9]*$ ]] || fail 'hostd receipt count is absent'
"$LOOM" verify-journal --journal "$(record_value "$PHASE_A" journal_file)" |
  grep -q 'phase=active' || fail 'semantic journal did not verify after hostd cycles'
"$LOOM" verify-guardian-journal \
  --journal "$(record_value "$PHASE_A" guardian_journal_file)" |
  grep -q 'phase=active' || fail 'Guardian journal did not verify after hostd cycles'

systemctl stop "$HOSTD_UNIT"
"$LOOM" stop --state-dir "$STATE_DIR" --cwd "$WORK_DIR" \
  --agent "$AGENT" --lane "$LANE" >/dev/null
wait_process_absent "$guardian"
wait_process_absent "$harness"
wait_process_absent "$daemon_recovered_c"
systemctl stop "$START_UNIT" >/dev/null 2>&1 || true
systemctl disable "$HOSTD_UNIT" >/dev/null 2>&1
rm -f "/etc/systemd/system/$HOSTD_UNIT"
systemctl daemon-reload
systemctl is-enabled --quiet "$HOSTD_UNIT" 2>/dev/null &&
  fail 'canary unit remained boot-enabled after cleanup'

receipt="sounio-loom-hostd-systemd-host-selftest: HOST_MEASUREMENT_PASS semantic_authority=Sounio actions=9030,9031,9041 operational_language=OCaml operational_role=EFFECT_PARITY material_platform=Linux+systemd host=$(hostname) kernel=$(uname -r) systemd_version=$(systemctl --version | sed -n '1s/^systemd //p') boot_id=$(record_value "$PHASE_A" boot_id) transport_a_uid=$TRANSPORT_A_UID transport_b_uid=$TRANSPORT_B_UID transport_replaced=true predecessor_transport_extinct=true unit=$HOSTD_UNIT boot_enabled=true real_systemd_activation=true exec_cell_boot_gate=true exec_cell_boot_gate_test_only=true exec_cell_boot_receipts=$exec_cell_boot_receipts_final exec_cell_outcome=DONE exec_cell_extinction=true exec_cell_command_mismatch=DENY492 exec_attached=false supervisor_pid_before=$supervisor_a supervisor_pid_after=$supervisor_b supervisor_restarted=true kill_mode_process_preserved_lane=true guardian_pid=$guardian guardian_start_tick=$guardian_start guardian_equal=true harness_pid=$harness harness_start_tick=$harness_start harness_equal=true instance_id=$instance instance_equal=true kernel_pid_before=$(record_value "$PHASE_A" daemon_pid) kernel_pid_after=$daemon_recovered_c automatic_recovery=true repeated_recovery=true output_prefix_preserved=true semantic_journal_verified=true guardian_journal_verified=true receipt_count=$receipt_count receipt_chain=PASS sabotage_decision=DENY545 sabotage_process_created=false causal_sabotage=PASS same_physical_recovery=true same_pty_claim_after_guardian_loss=false full_extinction=true tmux_used=false python_executed=false rust_executed=false production_activation=canary-only"
printf '%s\n' "$receipt" | tee "$HOST_RECEIPT"
