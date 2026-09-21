#!/usr/bin/env bash

set -euo pipefail
umask 077

fail() {
  printf 'sounio-loom-process-witness-effect-hypercube-v11-host-gate: FAIL reason=%s material_hypercube=false material_coverage=false complete_effects=false material_execution=false action_9025_judged=false claim_ready=false\n' "$*" >&2
  exit 1
}

unavailable() {
  printf 'sounio-loom-process-witness-effect-hypercube-v11-host-gate: HOST_GATE_UNAVAILABLE reason=%s material_hypercube=false material_coverage=false complete_effects=false material_execution=false action_9025_judged=false claim_ready=false\n' "$*" >&2
  exit 77
}

usage() {
  printf 'usage: %s --root ABSOLUTE --cell-sha256 HEX --tree-sha256 HEX --bundle ABSOLUTE --bundle-sha256 HEX\n' "$0" >&2
  exit 64
}

field() {
  local line="$1" key="$2" token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s' "${token#*=}"
      return 0
    fi
  done
  fail "receipt omitted field: $key"
}

ROOT=''
CELL_SHA256=''
TREE_SHA256=''
BUNDLE=''
BUNDLE_SHA256=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) [[ $# -ge 2 ]] || usage; ROOT="$2"; shift 2 ;;
    --cell-sha256) [[ $# -ge 2 ]] || usage; CELL_SHA256="$2"; shift 2 ;;
    --tree-sha256) [[ $# -ge 2 ]] || usage; TREE_SHA256="$2"; shift 2 ;;
    --bundle) [[ $# -ge 2 ]] || usage; BUNDLE="$2"; shift 2 ;;
    --bundle-sha256) [[ $# -ge 2 ]] || usage; BUNDLE_SHA256="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ "$ROOT" == /* && "$ROOT" != / && "$ROOT" =~ ^[A-Za-z0-9._/-]+$ &&
   "$BUNDLE" == /* && "$BUNDLE" != / && "$BUNDLE" =~ ^[A-Za-z0-9._/-]+$ &&
   "$CELL_SHA256" =~ ^[0-9a-f]{64}$ && "$TREE_SHA256" =~ ^[0-9a-f]{64}$ &&
   "$BUNDLE_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$(id -u)" == 0 && "$(id -g)" == 0 ]] || unavailable 'root identity is absent'
[[ "$(tr -d '\n' < /proc/1/comm 2>/dev/null)" == systemd ]] ||
  unavailable 'PID 1 is not systemd'
for tool in systemctl systemd-run sha256sum stat readelf find findmnt \
            mount umount journalctl mktemp chmod chown mknod; do
  command -v "$tool" >/dev/null 2>&1 || unavailable "required host tool is absent: $tool"
done
[[ "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)" == "$BUNDLE_SHA256" &&
   "$BUNDLE_SHA256" == 876dce5e9445a5c29236689699719e53ebf79930afae75f8ad5ff21544664394 ]] ||
  fail 'Sounio expected bundle hash drifted'
[[ "$(grep -c '^VERTEX ' "$BUNDLE" || true)" == 40 ]] ||
  fail 'Sounio expected bundle vertex count drifted'

expected_paths="$ROOT/dev
$ROOT/dev/null
$ROOT/loom
$ROOT/loom/effect-cell
$ROOT/loom/effect-policy-v11.freeze.v1
$ROOT/proc
$ROOT/run
$ROOT/run/systemd
$ROOT/run/systemd/incoming
$ROOT/sys
$ROOT/tmp
$ROOT/var
$ROOT/var/tmp"
actual_paths="$(find "$ROOT" -mindepth 1 -printf '%p\n' | sort)"
[[ "$actual_paths" == "$expected_paths" ]] || fail 'host capsule path set drifted'
for directory in "$ROOT" "$ROOT/loom" "$ROOT/dev" "$ROOT/proc" \
                 "$ROOT/tmp" "$ROOT/run" "$ROOT/run/systemd" \
                 "$ROOT/run/systemd/incoming" "$ROOT/sys" "$ROOT/var" \
                 "$ROOT/var/tmp"; do
  [[ "$(stat -c '%F:%u:%g:%a' "$directory")" == 'directory:0:0:555' ]] ||
    fail "host capsule directory metadata drifted: $directory"
done
[[ "$(stat -c '%F:%u:%g:%a:%h' "$ROOT/loom/effect-cell")" == \
   'regular file:0:0:555:1' ]] || fail 'host material cell metadata drifted'
[[ "$(stat -c '%F:%u:%g:%a:%h' "$ROOT/loom/effect-policy-v11.freeze.v1")" == \
   'regular file:0:0:444:1' ]] || fail 'host policy manifest metadata drifted'
[[ "$(stat -c '%F:%t:%T' "$ROOT/dev/null")" == 'character special file:1:3' ]] ||
  fail 'host capsule /dev/null is not character device 1:3'
[[ "$(sha256sum "$ROOT/loom/effect-cell" | cut -d ' ' -f 1)" == "$CELL_SHA256" ]] ||
  fail 'host material cell hash drifted'
[[ "$(sha256sum "$ROOT/loom/effect-policy-v11.freeze.v1" | cut -d ' ' -f 1)" == \
   adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c ]] ||
  fail 'host policy manifest hash drifted'
if readelf -l "$ROOT/loom/effect-cell" | grep -q INTERP; then
  fail 'host material cell retained a dynamic interpreter'
fi
TREE_RECORD="loom/effect-cell:0555:$CELL_SHA256
loom/effect-policy-v11.freeze.v1:0444:adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c
dev/null:host-character-device:1:3
proc:empty:0555
tmp:empty-mountpoint:0555
run/systemd/incoming:empty-systemd-mountpoint:0555
sys:empty-systemd-mountpoint:0555
var/tmp:empty-mountpoint:0555"
[[ "$(printf '%s\n' "$TREE_RECORD" | sha256sum | cut -d ' ' -f 1)" == \
   "$TREE_SHA256" ]] || fail 'host capsule tree identity drifted'

nonce="$$-$(date +%s%N)"
SCRATCH="/var/tmp/loom-effect-hypercube-v11-scratch-$nonce"
SERVER_OUT="/var/tmp/loom-effect-hypercube-v11-server-$nonce.out"
SERVER_ERR="/var/tmp/loom-effect-hypercube-v11-server-$nonce.err"
ROOT_MOUNTED=false
SERVER_PID=''
declare -a UNITS=()

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  for unit in "${UNITS[@]}"; do
    systemctl stop "$unit" >/dev/null 2>&1 || true
    systemctl reset-failed "$unit" >/dev/null 2>&1 || true
  done
  if [[ "$ROOT_MOUNTED" == true ]]; then
    umount "$ROOT" >/dev/null 2>&1 || true
    ROOT_MOUNTED=false
  fi
  rm -rf "$SCRATCH"
  rm -f "$SERVER_OUT" "$SERVER_ERR"
}
trap cleanup EXIT

install -d -m 1777 -o 0 -g 0 "$SCRATCH"
mount --bind "$ROOT" "$ROOT"
ROOT_MOUNTED=true
mount -o remount,bind,ro,nosuid,nodev "$ROOT"
mount_options="$(findmnt -n -o OPTIONS -T "$ROOT")"
[[ ",$mount_options," == *,ro,* && ",$mount_options," == *,nosuid,* &&
   ",$mount_options," == *,nodev,* ]] || fail 'host capsule root is not immutable'

"$ROOT/loom/effect-cell" --inet-server --port 0 \
  > "$SERVER_OUT" 2> "$SERVER_ERR" &
SERVER_PID=$!
for attempt in $(seq 1 200); do
  grep -q '^LOOM_EFFECT_INET_SERVER_V11 READY ' "$SERVER_OUT" 2>/dev/null && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID" || true
    fail "host endpoint server exited before READY: $(tr '\n' ' ' < "$SERVER_ERR" 2>/dev/null)"
  fi
  sleep 0.05
done
server_ready="$(sed -n '1p' "$SERVER_OUT")"
[[ "$server_ready" == LOOM_EFFECT_INET_SERVER_V11\ READY\ address=127.0.0.1\ port=* ]] ||
  fail 'host endpoint server did not become ready'
INET_PORT="$(field "$server_ready" port)"
[[ "$INET_PORT" =~ ^[0-9]+$ && "$INET_PORT" -ge 1 && "$INET_PORT" -le 65535 ]] ||
  fail 'host endpoint server port is malformed'

declare -A invariant_by_probe
declare -A delta_by_vertex
vertex_count=0
refusal_count=0
completion_count=0
extinction_count=0

while IFS= read -r expected_line; do
  family="$(field "$expected_line" family)"
  probe="$(field "$expected_line" probe)"
  bits="$(field "$expected_line" bits)"
  expected="$(field "$expected_line" expected)"
  expected_result="$(field "$expected_line" syscall_result)"
  expected_witness="$(field "$expected_line" witness_kind)"
  [[ "$family" =~ ^([1-9]|1[0-2])$ && "$probe" =~ ^[a-z0-9_]+$ &&
     "$bits" =~ ^(0|1|00|01|10|11)$ ]] || fail 'Sounio vertex identity is unsafe'
  first_bit="${bits:0:1}"
  private_network=yes
  lock_personality=yes
  restrict_namespaces=yes
  proc_source="$ROOT/proc"
  tmp_mode=readonly
  if [[ "$family" == 7 ]]; then
    [[ "$first_bit" == 1 ]] && private_network=yes || private_network=no
  fi
  if [[ "$family" == 10 ]]; then
    [[ "$first_bit" == 1 ]] && lock_personality=yes || lock_personality=no
  fi
  if [[ "$family" == 2 ]]; then
    restrict_namespaces=no
  fi
  if [[ "$family" == 11 && "$first_bit" == 0 ]]; then
    proc_source=/proc
  fi
  if [[ "$family" == 3 && "$first_bit" == 0 ]] || [[ "$family" == 8 ]]; then
    tmp_mode=readwrite
  fi

  unit="sounio-loom-v11-${family}-${probe:0:20}-${bits}-${nonce}.service"
  UNITS+=("$unit")
  common_properties=(
    --property="RootDirectory=$ROOT"
    --property=MountAPIVFS=no
    --property=DynamicUser=yes
    --property=UMask=0077
    --property=NoNewPrivileges=yes
    --property=PrivateTmp=no
    --property=PrivateDevices=no
    --property="PrivateNetwork=$private_network"
    --property=ProtectSystem=strict
    --property=ProtectHome=read-only
    --property="RestrictNamespaces=$restrict_namespaces"
    --property=RestrictSUIDSGID=yes
    --property="LockPersonality=$lock_personality"
    --property=MemoryDenyWriteExecute=yes
    --property=RestrictRealtime=yes
    --property=SystemCallArchitectures=native
    --property=KillMode=mixed
    --property=TimeoutStartSec=10s
    --property=TimeoutStopSec=2s
    --property="BindReadOnlyPaths=$proc_source:/proc /sys:/sys"
  )
  if [[ "$tmp_mode" == readwrite ]]; then
    common_properties+=(--property="BindPaths=$SCRATCH:/tmp $SCRATCH:/var/tmp")
  else
    common_properties+=(--property="BindReadOnlyPaths=$SCRATCH:/tmp $SCRATCH:/var/tmp")
  fi

  set +e
  observed="$(systemd-run --quiet --unit="$unit" --service-type=exec --pipe --wait \
    "${common_properties[@]}" -- \
    /loom/effect-cell --vertex --family "$family" --probe "$probe" --bits "$bits" \
    --policy-manifest /loom/effect-policy-v11.freeze.v1 \
    --cell-path /loom/effect-cell --cell-sha256 "$CELL_SHA256" \
    --root-tree-sha256 "$TREE_SHA256" --scratch-path /tmp/material-file \
    --inet-address 127.0.0.1 --inet-port "$INET_PORT" \
    --unix-path /tmp/material.sock --principal-class DYNAMIC_USER 2>&1)"
  status=$?
  set -e
  [[ $status -eq 0 ]] || {
    state="$(systemctl show "$unit" --property Result --property ExecMainStatus \
      --property ExecMainCode 2>/dev/null | tr '\n' ' ')"
    logs="$(journalctl --no-pager -n 8 -u "$unit" 2>/dev/null | tr '\n' ' ')"
    fail "vertex unit failed family=$family probe=$probe bits=$bits status=$status output=$observed state=$state logs=$logs"
  }
  observed="$(printf '%s\n' "$observed" | grep '^LOOM_EFFECT_VERTEX_V11 OBSERVED ' || true)"
  [[ -n "$observed" && "$(printf '%s\n' "$observed" | wc -l)" == 1 ]] ||
    fail "vertex receipt absent or duplicated: $family/$probe/$bits"
  observation="$(field "$observed" observation)"
  observed_result="$(field "$observed" syscall_result)"
  observed_witness="$(field "$observed" witness_kind)"
  [[ "$observation" == "$expected" && "$observed_result" == "$expected_result" &&
     "$observed_witness" == "$expected_witness" ]] ||
    fail "Sounio mismatch vertex=$family/$probe/$bits expected=$expected/$expected_result/$expected_witness observed=$observation/$observed_result/$observed_witness"
  [[ "$(field "$observed" semantic_authority)" == Sounio &&
     "$(field "$observed" semantic_decision)" == false ]] ||
    fail "material apparatus claimed semantic authority: $family/$probe/$bits"
  invariant="$(field "$observed" invariant_sha256)"
  delta="$(field "$observed" delta_sha256)"
  witness_sha="$(field "$observed" witness_sha256)"
  [[ "$invariant" =~ ^[0-9a-f]{64}$ && "$delta" =~ ^[0-9a-f]{64}$ &&
     "$witness_sha" =~ ^[0-9a-f]{64}$ ]] ||
    fail "vertex causal hashes are malformed: $family/$probe/$bits"
  key="$family/$probe"
  if [[ -n "${invariant_by_probe[$key]:-}" &&
        "${invariant_by_probe[$key]}" != "$invariant" ]]; then
    fail "probe invariant drifted: $key"
  fi
  invariant_by_probe[$key]="$invariant"
  [[ -z "${delta_by_vertex[$key/$bits]:-}" ]] ||
    fail "vertex delta was duplicated: $key/$bits"
  delta_by_vertex[$key/$bits]="$delta"
  if [[ "$observation" == REFUSED_BEFORE_EFFECT ]]; then
    refusal_count=$((refusal_count + 1))
  elif [[ "$observation" == EFFECT_COMPLETED ]]; then
    [[ "$(field "$observed" witness_extinct)" == true ]] ||
      fail "completed witness did not become extinct: $family/$probe/$bits"
    completion_count=$((completion_count + 1))
    extinction_count=$((extinction_count + 1))
  else
    fail "closed observation reached host receipt: $family/$probe/$bits/$observation"
  fi
  [[ -z "$(find "$SCRATCH" -mindepth 1 -print -quit)" ]] ||
    fail "scratch object survived vertex: $family/$probe/$bits"
  unit_state="$(systemctl show "$unit" --property ActiveState --property MainPID 2>/dev/null | tr '\n' ' ')"
  [[ "$unit_state" == *'ActiveState=inactive'* && "$unit_state" == *'MainPID=0'* ]] ||
    fail "vertex process did not become extinct: $family/$probe/$bits state=$unit_state"
  systemctl reset-failed "$unit" >/dev/null 2>&1 || true
  printf '%s\n' "$observed"
  vertex_count=$((vertex_count + 1))
done < <(grep '^VERTEX ' "$BUNDLE")

[[ "$vertex_count" == 40 && "$refusal_count" == 25 &&
   "$completion_count" == 15 && "$extinction_count" == 15 ]] ||
  fail "host cube count drifted vertices=$vertex_count refusals=$refusal_count completions=$completion_count extinctions=$extinction_count"
wait "$SERVER_PID" ||
  fail "host endpoint server failed: $(tr '\n' ' ' < "$SERVER_ERR" 2>/dev/null)"
SERVER_PID=''
grep -Fxq 'LOOM_EFFECT_INET_SERVER_V11 ACCEPTED extinction=true' "$SERVER_OUT" ||
  fail 'host endpoint completion or extinction is absent'
[[ -z "$(find "$SCRATCH" -mindepth 1 -print -quit)" ]] ||
  fail 'scratch objects survived the complete cube'

printf 'sounio-loom-process-witness-effect-hypercube-v11-host-gate: HOST_MEASUREMENT_PASS semantic_authority=Sounio producer=C++20 role=MATERIAL_PARITY transitory=true semantic_decision=false action=9025 policy_manifest_sha256=adbc7151da91bd12928cf059a4fce01de59b38096bb7bebe55be0402fab9972c expected_bundle_sha256=%s cell_sha256=%s tree_sha256=%s hardware_host=%s hardware_arch=%s hardware_kernel=%s systemd_version=%s families=12 probes=13 mechanism_dimensions=18 vertices=40 refusals=25 completions=15 extinctions=15 mincuts_expected=13 crossed_named_rule=0 experiment_unavailable=0 invariant_stable=true delta_distinct=true triple_hash_binding=true dynamic_user=true vfs_read_only_toggled=true private_network_toggled=true unix_endpoint_absence_toggled=true lock_personality_toggled=true proc_treatment_toggled=CAPSULE_EMPTY_BIND+LIVE_PROCFS endpoint_extinction=true process_extinction=true scratch_extinction=true material_hypercube=true material_coverage=false complete_effects=false material_execution=false action_9025_judged=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false claim_ready=false\n' \
  "$BUNDLE_SHA256" "$CELL_SHA256" "$TREE_SHA256" "$(hostname)" "$(uname -m)" \
  "$(uname -s)_$(uname -r)" "$(systemctl --version | sed -n '1s/^systemd //p')"
