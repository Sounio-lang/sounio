#!/usr/bin/env bash

set -euo pipefail
umask 077

fail() {
  printf 'promote-loom-host-capsule: REFUSE reason=%s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s --archive ABSOLUTE_PATH --expected-sha256 HEX --mode verify|preflight|promote\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

record_value() {
  local path="$1"
  local key="$2"
  local line name value found=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$found" ]] || fail "duplicate record field: $key"
      found="$value"
    fi
  done < "$path"
  [[ -n "$found" ]] || fail "record omitted field: $key"
  printf '%s\n' "$found"
}

ARCHIVE=''
EXPECTED_SHA256=''
MODE=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive)
      [[ $# -ge 2 ]] || usage
      ARCHIVE="$2"
      shift 2
      ;;
    --expected-sha256)
      [[ $# -ge 2 ]] || usage
      EXPECTED_SHA256="$2"
      shift 2
      ;;
    --mode)
      [[ $# -ge 2 ]] || usage
      MODE="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$ARCHIVE" == /* && -f "$ARCHIVE" && ! -L "$ARCHIVE" ]] || fail 'capsule archive is absent, linked, or non-absolute'
[[ "$EXPECTED_SHA256" =~ ^[0-9a-f]{64}$ ]] || fail 'expected capsule hash is not canonical SHA-256'
[[ "$MODE" == verify || "$MODE" == preflight || "$MODE" == promote ]] || usage

for tool in sha256sum stat mktemp find sort tar head sync cut grep wc; do
  command -v "$tool" >/dev/null 2>&1 || fail "required promotion tool is missing: $tool"
done

ACTUAL_SHA256="$(sha256_file "$ARCHIVE")"
[[ "$ACTUAL_SHA256" == "$EXPECTED_SHA256" ]] || fail 'capsule archive hash drifted'

member_count=0
while IFS= read -r member; do
  [[ -n "$member" && "$member" =~ ^[A-Za-z0-9._/-]+$ ]] || fail "capsule has an unsafe member name: $member"
  [[ "$member" == capsule-v1 || "$member" == capsule-v1/* ]] || fail "capsule member escaped its root: $member"
  [[ "/$member/" != *'/../'* && "/$member/" != *'/./'* ]] || fail "capsule member traverses a directory: $member"
  member_count=$((member_count + 1))
done < <(tar -tf "$ARCHIVE")
[[ $member_count -gt 0 ]] || fail 'capsule archive is empty'

while IFS= read -r verbose; do
  type="${verbose:0:1}"
  [[ "$type" == d || "$type" == - ]] || fail "capsule contains a non-file archive type: $type"
done < <(tar -tvf "$ARCHIVE")

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-promotion-verify.XXXXXX")"
cleanup() {
  find "$WORK" -type d -exec chmod u+rwx {} + 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT
tar --no-same-owner --same-permissions -xf "$ARCHIVE" -C "$WORK"

CAPSULE="$WORK/capsule-v1"
ROOTFS="$CAPSULE/rootfs"
META="$CAPSULE/meta"
MANIFEST="$META/capsule.manifest.v1"
ENTRIES="$META/payload.entries.v1"
HOST_GATE="$META/sounio_loom_kernel_principal_broker_host_gate.sh"
CAPSULE_PROMOTER="$META/promote_loom_host_capsule.sh"
[[ -d "$CAPSULE" && ! -L "$CAPSULE" && -d "$ROOTFS" && ! -L "$ROOTFS" && -d "$META" && ! -L "$META" ]] ||
  fail 'capsule root topology is invalid'
for required in "$MANIFEST" "$ENTRIES" "$HOST_GATE" "$CAPSULE_PROMOTER"; do
  [[ -f "$required" && ! -L "$required" ]] || fail "capsule metadata file is absent or linked: $required"
done
if find "$CAPSULE" -type l -print -quit | grep -q .; then
  fail 'capsule contains a symlink before host promotion'
fi
while IFS= read -r metadata_path; do
  metadata_name="${metadata_path#"$META"/}"
  [[ "$metadata_name" == capsule.manifest.v1 || "$metadata_name" == payload.entries.v1 || \
     "$metadata_name" == sounio_loom_kernel_principal_broker_host_gate.sh || \
     "$metadata_name" == promote_loom_host_capsule.sh ]] ||
    fail "undeclared capsule metadata entry: $metadata_name"
done < <(find "$META" -mindepth 1 -type f | sort)

[[ "$(record_value "$MANIFEST" schema)" == loom-host-promotion-capsule-v1 ]] || fail 'capsule schema is not v1'
[[ "$(record_value "$MANIFEST" stage)" == SEMANTICS_FROZEN ]] || fail 'capsule semantic stage is not frozen'
[[ "$(record_value "$MANIFEST" semantic_producer)" == Sounio ]] || fail 'capsule semantic producer is not Sounio'
[[ "$(record_value "$MANIFEST" semantic_role)" == SEMANTIC_AUTHORITY ]] || fail 'capsule semantic role is not authoritative'
[[ "$(record_value "$MANIFEST" semantic_actions)" == 9027+9028+9029 ]] || fail 'capsule semantic action lineage drifted'
[[ "$(record_value "$MANIFEST" transport_role)" == MECHANICAL_PACKAGING ]] || fail 'transport claimed a non-mechanical role'
[[ "$(record_value "$MANIFEST" transport_authority)" == false ]] || fail 'transport promoted itself to authority'
[[ "$(record_value "$MANIFEST" material_role)" == MATERIAL_PARITY ]] || fail 'material role drifted'
[[ "$(record_value "$MANIFEST" material_transitory)" == true ]] || fail 'material implementation lost its transitory marker'
for closed in parity_open claim_ready launch_open recycle_open material_broker material_capsule material_invocation same_uid_peer_isolation; do
  [[ "$(record_value "$MANIFEST" "$closed")" == false ]] || fail "capsule opened forbidden boundary: $closed"
done

SOURCE_COMMIT="$(record_value "$MANIFEST" source_commit)"
SOURCE_TREE_STATE="$(record_value "$MANIFEST" source_tree_state)"
RELEASE_ID="$(record_value "$MANIFEST" release_id)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail 'capsule source commit is not canonical'
[[ "$SOURCE_TREE_STATE" == CLEAN || "$SOURCE_TREE_STATE" == DIRTY_UNPROMOTABLE ]] || fail 'capsule source tree state is unknown'
[[ "$RELEASE_ID" =~ ^9029-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}$ ]] ||
  fail 'capsule release identity is invalid'

[[ "$(sha256_file "$ENTRIES")" == "$(record_value "$MANIFEST" payload_entries_sha256)" ]] ||
  fail 'payload entry manifest hash drifted'
[[ "$(sha256_file "$CAPSULE_PROMOTER")" == "$(record_value "$MANIFEST" promoter_sha256)" ]] ||
  fail 'capsule promoter hash drifted'
[[ "$(sha256_file "$0")" == "$(record_value "$MANIFEST" promoter_sha256)" ]] ||
  fail 'executing promoter differs from the capsule-bound promoter'
[[ "$(sha256_file "$HOST_GATE")" == "$(record_value "$MANIFEST" host_gate_sha256)" ]] ||
  fail 'host gate hash drifted'
[[ "$(head -n 1 "$HOST_GATE")" == '#!/usr/bin/env bash' ]] || fail 'host gate language is not the mechanical Bash installation boundary'

declare -A seen_paths=()
verified_entries=0
while IFS='|' read -r kind mode digest relative extra || [[ -n "${kind:-}" ]]; do
  [[ -z "${extra:-}" && -n "$kind" && "$mode" =~ ^[0-7]{3,4}$ && "$relative" =~ ^[A-Za-z0-9._/-]+$ ]] ||
    fail "malformed payload entry: $kind|$mode|$digest|$relative${extra:+|$extra}"
  [[ "/$relative/" != *'/../'* && "/$relative/" != *'/./'* && "$relative" != /* ]] ||
    fail "payload entry traverses a directory: $relative"
  [[ -z "${seen_paths[$relative]+present}" ]] || fail "duplicate payload entry: $relative"
  seen_paths[$relative]=1
  path="$ROOTFS/$relative"
  case "$kind" in
    D)
      [[ "$digest" == - && -d "$path" && ! -L "$path" ]] || fail "payload directory is absent or changed: $relative"
      ;;
    F)
      [[ "$digest" =~ ^[0-9a-f]{64}$ && -f "$path" && ! -L "$path" ]] || fail "payload file is absent or changed: $relative"
      [[ "$(sha256_file "$path")" == "$digest" ]] || fail "payload content hash drifted: $relative"
      ;;
    *) fail "payload entry has an unsupported kind: $kind" ;;
  esac
  [[ "$(stat -c '%a' "$path")" == "$mode" ]] || fail "payload mode drifted: $relative"
  verified_entries=$((verified_entries + 1))
done < "$ENTRIES"
[[ "$verified_entries" == "$(record_value "$MANIFEST" payload_entry_count)" ]] || fail 'payload entry count drifted'
actual_entries="$(find "$ROOTFS" -mindepth 1 -printf . | wc -c)"
[[ "$actual_entries" == "$verified_entries" ]] || fail 'payload contains an undeclared filesystem entry'

RELEASE="$ROOTFS/usr/lib/sounio/loom/releases/$RELEASE_ID"
RECEIPT="$RELEASE/install.receipt.v1"
BROKER="$RELEASE/loom-kernel-principal-broker"
LEASE_AUTHORITY="$RELEASE/sounio-loom-kernel-principal-lease-authority-runtime"
CAPSULE_AUTHORITY="$RELEASE/sounio-loom-kernel-principal-capsule-authority-runtime"
INVOCATION_AUTHORITY="$RELEASE/sounio-loom-kernel-invocation-cell-authority-runtime"
for required in "$RECEIPT" "$BROKER" "$LEASE_AUTHORITY" "$CAPSULE_AUTHORITY" "$INVOCATION_AUTHORITY" \
  "$ROOTFS/etc/sounio/loom-principal-broker.conf" \
  "$ROOTFS/etc/systemd/system/sounio-loom-principal-broker.socket" \
  "$ROOTFS/etc/systemd/system/sounio-loom-principal-broker.service"; do
  [[ -f "$required" && ! -L "$required" ]] || fail "required promotion payload is absent: $required"
done
[[ "$(sha256_file "$RECEIPT")" == "$(record_value "$MANIFEST" install_receipt_sha256)" ]] || fail 'install receipt hash drifted'
for binding in lease_manifest lease_authority capsule_manifest capsule_authority invocation_manifest invocation_authority broker; do
  [[ "$(record_value "$RECEIPT" "${binding}_sha256")" == "$(record_value "$MANIFEST" "${binding}_sha256")" ]] ||
    fail "capsule and install receipt disagree: $binding"
done
[[ "$(record_value "$RECEIPT" semantic_producer)" == Sounio ]] || fail 'install receipt semantic producer drifted'
[[ "$(record_value "$RECEIPT" semantic_role)" == SEMANTIC_AUTHORITY ]] || fail 'install receipt semantic role drifted'
[[ "$(record_value "$RECEIPT" material_role)" == MATERIAL_PARITY ]] || fail 'install receipt material role drifted'
for closed in launch_open recycle_open material_broker material_capsule material_invocation; do
  [[ "$(record_value "$RECEIPT" "$closed")" == false ]] || fail "install receipt opened forbidden boundary: $closed"
done

lease_selftest="$(printf '0\n' | "$LEASE_AUTHORITY")"
capsule_selftest="$(printf '0\n' | "$CAPSULE_AUTHORITY")"
invocation_selftest="$(printf '0\n' | "$INVOCATION_AUTHORITY")"
[[ "$lease_selftest" == 'SOUNIO_KERNEL_PRINCIPAL_LEASE_SELFTEST PASS cases=18' ]] || fail 'lease authority offline selftest failed'
[[ "$capsule_selftest" == 'SOUNIO_KERNEL_PRINCIPAL_CAPSULE_SELFTEST PASS cases=19' ]] || fail 'capsule authority offline selftest failed'
[[ "$invocation_selftest" == 'SOUNIO_KERNEL_INVOCATION_CELL_SELFTEST PASS cases=17' ]] || fail 'InvocationCell authority offline selftest failed'
BROKER_PROTOCOL_CHECK=offline-selftest
if [[ "$(id -u)" == 0 || "$(id -g)" == 0 ]]; then
  # The broker deliberately refuses its synthetic protocol selftest as root.
  # Root behavior is exercised only through the socket-activated live host gate.
  BROKER_PROTOCOL_CHECK=deferred-to-live-host-gate
else
  broker_selftest="$($BROKER --selftest-protocol)"
  [[ "$broker_selftest" == 'LOOM_KERNEL_PRINCIPAL_BROKER_PROTOCOL_SELFTEST PASS admission_without_context=denied malformed_admission=denied launch=closed recycle=closed unknown=denied partial_status=denied' ]] ||
    fail 'broker offline protocol selftest failed'
fi

if [[ "$MODE" == verify ]]; then
  printf 'LOOM_HOST_PROMOTION_CAPSULE_VERIFY PASS archive_sha256=%s release=%s source_commit=%s source_tree_state=%s payload_entries=%s broker_protocol=%s semantic_producer=Sounio semantic_role=SEMANTIC_AUTHORITY transport_authority=false parity_open=false claim_ready=false launch=closed material_broker=false material_capsule=false material_invocation=false\n' \
    "$ACTUAL_SHA256" "$RELEASE_ID" "$SOURCE_COMMIT" "$SOURCE_TREE_STATE" "$verified_entries" "$BROKER_PROTOCOL_CHECK"
  exit 0
fi

[[ "$SOURCE_TREE_STATE" == CLEAN ]] || fail 'dirty-source capsule cannot reach host preflight or promotion'
[[ "$(id -u)" == 0 && "$(id -g)" == 0 ]] || fail 'host promotion requires root identity'
[[ "$(tr -d '\n' < /proc/1/comm)" == systemd ]] || fail 'host promotion requires PID 1 systemd'
[[ -d /run/systemd/system ]] || fail 'host promotion requires the systemd runtime boundary'
command -v systemctl >/dev/null 2>&1 || fail 'host promotion requires systemctl'

if [[ "$MODE" == preflight ]]; then
  printf 'LOOM_HOST_PROMOTION_PREFLIGHT PASS archive_sha256=%s release=%s source_commit=%s payload_entries=%s broker_protocol=%s pid1=systemd semantic_producer=Sounio semantic_role=SEMANTIC_AUTHORITY transport_authority=false parity_open=false claim_ready=false launch=closed material_broker=false material_capsule=false material_invocation=false\n' \
    "$ACTUAL_SHA256" "$RELEASE_ID" "$SOURCE_COMMIT" "$verified_entries" "$BROKER_PROTOCOL_CHECK"
  exit 0
fi

for tool in install cp mv rm chown chmod readlink systemctl timeout; do
  command -v "$tool" >/dev/null 2>&1 || fail "required host installation tool is missing: $tool"
done

ensure_host_directory() {
  local path="$1"
  if [[ -e "$path" || -L "$path" ]]; then
    [[ -d "$path" && ! -L "$path" ]] || fail "host destination is not a real directory: $path"
  else
    install -d -m 0755 -o root -g root "$path"
  fi
  [[ "$(stat -c '%u:%g' "$path")" == 0:0 ]] || fail "host destination is not root-owned: $path"
  [[ -z "$(find "$path" -maxdepth 0 -perm /022 -print -quit)" ]] || fail "host destination is writable by group or world: $path"
}

atomic_file() {
  local source="$1" destination="$2" mode="$3"
  local directory temporary
  directory="$(dirname "$destination")"
  ensure_host_directory "$directory"
  temporary="$(mktemp "$directory/.loom-promotion.XXXXXX")"
  install -m "$mode" -o root -g root "$source" "$temporary"
  sync -f "$temporary" 2>/dev/null || sync
  mv -fT "$temporary" "$destination"
  sync -f "$directory" 2>/dev/null || sync
}

atomic_link() {
  local target="$1" destination="$2"
  local directory temporary_directory temporary
  directory="$(dirname "$destination")"
  ensure_host_directory "$directory"
  temporary_directory="$(mktemp -d "$directory/.loom-link.XXXXXX")"
  temporary="$temporary_directory/link"
  ln -s "$target" "$temporary"
  chown -h root:root "$temporary"
  mv -fT "$temporary" "$destination"
  rmdir "$temporary_directory"
  sync -f "$directory" 2>/dev/null || sync
}

compare_release() {
  local source="$1" destination="$2"
  local source_count destination_count source_path relative destination_path
  source_count="$(find "$source" -mindepth 1 -printf . | wc -c)"
  destination_count="$(find "$destination" -mindepth 1 -printf . | wc -c)"
  [[ "$source_count" == "$destination_count" ]] || fail 'existing immutable release entry count drifted'
  [[ "$(stat -c '%a' "$source")" == "$(stat -c '%a' "$destination")" ]] ||
    fail 'existing immutable release root mode drifted'
  [[ "$(stat -c '%u:%g' "$destination")" == 0:0 ]] ||
    fail 'existing immutable release root ownership drifted'
  while IFS= read -r -d '' source_path; do
    relative="${source_path#"$source"/}"
    destination_path="$destination/$relative"
    if [[ -d "$source_path" ]]; then
      [[ -d "$destination_path" && ! -L "$destination_path" ]] || fail "existing immutable release directory drifted: $relative"
    else
      [[ -f "$destination_path" && ! -L "$destination_path" ]] || fail "existing immutable release file drifted: $relative"
      [[ "$(sha256_file "$source_path")" == "$(sha256_file "$destination_path")" ]] || fail "existing immutable release content drifted: $relative"
    fi
    [[ "$(stat -c '%a' "$source_path")" == "$(stat -c '%a' "$destination_path")" ]] || fail "existing immutable release mode drifted: $relative"
    [[ "$(stat -c '%u:%g' "$destination_path")" == 0:0 ]] || fail "existing immutable release ownership drifted: $relative"
  done < <(find "$source" -mindepth 1 -print0 | sort -z)
}

RELEASE_PARENT=/usr/lib/sounio/loom/releases
HOST_RELEASE="$RELEASE_PARENT/$RELEASE_ID"
BROKER_LINK=/usr/libexec/sounio/loom-kernel-principal-broker
STABLE_PATHS=(
  /etc/systemd/system/sounio-loom-principal-broker.socket
  /etc/systemd/system/sounio-loom-principal-broker.service
  /etc/sounio/loom-principal-broker.conf
  /usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_BOOTSTRAP_V1.md
  /usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md
  /usr/share/doc/sounio/loom/INVOCATION_CELL_MATERIAL_ADMISSION_V1.md
)
SOURCE_PATHS=(
  "$ROOTFS/etc/systemd/system/sounio-loom-principal-broker.socket"
  "$ROOTFS/etc/systemd/system/sounio-loom-principal-broker.service"
  "$ROOTFS/etc/sounio/loom-principal-broker.conf"
  "$ROOTFS/usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_BOOTSTRAP_V1.md"
  "$ROOTFS/usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md"
  "$ROOTFS/usr/share/doc/sounio/loom/INVOCATION_CELL_MATERIAL_ADMISSION_V1.md"
)
STABLE_MODES=(0644 0644 0600 0444 0444 0444)

TRANSACTION="$WORK/transaction"
BACKUP="$TRANSACTION/backup"
mkdir -p "$BACKUP"
previous_link=ABSENT
if [[ -L "$BROKER_LINK" ]]; then
  previous_link="$(readlink "$BROKER_LINK")"
elif [[ -e "$BROKER_LINK" ]]; then
  fail 'existing stable broker path is not a symlink'
fi
for index in "${!STABLE_PATHS[@]}"; do
  destination="${STABLE_PATHS[$index]}"
  if [[ -e "$destination" || -L "$destination" ]]; then
    [[ -f "$destination" && ! -L "$destination" ]] || fail "existing stable path is not a regular file: $destination"
    backup_path="$BACKUP$destination"
    mkdir -p "$(dirname "$backup_path")"
    cp -a "$destination" "$backup_path"
  fi
done

promotion_started=0
promotion_committed=0
rollback() {
  local status="$1"
  trap - EXIT
  set +e
  systemctl stop sounio-loom-principal-broker.socket sounio-loom-principal-broker.service >/dev/null 2>&1
  for index in "${!STABLE_PATHS[@]}"; do
    destination="${STABLE_PATHS[$index]}"
    backup_path="$BACKUP$destination"
    if [[ -f "$backup_path" ]]; then
      atomic_file "$backup_path" "$destination" "${STABLE_MODES[$index]}"
    else
      rm -f "$destination"
    fi
  done
  if [[ "$previous_link" == ABSENT ]]; then
    rm -f "$BROKER_LINK"
  else
    atomic_link "$previous_link" "$BROKER_LINK"
  fi
  systemctl daemon-reload >/dev/null 2>&1
  if [[ "$previous_link" != ABSENT ]]; then
    systemctl enable --now sounio-loom-principal-broker.socket >/dev/null 2>&1
  fi
  printf 'promote-loom-host-capsule: ROLLBACK status=%s previous_link=%s\n' "$status" "$previous_link" >&2
  cleanup
  exit "$status"
}
on_exit() {
  local status=$?
  if [[ $promotion_started -eq 1 && $promotion_committed -eq 0 ]]; then
    rollback "$status"
  fi
  cleanup
  exit "$status"
}
trap on_exit EXIT

ensure_host_directory /usr
ensure_host_directory /usr/lib
ensure_host_directory /usr/lib/sounio
ensure_host_directory /usr/lib/sounio/loom
ensure_host_directory "$RELEASE_PARENT"
ensure_host_directory /usr/libexec
ensure_host_directory /usr/libexec/sounio
ensure_host_directory /etc
ensure_host_directory /etc/sounio
ensure_host_directory /etc/systemd
ensure_host_directory /etc/systemd/system
ensure_host_directory /usr/share
ensure_host_directory /usr/share/doc
ensure_host_directory /usr/share/doc/sounio
ensure_host_directory /usr/share/doc/sounio/loom

if [[ -e "$HOST_RELEASE" || -L "$HOST_RELEASE" ]]; then
  [[ -d "$HOST_RELEASE" && ! -L "$HOST_RELEASE" ]] || fail 'existing immutable release path is not a real directory'
  compare_release "$RELEASE" "$HOST_RELEASE"
else
  release_stage="$(mktemp -d "$RELEASE_PARENT/.${RELEASE_ID}.stage.XXXXXX")"
  cp -a "$RELEASE/." "$release_stage/"
  chown -R root:root "$release_stage"
  chmod 0555 "$release_stage"
  compare_release "$RELEASE" "$release_stage"
  sync -f "$release_stage" 2>/dev/null || sync
  mv "$release_stage" "$HOST_RELEASE"
  sync -f "$RELEASE_PARENT" 2>/dev/null || sync
fi

promotion_started=1
systemctl stop sounio-loom-principal-broker.socket sounio-loom-principal-broker.service >/dev/null 2>&1 || true
for index in "${!STABLE_PATHS[@]}"; do
  atomic_file "${SOURCE_PATHS[$index]}" "${STABLE_PATHS[$index]}" "${STABLE_MODES[$index]}"
done
STABLE_BROKER_TARGET="$(record_value "$MANIFEST" stable_broker_target)"
[[ "$STABLE_BROKER_TARGET" == "/usr/lib/sounio/loom/releases/$RELEASE_ID/loom-kernel-principal-broker" ]] ||
  fail 'stable broker target does not bind the promoted release'
atomic_link "$STABLE_BROKER_TARGET" "$BROKER_LINK"
systemctl daemon-reload
systemctl enable --now sounio-loom-principal-broker.socket

set +e
HOST_GATE_OUTPUT="$(timeout --signal=TERM --kill-after=5s 45s "$HOST_GATE" 2>&1)"
HOST_GATE_STATUS=$?
set -e
[[ $HOST_GATE_STATUS -eq 0 ]] ||
  fail "host activation gate failed or timed out status=$HOST_GATE_STATUS output=$HOST_GATE_OUTPUT"
[[ "$HOST_GATE_OUTPUT" == 'sounio-loom-kernel-principal-broker-host-gate: HOST_ACTIVATION_PASS '* ]] ||
  fail "host activation gate did not pass: $HOST_GATE_OUTPUT"
promotion_committed=1
trap - EXIT
cleanup
trap - EXIT

printf '%s\n' "$HOST_GATE_OUTPUT"
printf 'LOOM_HOST_PROMOTION PASS archive_sha256=%s release=%s source_commit=%s transport_authority=false rollback=armed host_gate=PASS parity_open=false claim_ready=false launch=closed material_broker=false material_capsule=false material_invocation=false\n' \
  "$ACTUAL_SHA256" "$RELEASE_ID" "$SOURCE_COMMIT"
