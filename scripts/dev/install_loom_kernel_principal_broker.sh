#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
MANIFEST="$ROOT_DIR/tools/loom/kernel_principal_lease_authority.freeze.v1"
CAPSULE_MANIFEST="$ROOT_DIR/tools/loom/kernel_principal_capsule_authority.freeze.v1"
INVOCATION_MANIFEST="$ROOT_DIR/tools/loom/kernel_invocation_cell_authority.freeze.v1"
BROKER_SOURCE="$ROOT_DIR/tools/loom/src/loom_kernel_principal_broker.cpp"
SOCKET_UNIT="$ROOT_DIR/tools/loom/systemd/sounio-loom-principal-broker.socket"
SERVICE_UNIT="$ROOT_DIR/tools/loom/systemd/sounio-loom-principal-broker.service"
BOOTSTRAP_DOC="$ROOT_DIR/tools/loom/HOST_KERNEL_PRINCIPAL_BROKER_BOOTSTRAP_V1.md"
INSTALL_DOC="$ROOT_DIR/tools/loom/HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md"
INVOCATION_DOC="$ROOT_DIR/tools/loom/INVOCATION_CELL_MATERIAL_ADMISSION_V1.md"

fail() {
  printf 'install-loom-kernel-principal-broker: REFUSE reason=%s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s (--host-install | --staging-root ABSOLUTE_PATH)\n' "$0" >&2
  exit 64
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
      [[ -z "$found" ]] || fail "duplicate manifest field: $key"
      found="$value"
    fi
  done < "$path"
  [[ -n "$found" ]] || fail "manifest omitted field: $key"
  printf '%s\n' "$found"
}

manifest_value() {
  record_value "$MANIFEST" "$1"
}

capsule_manifest_value() {
  record_value "$CAPSULE_MANIFEST" "$1"
}

invocation_manifest_value() {
  record_value "$INVOCATION_MANIFEST" "$1"
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

mode_of() {
  stat -c '%a' "$1"
}

sync_path() {
  sync -f "$1" 2>/dev/null || sync
}

atomic_file() {
  local source="$1"
  local destination="$2"
  local mode="$3"
  local directory temporary
  directory="$(dirname "$destination")"
  mkdir -p "$directory"
  [[ ! -L "$directory" ]] || fail "destination directory is a symlink: $directory"
  temporary="$(mktemp "$directory/.loom-install.XXXXXX")"
  install -m "$mode" "$source" "$temporary"
  if [[ "$INSTALL_MODE" == HOST ]]; then
    chown root:root "$temporary"
  fi
  sync_path "$temporary"
  mv -fT "$temporary" "$destination"
  sync_path "$directory"
}

atomic_text() {
  local contents="$1"
  local destination="$2"
  local mode="$3"
  local staged
  staged="$(mktemp "${TMPDIR:-/tmp}/loom-install-text.XXXXXX")"
  printf '%s' "$contents" > "$staged"
  atomic_file "$staged" "$destination" "$mode"
  rm -f "$staged"
}

atomic_symlink() {
  local target="$1"
  local destination="$2"
  local directory temporary_directory temporary
  directory="$(dirname "$destination")"
  mkdir -p "$directory"
  temporary_directory="$(mktemp -d "$directory/.loom-link.XXXXXX")"
  temporary="$temporary_directory/link"
  ln -s "$target" "$temporary"
  if [[ "$INSTALL_MODE" == HOST ]]; then
    chown -h root:root "$temporary"
  fi
  mv -fT "$temporary" "$destination"
  rmdir "$temporary_directory"
  sync_path "$directory"
}

ensure_destination_directory() {
  local relative="$1"
  local path="$DEST_ROOT$relative"
  if [[ -e "$path" || -L "$path" ]]; then
    [[ -d "$path" && ! -L "$path" ]] || fail "destination ancestor is not a real directory: $path"
  else
    mkdir "$path"
  fi
  if [[ "$INSTALL_MODE" == HOST ]]; then
    [[ "$(stat -c '%u:%g' "$path")" == 0:0 ]] || fail "destination ancestor is not root-owned: $path"
    [[ -z "$(find "$path" -maxdepth 0 -perm /022 -print -quit)" ]] ||
      fail "destination ancestor is group/world writable: $path"
  fi
}

[[ $# -ge 1 ]] || usage
case "$1" in
  --host-install)
    [[ $# -eq 1 ]] || usage
    INSTALL_MODE=HOST
    DEST_ROOT=/
    ;;
  --staging-root)
    [[ $# -eq 2 ]] || usage
    INSTALL_MODE=STAGING_ONLY
    DEST_ROOT="$2"
    [[ "$DEST_ROOT" == /* ]] || fail 'staging root must be absolute'
    [[ "$DEST_ROOT" != / ]] || fail 'staging mode refuses the real root'
    [[ ! -L "$DEST_ROOT" ]] || fail 'staging root must not be a symlink'
    mkdir -p "$DEST_ROOT"
    DEST_ROOT="$(cd "$DEST_ROOT" && pwd -P)"
    ;;
  *) usage ;;
esac

for tool in sha256sum stat install mktemp sync; do
  command -v "$tool" >/dev/null 2>&1 || fail "required installation tool is missing: $tool"
done
for input in "$MANIFEST" "$CAPSULE_MANIFEST" "$INVOCATION_MANIFEST" "$BROKER_SOURCE" \
  "$SOCKET_UNIT" "$SERVICE_UNIT" "$BOOTSTRAP_DOC" "$INSTALL_DOC" "$INVOCATION_DOC"; do
  [[ -f "$input" && ! -L "$input" ]] || fail "installation input is missing or a symlink: $input"
done

if [[ "$INSTALL_MODE" == HOST ]]; then
  [[ "$(id -u)" == 0 && "$(id -g)" == 0 ]] || fail 'host install requires root identity'
  [[ "$(tr -d '\n' < /proc/1/comm)" == systemd ]] || fail 'host install requires PID 1 systemd'
  [[ -d /run/systemd/system ]] || fail 'host install requires the systemd runtime boundary'
  command -v systemctl >/dev/null 2>&1 || fail 'host install requires systemctl'
fi

for relative in /usr /usr/lib /usr/lib/sounio /usr/lib/sounio/loom \
  /usr/lib/sounio/loom/releases /usr/libexec /usr/libexec/sounio \
  /usr/share /usr/share/doc /usr/share/doc/sounio /usr/share/doc/sounio/loom \
  /etc /etc/sounio /etc/systemd /etc/systemd/system; do
  ensure_destination_directory "$relative"
done

[[ "$(manifest_value schema)" == loom-kernel-principal-lease-authority-freeze-v1 ]] ||
  fail 'manifest schema is not the frozen action 9027 schema'
[[ "$(manifest_value stage)" == SEMANTICS_FROZEN ]] || fail 'manifest stage is not frozen'
[[ "$(manifest_value producing_language)" == Sounio ]] || fail 'manifest producer is not Sounio'
[[ "$(manifest_value language_role)" == SEMANTIC_AUTHORITY ]] ||
  fail 'manifest role is not semantic authority'
[[ "$(manifest_value action)" == 9027 ]] || fail 'manifest action is not 9027'
[[ "$(manifest_value material_broker)" == false ]] || fail 'bootstrap manifest opened material broker'
[[ "$(capsule_manifest_value schema)" == loom-kernel-principal-capsule-authority-freeze-v1 ]] ||
  fail 'capsule manifest schema is not the frozen action 9028 schema'
[[ "$(capsule_manifest_value stage)" == SEMANTICS_FROZEN ]] ||
  fail 'capsule manifest stage is not frozen'
[[ "$(capsule_manifest_value producing_language)" == Sounio ]] ||
  fail 'capsule manifest producer is not Sounio'
[[ "$(capsule_manifest_value language_role)" == SEMANTIC_AUTHORITY ]] ||
  fail 'capsule manifest role is not semantic authority'
[[ "$(capsule_manifest_value action)" == 9028 ]] || fail 'capsule manifest action is not 9028'
[[ "$(capsule_manifest_value parent_action)" == 9027 ]] ||
  fail 'capsule manifest parent action is not 9027'
[[ "$(capsule_manifest_value material_capsule)" == false ]] ||
  fail 'capsule manifest opened material capsule'
[[ "$(invocation_manifest_value schema)" == loom-kernel-invocation-cell-authority-freeze-v1 ]] ||
  fail 'InvocationCell manifest schema is not the frozen action 9029 schema'
[[ "$(invocation_manifest_value stage)" == SEMANTICS_FROZEN ]] ||
  fail 'InvocationCell manifest stage is not frozen'
[[ "$(invocation_manifest_value producing_language)" == Sounio ]] ||
  fail 'InvocationCell manifest producer is not Sounio'
[[ "$(invocation_manifest_value language_role)" == SEMANTIC_AUTHORITY ]] ||
  fail 'InvocationCell manifest role is not semantic authority'
[[ "$(invocation_manifest_value action)" == 9029 ]] ||
  fail 'InvocationCell manifest action is not 9029'
[[ "$(invocation_manifest_value material_invocation)" == false ]] ||
  fail 'InvocationCell manifest opened material invocation'

bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_principal_lease_authority_freeze_selftest.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_principal_capsule_authority_freeze_selftest.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_kernel_invocation_cell_authority_freeze_selftest.sh" >/dev/null

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-kernel-principal-install.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
AUTHORITY_BUILD="$WORK/sounio-loom-kernel-principal-lease-authority-runtime"
CAPSULE_AUTHORITY_BUILD="$WORK/sounio-loom-kernel-principal-capsule-authority-runtime"
INVOCATION_AUTHORITY_BUILD="$WORK/sounio-loom-kernel-invocation-cell-authority-runtime"
BROKER_BUILD="$WORK/loom-kernel-principal-broker"
SOUNIO_LOOM_KERNEL_PRINCIPAL_LEASE_OUTPUT="$AUTHORITY_BUILD" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_principal_lease_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_CAPSULE_OUTPUT="$CAPSULE_AUTHORITY_BUILD" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_principal_capsule_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_INVOCATION_CELL_OUTPUT="$INVOCATION_AUTHORITY_BUILD" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_invocation_cell_authority.sh" >/dev/null
SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BROKER_BUILD" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null

MANIFEST_SHA256="$(sha256_file "$MANIFEST")"
AUTHORITY_SHA256="$(sha256_file "$AUTHORITY_BUILD")"
CAPSULE_MANIFEST_SHA256="$(sha256_file "$CAPSULE_MANIFEST")"
CAPSULE_AUTHORITY_SHA256="$(sha256_file "$CAPSULE_AUTHORITY_BUILD")"
INVOCATION_MANIFEST_SHA256="$(sha256_file "$INVOCATION_MANIFEST")"
INVOCATION_AUTHORITY_SHA256="$(sha256_file "$INVOCATION_AUTHORITY_BUILD")"
BROKER_SHA256="$(sha256_file "$BROKER_BUILD")"
BROKER_SOURCE_SHA256="$(sha256_file "$BROKER_SOURCE")"
INSTALLER_SHA256="$(sha256_file "$ROOT_DIR/scripts/dev/install_loom_kernel_principal_broker.sh")"
SOCKET_UNIT_SHA256="$(sha256_file "$SOCKET_UNIT")"
SERVICE_UNIT_SHA256="$(sha256_file "$SERVICE_UNIT")"
BOOTSTRAP_DOC_SHA256="$(sha256_file "$BOOTSTRAP_DOC")"
INSTALL_DOC_SHA256="$(sha256_file "$INSTALL_DOC")"
INVOCATION_DOC_SHA256="$(sha256_file "$INVOCATION_DOC")"
[[ "$MANIFEST_SHA256" == 7bb5bbf30106d269644b0f9e6d80ee09f43eecf0e4a840bc3f429cfb6eca7cb5 ]] ||
  fail 'frozen manifest hash drifted from the broker contract'
[[ "$AUTHORITY_SHA256" == "$(manifest_value executable_sha256)" ]] ||
  fail 'source-fresh Sounio authority hash differs from frozen manifest'
[[ "$CAPSULE_MANIFEST_SHA256" == 76ac860306c8cc00517f81f3fe2a4a2742a1cd4b9c4b4bb34b144b25fbcdf26f ]] ||
  fail 'frozen capsule manifest hash drifted from the broker contract'
[[ "$CAPSULE_AUTHORITY_SHA256" == "$(capsule_manifest_value executable_sha256)" ]] ||
  fail 'source-fresh Sounio capsule authority hash differs from frozen manifest'
[[ "$INVOCATION_MANIFEST_SHA256" == 61918604bf177753c6141f6cd0f05d342a1869ab8fc08d187306a481de33d70e ]] ||
  fail 'frozen InvocationCell manifest hash drifted from the broker contract'
[[ "$INVOCATION_AUTHORITY_SHA256" == "$(invocation_manifest_value executable_sha256)" ]] ||
  fail 'source-fresh Sounio InvocationCell authority hash differs from frozen manifest'

BUNDLE_RECORD="installer_sha256=$INSTALLER_SHA256
socket_unit_sha256=$SOCKET_UNIT_SHA256
service_unit_sha256=$SERVICE_UNIT_SHA256
bootstrap_doc_sha256=$BOOTSTRAP_DOC_SHA256
install_doc_sha256=$INSTALL_DOC_SHA256
invocation_doc_sha256=$INVOCATION_DOC_SHA256
"
BUNDLE_SHA256="$(printf '%s' "$BUNDLE_RECORD" | sha256sum | cut -d ' ' -f 1)"
RELEASE_ID="9029-${MANIFEST_SHA256:0:16}-${CAPSULE_MANIFEST_SHA256:0:16}-${INVOCATION_MANIFEST_SHA256:0:16}-${BROKER_SHA256:0:16}-${BUNDLE_SHA256:0:16}"
RELEASE_PARENT="$DEST_ROOT/usr/lib/sounio/loom/releases"
RELEASE_DIR="$RELEASE_PARENT/$RELEASE_ID"
RELEASE_STAGE="$RELEASE_PARENT/.${RELEASE_ID}.stage.$$"
mkdir -p "$RELEASE_PARENT"
[[ ! -L "$RELEASE_PARENT" ]] || fail 'release parent must not be a symlink'
RECEIPT="schema=loom-kernel-principal-broker-install-receipt-v1
release_id=$RELEASE_ID
semantic_actions=9027+9028+9029
semantic_producer=Sounio
semantic_role=SEMANTIC_AUTHORITY
lease_manifest_sha256=$MANIFEST_SHA256
lease_authority_sha256=$AUTHORITY_SHA256
capsule_manifest_sha256=$CAPSULE_MANIFEST_SHA256
capsule_authority_sha256=$CAPSULE_AUTHORITY_SHA256
invocation_manifest_sha256=$INVOCATION_MANIFEST_SHA256
invocation_authority_sha256=$INVOCATION_AUTHORITY_SHA256
material_producer=C++20
material_role=MATERIAL_PARITY
material_transitory=true
broker_source_sha256=$BROKER_SOURCE_SHA256
broker_sha256=$BROKER_SHA256
installer_sha256=$INSTALLER_SHA256
socket_unit_sha256=$SOCKET_UNIT_SHA256
service_unit_sha256=$SERVICE_UNIT_SHA256
bootstrap_doc_sha256=$BOOTSTRAP_DOC_SHA256
install_doc_sha256=$INSTALL_DOC_SHA256
invocation_doc_sha256=$INVOCATION_DOC_SHA256
bundle_sha256=$BUNDLE_SHA256
admission_open=true
launch_open=false
recycle_open=false
material_broker=false
material_capsule=false
material_invocation=false
"
RECEIPT_SHA256="$(printf '%s' "$RECEIPT" | sha256sum | cut -d ' ' -f 1)"

if [[ -e "$RELEASE_DIR" ]]; then
  [[ -d "$RELEASE_DIR" && ! -L "$RELEASE_DIR" ]] || fail 'existing release is not one directory'
  [[ "$(mode_of "$RELEASE_DIR")" == 555 ]] || fail 'existing immutable release directory mode drifted'
  [[ "$(sha256_file "$RELEASE_DIR/kernel_principal_lease_authority.freeze.v1")" == "$MANIFEST_SHA256" ]] ||
    fail 'existing immutable release manifest drifted'
  [[ "$(mode_of "$RELEASE_DIR/kernel_principal_lease_authority.freeze.v1")" == 444 ]] ||
    fail 'existing immutable release manifest mode drifted'
  [[ "$(sha256_file "$RELEASE_DIR/sounio-loom-kernel-principal-lease-authority-runtime")" == "$AUTHORITY_SHA256" ]] ||
    fail 'existing immutable release authority drifted'
  [[ "$(mode_of "$RELEASE_DIR/sounio-loom-kernel-principal-lease-authority-runtime")" == 555 ]] ||
    fail 'existing immutable release authority mode drifted'
  [[ "$(sha256_file "$RELEASE_DIR/kernel_principal_capsule_authority.freeze.v1")" == "$CAPSULE_MANIFEST_SHA256" ]] ||
    fail 'existing immutable release capsule manifest drifted'
  [[ "$(mode_of "$RELEASE_DIR/kernel_principal_capsule_authority.freeze.v1")" == 444 ]] ||
    fail 'existing immutable release capsule manifest mode drifted'
  [[ "$(sha256_file "$RELEASE_DIR/sounio-loom-kernel-principal-capsule-authority-runtime")" == "$CAPSULE_AUTHORITY_SHA256" ]] ||
    fail 'existing immutable release capsule authority drifted'
  [[ "$(mode_of "$RELEASE_DIR/sounio-loom-kernel-principal-capsule-authority-runtime")" == 555 ]] ||
    fail 'existing immutable release capsule authority mode drifted'
  [[ "$(sha256_file "$RELEASE_DIR/kernel_invocation_cell_authority.freeze.v1")" == "$INVOCATION_MANIFEST_SHA256" ]] ||
    fail 'existing immutable release InvocationCell manifest drifted'
  [[ "$(mode_of "$RELEASE_DIR/kernel_invocation_cell_authority.freeze.v1")" == 444 ]] ||
    fail 'existing immutable release InvocationCell manifest mode drifted'
  [[ "$(sha256_file "$RELEASE_DIR/sounio-loom-kernel-invocation-cell-authority-runtime")" == "$INVOCATION_AUTHORITY_SHA256" ]] ||
    fail 'existing immutable release InvocationCell authority drifted'
  [[ "$(mode_of "$RELEASE_DIR/sounio-loom-kernel-invocation-cell-authority-runtime")" == 555 ]] ||
    fail 'existing immutable release InvocationCell authority mode drifted'
  [[ "$(sha256_file "$RELEASE_DIR/loom-kernel-principal-broker")" == "$BROKER_SHA256" ]] ||
    fail 'existing immutable release broker drifted'
  [[ "$(mode_of "$RELEASE_DIR/loom-kernel-principal-broker")" == 555 ]] ||
    fail 'existing immutable release broker mode drifted'
  [[ "$(mode_of "$RELEASE_DIR/install.receipt.v1")" == 444 ]] ||
    fail 'existing immutable release receipt mode drifted'
  [[ "$(sha256_file "$RELEASE_DIR/install.receipt.v1")" == "$RECEIPT_SHA256" ]] ||
    fail 'existing immutable release receipt drifted'
  [[ "$(sha256_file "$RELEASE_DIR/HOST_KERNEL_PRINCIPAL_BROKER_BOOTSTRAP_V1.md")" == "$BOOTSTRAP_DOC_SHA256" ]] ||
    fail 'existing immutable release bootstrap contract drifted'
  [[ "$(sha256_file "$RELEASE_DIR/HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md")" == "$INSTALL_DOC_SHA256" ]] ||
    fail 'existing immutable release install contract drifted'
  [[ "$(sha256_file "$RELEASE_DIR/INVOCATION_CELL_MATERIAL_ADMISSION_V1.md")" == "$INVOCATION_DOC_SHA256" ]] ||
    fail 'existing immutable release InvocationCell admission contract drifted'
  [[ "$(sha256_file "$RELEASE_DIR/install_loom_kernel_principal_broker.sh")" == "$INSTALLER_SHA256" ]] ||
    fail 'existing immutable release installer drifted'
  if [[ "$INSTALL_MODE" == HOST ]]; then
    [[ "$(stat -c '%u:%g' "$RELEASE_DIR" "$RELEASE_DIR"/* | sort -u)" == 0:0 ]] ||
      fail 'existing immutable release ownership drifted'
  fi
else
  mkdir "$RELEASE_STAGE"
  install -m 0555 "$BROKER_BUILD" "$RELEASE_STAGE/loom-kernel-principal-broker"
  install -m 0555 "$AUTHORITY_BUILD" "$RELEASE_STAGE/sounio-loom-kernel-principal-lease-authority-runtime"
  install -m 0555 "$CAPSULE_AUTHORITY_BUILD" \
    "$RELEASE_STAGE/sounio-loom-kernel-principal-capsule-authority-runtime"
  install -m 0555 "$INVOCATION_AUTHORITY_BUILD" \
    "$RELEASE_STAGE/sounio-loom-kernel-invocation-cell-authority-runtime"
  install -m 0444 "$MANIFEST" "$RELEASE_STAGE/kernel_principal_lease_authority.freeze.v1"
  install -m 0444 "$CAPSULE_MANIFEST" "$RELEASE_STAGE/kernel_principal_capsule_authority.freeze.v1"
  install -m 0444 "$INVOCATION_MANIFEST" "$RELEASE_STAGE/kernel_invocation_cell_authority.freeze.v1"
  install -m 0444 "$BOOTSTRAP_DOC" "$RELEASE_STAGE/HOST_KERNEL_PRINCIPAL_BROKER_BOOTSTRAP_V1.md"
  install -m 0444 "$INSTALL_DOC" "$RELEASE_STAGE/HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md"
  install -m 0444 "$INVOCATION_DOC" "$RELEASE_STAGE/INVOCATION_CELL_MATERIAL_ADMISSION_V1.md"
  install -m 0444 "$ROOT_DIR/scripts/dev/install_loom_kernel_principal_broker.sh" \
    "$RELEASE_STAGE/install_loom_kernel_principal_broker.sh"
  printf '%s' "$RECEIPT" > "$RELEASE_STAGE/install.receipt.v1"
  chmod 0444 "$RELEASE_STAGE/install.receipt.v1"
  if [[ "$INSTALL_MODE" == HOST ]]; then
    chown -R root:root "$RELEASE_STAGE"
  fi
  sync_path "$RELEASE_STAGE/loom-kernel-principal-broker"
  sync_path "$RELEASE_STAGE/sounio-loom-kernel-principal-lease-authority-runtime"
  sync_path "$RELEASE_STAGE/sounio-loom-kernel-principal-capsule-authority-runtime"
  sync_path "$RELEASE_STAGE/sounio-loom-kernel-invocation-cell-authority-runtime"
  sync_path "$RELEASE_STAGE/kernel_principal_lease_authority.freeze.v1"
  sync_path "$RELEASE_STAGE/kernel_principal_capsule_authority.freeze.v1"
  sync_path "$RELEASE_STAGE/kernel_invocation_cell_authority.freeze.v1"
  sync_path "$RELEASE_STAGE/install.receipt.v1"
  chmod 0555 "$RELEASE_STAGE"
  sync_path "$RELEASE_STAGE"
  mv "$RELEASE_STAGE" "$RELEASE_DIR"
  sync_path "$RELEASE_PARENT"
fi

if [[ "$INSTALL_MODE" == HOST ]]; then
  systemctl stop sounio-loom-principal-broker.socket sounio-loom-principal-broker.service \
    >/dev/null 2>&1 || true
  ! systemctl is-active --quiet sounio-loom-principal-broker.socket ||
    fail 'existing broker socket could not be stopped'
  ! systemctl is-active --quiet sounio-loom-principal-broker.service ||
    fail 'existing broker service could not be stopped'
fi

BROKER_TARGET="/usr/lib/sounio/loom/releases/$RELEASE_ID/loom-kernel-principal-broker"
MANIFEST_TARGET="/usr/lib/sounio/loom/releases/$RELEASE_ID/kernel_principal_lease_authority.freeze.v1"
AUTHORITY_TARGET="/usr/lib/sounio/loom/releases/$RELEASE_ID/sounio-loom-kernel-principal-lease-authority-runtime"
CAPSULE_MANIFEST_TARGET="/usr/lib/sounio/loom/releases/$RELEASE_ID/kernel_principal_capsule_authority.freeze.v1"
CAPSULE_AUTHORITY_TARGET="/usr/lib/sounio/loom/releases/$RELEASE_ID/sounio-loom-kernel-principal-capsule-authority-runtime"
INVOCATION_MANIFEST_TARGET="/usr/lib/sounio/loom/releases/$RELEASE_ID/kernel_invocation_cell_authority.freeze.v1"
INVOCATION_AUTHORITY_TARGET="/usr/lib/sounio/loom/releases/$RELEASE_ID/sounio-loom-kernel-invocation-cell-authority-runtime"
atomic_symlink "$BROKER_TARGET" "$DEST_ROOT/usr/libexec/sounio/loom-kernel-principal-broker"
atomic_file "$SOCKET_UNIT" "$DEST_ROOT/etc/systemd/system/sounio-loom-principal-broker.socket" 0644
atomic_file "$SERVICE_UNIT" "$DEST_ROOT/etc/systemd/system/sounio-loom-principal-broker.service" 0644
atomic_file "$BOOTSTRAP_DOC" "$DEST_ROOT/usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_BOOTSTRAP_V1.md" 0444
atomic_file "$INSTALL_DOC" "$DEST_ROOT/usr/share/doc/sounio/loom/HOST_KERNEL_PRINCIPAL_BROKER_INSTALL_V1.md" 0444
atomic_file "$INVOCATION_DOC" "$DEST_ROOT/usr/share/doc/sounio/loom/INVOCATION_CELL_MATERIAL_ADMISSION_V1.md" 0444
CONFIG="LOOM_PRINCIPAL_MANIFEST=$MANIFEST_TARGET
LOOM_PRINCIPAL_AUTHORITY=$AUTHORITY_TARGET
LOOM_PRINCIPAL_CAPSULE_MANIFEST=$CAPSULE_MANIFEST_TARGET
LOOM_PRINCIPAL_CAPSULE_AUTHORITY=$CAPSULE_AUTHORITY_TARGET
LOOM_PRINCIPAL_INVOCATION_MANIFEST=$INVOCATION_MANIFEST_TARGET
LOOM_PRINCIPAL_INVOCATION_AUTHORITY=$INVOCATION_AUTHORITY_TARGET
LOOM_PRINCIPAL_JOURNAL=/var/lib/sounio/loom-principal-broker/leases.v1
"
atomic_text "$CONFIG" "$DEST_ROOT/etc/sounio/loom-principal-broker.conf" 0600

if [[ "$INSTALL_MODE" == HOST ]]; then
  chown root:root "$RELEASE_PARENT" "$DEST_ROOT/usr/lib/sounio/loom" \
    "$DEST_ROOT/usr/libexec/sounio" "$DEST_ROOT/etc/sounio" \
    "$DEST_ROOT/usr/share/doc/sounio/loom"
  chmod 0755 "$RELEASE_PARENT" "$DEST_ROOT/usr/lib/sounio/loom" \
    "$DEST_ROOT/usr/libexec/sounio" "$DEST_ROOT/etc/sounio" \
    "$DEST_ROOT/usr/share/doc/sounio/loom"
  systemctl daemon-reload
  systemctl enable --now sounio-loom-principal-broker.socket
fi

printf 'LOOM_KERNEL_PRINCIPAL_BROKER_INSTALL PASS mode=%s release=%s lease_manifest_sha256=%s lease_authority_sha256=%s capsule_manifest_sha256=%s capsule_authority_sha256=%s invocation_manifest_sha256=%s invocation_authority_sha256=%s broker_sha256=%s bundle_sha256=%s activated=%s admission=decision-only material_broker=false material_capsule=false material_invocation=false launch=closed recycle=closed\n' \
  "$INSTALL_MODE" "$RELEASE_ID" "$MANIFEST_SHA256" "$AUTHORITY_SHA256" \
  "$CAPSULE_MANIFEST_SHA256" "$CAPSULE_AUTHORITY_SHA256" "$INVOCATION_MANIFEST_SHA256" \
  "$INVOCATION_AUTHORITY_SHA256" "$BROKER_SHA256" "$BUNDLE_SHA256" \
  "$([[ "$INSTALL_MODE" == HOST ]] && printf true || printf false)"
