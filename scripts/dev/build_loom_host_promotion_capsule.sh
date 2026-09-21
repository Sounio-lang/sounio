#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
INSTALLER="$ROOT_DIR/scripts/dev/install_loom_kernel_principal_broker.sh"
PROMOTER="$ROOT_DIR/scripts/dev/promote_loom_host_capsule.sh"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_kernel_principal_broker_host_gate.sh"

fail() {
  printf 'build-loom-host-promotion-capsule: REFUSE reason=%s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s --output ABSOLUTE_PATH\n' "$0" >&2
  exit 64
}

field() {
  local line="$1"
  local key="$2"
  local token
  for token in $line; do
    if [[ "$token" == "$key="* ]]; then
      printf '%s\n' "${token#*=}"
      return 0
    fi
  done
  fail "installer output omitted field: $key"
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

OUTPUT=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      [[ $# -ge 2 ]] || usage
      OUTPUT="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ -n "$OUTPUT" && "$OUTPUT" == /* ]] || usage
[[ ! -L "$OUTPUT" ]] || fail 'output path must not be a symlink'
mkdir -p "$(dirname "$OUTPUT")"

for tool in git sha256sum stat install mktemp find sort tar sync cut readlink rm chmod mv; do
  command -v "$tool" >/dev/null 2>&1 || fail "required packaging tool is missing: $tool"
done
for input in "$INSTALLER" "$PROMOTER" "$HOST_GATE"; do
  [[ -f "$input" && ! -L "$input" && -x "$input" ]] ||
    fail "required capsule input is absent, linked, or non-executable: $input"
done

SOURCE_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail 'source commit is not a full Git object id'
SOURCE_TREE_STATE=CLEAN
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal)" ]]; then
  [[ "${SOUNIO_LOOM_ALLOW_DIRTY_CAPSULE:-0}" == 1 ]] ||
    fail 'source tree is dirty; commit the capsule implementation before promotion'
  SOURCE_TREE_STATE=DIRTY_UNPROMOTABLE
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-promotion-build.XXXXXX")"
cleanup() {
  find "$WORK" -type d -exec chmod u+rwx {} + 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

CAPSULE="$WORK/capsule-v1"
ROOTFS="$CAPSULE/rootfs"
META="$CAPSULE/meta"
mkdir -p "$ROOTFS" "$META"

install_output="$($INSTALLER --staging-root "$ROOTFS")"
[[ "$install_output" == 'LOOM_KERNEL_PRINCIPAL_BROKER_INSTALL PASS '* ]] ||
  fail "staging installer did not pass: $install_output"
[[ "$(field "$install_output" mode)" == STAGING_ONLY ]] || fail 'staging installer claimed host activation'
[[ "$(field "$install_output" activated)" == false ]] || fail 'staging installer activated a service'
[[ "$(field "$install_output" material_broker)" == false ]] || fail 'staging installer opened material broker'
[[ "$(field "$install_output" material_capsule)" == false ]] || fail 'staging installer opened material capsule'
[[ "$(field "$install_output" material_invocation)" == false ]] || fail 'staging installer opened material invocation'
[[ "$(field "$install_output" material_grant)" == false ]] || fail 'staging installer opened material grant'
[[ "$(field "$install_output" material_execution)" == false ]] || fail 'staging installer opened material execution'
[[ "$(field "$install_output" barrier_release)" == false ]] || fail 'staging installer released host barrier'
[[ "$(field "$install_output" resident_action_9030_attached)" == true ]] ||
  fail 'staging installer omitted resident action 9030'

RELEASE_ID="$(field "$install_output" release)"
[[ "$RELEASE_ID" =~ ^9030-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}-[0-9a-f]{16}$ ]] ||
  fail 'staging installer returned an invalid immutable release identity'
RELEASE="$ROOTFS/usr/lib/sounio/loom/releases/$RELEASE_ID"
RECEIPT="$RELEASE/install.receipt.v1"
[[ -f "$RECEIPT" && ! -L "$RECEIPT" ]] || fail 'staged install receipt is absent'

BROKER_LINK="$ROOTFS/usr/libexec/sounio/loom-kernel-principal-broker"
[[ -L "$BROKER_LINK" ]] || fail 'staged stable broker link is absent'
BROKER_TARGET="$(readlink "$BROKER_LINK")"
[[ "$BROKER_TARGET" == "/usr/lib/sounio/loom/releases/$RELEASE_ID/loom-kernel-principal-broker" ]] ||
  fail 'staged stable broker target escaped the immutable release'
rm "$BROKER_LINK"

install -m 0555 "$PROMOTER" "$META/promote_loom_host_capsule.sh"
install -m 0555 "$HOST_GATE" "$META/sounio_loom_kernel_principal_broker_host_gate.sh"

ENTRIES="$META/payload.entries.v1"
: > "$ENTRIES"
entry_count=0
while IFS= read -r -d '' path; do
  relative="${path#"$ROOTFS"/}"
  [[ "$relative" != "$path" && "$relative" =~ ^[A-Za-z0-9._/-]+$ ]] ||
    fail "payload path is not representable in capsule v1: $relative"
  mode="$(stat -c '%a' "$path")"
  if [[ -d "$path" && ! -L "$path" ]]; then
    printf 'D|%s|-|%s\n' "$mode" "$relative" >> "$ENTRIES"
  elif [[ -f "$path" && ! -L "$path" ]]; then
    printf 'F|%s|%s|%s\n' "$mode" "$(sha256_file "$path")" "$relative" >> "$ENTRIES"
  else
    fail "payload contains a non-regular, non-directory entry: $relative"
  fi
  entry_count=$((entry_count + 1))
done < <(find "$ROOTFS" -mindepth 1 -print0 | sort -z)
chmod 0444 "$ENTRIES"

ENTRIES_SHA256="$(sha256_file "$ENTRIES")"
PROMOTER_SHA256="$(sha256_file "$META/promote_loom_host_capsule.sh")"
HOST_GATE_SHA256="$(sha256_file "$META/sounio_loom_kernel_principal_broker_host_gate.sh")"
RECEIPT_SHA256="$(sha256_file "$RECEIPT")"
LEASE_MANIFEST_SHA256="$(record_value "$RECEIPT" lease_manifest_sha256)"
LEASE_AUTHORITY_SHA256="$(record_value "$RECEIPT" lease_authority_sha256)"
CAPSULE_MANIFEST_SHA256="$(record_value "$RECEIPT" capsule_manifest_sha256)"
CAPSULE_AUTHORITY_SHA256="$(record_value "$RECEIPT" capsule_authority_sha256)"
INVOCATION_MANIFEST_SHA256="$(record_value "$RECEIPT" invocation_manifest_sha256)"
INVOCATION_AUTHORITY_SHA256="$(record_value "$RECEIPT" invocation_authority_sha256)"
EXEC_GRANT_MANIFEST_SHA256="$(record_value "$RECEIPT" exec_grant_manifest_sha256)"
RESIDENT_MANIFEST_SHA256="$(record_value "$RECEIPT" resident_manifest_sha256)"
RESIDENT_RUNTIME_SHA256="$(record_value "$RECEIPT" resident_runtime_sha256)"
BROKER_SHA256="$(record_value "$RECEIPT" broker_sha256)"

cat > "$META/capsule.manifest.v1" <<EOF
schema=loom-host-promotion-capsule-v1
stage=SEMANTICS_FROZEN
source_commit=$SOURCE_COMMIT
source_tree_state=$SOURCE_TREE_STATE
semantic_producer=Sounio
semantic_role=SEMANTIC_AUTHORITY
semantic_actions=9027+9028+9029+9030
transport_producer=Bash+GNU-tar
transport_role=MECHANICAL_PACKAGING
transport_authority=false
material_producer=C++20
material_role=MATERIAL_PARITY
material_transitory=true
release_id=$RELEASE_ID
stable_broker_target=$BROKER_TARGET
payload_entry_count=$entry_count
payload_entries_sha256=$ENTRIES_SHA256
promoter_sha256=$PROMOTER_SHA256
host_gate_sha256=$HOST_GATE_SHA256
install_receipt_sha256=$RECEIPT_SHA256
lease_manifest_sha256=$LEASE_MANIFEST_SHA256
lease_authority_sha256=$LEASE_AUTHORITY_SHA256
capsule_manifest_sha256=$CAPSULE_MANIFEST_SHA256
capsule_authority_sha256=$CAPSULE_AUTHORITY_SHA256
invocation_manifest_sha256=$INVOCATION_MANIFEST_SHA256
invocation_authority_sha256=$INVOCATION_AUTHORITY_SHA256
exec_grant_manifest_sha256=$EXEC_GRANT_MANIFEST_SHA256
resident_manifest_sha256=$RESIDENT_MANIFEST_SHA256
resident_runtime_sha256=$RESIDENT_RUNTIME_SHA256
broker_sha256=$BROKER_SHA256
parity_open=false
claim_ready=false
launch_open=false
recycle_open=false
material_broker=false
material_capsule=false
material_invocation=false
resident_action_9030_attached=true
decision_transport_material=true
material_grant=false
material_execution=false
barrier_release=false
same_uid_peer_isolation=false
EOF
chmod 0444 "$META/capsule.manifest.v1"

ARCHIVE_STAGE="$WORK/capsule.tar"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  --format=posix --pax-option=delete=atime,delete=ctime \
  -C "$WORK" -cf "$ARCHIVE_STAGE" capsule-v1
ARCHIVE_SHA256="$(sha256_file "$ARCHIVE_STAGE")"

OUTPUT_STAGE="$(mktemp "$(dirname "$OUTPUT")/.loom-capsule.XXXXXX")"
install -m 0600 "$ARCHIVE_STAGE" "$OUTPUT_STAGE"
sync -f "$OUTPUT_STAGE" 2>/dev/null || sync
mv -fT "$OUTPUT_STAGE" "$OUTPUT"
sync -f "$(dirname "$OUTPUT")" 2>/dev/null || sync
printf '%s  %s\n' "$ARCHIVE_SHA256" "$(basename "$OUTPUT")" > "$OUTPUT.sha256"
chmod 0600 "$OUTPUT.sha256"

printf 'LOOM_HOST_PROMOTION_CAPSULE_BUILD PASS archive=%s archive_sha256=%s release=%s source_commit=%s source_tree_state=%s payload_entries=%s payload_entries_sha256=%s promoter_sha256=%s host_gate_sha256=%s semantic_producer=Sounio semantic_role=SEMANTIC_AUTHORITY transport_role=MECHANICAL_PACKAGING resident_action_9030_attached=true decision_transport_material=true parity_open=false claim_ready=false launch=closed material_broker=false material_capsule=false material_invocation=false material_grant=false material_execution=false barrier_release=false\n' \
  "$OUTPUT" "$ARCHIVE_SHA256" "$RELEASE_ID" "$SOURCE_COMMIT" "$SOURCE_TREE_STATE" \
  "$entry_count" "$ENTRIES_SHA256" "$PROMOTER_SHA256" "$HOST_GATE_SHA256"
