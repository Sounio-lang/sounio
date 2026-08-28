#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="${SOUNIO_SOURCE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd -P)}"
PROMOTER="$ROOT_DIR/scripts/dev/promote_loom_host_exec_quorum_capsule.sh"
HOST_GATE="$ROOT_DIR/scripts/ci/sounio_loom_host_exec_quorum_host_gate.sh"

fail() {
  printf 'build-loom-host-exec-quorum-capsule: REFUSE reason=%s\n' "$*" >&2
  exit 70
}

usage() {
  printf 'usage: %s --output ABSOLUTE_PATH\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
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
[[ "$OUTPUT" == /* && ! -L "$OUTPUT" ]] || usage

for tool in git sha256sum stat install mktemp tar find sort chmod sync mv; do
  command -v "$tool" >/dev/null 2>&1 || fail "required packaging tool is absent: $tool"
done
for input in "$PROMOTER" "$HOST_GATE"; do
  [[ -f "$input" && ! -L "$input" && -x "$input" ]] || fail "required capsule input is absent, linked, or non-executable: $input"
done

SOURCE_COMMIT="$(git -C "$ROOT_DIR" rev-parse HEAD)"
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail 'source commit is not canonical'
SOURCE_TREE_STATE=CLEAN
if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal)" ]]; then
  [[ "${SOUNIO_LOOM_ALLOW_DIRTY_CAPSULE:-0}" == 1 ]] || fail 'source tree is dirty'
  SOURCE_TREE_STATE=DIRTY_UNPROMOTABLE
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-host-exec-quorum-capsule.XXXXXX")"
cleanup() {
  find "$WORK" -type d -exec chmod u+rwx {} + 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT
CAPSULE="$WORK/capsule-v1"
RELEASE_STAGE="$WORK/release"
BIN="$RELEASE_STAGE/bin"
DATA="$RELEASE_STAGE/data"
AUTHORITY_ROOT="$RELEASE_STAGE/authority-root"
META="$CAPSULE/meta"
mkdir -p "$BIN" "$DATA" "$AUTHORITY_ROOT/.git" "$META"
chmod 0700 "$AUTHORITY_ROOT/.git"

SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$BIN/loom-kernel-principal-broker" \
  bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null
SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_OUTPUT="$BIN/loom-exec-grant-controller" \
  bash "$ROOT_DIR/scripts/dev/build_loom_exec_grant_controller.sh" >/dev/null
SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_OUTPUT="$BIN/sounio-loom-resident-membrane-runtime-v4" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v4.sh" >/dev/null
SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_INTEGRATED_OUTPUT="$BIN/loom-principal-cell-barrier-integrated" \
  bash "$ROOT_DIR/scripts/dev/build_loom_principal_cell_barrier_integrated.sh" >/dev/null
SOUNIO_LOOM_HOST_EXEC_QUORUM_PRINCIPAL_CELL_OUTPUT="$BIN/loom-host-exec-quorum-principal-cell" \
  bash "$ROOT_DIR/scripts/dev/build_loom_host_exec_quorum_principal_cell.sh" >/dev/null
fixture_runtime="$WORK/sounio-loom-host-exec-quorum-fixture"
SOUNIO_LOOM_HOST_EXEC_QUORUM_FIXTURE_OUTPUT="$fixture_runtime" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_exec_quorum_fixture.sh" >/dev/null
"$fixture_runtime" > "$DATA/host-exec-quorum-fixtures.v1"
[[ "$(sha256_file "$DATA/host-exec-quorum-fixtures.v1")" == 523e132c4ab6a41ade56c2421472b092171627087fe4cf55ba4c74ac1f5d98fe ]] ||
  fail 'source-fresh Sounio fixture bundle drifted'
chmod 0555 "$BIN"/*
chmod 0444 "$DATA/host-exec-quorum-fixtures.v1"

install_root_file() {
  local relative="$1" source="$ROOT_DIR/$1" destination="$AUTHORITY_ROOT/$1" mode=0444
  [[ -f "$source" && ! -L "$source" ]] || fail "authority-root input is absent or linked: $relative"
  [[ -x "$source" ]] && mode=0555
  install -d -m 0755 "$(dirname "$destination")"
  install -m "$mode" "$source" "$destination"
}

AUTHORITY_FILES=(
  tools/loom/exec_grant_controller.runtime.v1
  tools/loom/host_exec_quorum_fixture.freeze.v1
  tools/loom/kernel_exec_grant_cell_authority.freeze.v1
  tools/loom/GARDEN_KERNEL_EXEC_GRANT_CELL_V1.md
  stdlib/coordination/loom_kernel_exec_grant_cell_authority.sio
  tools/loom/kernel_exec_grant_cell_authority_main.sio
  tools/loom/kernel_invocation_cell_authority.freeze.v1
  tools/loom/execution_authority.freeze.v2
  tools/loom/execution_outcome.freeze.v1
  tools/loom/resident_membrane.runtime.v4
  tools/loom/subprocess_membrane.freeze.v1
  tools/loom/resident_authority.freeze.v1
  tools/loom/effect_closure_authority.freeze.v1
  tools/loom/resident_membrane.runtime.v3
  tools/loom/resident_membrane_v4_main.sio
  scripts/dev/build_sounio_loom_resident_membrane_v4.sh
  scripts/ci/sounio_loom_resident_transport_v4_selftest.sh
)
for relative in "${AUTHORITY_FILES[@]}"; do
  install_root_file "$relative"
done
while IFS= read -r -d '' directory; do
  [[ "$directory" == "$AUTHORITY_ROOT/.git" ]] || chmod 0555 "$directory"
done < <(find "$AUTHORITY_ROOT" -type d -print0)

BROKER_SHA256="$(sha256_file "$BIN/loom-kernel-principal-broker")"
CONTROLLER_SHA256="$(sha256_file "$BIN/loom-exec-grant-controller")"
RESIDENT_SHA256="$(sha256_file "$BIN/sounio-loom-resident-membrane-runtime-v4")"
LOCAL_BARRIER_SHA256="$(sha256_file "$BIN/loom-principal-cell-barrier-integrated")"
HOST_BARRIER_SHA256="$(sha256_file "$BIN/loom-host-exec-quorum-principal-cell")"
FIXTURE_MANIFEST_SHA256="$(sha256_file "$AUTHORITY_ROOT/tools/loom/host_exec_quorum_fixture.freeze.v1")"
CONTROLLER_MANIFEST_SHA256="$(sha256_file "$AUTHORITY_ROOT/tools/loom/exec_grant_controller.runtime.v1")"
DERIVED_GARDEN_SHA256="$(sha256_file "$ROOT_DIR/tools/loom/GARDEN_HOST_EXEC_QUORUM_DYNAMIC_PRINCIPAL_V1.md")"
RELEASE_DIGEST="$(printf '%s\n' "$SOURCE_COMMIT" "$BROKER_SHA256" "$CONTROLLER_SHA256" "$RESIDENT_SHA256" "$LOCAL_BARRIER_SHA256" "$HOST_BARRIER_SHA256" "$DERIVED_GARDEN_SHA256" | sha256sum | cut -d ' ' -f 1)"
RELEASE_ID="9030-hostq-${RELEASE_DIGEST:0:32}"

cat > "$RELEASE_STAGE/release.manifest.v1" <<EOF
schema=loom-host-exec-quorum-experiment-release-v1
stage=PARITY_OPEN_CANDIDATE
release_id=$RELEASE_ID
source_commit=$SOURCE_COMMIT
source_tree_state=$SOURCE_TREE_STATE
semantic_authority=Sounio
semantic_action=9030
controller_language=OCaml
controller_role=EFFECT_PARITY
material_language=C++20+Linux+systemd
material_role=MATERIAL_PARITY
material_transitory=true
derived_garden_path=tools/loom/GARDEN_HOST_EXEC_QUORUM_DYNAMIC_PRINCIPAL_V1.md
derived_garden_sha256=$DERIVED_GARDEN_SHA256
parent_quorum_manifest_sha256=$(sha256_file "$ROOT_DIR/tools/loom/host_exec_quorum.runtime.v1")
authority_root_path=authority-root
broker_path=bin/loom-kernel-principal-broker
broker_sha256=$BROKER_SHA256
controller_manifest_path=authority-root/tools/loom/exec_grant_controller.runtime.v1
controller_manifest_sha256=$CONTROLLER_MANIFEST_SHA256
controller_runtime_path=bin/loom-exec-grant-controller
controller_runtime_sha256=$CONTROLLER_SHA256
fixture_manifest_path=authority-root/tools/loom/host_exec_quorum_fixture.freeze.v1
fixture_manifest_sha256=$FIXTURE_MANIFEST_SHA256
fixture_bundle_path=data/host-exec-quorum-fixtures.v1
fixture_bundle_sha256=$(sha256_file "$DATA/host-exec-quorum-fixtures.v1")
resident_runtime_path=bin/sounio-loom-resident-membrane-runtime-v4
resident_runtime_sha256=$RESIDENT_SHA256
local_barrier_path=bin/loom-principal-cell-barrier-integrated
local_barrier_sha256=$LOCAL_BARRIER_SHA256
host_barrier_path=bin/loom-host-exec-quorum-principal-cell
host_barrier_sha256=$HOST_BARRIER_SHA256
non_bearer_transport=host-measurement-pending
material_grant=false
material_execution=false
launch_open=false
recycle_open=false
exec_attached=false
commit_attached=false
ci_attached=false
parity_open=false
claim_ready=false
EOF
chmod 0444 "$RELEASE_STAGE/release.manifest.v1"
MANIFEST_SHA256="$(sha256_file "$RELEASE_STAGE/release.manifest.v1")"
chmod 0555 "$BIN" "$DATA" "$AUTHORITY_ROOT" "$RELEASE_STAGE"

mv "$RELEASE_STAGE" "$CAPSULE/release"
install -m 0555 "$PROMOTER" "$META/promote_loom_host_exec_quorum_capsule.sh"
install -m 0555 "$HOST_GATE" "$META/sounio_loom_host_exec_quorum_host_gate.sh"
cat > "$META/capsule.manifest.v1" <<EOF
schema=loom-host-exec-quorum-experiment-capsule-v1
release_id=$RELEASE_ID
release_manifest_sha256=$MANIFEST_SHA256
source_commit=$SOURCE_COMMIT
source_tree_state=$SOURCE_TREE_STATE
promoter_sha256=$(sha256_file "$META/promote_loom_host_exec_quorum_capsule.sh")
host_gate_sha256=$(sha256_file "$META/sounio_loom_host_exec_quorum_host_gate.sh")
production_activation=false
material_grant=false
material_execution=false
EOF
chmod 0444 "$META/capsule.manifest.v1"

archive_stage="$WORK/capsule.tar"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  --format=posix --pax-option=delete=atime,delete=ctime \
  -C "$WORK" -cf "$archive_stage" capsule-v1
ARCHIVE_SHA256="$(sha256_file "$archive_stage")"
mkdir -p "$(dirname "$OUTPUT")"
output_stage="$(mktemp "$(dirname "$OUTPUT")/.loom-hostq-capsule.XXXXXX")"
install -m 0600 "$archive_stage" "$output_stage"
sync -f "$output_stage" 2>/dev/null || sync
mv -fT "$output_stage" "$OUTPUT"
printf '%s  %s\n' "$ARCHIVE_SHA256" "$(basename "$OUTPUT")" > "$OUTPUT.sha256"
chmod 0600 "$OUTPUT.sha256"

printf 'LOOM_HOST_EXEC_QUORUM_CAPSULE_BUILD PASS archive=%s archive_sha256=%s release_id=%s release_manifest_sha256=%s source_commit=%s source_tree_state=%s semantic_authority=Sounio controller_language=OCaml material_role=MATERIAL_PARITY production_activation=false material_grant=false material_execution=false launch_open=false parity_open=false claim_ready=false\n' \
  "$OUTPUT" "$ARCHIVE_SHA256" "$RELEASE_ID" "$MANIFEST_SHA256" "$SOURCE_COMMIT" "$SOURCE_TREE_STATE"
