#!/usr/bin/env bash

set -euo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
INSTALL_ROOT=/
PREFIX=/opt/sounio/loom-hostd
STATE_DIR=/var/lib/sounio/loom
UNIT_DIR=/etc/systemd/system
UNIT_NAME=sounio-loom-hostd.service
SERVICE_USER="$(id -un)"
SERVICE_UID=''
SERVICE_GID=''
SOCKET_ROOT=''
RUNTIME=''
AUTHORITY=''
RESIDENT=''
POLICY_ROOT="$ROOT_DIR"
EXEC_CELL_CAPSULE=''
EXEC_CELL_CAPSULE_SHA256=''
ACTIVATE=0

POLICY_FILES=(
  tools/loom/kernel_peer_activation_capsule_authority.freeze.v1
  tools/loom/kernel_peer_activation_capsule.runtime.v1
  tools/loom/kernel_peer_activation_capsule.current.v1
  tools/loom/resident_membrane.runtime.v5
  tools/loom/GARDEN_KERNEL_PEER_ACTIVATION_CAPSULE_V1.md
  stdlib/coordination/loom_kernel_peer_activation_capsule_authority.sio
  tools/loom/kernel_peer_activation_capsule_authority_main.sio
  tools/loom/kernel_peer_material_judgment_v13.freeze.v1
  tools/loom/kernel_exec_grant_cell_authority.freeze.v1
  tools/loom/subprocess_membrane.freeze.v1
  tools/loom/resident_authority.freeze.v1
  tools/loom/effect_closure_authority.freeze.v1
  tools/loom/kernel_invocation_cell_authority.freeze.v1
  tools/loom/resident_membrane.runtime.v4
  tools/loom/resident_membrane_v5_main.sio
  scripts/dev/build_sounio_loom_resident_membrane_v5.sh
  scripts/dev/promote_loom_host_exec_quorum_capsule.sh
  scripts/ci/sounio_loom_resident_transport_v5_selftest.sh
)

fail() {
  printf 'install-loom-hostd: FAIL: %s\n' "$*" >&2
  exit 1
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

manifest_value() {
  local manifest="$1" key="$2" line name value found=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$found" ]] || fail "duplicate manifest field: $key"
      found="$value"
    fi
  done < "$manifest"
  [[ -n "$found" ]] || fail "manifest omitted field: $key"
  printf '%s\n' "$found"
}

tree_sha256() {
  local root="$1" path relative material=''
  while IFS= read -r -d '' path; do
    relative="${path#"$root"/}"
    if [[ -d "$path" && ! -L "$path" ]]; then
      material+="D $(stat -c %a "$path") $relative"$'\n'
    elif [[ -f "$path" && ! -L "$path" ]]; then
      material+="F $(stat -c %a "$path") $(sha256_file "$path") $relative"$'\n'
    else
      fail "ExecCell release contains an unsupported node: $relative"
    fi
  done < <(find "$root" -mindepth 1 -print0 | sort -z)
  printf '%s' "$material" | sha256sum | cut -d ' ' -f 1
}

usage() {
  cat >&2 <<'EOF'
usage: install_loom_hostd.sh [--install-root DIR] [--prefix ABS]
       [--state-dir ABS] [--unit-dir ABS] [--unit-name NAME.service] [--user USER]
       [--runtime PATH] [--authority PATH] [--resident PATH]
       [--policy-root DIR]
       [--exec-cell-capsule PATH --exec-cell-capsule-sha256 HEX]
       [--activate]

Installation is disabled by default. --activate is accepted only for the real
host root and performs systemctl daemon-reload followed by enable --now.
EOF
  exit 2
}

while (($#)); do
  case "$1" in
    --install-root) [[ $# -ge 2 ]] || usage; INSTALL_ROOT="$2"; shift 2 ;;
    --prefix) [[ $# -ge 2 ]] || usage; PREFIX="$2"; shift 2 ;;
    --state-dir) [[ $# -ge 2 ]] || usage; STATE_DIR="$2"; shift 2 ;;
    --unit-dir) [[ $# -ge 2 ]] || usage; UNIT_DIR="$2"; shift 2 ;;
    --unit-name) [[ $# -ge 2 ]] || usage; UNIT_NAME="$2"; shift 2 ;;
    --user) [[ $# -ge 2 ]] || usage; SERVICE_USER="$2"; shift 2 ;;
    --runtime) [[ $# -ge 2 ]] || usage; RUNTIME="$2"; shift 2 ;;
    --authority) [[ $# -ge 2 ]] || usage; AUTHORITY="$2"; shift 2 ;;
    --resident) [[ $# -ge 2 ]] || usage; RESIDENT="$2"; shift 2 ;;
    --policy-root) [[ $# -ge 2 ]] || usage; POLICY_ROOT="$2"; shift 2 ;;
    --exec-cell-capsule) [[ $# -ge 2 ]] || usage; EXEC_CELL_CAPSULE="$2"; shift 2 ;;
    --exec-cell-capsule-sha256) [[ $# -ge 2 ]] || usage; EXEC_CELL_CAPSULE_SHA256="$2"; shift 2 ;;
    --activate) ACTIVATE=1; shift ;;
    *) usage ;;
  esac
done

[[ "$INSTALL_ROOT" == /* && "$PREFIX" == /* && "$STATE_DIR" == /* && "$UNIT_DIR" == /* ]] ||
  fail 'install root, prefix, state directory, and unit directory must be absolute'
[[ "$PREFIX" != / && "$STATE_DIR" != / && "$UNIT_DIR" != / ]] ||
  fail 'prefix, state directory, and unit directory cannot be the filesystem root'
[[ "$SERVICE_USER" =~ ^[a-zA-Z_][a-zA-Z0-9_.-]*$ ]] || fail 'service user is invalid'
[[ "$UNIT_NAME" =~ ^[a-zA-Z0-9_.@-]+\.service$ ]] || fail 'systemd unit name is invalid'
SERVICE_UID="$(id -u "$SERVICE_USER" 2>/dev/null)" ||
  fail "service user does not exist: $SERVICE_USER"
SERVICE_GID="$(id -g "$SERVICE_USER" 2>/dev/null)" ||
  fail "service group does not exist: $SERVICE_USER"
[[ "$SERVICE_UID" =~ ^[0-9]+$ && "$SERVICE_GID" =~ ^[0-9]+$ ]] ||
  fail "service identity is non-canonical: user=$SERVICE_USER uid=$SERVICE_UID gid=$SERVICE_GID"
SOCKET_ROOT="/tmp/sounio-loom-$SERVICE_UID"
INSTALL_ROOT="$(cd "$INSTALL_ROOT" && pwd -P)"
if [[ $ACTIVATE -eq 1 && "$INSTALL_ROOT" != / ]]; then
  fail '--activate is forbidden for a staged install root'
fi
if [[ -n "$EXEC_CELL_CAPSULE" || -n "$EXEC_CELL_CAPSULE_SHA256" ]]; then
  [[ "$EXEC_CELL_CAPSULE" == /* && -f "$EXEC_CELL_CAPSULE" &&
     ! -L "$EXEC_CELL_CAPSULE" ]] ||
    fail 'ExecCell capsule is absent, linked, or non-absolute'
  [[ "$EXEC_CELL_CAPSULE_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
    fail 'ExecCell capsule SHA-256 is absent or malformed'
elif [[ $ACTIVATE -eq 1 ]]; then
  fail '--activate requires a frozen ExecCell capsule and expected SHA-256'
fi
if [[ $ACTIVATE -eq 1 && "$SERVICE_UID" != 0 ]]; then
  fail '--activate with the ExecCell boot gate requires the root service identity'
fi

if [[ -z "$RUNTIME" ]]; then
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom.sh" >/dev/null
  RUNTIME="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
fi
if [[ -z "$AUTHORITY" ]]; then
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_boot_reconciler.sh" >/dev/null
  AUTHORITY="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-host-boot-reconciler"
fi
if [[ -z "$RESIDENT" ]]; then
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_membrane_v5.sh" >/dev/null
  RESIDENT="$ROOT_DIR/tools/loom/.runtime/sounio-loom-resident-membrane-runtime-v5"
fi
RUNTIME="$(readlink -f "$RUNTIME")"
AUTHORITY="$(readlink -f "$AUTHORITY")"
RESIDENT="$(readlink -f "$RESIDENT")"
POLICY_ROOT="$(readlink -f "$POLICY_ROOT")"
[[ -x "$RUNTIME" && -f "$RUNTIME" && ! -L "$RUNTIME" ]] ||
  fail "OCaml runtime is absent, linked, or non-executable: $RUNTIME"
[[ -x "$AUTHORITY" && -f "$AUTHORITY" && ! -L "$AUTHORITY" ]] ||
  fail "Sounio authority is absent, linked, or non-executable: $AUTHORITY"
[[ -x "$RESIDENT" && -f "$RESIDENT" && ! -L "$RESIDENT" ]] ||
  fail "Sounio resident is absent, linked, or non-executable: $RESIDENT"
[[ -d "$POLICY_ROOT" && ! -L "$POLICY_ROOT" ]] ||
  fail "product activation policy root is absent or linked: $POLICY_ROOT"

runtime_identity="$($RUNTIME runtime-version)"
grep -Fxq 'language=OCaml' <<< "$runtime_identity" || fail 'runtime is not OCaml'
grep -Fxq 'runtime_version=2026.08.30.42' <<< "$runtime_identity" ||
  fail 'runtime version does not implement loom-hostd v1'
runtime_sha256="$(sha256sum "$RUNTIME" | cut -d ' ' -f 1)"
authority_sha256="$(sha256sum "$AUTHORITY" | cut -d ' ' -f 1)"
resident_sha256="$(sha256sum "$RESIDENT" | cut -d ' ' -f 1)"
[[ "$authority_sha256" == \
   '99f5062729a171ac2d8c1b9b181497fbe1b8c9317859ee0fdc4d2cd4acaedb5b' ]] ||
  fail "Sounio authority hash drifted: $authority_sha256"
authority_probe="$(printf '0\n' | "$AUTHORITY")"
[[ "$authority_probe" == 'SOUNIO_HOST_BOOT_RECONCILER_SELFTEST PASS cases=14' ]] ||
  fail "Sounio authority selftest diverged: $authority_probe"
resident_expected_sha256="$(sed -n 's/^runtime_sha256=//p' \
  "$POLICY_ROOT/tools/loom/resident_membrane.runtime.v5")"
[[ "$resident_expected_sha256" =~ ^[0-9a-f]{64}$ && \
   "$resident_sha256" == "$resident_expected_sha256" ]] ||
  fail "Sounio resident hash drifted: expected=$resident_expected_sha256 actual=$resident_sha256"

policy_material=''
for relative in "${POLICY_FILES[@]}"; do
  source_path="$POLICY_ROOT/$relative"
  [[ -f "$source_path" && ! -L "$source_path" ]] ||
    fail "product activation policy input is absent or linked: $relative"
  file_sha256="$(sha256sum "$source_path" | cut -d ' ' -f 1)"
  policy_material+="$file_sha256  $relative"$'\n'
done
policy_tree_sha256="$(printf '%s' "$policy_material" | sha256sum | cut -d ' ' -f 1)"

exec_cell_bundle_present=false
exec_cell_release_id=absent
exec_cell_release_manifest_sha256=absent
exec_cell_release_tree_sha256=absent
exec_cell_release=''
exec_cell_exec_start_pre=''
capsule_work=''
cleanup_capsule_work() {
  if [[ -n "$capsule_work" && -d "$capsule_work" ]]; then
    find "$capsule_work" -type d -exec chmod u+rwx {} + 2>/dev/null || true
    rm -rf "$capsule_work"
  fi
}
trap cleanup_capsule_work EXIT

if [[ -n "$EXEC_CELL_CAPSULE" ]]; then
  promoter="$POLICY_ROOT/scripts/dev/promote_loom_host_exec_quorum_capsule.sh"
  [[ -f "$promoter" && -x "$promoter" && ! -L "$promoter" ]] ||
    fail 'ExecCell capsule verifier is absent, linked, or non-executable'
  capsule_work="$(mktemp -d "${TMPDIR:-/tmp}/loom-hostd-exec-cell.XXXXXX")"
  capsule_copy="$capsule_work/capsule.tar"
  install -m 0400 "$EXEC_CELL_CAPSULE" "$capsule_copy"
  [[ "$(sha256_file "$capsule_copy")" == "$EXEC_CELL_CAPSULE_SHA256" ]] ||
    fail 'ExecCell capsule archive hash drifted'
  capsule_verify="$(bash "$promoter" --archive "$capsule_copy" \
    --expected-sha256 "$EXEC_CELL_CAPSULE_SHA256" --mode verify)" ||
    fail 'ExecCell capsule verifier refused the archive'
  [[ "$capsule_verify" == 'LOOM_HOST_EXEC_QUORUM_CAPSULE_VERIFY PASS '* ]] ||
    fail 'ExecCell capsule verifier returned a non-canonical receipt'
  install -d -m 0700 "$capsule_work/extracted"
  tar --no-same-owner --same-permissions -xf "$capsule_copy" \
    -C "$capsule_work/extracted"
  [[ "$(sha256_file "$capsule_copy")" == "$EXEC_CELL_CAPSULE_SHA256" ]] ||
    fail 'ExecCell capsule archive changed during verification'

  exec_cell_release="$capsule_work/extracted/capsule-v1/release"
  exec_cell_manifest="$exec_cell_release/release.manifest.v1"
  [[ -d "$exec_cell_release" && ! -L "$exec_cell_release" &&
     -f "$exec_cell_manifest" && ! -L "$exec_cell_manifest" ]] ||
    fail 'ExecCell capsule release topology is incomplete'
  exec_cell_release_manifest_sha256="$(sha256_file "$exec_cell_manifest")"
  [[ "$(manifest_value "$exec_cell_manifest" schema)" == \
       loom-host-exec-quorum-experiment-release-v1 &&
     "$(manifest_value "$exec_cell_manifest" stage)" == PARITY_OPEN_CANDIDATE &&
     "$(manifest_value "$exec_cell_manifest" semantic_authority)" == Sounio &&
     "$(manifest_value "$exec_cell_manifest" semantic_action)" == 9030 &&
     "$(manifest_value "$exec_cell_manifest" product_exec_ingress_action)" == 9031 &&
     "$(manifest_value "$exec_cell_manifest" controller_language)" == OCaml &&
     "$(manifest_value "$exec_cell_manifest" controller_role)" == EFFECT_PARITY &&
     "$(manifest_value "$exec_cell_manifest" material_language)" == C++20+Linux+systemd &&
     "$(manifest_value "$exec_cell_manifest" material_role)" == MATERIAL_PARITY &&
     "$(manifest_value "$exec_cell_manifest" material_transitory)" == true ]] ||
    fail 'ExecCell capsule language authority or stage drifted'
  for closed in material_grant material_execution launch_open recycle_open \
    exec_attached commit_attached ci_attached parity_open claim_ready; do
    [[ "$(manifest_value "$exec_cell_manifest" "$closed")" == false ]] ||
      fail "ExecCell capsule preclaimed $closed"
  done

  verify_exec_cell_binding() {
    local path_key="$1" hash_key="$2" mode="$3" relative path
    relative="$(manifest_value "$exec_cell_manifest" "$path_key")"
    [[ "$relative" =~ ^[A-Za-z0-9._/-]+$ && "$relative" != /* &&
       "/$relative/" != *'/../'* ]] ||
      fail "ExecCell binding path is unsafe: $path_key"
    path="$exec_cell_release/$relative"
    [[ -f "$path" && ! -L "$path" ]] ||
      fail "ExecCell binding is absent or linked: $relative"
    [[ "$(sha256_file "$path")" == \
       "$(manifest_value "$exec_cell_manifest" "$hash_key")" ]] ||
      fail "ExecCell binding hash drifted: $relative"
    [[ "$(stat -c %a "$path")" == "$mode" ]] ||
      fail "ExecCell binding mode drifted: $relative"
    printf '%s\n' "$path"
  }

  exec_cell_broker="$(verify_exec_cell_binding broker_path broker_sha256 555)"
  exec_cell_controller_manifest="$(verify_exec_cell_binding \
    controller_manifest_path controller_manifest_sha256 444)"
  exec_cell_controller_runtime="$(verify_exec_cell_binding \
    controller_runtime_path controller_runtime_sha256 555)"
  exec_cell_witness_cell="$(verify_exec_cell_binding \
    process_witness_cell_path process_witness_cell_sha256 555)"
  exec_cell_witness_payload="$(verify_exec_cell_binding \
    process_witness_payload_path process_witness_payload_sha256 555)"
  exec_cell_witness_manifest="$(verify_exec_cell_binding \
    process_witness_manifest_path process_witness_manifest_sha256 444)"
  exec_cell_fixture_manifest="$(verify_exec_cell_binding \
    product_exec_cell_fixture_manifest_path \
    product_exec_cell_fixture_manifest_sha256 444)"
  exec_cell_fixture_bundle="$(verify_exec_cell_binding \
    product_exec_cell_fixture_bundle_path \
    product_exec_cell_fixture_bundle_sha256 444)"
  exec_cell_product_runtime="$(verify_exec_cell_binding \
    product_exec_ingress_runtime_path product_exec_ingress_runtime_sha256 555)"
  exec_cell_language_runtime="$(verify_exec_cell_binding \
    product_language_runtime_path product_language_runtime_sha256 555)"
  exec_cell_resident_runtime="$(verify_exec_cell_binding \
    product_resident_runtime_path product_resident_runtime_sha256 555)"
  : "$exec_cell_broker" "$exec_cell_controller_runtime" \
    "$exec_cell_witness_cell" "$exec_cell_product_runtime" \
    "$exec_cell_language_runtime" "$exec_cell_resident_runtime"
  [[ "$(manifest_value "$exec_cell_controller_manifest" semantic_authority)" == Sounio &&
     "$(manifest_value "$exec_cell_controller_manifest" action)" == 9030 &&
     "$(manifest_value "$exec_cell_controller_manifest" producing_language)" == OCaml &&
     "$(manifest_value "$exec_cell_controller_manifest" language_role)" == EFFECT_PARITY &&
     "$(manifest_value "$exec_cell_controller_manifest" python_executable_invoked)" == false &&
     "$(manifest_value "$exec_cell_controller_manifest" rust_executable_invoked)" == false &&
     "$(manifest_value "$exec_cell_fixture_manifest" stage)" == SEMANTICS_FROZEN &&
     "$(manifest_value "$exec_cell_fixture_manifest" semantic_authority)" == Sounio &&
     "$(manifest_value "$exec_cell_fixture_manifest" producing_language)" == Sounio &&
     "$(manifest_value "$exec_cell_fixture_manifest" action)" == 9030 &&
     "$(manifest_value "$exec_cell_fixture_manifest" python_executable_invoked)" == false &&
     "$(manifest_value "$exec_cell_fixture_manifest" rust_executable_invoked)" == false &&
     "$(manifest_value "$exec_cell_fixture_manifest" bundle_sha256)" == \
       "$(sha256_file "$exec_cell_fixture_bundle")" &&
     "$(manifest_value "$exec_cell_fixture_manifest" payload_sha256)" == \
       "$(sha256_file "$exec_cell_witness_payload")" &&
     "$(manifest_value "$exec_cell_fixture_manifest" payload_manifest_sha256)" == \
       "$(sha256_file "$exec_cell_witness_manifest")" ]] ||
    fail 'ExecCell Sounio/OCaml/ProcessWitness provenance drifted'
  exec_cell_release_id="$(manifest_value "$exec_cell_manifest" release_id)"
  [[ "$exec_cell_release_id" =~ ^9030-hostq-[0-9a-f]{32}$ ]] ||
    fail 'ExecCell release identity is non-canonical'
  exec_cell_authority_root_relative="$(manifest_value \
    "$exec_cell_manifest" authority_root_path)"
  exec_cell_product_root_relative="$(manifest_value \
    "$exec_cell_manifest" product_authority_root_path)"
  [[ "$exec_cell_authority_root_relative" =~ ^[A-Za-z0-9._/-]+$ &&
     "$exec_cell_authority_root_relative" != /* &&
     "/$exec_cell_authority_root_relative/" != *'/../'* &&
     "$exec_cell_product_root_relative" == \
       "$exec_cell_authority_root_relative" &&
     -d "$exec_cell_release/$exec_cell_authority_root_relative" &&
     ! -L "$exec_cell_release/$exec_cell_authority_root_relative" ]] ||
    fail 'ExecCell authority-root binding is unsafe or divergent'
  systemd_run_path="$(readlink -f "$(command -v systemd-run 2>/dev/null || true)")"
  systemctl_path="$(readlink -f "$(command -v systemctl 2>/dev/null || true)")"
  [[ "$systemd_run_path" == /* && -f "$systemd_run_path" &&
     -x "$systemd_run_path" && ! -L "$systemd_run_path" &&
     "$systemctl_path" == /* && -f "$systemctl_path" &&
     -x "$systemctl_path" && ! -L "$systemctl_path" ]] ||
    fail 'ExecCell boot gate requires canonical systemd-run and systemctl'
  exec_cell_release_tree_sha256="$(tree_sha256 "$exec_cell_release")"
  exec_cell_bundle_present=true
fi

dest_prefix="$INSTALL_ROOT$PREFIX"
dest_state="$INSTALL_ROOT$STATE_DIR"
dest_unit_dir="$INSTALL_ROOT$UNIT_DIR"
dest_socket_root="$INSTALL_ROOT$SOCKET_ROOT"
dest_policy_root="$dest_prefix/policy/product-activation"
install -d -m 0755 "$dest_prefix/bin" "$dest_prefix/share" "$dest_unit_dir" \
  "$dest_policy_root"
if [[ "$INSTALL_ROOT" == / ]]; then
  install -d -m 0700 -o "$SERVICE_UID" -g "$SERVICE_GID" \
    "$dest_state" "$dest_socket_root"
else
  install -d -m 0700 "$dest_state" "$dest_socket_root"
fi
install -m 0755 "$RUNTIME" "$dest_prefix/bin/sounio-loom-runtime"
install -m 0755 "$AUTHORITY" "$dest_prefix/bin/sounio-loom-host-boot-reconciler"
install -m 0755 "$RESIDENT" \
  "$dest_prefix/bin/sounio-loom-resident-membrane-runtime-v5"
for relative in "${POLICY_FILES[@]}"; do
  destination="$dest_policy_root/$relative"
  policy_mode=0444
  [[ -x "$POLICY_ROOT/$relative" ]] && policy_mode=0555
  install -d -m 0755 "$(dirname "$destination")"
  install -m "$policy_mode" "$POLICY_ROOT/$relative" "$destination"
done
if [[ "$exec_cell_bundle_present" == true ]]; then
  dest_exec_cell_parent="$dest_prefix/exec-cell/releases"
  dest_exec_cell_release="$dest_exec_cell_parent/$exec_cell_release_id"
  install -d -m 0755 "$dest_exec_cell_parent"
  if [[ -e "$dest_exec_cell_release" || -L "$dest_exec_cell_release" ]]; then
    [[ -d "$dest_exec_cell_release" && ! -L "$dest_exec_cell_release" &&
       "$(tree_sha256 "$dest_exec_cell_release")" == \
         "$exec_cell_release_tree_sha256" ]] ||
      fail 'installed ExecCell release drifted'
  else
    dest_exec_cell_stage="$(mktemp -d \
      "$dest_exec_cell_parent/.${exec_cell_release_id}.XXXXXX")"
    cp -a "$exec_cell_release/." "$dest_exec_cell_stage/"
    if [[ "$INSTALL_ROOT" == / ]]; then
      chown -R root:root "$dest_exec_cell_stage"
    fi
    [[ "$(tree_sha256 "$dest_exec_cell_stage")" == \
       "$exec_cell_release_tree_sha256" ]] ||
      fail 'installed ExecCell release copy drifted'
    mv -T "$dest_exec_cell_stage" "$dest_exec_cell_release"
  fi
fi
policy_manifest="$dest_prefix/share/product-activation-policy.v1"
policy_manifest_stage="$(mktemp "$dest_prefix/share/.product-activation-policy.v1.XXXXXX")"
{
  printf '%s\n' \
    'schema=loom-product-activation-policy-install-v1' \
    'semantic_authority=Sounio' \
    'semantic_action=9031' \
    "policy_file_count=${#POLICY_FILES[@]}" \
    "policy_tree_sha256=$policy_tree_sha256" \
    "resident_runtime_sha256=$resident_sha256" \
    'production_activation=false'
  printf '%s' "$policy_material"
} > "$policy_manifest_stage"
chmod 0444 "$policy_manifest_stage"
mv -f "$policy_manifest_stage" "$policy_manifest"

exec_cell_manifest_install="$dest_prefix/share/exec-cell-bundle.v1"
exec_cell_manifest_stage="$(mktemp "$dest_prefix/share/.exec-cell-bundle.v1.XXXXXX")"
cat > "$exec_cell_manifest_stage" <<EOF
schema=loom-hostd-exec-cell-bundle-v1
semantic_authority=Sounio
semantic_actions=9030,9031
bundle_present=$exec_cell_bundle_present
capsule_sha256=${EXEC_CELL_CAPSULE_SHA256:-absent}
release_id=$exec_cell_release_id
release_manifest_sha256=$exec_cell_release_manifest_sha256
release_tree_sha256=$exec_cell_release_tree_sha256
controller_language=OCaml
controller_role=EFFECT_PARITY
material_language=C++20+Linux+systemd
material_role=MATERIAL_PARITY
material_transitory=true
python_executable_invoked=false
rust_executable_invoked=false
exec_cell_canary_frozen=$exec_cell_bundle_present
exec_cell_boot_gate_configured=$exec_cell_bundle_present
exec_cell_boot_gate_test_only=true
exec_attached=false
production_activation=false
EOF
chmod 0444 "$exec_cell_manifest_stage"
mv -f "$exec_cell_manifest_stage" "$exec_cell_manifest_install"

if [[ "$exec_cell_bundle_present" == true ]]; then
  unit_exec_cell_release="$PREFIX/exec-cell/releases/$exec_cell_release_id"
  unit_exec_cell_manifest="$unit_exec_cell_release/release.manifest.v1"
  unit_exec_cell_gate="${UNIT_NAME%.service}-exec-cell-gate"
  unit_value() {
    manifest_value "$exec_cell_manifest" "$1"
  }
  exec_cell_exec_start_pre="ExecStartPre=$systemd_run_path --quiet --wait --pipe --collect --unit=$unit_exec_cell_gate --service-type=exec --property=UMask=0077 --property=NoNewPrivileges=yes --property=PrivateTmp=yes --property=PrivateDevices=yes --property=PrivateNetwork=yes --property=ProtectSystem=strict --property=ProtectHome=yes --property=ReadWritePaths=/run --property=RestrictAddressFamilies=AF_UNIX --property=TimeoutStartSec=220s -- $unit_exec_cell_release/$(unit_value broker_path) --selftest-product-exec-cell-host --controller-manifest $unit_exec_cell_release/$(unit_value controller_manifest_path) --controller-runtime $unit_exec_cell_release/$(unit_value controller_runtime_path) --controller-root $unit_exec_cell_release/$exec_cell_authority_root_relative --resident-runtime $unit_exec_cell_release/$(unit_value resident_runtime_path) --process-witness-runtime $unit_exec_cell_release/$(unit_value process_witness_cell_path) --process-witness-payload $unit_exec_cell_release/$(unit_value process_witness_payload_path) --process-witness-manifest $unit_exec_cell_release/$(unit_value process_witness_manifest_path) --product-root $unit_exec_cell_release/$exec_cell_product_root_relative --product-runtime $unit_exec_cell_release/$(unit_value product_exec_ingress_runtime_path) --product-language-runtime $unit_exec_cell_release/$(unit_value product_language_runtime_path) --product-resident-runtime $unit_exec_cell_release/$(unit_value product_resident_runtime_path) --product-exec-cell-fixture-manifest $unit_exec_cell_release/$(unit_value product_exec_cell_fixture_manifest_path) --product-exec-cell-fixture-bundle $unit_exec_cell_release/$(unit_value product_exec_cell_fixture_bundle_path) --systemd-run $systemd_run_path --systemctl $systemctl_path"
fi

unit="$dest_unit_dir/$UNIT_NAME"
unit_stage="$(mktemp "$dest_unit_dir/.${UNIT_NAME}.XXXXXX")"
cat > "$unit_stage" <<EOF
[Unit]
Description=Sounio Loom host lane reconciler
After=local-fs.target
ConditionPathIsDirectory=$STATE_DIR

[Service]
Type=simple
User=$SERVICE_USER
Environment=SOUNIO_LOOM_HOST_BOOT_AUTHORITY=$PREFIX/bin/sounio-loom-host-boot-reconciler
Environment=XDG_RUNTIME_DIR=/tmp
$exec_cell_exec_start_pre
ExecStart=$PREFIX/bin/sounio-loom-runtime host-supervise --state-dir $STATE_DIR --service-enabled --apply
Restart=on-failure
RestartSec=2s
TimeoutStartSec=240s
KillMode=process
NoNewPrivileges=true
PrivateTmp=false
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$STATE_DIR $SOCKET_ROOT
UMask=0077

[Install]
WantedBy=multi-user.target
EOF
chmod 0644 "$unit_stage"
mv -f "$unit_stage" "$unit"

manifest="$dest_prefix/manifest.v1"
manifest_stage="$(mktemp "$dest_prefix/.manifest.v1.XXXXXX")"
cat > "$manifest_stage" <<EOF
schema=loom-hostd-install-v1
language=OCaml
language_role=EFFECT_PARITY
semantic_authority=Sounio
semantic_action=9041
semantics_sha256=0d5174cd87b8c18b5f3bbfa7ed44d0258795a96f146730c879c46167abdddf7d
authority_runtime_sha256=$authority_sha256
ocaml_runtime_sha256=$runtime_sha256
resident_runtime_sha256=$resident_sha256
product_activation_policy_sha256=$policy_tree_sha256
product_activation_policy_files=${#POLICY_FILES[@]}
exec_cell_bundle_present=$exec_cell_bundle_present
exec_cell_capsule_sha256=${EXEC_CELL_CAPSULE_SHA256:-absent}
exec_cell_release_id=$exec_cell_release_id
exec_cell_release_manifest_sha256=$exec_cell_release_manifest_sha256
exec_cell_release_tree_sha256=$exec_cell_release_tree_sha256
exec_cell_canary_frozen=$exec_cell_bundle_present
exec_cell_boot_gate_configured=$exec_cell_bundle_present
exec_cell_boot_gate_test_only=true
exec_attached=false
prefix=$PREFIX
state_dir=$STATE_DIR
unit_path=$UNIT_DIR/$UNIT_NAME
service_user=$SERVICE_USER
service_uid=$SERVICE_UID
service_gid=$SERVICE_GID
install_default=disabled
service_enabled=false
production_activation=false
automatic_lineage_resurrection=false
socket_namespace=host-shared-tmp
socket_root=$SOCKET_ROOT
private_tmp=false
python_executable_invoked=false
rust_executable_invoked=false
EOF
chmod 0444 "$manifest_stage"
mv -f "$manifest_stage" "$manifest"

activated=false
if [[ $ACTIVATE -eq 1 ]]; then
  systemctl daemon-reload
  systemctl enable --now "$UNIT_NAME"
  activated=true
  manifest_stage="$(mktemp "$dest_prefix/.manifest.v1.XXXXXX")"
  sed -e 's/^service_enabled=false$/service_enabled=true/' \
      -e 's/^production_activation=false$/production_activation=true/' \
      "$manifest" > "$manifest_stage"
  chmod 0444 "$manifest_stage"
  mv -f "$manifest_stage" "$manifest"
fi

printf 'LOOM_HOSTD_INSTALLED prefix=%s state_dir=%s socket_root=%s unit=%s language=OCaml role=EFFECT_PARITY semantic_authority=Sounio actions=9030,9031,9041 semantics_sha256=%s authority_runtime_sha256=%s ocaml_runtime_sha256=%s resident_runtime_sha256=%s product_activation_policy_sha256=%s exec_cell_bundle_present=%s exec_cell_release_id=%s exec_cell_release_manifest_sha256=%s exec_cell_release_tree_sha256=%s exec_cell_canary_frozen=%s exec_cell_boot_gate_configured=%s exec_cell_boot_gate_test_only=true exec_attached=false activated=%s automatic_lineage_resurrection=false python_executed=false rust_executed=false\n' \
  "$PREFIX" "$STATE_DIR" "$SOCKET_ROOT" "$UNIT_DIR/$UNIT_NAME" \
  '0d5174cd87b8c18b5f3bbfa7ed44d0258795a96f146730c879c46167abdddf7d' \
  "$authority_sha256" "$runtime_sha256" "$resident_sha256" \
  "$policy_tree_sha256" "$exec_cell_bundle_present" "$exec_cell_release_id" \
  "$exec_cell_release_manifest_sha256" "$exec_cell_release_tree_sha256" \
  "$exec_cell_bundle_present" "$exec_cell_bundle_present" "$activated"
