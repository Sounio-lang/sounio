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
RUNTIME=''
AUTHORITY=''
RESIDENT=''
POLICY_ROOT="$ROOT_DIR"
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
  scripts/ci/sounio_loom_resident_transport_v5_selftest.sh
)

fail() {
  printf 'install-loom-hostd: FAIL: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat >&2 <<'EOF'
usage: install_loom_hostd.sh [--install-root DIR] [--prefix ABS]
       [--state-dir ABS] [--unit-dir ABS] [--unit-name NAME.service] [--user USER]
       [--runtime PATH] [--authority PATH] [--resident PATH]
       [--policy-root DIR] [--activate]

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
INSTALL_ROOT="$(cd "$INSTALL_ROOT" && pwd -P)"
if [[ $ACTIVATE -eq 1 && "$INSTALL_ROOT" != / ]]; then
  fail '--activate is forbidden for a staged install root'
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

dest_prefix="$INSTALL_ROOT$PREFIX"
dest_state="$INSTALL_ROOT$STATE_DIR"
dest_unit_dir="$INSTALL_ROOT$UNIT_DIR"
dest_policy_root="$dest_prefix/policy/product-activation"
install -d -m 0755 "$dest_prefix/bin" "$dest_prefix/share" "$dest_unit_dir" \
  "$dest_policy_root"
install -d -m 0700 "$dest_state"
install -m 0755 "$RUNTIME" "$dest_prefix/bin/sounio-loom-runtime"
install -m 0755 "$AUTHORITY" "$dest_prefix/bin/sounio-loom-host-boot-reconciler"
install -m 0755 "$RESIDENT" \
  "$dest_prefix/bin/sounio-loom-resident-membrane-runtime-v5"
for relative in "${POLICY_FILES[@]}"; do
  destination="$dest_policy_root/$relative"
  install -d -m 0755 "$(dirname "$destination")"
  install -m 0444 "$POLICY_ROOT/$relative" "$destination"
done
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
ExecStart=$PREFIX/bin/sounio-loom-runtime host-supervise --state-dir $STATE_DIR --service-enabled --apply
Restart=on-failure
RestartSec=2s
KillMode=process
NoNewPrivileges=true
PrivateTmp=false
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=$STATE_DIR
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
prefix=$PREFIX
state_dir=$STATE_DIR
unit_path=$UNIT_DIR/$UNIT_NAME
service_user=$SERVICE_USER
install_default=disabled
service_enabled=false
production_activation=false
automatic_lineage_resurrection=false
socket_namespace=host-shared-tmp
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

printf 'LOOM_HOSTD_INSTALLED prefix=%s state_dir=%s unit=%s language=OCaml role=EFFECT_PARITY semantic_authority=Sounio action=9041 semantics_sha256=%s authority_runtime_sha256=%s ocaml_runtime_sha256=%s resident_runtime_sha256=%s product_activation_policy_sha256=%s activated=%s automatic_lineage_resurrection=false python_executed=false rust_executed=false\n' \
  "$PREFIX" "$STATE_DIR" "$UNIT_DIR/$UNIT_NAME" \
  '0d5174cd87b8c18b5f3bbfa7ed44d0258795a96f146730c879c46167abdddf7d' \
  "$authority_sha256" "$runtime_sha256" "$resident_sha256" \
  "$policy_tree_sha256" "$activated"
