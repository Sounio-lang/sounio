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
ACTIVATE=0

fail() {
  printf 'install-loom-hostd: FAIL: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat >&2 <<'EOF'
usage: install_loom_hostd.sh [--install-root DIR] [--prefix ABS]
       [--state-dir ABS] [--unit-dir ABS] [--unit-name NAME.service] [--user USER]
       [--runtime PATH] [--authority PATH] [--activate]

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
RUNTIME="$(readlink -f "$RUNTIME")"
AUTHORITY="$(readlink -f "$AUTHORITY")"
[[ -x "$RUNTIME" && -f "$RUNTIME" && ! -L "$RUNTIME" ]] ||
  fail "OCaml runtime is absent, linked, or non-executable: $RUNTIME"
[[ -x "$AUTHORITY" && -f "$AUTHORITY" && ! -L "$AUTHORITY" ]] ||
  fail "Sounio authority is absent, linked, or non-executable: $AUTHORITY"

runtime_identity="$($RUNTIME runtime-version)"
grep -Fxq 'language=OCaml' <<< "$runtime_identity" || fail 'runtime is not OCaml'
grep -Fxq 'runtime_version=2026.08.30.41' <<< "$runtime_identity" ||
  fail 'runtime version does not implement loom-hostd v1'
runtime_sha256="$(sha256sum "$RUNTIME" | cut -d ' ' -f 1)"
authority_sha256="$(sha256sum "$AUTHORITY" | cut -d ' ' -f 1)"
[[ "$authority_sha256" == \
   '99f5062729a171ac2d8c1b9b181497fbe1b8c9317859ee0fdc4d2cd4acaedb5b' ]] ||
  fail "Sounio authority hash drifted: $authority_sha256"
authority_probe="$(printf '0\n' | "$AUTHORITY")"
[[ "$authority_probe" == 'SOUNIO_HOST_BOOT_RECONCILER_SELFTEST PASS cases=14' ]] ||
  fail "Sounio authority selftest diverged: $authority_probe"

dest_prefix="$INSTALL_ROOT$PREFIX"
dest_state="$INSTALL_ROOT$STATE_DIR"
dest_unit_dir="$INSTALL_ROOT$UNIT_DIR"
install -d -m 0755 "$dest_prefix/bin" "$dest_prefix/share" "$dest_unit_dir"
install -d -m 0700 "$dest_state"
install -m 0755 "$RUNTIME" "$dest_prefix/bin/sounio-loom-runtime"
install -m 0755 "$AUTHORITY" "$dest_prefix/bin/sounio-loom-host-boot-reconciler"

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
PrivateTmp=true
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
prefix=$PREFIX
state_dir=$STATE_DIR
unit_path=$UNIT_DIR/$UNIT_NAME
service_user=$SERVICE_USER
install_default=disabled
service_enabled=false
production_activation=false
automatic_lineage_resurrection=false
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

printf 'LOOM_HOSTD_INSTALLED prefix=%s state_dir=%s unit=%s language=OCaml role=EFFECT_PARITY semantic_authority=Sounio action=9041 semantics_sha256=%s authority_runtime_sha256=%s ocaml_runtime_sha256=%s activated=%s automatic_lineage_resurrection=false python_executed=false rust_executed=false\n' \
  "$PREFIX" "$STATE_DIR" "$UNIT_DIR/$UNIT_NAME" \
  '0d5174cd87b8c18b5f3bbfa7ed44d0258795a96f146730c879c46167abdddf7d' \
  "$authority_sha256" "$runtime_sha256" "$activated"
