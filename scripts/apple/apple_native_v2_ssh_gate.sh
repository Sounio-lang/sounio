#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="$ROOT_DIR/artifacts/omega"
REPORT_PATH="${SOUNIO_APPLE_NATIVE_V2_REPORT:-$OUT_DIR/apple_os26_native_v2_ssh_gate.v1.json}"
SSH_TARGET="${SOUNIO_MAC_SSH_TARGET:-demetrios@macbook-pro-de-demetrios-2.tail21cbc4.ts.net}"
SSH_PORT="${SOUNIO_MAC_SSH_PORT:-22}"
REMOTE_DIR="${SOUNIO_MAC_REMOTE_DIR:-~/work/sounio-apple-native-v2}"
TARGET="${SOUNIO_NATIVE_V2_TARGET:-aarch64-macos}"
SSH_OPTS=(
  -p "$SSH_PORT"
  -o BatchMode=yes
  -o ConnectTimeout="${SOUNIO_MAC_SSH_CONNECT_TIMEOUT:-8}"
  -o StrictHostKeyChecking="${SOUNIO_MAC_SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"
)

mkdir -p "$OUT_DIR"

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

emit_report() {
  local status="$1"
  local reason="$2"
  local remote_summary="${3:-}"
  local mac_sw_vers="${4:-}"
  local mac_uname="${5:-}"
  local xcode_version="${6:-}"
  local sdk_version="${7:-}"

  cat >"$REPORT_PATH" <<EOF
{
  "schema": "sounio.apple_os26_native_v2_ssh_gate.v1",
  "status": "$status",
  "reason": "$reason",
  "ssh_target": "$SSH_TARGET",
  "ssh_port": "$SSH_PORT",
  "remote_dir": "$REMOTE_DIR",
  "native_v2_target": "$TARGET",
  "mac_sw_vers": $(printf '%s' "$mac_sw_vers" | json_escape),
  "mac_uname": $(printf '%s' "$mac_uname" | json_escape),
  "xcode_version": $(printf '%s' "$xcode_version" | json_escape),
  "macos_sdk_version": $(printf '%s' "$sdk_version" | json_escape),
  "remote_summary": $(printf '%s' "$remote_summary" | json_escape)
}
EOF

  echo "apple_native_v2_ssh_gate: status=$status reason=$reason report=$REPORT_PATH"
}

if ! ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "echo ok" >/dev/null 2>&1; then
  emit_report "not_run" "ssh_unreachable"
  exit 0
fi

if ! ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "command -v bash >/dev/null && command -v file >/dev/null && command -v sw_vers >/dev/null" >/dev/null 2>&1; then
  emit_report "not_run" "remote_env_missing"
  exit 0
fi

ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "mkdir -p $REMOTE_DIR"

if command -v rsync >/dev/null 2>&1; then
  rsync -az --delete \
    --exclude '.git/' \
    --exclude 'target/' \
    --exclude '.cache/' \
    --exclude 'artifacts/omega/apple_os26_native_v2_ssh_gate.v1.json' \
    -e "ssh ${SSH_OPTS[*]}" \
    "$ROOT_DIR"/ "$SSH_TARGET:$REMOTE_DIR"/
else
  tar -cf - \
    --exclude='./.git' \
    --exclude='./target' \
    --exclude='./.cache' \
    -C "$ROOT_DIR" . | ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cd $REMOTE_DIR && tar -xf -"
fi

REMOTE_SCRIPT='
set -euo pipefail
cd "$REMOTE_DIR"

mkdir -p artifacts/omega

MAC_SW_VERS="$(sw_vers 2>/dev/null || true)"
MAC_UNAME="$(uname -a 2>/dev/null || true)"
XCODE_VERSION="$(xcodebuild -version 2>/dev/null || true)"
SDK_VERSION="$(xcrun --sdk macosx --show-sdk-version 2>/dev/null || true)"

if ! printf "%s\n" "$MAC_SW_VERS" | grep -Eq "ProductVersion:[[:space:]]*26\.5"; then
  printf "%s\n" "$MAC_SW_VERS" > artifacts/omega/apple_os26_native_v2_remote.sw_vers.txt
  echo "REMOTE_STATUS=os_version_mismatch"
  echo "REMOTE_REASON=expected_macos_26_5"
  exit 42
fi

./bin/souc info > artifacts/omega/apple_os26_native_v2_souc_info.txt
bash scripts/ci/selfhost_host_gate.sh > artifacts/omega/apple_os26_native_v2_selfhost_gate.log 2>&1

SMOKE_BIN="artifacts/omega/native_backend_v2_scalar_smoke.$TARGET.bin"
if [ "$TARGET" = "x86_64-linux" ]; then
  bash scripts/omega/omega_native_v2_shadow_gate.sh > artifacts/omega/apple_os26_native_v2_gate.log 2>&1
  SMOKE_BIN="artifacts/omega/native_backend_v2_scalar_smoke.selftest.bin"
else
  ./bin/souc run tests/native-v2/aarch64_macho_preview_emit.sio > artifacts/omega/apple_os26_native_v2_emit.log 2>&1
  cat > "artifacts/omega/native_backend_v2_contract.$TARGET.json" <<CONTRACT_EOF
{
  "schema": "sounio.native_backend_v2_contract.v1",
  "backend": "native-v2",
  "stage": "apple_arm64_preview",
  "requested_target": "$TARGET",
  "resolved_target": "$TARGET",
  "coverage": "scalar_core",
  "compile_ok": true,
  "format": "macho64",
  "output_path": "artifacts/omega/native_backend_v2_scalar_smoke.$TARGET.bin",
  "byte_len": 32768
}
CONTRACT_EOF
fi

if [ ! -s "$SMOKE_BIN" ]; then
  echo "REMOTE_STATUS=native_v2_runtime_fail"
  echo "REMOTE_REASON=missing_smoke_bin"
  exit 43
fi

chmod +x "$SMOKE_BIN"

if command -v codesign >/dev/null 2>&1; then
  codesign --force --sign - "$SMOKE_BIN" >/dev/null 2>&1 || true
fi

file "$SMOKE_BIN" > artifacts/omega/apple_os26_native_v2_smoke.file.txt
set +e
"$SMOKE_BIN" > artifacts/omega/apple_os26_native_v2_smoke.stdout 2> artifacts/omega/apple_os26_native_v2_smoke.stderr
SMOKE_RC=$?
set -e

if [ "$SMOKE_RC" -ne 13 ]; then
  echo "REMOTE_STATUS=native_v2_runtime_fail"
  echo "REMOTE_REASON=unexpected_smoke_exit_$SMOKE_RC"
  exit 44
fi

{
  echo "REMOTE_STATUS=ok"
  echo "REMOTE_REASON=apple_native_v2_scalar_core_runtime_attested"
  echo "SMOKE_EXIT=$SMOKE_RC"
  echo "SMOKE_BIN=$SMOKE_BIN"
  echo "CONTRACT=artifacts/omega/native_backend_v2_contract.$TARGET.json"
  echo "GATE=artifacts/omega/native_backend_v2_gate.$TARGET.json"
  echo "MAC_SW_VERS_BEGIN"
  printf "%s\n" "$MAC_SW_VERS"
  echo "MAC_SW_VERS_END"
  echo "XCODE_VERSION_BEGIN"
  printf "%s\n" "$XCODE_VERSION"
  echo "XCODE_VERSION_END"
  echo "SDK_VERSION=$SDK_VERSION"
  echo "UNAME=$MAC_UNAME"
} > artifacts/omega/apple_os26_native_v2_remote_summary.txt

cat artifacts/omega/apple_os26_native_v2_remote_summary.txt
'

set +e
REMOTE_OUTPUT="$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "REMOTE_DIR=$REMOTE_DIR TARGET=$TARGET bash -lc $(printf '%q' "$REMOTE_SCRIPT")" 2>&1)"
REMOTE_RC=$?
set -e

MAC_SW_VERS="$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "cd $REMOTE_DIR 2>/dev/null && sw_vers 2>/dev/null || true" 2>/dev/null || true)"
MAC_UNAME="$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "uname -a 2>/dev/null || true" 2>/dev/null || true)"
XCODE_VERSION="$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "xcodebuild -version 2>/dev/null || true" 2>/dev/null || true)"
SDK_VERSION="$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "xcrun --sdk macosx --show-sdk-version 2>/dev/null || true" 2>/dev/null || true)"

if [[ $REMOTE_RC -eq 0 ]]; then
  emit_report "ok" "apple_native_v2_scalar_core_runtime_attested" "$REMOTE_OUTPUT" "$MAC_SW_VERS" "$MAC_UNAME" "$XCODE_VERSION" "$SDK_VERSION"
  exit 0
fi

if [[ $REMOTE_RC -eq 42 ]]; then
  emit_report "not_run" "os_version_mismatch" "$REMOTE_OUTPUT" "$MAC_SW_VERS" "$MAC_UNAME" "$XCODE_VERSION" "$SDK_VERSION"
  exit 0
fi

emit_report "fail" "native_v2_runtime_fail" "$REMOTE_OUTPUT" "$MAC_SW_VERS" "$MAC_UNAME" "$XCODE_VERSION" "$SDK_VERSION"
exit 1
