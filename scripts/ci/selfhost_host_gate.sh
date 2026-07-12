#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MACOS_CODESIGN_HELPER="$ROOT_DIR/scripts/lib/macos_codesign.sh"
if [[ -f "$MACOS_CODESIGN_HELPER" ]]; then
  # shellcheck source=/dev/null
  source "$MACOS_CODESIGN_HELPER"
fi

host_platform() {
  local host_os="${SOUNIO_HOST_OS_OVERRIDE:-$(uname -s 2>/dev/null || echo unknown)}"
  local host_arch="${SOUNIO_HOST_ARCH_OVERRIDE:-$(uname -m 2>/dev/null || echo unknown)}"
  printf '%s:%s\n' "$host_os" "$host_arch"
}

host_target() {
  case "$1" in
    Linux:x86_64|Linux:amd64) printf '%s\n' "x86_64-linux" ;;
    Linux:arm64|Linux:aarch64) printf '%s\n' "aarch64-linux" ;;
    Darwin:arm64|Darwin:aarch64) printf '%s\n' "aarch64-macos" ;;
    Darwin:x86_64|Darwin:amd64) printf '%s\n' "x86_64-macos" ;;
    *)
      echo "error: unsupported host platform: $1" >&2
      return 1
      ;;
  esac
}

expected_file_kind() {
  case "$1" in
    x86_64-linux) printf '%s\n' "ELF 64-bit LSB executable, x86-64" ;;
    aarch64-linux) printf '%s\n' "ELF 64-bit LSB executable, ARM aarch64" ;;
    aarch64-macos) printf '%s\n' "Mach-O 64-bit arm64 executable|Mach-O 64-bit executable arm64" ;;
    x86_64-macos) printf '%s\n' "Mach-O 64-bit x86_64 executable|Mach-O 64-bit executable x86_64" ;;
    *)
      echo "error: unsupported target triple: $1" >&2
      return 1
      ;;
  esac
}

secondary_target() {
  case "$1" in
    Linux:x86_64|Linux:amd64) printf '%s\n' "aarch64-macos" ;;
    Linux:arm64|Linux:aarch64) printf '%s\n' "x86_64-macos" ;;
    Darwin:arm64|Darwin:aarch64) printf '%s\n' "x86_64-macos" ;;
    Darwin:x86_64|Darwin:amd64) printf '%s\n' "aarch64-macos" ;;
    *)
      return 1
      ;;
  esac
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

portable_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

mk_work_dir() {
  local base="${TMPDIR:-/tmp}/sounio-selfhost-host-XXXXXX"
  mktemp -d "$base"
}

assert_file_kind() {
  local path="$1"
  local expected="$2"
  local actual
  local variant

  actual="$(file "$path" 2>/dev/null || true)"
  IFS='|' read -r -a expected_variants <<<"$expected"
  for variant in "${expected_variants[@]}"; do
    if [[ "$actual" == *"$variant"* ]]; then
      return 0
    fi
  done

  echo "error: unexpected artifact kind for $path" >&2
  echo "expected fragment: $expected" >&2
  echo "actual: $actual" >&2
  exit 1
}

assert_output_equals() {
  local expected="$1"
  local actual="$2"
  local label="$3"

  if [[ "$actual" != "$expected" ]]; then
    echo "error: unexpected output for $label" >&2
    echo "expected: $(printf '%q' "$expected")" >&2
    echo "actual:   $(printf '%q' "$actual")" >&2
    exit 1
  fi
}

maybe_codesign() {
  local path="$1"
  if declare -F sounio_ad_hoc_codesign >/dev/null 2>&1; then
    sounio_ad_hoc_codesign "$path"
  fi
}

emit_summary() {
  local summary_path="$1"
  local host_platform="$2"
  local host_target="$3"
  local native_bin="$4"
  local stage2_bin="$5"
  local stage3_bin="$6"
  local host_bin="$7"
  local cross_bin="$8"
  local cross_target="$9"

  cat >"$summary_path" <<EOF
host_platform=$host_platform
host_target=$host_target
native_bin=$native_bin
native_bin_sha256=$(portable_sha256 "$native_bin")
native_bin_bytes=$(portable_size "$native_bin")
stage2_bin=$stage2_bin
stage2_sha256=$(portable_sha256 "$stage2_bin")
stage3_bin=$stage3_bin
stage3_sha256=$(portable_sha256 "$stage3_bin")
host_smoke_bin=$host_bin
host_smoke_sha256=$(portable_sha256 "$host_bin")
cross_target=$cross_target
cross_smoke_bin=$cross_bin
cross_smoke_sha256=$(portable_sha256 "$cross_bin")
EOF
}

HOST_PLATFORM="$(host_platform)"
HOST_TARGET="$(host_target "$HOST_PLATFORM")"
HOST_FILE_KIND="$(expected_file_kind "$HOST_TARGET")"
SECONDARY_TARGET="$(secondary_target "$HOST_PLATFORM")"
SECONDARY_FILE_KIND="$(expected_file_kind "$SECONDARY_TARGET")"

WORK_DIR="${SOUNIO_SELFHOST_HOST_GATE_DIR:-$(mk_work_dir)}"
ARTIFACT_DIR="$WORK_DIR/artifacts"
LOG_DIR="$WORK_DIR/logs"
SUMMARY_PATH="$WORK_DIR/summary.txt"

mkdir -p "$ARTIFACT_DIR" "$LOG_DIR"

NATIVE_BIN="$ARTIFACT_DIR/souc-host"
STAGE2_BIN="$ARTIFACT_DIR/souc-stage2"
STAGE3_BIN="$ARTIFACT_DIR/souc-stage3"
HOST_SMOKE_BIN="$ARTIFACT_DIR/host-smoke"
PRINT_SMOKE_BIN="$ARTIFACT_DIR/print-f64-smoke"
READ_SMOKE_BIN="$ARTIFACT_DIR/read64-smoke"
READ_DATA_BIN="$ARTIFACT_DIR/read64.bin"
CROSS_SMOKE_BIN="$ARTIFACT_DIR/cross-smoke"

echo "SELFHOST_HOST_GATE_START host_platform=$HOST_PLATFORM host_target=$HOST_TARGET work_dir=$WORK_DIR"

bash "$ROOT_DIR/scripts/ci/build_native_souc.sh" "$NATIVE_BIN" >"$LOG_DIR/build-native.log" 2>&1
chmod +x "$NATIVE_BIN" 2>/dev/null || true
maybe_codesign "$NATIVE_BIN"
assert_file_kind "$NATIVE_BIN" "$HOST_FILE_KIND"

if [[ "$HOST_PLATFORM" == Darwin:* && "${SOUNIO_DARWIN_SELFHOST_EXEC_MODE:-full}" == "attest" ]]; then
  if command -v codesign >/dev/null 2>&1; then
    codesign --verify "$NATIVE_BIN" >"$LOG_DIR/native-codesign-verify.log" 2>&1 || true
  fi
  cat >"$SUMMARY_PATH" <<EOF
host_platform=$HOST_PLATFORM
host_target=$HOST_TARGET
mode=attest
reason=darwin_host_execution_blocked
native_bin=$NATIVE_BIN
native_bin_sha256=$(portable_sha256 "$NATIVE_BIN")
native_bin_bytes=$(portable_size "$NATIVE_BIN")
EOF
  echo "SELFHOST_HOST_GATE_ATTEST host_platform=$HOST_PLATFORM host_target=$HOST_TARGET native_sha256=$(portable_sha256 "$NATIVE_BIN") reason=darwin_host_execution_blocked"
  echo "SELFHOST_HOST_GATE_ARTIFACT_DIR=$WORK_DIR"
  exit 0
fi

"$NATIVE_BIN" self-hosted/compiler/lean_single.sio "$STAGE2_BIN" --target "$HOST_TARGET" >"$LOG_DIR/stage2.log" 2>&1
chmod +x "$STAGE2_BIN" 2>/dev/null || true
maybe_codesign "$STAGE2_BIN"
assert_file_kind "$STAGE2_BIN" "$HOST_FILE_KIND"

"$STAGE2_BIN" self-hosted/compiler/lean_single.sio "$STAGE3_BIN" --target "$HOST_TARGET" >"$LOG_DIR/stage3.log" 2>&1
chmod +x "$STAGE3_BIN" 2>/dev/null || true
maybe_codesign "$STAGE3_BIN"
assert_file_kind "$STAGE3_BIN" "$HOST_FILE_KIND"

if ! bash "$ROOT_DIR/scripts/lib/compare_executable_payloads.sh" "$STAGE2_BIN" "$STAGE3_BIN"; then
  echo "error: self-host fixed-point mismatch for $HOST_TARGET" >&2
  exit 1
fi

"$STAGE2_BIN" examples/hello.sio "$HOST_SMOKE_BIN" --target "$HOST_TARGET" >"$LOG_DIR/hello-host.log" 2>&1
chmod +x "$HOST_SMOKE_BIN" 2>/dev/null || true
maybe_codesign "$HOST_SMOKE_BIN"
assert_file_kind "$HOST_SMOKE_BIN" "$HOST_FILE_KIND"

"$STAGE2_BIN" self-hosted/compiler/native_print_f64_smoke.sio "$PRINT_SMOKE_BIN" --target "$HOST_TARGET" >"$LOG_DIR/print-f64.log" 2>&1
chmod +x "$PRINT_SMOKE_BIN" 2>/dev/null || true
maybe_codesign "$PRINT_SMOKE_BIN"
assert_file_kind "$PRINT_SMOKE_BIN" "$HOST_FILE_KIND"
PRINT_OUTPUT="$("$PRINT_SMOKE_BIN" 2>&1)"
assert_output_equals $'3.141590\n-0.500000\n2.000000' "$PRINT_OUTPUT" "native_print_f64_smoke"

# Write test binary: little-endian f64(3.14159) f64(-0.5) i64(42)
printf '\x6e\x86\x1b\xf0\xf9\x21\x09\x40\x00\x00\x00\x00\x00\x00\xe0\xbf\x2a\x00\x00\x00\x00\x00\x00\x00' > "$READ_DATA_BIN"

"$STAGE2_BIN" self-hosted/compiler/native_read64_smoke.sio "$READ_SMOKE_BIN" --target "$HOST_TARGET" >"$LOG_DIR/read64.log" 2>&1
chmod +x "$READ_SMOKE_BIN" 2>/dev/null || true
maybe_codesign "$READ_SMOKE_BIN"
assert_file_kind "$READ_SMOKE_BIN" "$HOST_FILE_KIND"
READ_OUTPUT="$("$READ_SMOKE_BIN" "$READ_DATA_BIN" 2>&1)"
assert_output_equals $'3.141590\n-0.500000\n42' "$READ_OUTPUT" "native_read64_smoke"

"$STAGE2_BIN" examples/hello.sio "$CROSS_SMOKE_BIN" --target "$SECONDARY_TARGET" >"$LOG_DIR/cross-smoke.log" 2>&1
chmod +x "$CROSS_SMOKE_BIN" 2>/dev/null || true
maybe_codesign "$CROSS_SMOKE_BIN"
assert_file_kind "$CROSS_SMOKE_BIN" "$SECONDARY_FILE_KIND"

emit_summary \
  "$SUMMARY_PATH" \
  "$HOST_PLATFORM" \
  "$HOST_TARGET" \
  "$NATIVE_BIN" \
  "$STAGE2_BIN" \
  "$STAGE3_BIN" \
  "$HOST_SMOKE_BIN" \
  "$CROSS_SMOKE_BIN" \
  "$SECONDARY_TARGET"

echo "SELFHOST_HOST_GATE_PASS host_platform=$HOST_PLATFORM host_target=$HOST_TARGET stage2_sha256=$(portable_sha256 "$STAGE2_BIN")"
echo "SELFHOST_HOST_GATE_ARTIFACT_DIR=$WORK_DIR"
