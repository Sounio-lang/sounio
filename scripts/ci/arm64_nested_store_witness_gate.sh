#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${SOUNIO_ARM64_WITNESS_MODE:-attest}"
COMPILER="${SOUNIO_ARM64_WITNESS_COMPILER:-${1:-}}"
OUT_DIR="${SOUNIO_ARM64_WITNESS_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-arm64-nested-store.XXXXXX")}"
SOURCE="${SOUNIO_ARM64_WITNESS_SOURCE:-tests/run-pass/arm64_nested_deref_store.sio}"
TARGET="aarch64-macos"
BIN="$OUT_DIR/nested-ref-field-array-store"
COMPILE_LOG="$OUT_DIR/compile.log"
RUN_LOG="$OUT_DIR/run.log"
SUMMARY="$OUT_DIR/summary.txt"

portable_sha256() {
  shasum -a 256 "$1" 2>/dev/null | awk '{print $1}' || sha256sum "$1" | awk '{print $1}'
}

if [[ -z "$COMPILER" || ! -x "$COMPILER" ]]; then
  echo "error: set SOUNIO_ARM64_WITNESS_COMPILER to an executable source-fresh compiler" >&2
  exit 2
fi
if [[ "$MODE" != "attest" && "$MODE" != "execute" ]]; then
  echo "error: SOUNIO_ARM64_WITNESS_MODE must be attest or execute" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
"$COMPILER" "$SOURCE" "$BIN" --target "$TARGET" >"$COMPILE_LOG" 2>&1
chmod +x "$BIN" 2>/dev/null || true

FILE_KIND="$(file "$BIN")"
if [[ ! "$FILE_KIND" =~ Mach-O\ 64-bit.*arm64.*executable && ! "$FILE_KIND" =~ Mach-O\ 64-bit.*executable.*arm64 ]]; then
  echo "error: witness is not a Mach-O arm64 executable: $FILE_KIND" >&2
  exit 1
fi

if [[ "$MODE" == "attest" ]]; then
  cat >"$SUMMARY" <<EOF
status=ATTESTED
mode=attest
target=$TARGET
source=$SOURCE
compiler=$COMPILER
compiler_sha256=$(portable_sha256 "$COMPILER")
artifact=$BIN
artifact_sha256=$(portable_sha256 "$BIN")
semantic_execution=not_run
EOF
  echo "ARM64_NESTED_STORE_WITNESS_ATTESTED target=$TARGET semantic_execution=not_run"
  echo "ARM64_NESTED_STORE_WITNESS_ARTIFACT_DIR=$OUT_DIR"
  exit 0
fi

HOST_OS="$(uname -s 2>/dev/null || echo unknown)"
HOST_ARCH="$(uname -m 2>/dev/null || echo unknown)"
if [[ "$HOST_OS:$HOST_ARCH" != "Darwin:arm64" && "$HOST_OS:$HOST_ARCH" != "Darwin:aarch64" ]]; then
  echo "error: execute mode requires Apple Silicon; host=$HOST_OS:$HOST_ARCH" >&2
  exit 2
fi

if [[ -f scripts/lib/macos_codesign.sh ]]; then
  # shellcheck source=/dev/null
  source scripts/lib/macos_codesign.sh
  sounio_ad_hoc_codesign "$BIN"
fi

set +e
"$BIN" >"$RUN_LOG" 2>&1
RUN_RC=$?
set -e
if [[ $RUN_RC -ne 0 || "$(cat "$RUN_LOG")" != "PASS" ]]; then
  echo "error: ARM64 nested-store witness failed with rc=$RUN_RC" >&2
  cat "$RUN_LOG" >&2
  exit 1
fi

cat >"$SUMMARY" <<EOF
status=EXECUTED
mode=execute
target=$TARGET
source=$SOURCE
compiler=$COMPILER
compiler_sha256=$(portable_sha256 "$COMPILER")
artifact=$BIN
artifact_sha256=$(portable_sha256 "$BIN")
semantic_execution=pass
EOF
echo "ARM64_NESTED_STORE_WITNESS_EXECUTED target=$TARGET semantic_execution=pass"
echo "ARM64_NESTED_STORE_WITNESS_ARTIFACT_DIR=$OUT_DIR"
