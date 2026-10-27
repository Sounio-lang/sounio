#!/usr/bin/env bash
# Build from an explicit current-source lean_single Mach-O seed and publish the
# local artifact only after executing the end-to-end acceptance gate.
# Prerequisites: Apple command-line tools and GNU timeout (brew install coreutils).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$(uname -s)/$(uname -m)" != Darwin/arm64 ]]; then
    echo 'ERROR: this build requires macOS arm64' >&2
    exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo 'ERROR: GNU timeout is required before building; install it with: brew install coreutils' >&2
    exit 2
fi
SEED_BIN="${1:?pass the current-source lean_single macOS arm64 seed}"
case "$SEED_BIN" in /*) ;; *) SEED_BIN="$PWD/$SEED_BIN" ;; esac
OUT="${2:-$ROOT_DIR/artifacts/self-hosted/madaros-arm64-macos}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
test -x "$SEED_BIN"
file "$SEED_BIN" | grep -Eq 'Mach-O 64-bit.*arm64'
codesign --verify "$SEED_BIN"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/madaros-macos-build.XXXXXX")"
echo "build_artifacts=$BUILD_DIR"
persist_madaros_build_log() {
    local gate_dir="${SOUNIO_SELFHOST_HOST_GATE_DIR:-}"
    if [[ -n "$gate_dir" && -f "$BUILD_DIR/build.log" ]]; then
        mkdir -p "$gate_dir/logs"
        cp -f "$BUILD_DIR/build.log" "$gate_dir/logs/madaros-build.log" || true
    fi
}
trap persist_madaros_build_log EXIT
cd "$ROOT_DIR"
export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
"$SEED_BIN" tests/run-pass/a64_macos_mapped_stack.sio "$BUILD_DIR/stack-probe" --target aarch64-macos >"$BUILD_DIR/stack-build.log" 2>&1
chmod +x "$BUILD_DIR/stack-probe"
codesign -s - -f "$BUILD_DIR/stack-probe"
"$BUILD_DIR/stack-probe" >"$BUILD_DIR/stack.stdout.log"
test "$(<"$BUILD_DIR/stack.stdout.log")" = 20000
"$SEED_BIN" self-hosted/compiler/main.sio "$BUILD_DIR/madaros" --target aarch64-macos >"$BUILD_DIR/build.log" 2>&1
test -s "$BUILD_DIR/madaros"
file "$BUILD_DIR/madaros" | grep -Eq 'Mach-O 64-bit.*arm64'
chmod +x "$BUILD_DIR/madaros"
codesign -s - -f "$BUILD_DIR/madaros"
codesign --verify "$BUILD_DIR/madaros"
timeout 120 bash "$ROOT_DIR/scripts/ci/madaros_macos_arm64_acceptance_gate.sh" "$BUILD_DIR/madaros" | tee "$BUILD_DIR/acceptance.log"
# A scratch candidate may be inspected after the bounded ABI gate. Publishing
# the public host requires the full pinned-member corpus and real Metal receipts.
if [[ "$OUT" == "$ROOT_DIR/artifacts/self-hosted/madaros-arm64-macos" ]]; then
    bash "$ROOT_DIR/scripts/ci/madaros_macos_wide_ir_gate.sh" "$SEED_BIN" "$BUILD_DIR/wide-ir" | tee "$BUILD_DIR/wide-ir.log"
    bash "$ROOT_DIR/scripts/ci/madaros_macos_port_regressions_gate.sh" "$BUILD_DIR/madaros" "$BUILD_DIR/regressions" | tee "$BUILD_DIR/regressions.log"
    bash "$ROOT_DIR/scripts/ci/madaros_macos_corpus_gate.sh" "$BUILD_DIR/madaros" "$BUILD_DIR/corpus" | tee "$BUILD_DIR/corpus.log"
fi
mkdir -p "$(dirname "$OUT")"
install -m 755 "$BUILD_DIR/madaros" "$OUT"
codesign --verify "$OUT"
echo "Madaros macOS arm64 ready: $OUT"
