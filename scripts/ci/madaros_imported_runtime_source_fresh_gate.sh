#!/usr/bin/env bash
# #901: bind imported-layout runtime acceptance to a compiler built in this tree.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_SCRIPT="$ROOT_DIR/scripts/ci/build_modular_madaros.sh"
ACCEPTANCE_GATE="$ROOT_DIR/scripts/ci/madaros_imported_runtime_acceptance_gate.sh"
KEEP_WORK="${SOUNIO_MADAROS_ISSUE901_SOURCE_FRESH_KEEP:-0}"

fail() {
  echo "[madaros-issue901-source-fresh] FAIL: $*" >&2
  exit 1
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

if [[ "${1:-}" == '--structural-only' ]]; then
  [[ $# -eq 1 ]] || fail 'usage: madaros_imported_runtime_source_fresh_gate.sh [--structural-only]'
  [[ -x "$BUILD_SCRIPT" ]] || fail "missing source build script: $BUILD_SCRIPT"
  [[ -x "$ACCEPTANCE_GATE" ]] || fail "missing raw-ELF acceptance gate: $ACCEPTANCE_GATE"
  [[ -f "$ROOT_DIR/self-hosted/compiler/main.sio" ]] || fail 'missing modular compiler entry source'
  [[ -f "$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_nested_field_chain_main.sio" ]] || fail 'missing #901 positive witness'
  [[ -f "$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_known_layout_miss_main.sio" ]] || fail 'missing #901 negative witness'
  echo '[madaros-issue901-source-fresh] PASS: structural source-build and direct-raw acceptance wiring is present'
  exit 0
fi
[[ $# -eq 0 ]] || fail 'usage: madaros_imported_runtime_source_fresh_gate.sh [--structural-only]'

cd "$ROOT_DIR"
[[ -z "$(git status --porcelain)" ]] || fail 'source tree must be clean; commit the source under test before claiming source-fresh evidence'

if [[ -n "${SOUNIO_MADAROS_ISSUE901_SOURCE_FRESH_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_ISSUE901_SOURCE_FRESH_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-issue901-source-fresh.XXXXXX)"
fi
if [[ "$KEEP_WORK" != '1' ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

SOURCE_HEAD="$(git rev-parse HEAD)"
SOURCE_TREE="$(git rev-parse 'HEAD^{tree}')"
MAIN_SHA256="$(portable_sha256 self-hosted/compiler/main.sio)"
LEAN_SHA256="$(portable_sha256 self-hosted/compiler/lean_single.sio)"
BUILDER_SHA256="$(portable_sha256 "$BUILD_SCRIPT")"
RAW_MADAROS="$WORK/madaros"
BUILD_LOG="$WORK/source-build.log"
RECEIPT="$WORK/source-build-receipt.tsv"

if ! env \
  -u MADAROS_RAW_BIN \
  -u SOUNIO_MADAROS_BIN \
  -u SOUC_BIN \
  -u SOUNIO_SOUC_BIN \
  SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
  bash "$BUILD_SCRIPT" "$RAW_MADAROS" >"$BUILD_LOG" 2>&1; then
  tail -n 120 "$BUILD_LOG" >&2 || true
  fail 'current-source Madaros build failed'
fi

[[ -z "$(git status --porcelain)" ]] || fail 'source tree changed while building the claimed current-source compiler'
[[ "$(git rev-parse HEAD)" == "$SOURCE_HEAD" ]] || fail 'source HEAD changed while building the claimed compiler'
[[ "$(git rev-parse 'HEAD^{tree}')" == "$SOURCE_TREE" ]] || fail 'source tree changed while building the claimed compiler'
[[ -x "$RAW_MADAROS" ]] || fail "source build did not emit an executable Madaros ELF: $RAW_MADAROS"
[[ "$(head -c2 "$RAW_MADAROS" 2>/dev/null)" != '#!' ]] || fail 'source build emitted a wrapper instead of an ELF'
RAW_SHA256="$(portable_sha256 "$RAW_MADAROS")"

printf 'receipt_version\tissue901-source-fresh-v1\n' >"$RECEIPT"
printf 'source_head\t%s\n' "$SOURCE_HEAD" >>"$RECEIPT"
printf 'source_tree\t%s\n' "$SOURCE_TREE" >>"$RECEIPT"
printf 'main_sio_sha256\t%s\n' "$MAIN_SHA256" >>"$RECEIPT"
printf 'lean_single_sio_sha256\t%s\n' "$LEAN_SHA256" >>"$RECEIPT"
printf 'build_script_sha256\t%s\n' "$BUILDER_SHA256" >>"$RECEIPT"
printf 'raw_madaros_sha256\t%s\n' "$RAW_SHA256" >>"$RECEIPT"
printf 'acceptance_mode\tdirect-raw-elf-no-wrapper\n' >>"$RECEIPT"

MADAROS_RAW_BIN="$RAW_MADAROS" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_EXPECTED_SHA256="$RAW_SHA256" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_DIR="$WORK/acceptance" \
SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_KEEP=1 \
  bash "$ACCEPTANCE_GATE" >"$WORK/acceptance.log" 2>&1 || {
    cat "$WORK/acceptance.log" >&2
    fail 'source-built Madaros did not satisfy the direct raw #901 acceptance gate'
  }

[[ -z "$(git status --porcelain)" ]] || fail 'source tree changed during direct raw #901 acceptance'
[[ "$(git rev-parse HEAD)" == "$SOURCE_HEAD" ]] || fail 'source HEAD changed during direct raw #901 acceptance'
[[ "$(git rev-parse 'HEAD^{tree}')" == "$SOURCE_TREE" ]] || fail 'source tree changed during direct raw #901 acceptance'

cat "$RECEIPT"
echo "[madaros-issue901-source-fresh] PASS: source_head=$SOURCE_HEAD source_tree=$SOURCE_TREE raw_sha256=$RAW_SHA256 receipt=$RECEIPT"
