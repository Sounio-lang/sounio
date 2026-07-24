#!/usr/bin/env bash
# Build Madaros from this clean checkout, then exercise the epistemic receipt
# imports directly on that raw ELF. This is provenance evidence, not a
# psychiatric, medical, or clinical decision procedure.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_SCRIPT="$ROOT_DIR/scripts/ci/build_modular_madaros.sh"
KEEP_WORK="${SOUNIO_EPISTEMIC_RECEIPT_SOURCE_FRESH_KEEP:-0}"

OBSERVATION_SMOKE="$ROOT_DIR/tests/run-pass/epistemic_observation_provenance_import_smoke.sio"
PARENTHESIZATION_SMOKE="$ROOT_DIR/tests/run-pass/epistemic_parenthesization_receipts_import_smoke.sio"
STATE_ALIASING_SMOKE="$ROOT_DIR/tests/run-pass/epistemic_state_aliasing_receipts_import_smoke.sio"

fail() {
  echo "[epistemic-receipt-source-fresh] FAIL: $*" >&2
  exit 1
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

assert_raw_elf() {
  local path="$1"
  local magic

  [[ -f "$path" && -s "$path" && -x "$path" ]] || fail "current-source build did not emit an executable ELF: $path"
  magic="$(od -An -tx1 -N4 "$path" 2>/dev/null | tr -d ' \n')"
  [[ "$magic" == 7f454c46 ]] || fail "current-source build did not emit an ELF: $path"
  [[ "$(head -c2 "$path" 2>/dev/null)" != '#!' ]] || fail 'current-source build emitted a wrapper instead of an ELF'
}

assert_no_fallback_marker() {
  local log="$1"

  if grep -Eiq 'source=fallback|fallback=|SELFHOST=fallback|native_prebundle:|compact modular IR table path' "$log"; then
    cat "$log" >&2
    fail "fallback or compact-imported-IR marker observed in $log"
  fi
}

run_direct_smoke() {
  local label="$1"
  local source="$2"
  local marker="$3"
  local log="$WORK/$label.run.log"
  local cwd="$WORK/$label.run-cwd"
  local rc

  mkdir -p "$cwd"
  set +e
  (
    cd "$cwd"
    exec env \
      -u MADAROS_RAW_BIN \
      -u SOUNIO_MADAROS_BIN \
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
      "$RAW_MADAROS" run "$source"
  ) >"$log" 2>&1
  rc=$?
  set -e

  [[ "$rc" -eq 0 ]] || {
    cat "$log" >&2
    fail "$label direct raw smoke exited rc=$rc"
  }
  assert_no_fallback_marker "$log"
  grep -Fxq "$marker" "$log" || {
    cat "$log" >&2
    fail "$label direct raw smoke omitted exact marker $marker"
  }
}

if [[ "${1:-}" == '--structural-only' ]]; then
  [[ $# -eq 1 ]] || fail 'usage: epistemic_receipt_source_fresh_gate.sh [--structural-only]'
  [[ -x "$BUILD_SCRIPT" ]] || fail "missing current-source builder: $BUILD_SCRIPT"
  [[ -f "$ROOT_DIR/self-hosted/compiler/main.sio" ]] || fail 'missing modular compiler entry source'
  [[ -f "$OBSERVATION_SMOKE" ]] || fail "missing observation smoke: $OBSERVATION_SMOKE"
  [[ -f "$PARENTHESIZATION_SMOKE" ]] || fail "missing parenthesization smoke: $PARENTHESIZATION_SMOKE"
  [[ -f "$STATE_ALIASING_SMOKE" ]] || fail "missing state-aliasing smoke: $STATE_ALIASING_SMOKE"
  echo '[epistemic-receipt-source-fresh] PASS: structural source-build and direct-raw receipt wiring is present'
  exit 0
fi
[[ $# -eq 0 ]] || fail 'usage: epistemic_receipt_source_fresh_gate.sh [--structural-only]'

cd "$ROOT_DIR"
[[ -z "$(git status --porcelain)" ]] || fail 'source tree must be clean before source-fresh evidence can be claimed'

if [[ -n "${SOUNIO_EPISTEMIC_RECEIPT_SOURCE_FRESH_DIR:-}" ]]; then
  WORK="$SOUNIO_EPISTEMIC_RECEIPT_SOURCE_FRESH_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-epistemic-receipt-source-fresh.XXXXXX)"
fi
if [[ "$KEEP_WORK" != '1' ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

SOURCE_HEAD="$(git rev-parse HEAD)"
SOURCE_TREE="$(git rev-parse 'HEAD^{tree}')"
MAIN_SHA256="$(portable_sha256 self-hosted/compiler/main.sio)"
LEAN_SHA256="$(portable_sha256 self-hosted/compiler/lean_single.sio)"
BUILDER_SHA256="$(portable_sha256 "$BUILD_SCRIPT")"
OBSERVATION_SHA256="$(portable_sha256 stdlib/epistemic/observation_provenance.sio)"
PARENTHESIZATION_SHA256="$(portable_sha256 stdlib/epistemic/parenthesization_receipts.sio)"
STATE_ALIASING_SHA256="$(portable_sha256 stdlib/epistemic/state_aliasing_receipts.sio)"
RAW_MADAROS="$WORK/madaros"
BUILD_LOG="$WORK/source-build.log"
RECEIPT="$WORK/source-fresh-receipt.tsv"

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

[[ -z "$(git status --porcelain)" ]] || fail 'source tree changed while building the claimed compiler'
[[ "$(git rev-parse HEAD)" == "$SOURCE_HEAD" ]] || fail 'source HEAD changed while building the claimed compiler'
[[ "$(git rev-parse 'HEAD^{tree}')" == "$SOURCE_TREE" ]] || fail 'source tree changed while building the claimed compiler'
assert_raw_elf "$RAW_MADAROS"
RAW_SHA256="$(portable_sha256 "$RAW_MADAROS")"

run_direct_smoke observation "$OBSERVATION_SMOKE" 'EPISTEMIC_OBSERVATION_PROVENANCE_IMPORT_SMOKE_PASS'
run_direct_smoke parenthesization "$PARENTHESIZATION_SMOKE" 'EPISTEMIC_PARENTHESIZATION_RECEIPTS_IMPORT_SMOKE_PASS'
run_direct_smoke state_aliasing "$STATE_ALIASING_SMOKE" 'EPISTEMIC_STATE_ALIASING_RECEIPTS_IMPORT_SMOKE_PASS'

[[ -z "$(git status --porcelain)" ]] || fail 'source tree changed during direct raw receipt evidence'
[[ "$(git rev-parse HEAD)" == "$SOURCE_HEAD" ]] || fail 'source HEAD changed during direct raw receipt evidence'
[[ "$(git rev-parse 'HEAD^{tree}')" == "$SOURCE_TREE" ]] || fail 'source tree changed during direct raw receipt evidence'

printf 'receipt_version\tepistemic-receipt-source-fresh-v1\n' >"$RECEIPT"
printf 'source_head\t%s\n' "$SOURCE_HEAD" >>"$RECEIPT"
printf 'source_tree\t%s\n' "$SOURCE_TREE" >>"$RECEIPT"
printf 'main_sio_sha256\t%s\n' "$MAIN_SHA256" >>"$RECEIPT"
printf 'lean_single_sio_sha256\t%s\n' "$LEAN_SHA256" >>"$RECEIPT"
printf 'build_script_sha256\t%s\n' "$BUILDER_SHA256" >>"$RECEIPT"
printf 'observation_provenance_sha256\t%s\n' "$OBSERVATION_SHA256" >>"$RECEIPT"
printf 'parenthesization_receipts_sha256\t%s\n' "$PARENTHESIZATION_SHA256" >>"$RECEIPT"
printf 'state_aliasing_receipts_sha256\t%s\n' "$STATE_ALIASING_SHA256" >>"$RECEIPT"
printf 'raw_madaros_sha256\t%s\n' "$RAW_SHA256" >>"$RECEIPT"
printf 'execution_mode\tdirect-raw-elf-no-wrapper\n' >>"$RECEIPT"

cat "$RECEIPT"
echo "[epistemic-receipt-source-fresh] PASS: source_head=$SOURCE_HEAD source_tree=$SOURCE_TREE raw_sha256=$RAW_SHA256 receipt=$RECEIPT"
