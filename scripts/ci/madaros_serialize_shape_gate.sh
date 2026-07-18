#!/usr/bin/env bash
# Prove ir::serialize has no stale IR shape diagnostics under normal import.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${SOUNIO_MADAROS_SERIALIZE_SHAPE_BIN:-$ROOT_DIR/bin/madaros}"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
WORK="$(mktemp -d /tmp/sounio-madaros-serialize-shape.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "[madaros-serialize-shape] FAIL: $*" >&2
  exit 1
}

count_marker() {
  local marker="$1"
  local log="$2"
  grep -Fc "$marker" "$log" || true
}

[[ -x "$MADAROS" ]] || fail "Madaros wrapper is missing or not executable: $MADAROS"
[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit Madaros ELF'
[[ -x "$RAW_MADAROS" ]] || fail "explicit Madaros ELF is missing or not executable: $RAW_MADAROS"

SOURCE="$ROOT_DIR/tests/native-v2/soir_serialize_module_activation_witness.sio"
LOG="$WORK/check.log"
rc=0
set +e
SOUNIO_STDLIB_PATH="$ROOT_DIR/self-hosted" \
  MADAROS_RAW_BIN="$RAW_MADAROS" \
  "$MADAROS" check "$SOURCE" >"$LOG" 2>&1
rc=$?
set -e

grep -Fq 'could not resolve import' "$LOG" && {
  cat "$LOG" >&2
  fail 'normal ir::serialize import did not resolve'
}

module_count="$(sed -n 's/^run_check_mode: about to check \([0-9][0-9]*\)$/\1/p' "$LOG")"
[[ -n "$module_count" && "$module_count" -ge 2 ]] || {
  cat "$LOG" >&2
  fail "normal import did not activate a module closure (modules=${module_count:-missing})"
}

for code in E002 E016 E046 E137; do
  count="$(count_marker "error[$code" "$LOG")"
  [[ "$count" -eq 0 ]] || {
    cat "$LOG" >&2
    fail "mechanical shape diagnostic $code reappeared count=$count"
  }
done

diagnostics="$(count_marker 'error[E' "$LOG")"
e175="$(count_marker 'error[E175' "$LOG")"
e177="$(count_marker 'error[E177' "$LOG")"
compiler_sha256="$(sha256sum "$RAW_MADAROS" | awk '{print $1}')"

if [[ "$rc" -eq 1 && "$diagnostics" -eq 32 && "$e175" -eq 25 && "$e177" -eq 7 ]]; then
  echo "[madaros-serialize-shape] receipt state=visibility-blocked modules=$module_count diagnostics=$diagnostics E175=$e175 E177=$e177 compiler_sha256=$compiler_sha256"
  exit 0
fi

if [[ "$rc" -eq 0 && "$diagnostics" -eq 0 ]] && grep -Fq 'check: OK' "$LOG"; then
  echo "[madaros-serialize-shape] receipt state=resolved modules=$module_count diagnostics=0 compiler_sha256=$compiler_sha256"
  exit 0
fi

cat "$LOG" >&2
fail "unexpected diagnostic state rc=$rc diagnostics=$diagnostics E175=$e175 E177=$e177"
