#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$(uname -s 2>/dev/null || echo unknown):$(uname -m 2>/dev/null || echo unknown)" in
  Linux:x86_64|Linux:amd64) ;;
  *)
    echo "[native-v2-global-addr-ptr] SKIP: x86-64 Linux-only gate"
    exit 0
    ;;
esac

MADAROS="${MADAROS_LAUNCHER:-$ROOT_DIR/bin/madaros}"
TMP_DIR="$(mktemp -d /tmp/sounio-global-addr-ptr.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[native-v2-global-addr-ptr] FAIL: $*" >&2
  exit 1
}

[[ -x "$MADAROS" ]] || fail "Madaros launcher is not executable: $MADAROS"

ELF="$TMP_DIR/global_addr_ptr.elf"
LOG="$TMP_DIR/emit.log"

"$MADAROS" --native-v2-emit-global-aggregate "$ELF" >"$LOG" 2>&1 || {
  cat "$LOG" >&2
  fail "global aggregate IR emission failed"
}

grep -q 'label=global_aggregate' "$LOG" || {
  cat "$LOG" >&2
  fail "global aggregate emission did not report the expected label"
}

[[ -s "$ELF" ]] || fail "global aggregate witness ELF was not created"
chmod +x "$ELF"

set +e
"$ELF"
RC=$?
set -e

[[ "$RC" -eq 42 ]] || fail "global aggregate witness expected exit 42, got $RC"

echo "NATIVE_V2_GLOBAL_ADDR_PTR_PASS"
