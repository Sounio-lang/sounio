#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() {
  printf '[duplicate-module-name] FAIL: %s\n' "$*" >&2
  exit 1
}

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]] ||
   [[ ! "$(uname -m 2>/dev/null || echo unknown)" =~ ^(x86_64|amd64)$ ]]; then
  printf '[duplicate-module-name] SKIP: Linux x86-64 compiler gate\n'
  exit 0
fi

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros

SOURCE="tests/compiler/duplicate_module_name/main.sio"
DEPENDENCY="tests/compiler/duplicate_module_name/dep.sio"
KEEP_WORK="${SOUNIO_DUPLICATE_MODULE_NAME_GATE_KEEP:-0}"

[[ -f "$SOURCE" ]] || fail "missing main fixture: $SOURCE"
[[ -f "$DEPENDENCY" ]] || fail "missing dependency fixture: $DEPENDENCY"

if [[ -n "${SOUNIO_DUPLICATE_MODULE_NAME_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_DUPLICATE_MODULE_NAME_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-duplicate-module-name.XXXXXX)"
fi

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

CHECK_LOG="$WORK/check.log"
COMPILE_LOG="$WORK/compile.log"
ELF="$WORK/duplicate-module-name.elf"

rm -f "$ELF"

set +e
"$MADAROS_BIN" check "$SOURCE" >"$CHECK_LOG" 2>&1
check_rc=$?
"$MADAROS_BIN" compile "$SOURCE" -o "$ELF" >"$COMPILE_LOG" 2>&1
compile_rc=$?
set -e

elf_state=absent
if [[ -e "$ELF" ]]; then
  elf_state=present
fi

printf 'DUPLICATE_MODULE_NAME_RESULT check_rc=%s compile_rc=%s elf=%s\n' \
  "$check_rc" "$compile_rc" "$elf_state"

contract_failed=0
if [[ "$check_rc" -ne 1 ]]; then
  printf '[duplicate-module-name] FAIL: check must reject with rc=1, got rc=%s\n' "$check_rc" >&2
  contract_failed=1
fi
if [[ "$compile_rc" -ne 1 ]]; then
  printf '[duplicate-module-name] FAIL: compile must reject with rc=1, got rc=%s\n' "$compile_rc" >&2
  contract_failed=1
fi
if [[ "$elf_state" != absent ]]; then
  printf '[duplicate-module-name] FAIL: rejected compile left an ELF: %s\n' "$ELF" >&2
  contract_failed=1
fi

if [[ "$contract_failed" -ne 0 ]]; then
  printf '%s\n' '--- check.log ---' >&2
  cat "$CHECK_LOG" >&2 || true
  printf '%s\n' '--- compile.log ---' >&2
  cat "$COMPILE_LOG" >&2 || true
  exit 1
fi

printf '[duplicate-module-name] PASS: duplicate logical module declarations reject with rc=1 and no ELF\n'
