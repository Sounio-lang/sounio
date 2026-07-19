#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
FRONTEND="$ROOT_DIR/self-hosted/compiler/module_frontend.sio"
SOURCE="tests/compiler/duplicate_module_name/main.sio"
DEPENDENCY="tests/compiler/duplicate_module_name/dep.sio"
SOURCE_ONLY=0

fail() {
  printf '[duplicate-module-name] FAIL: %s\n' "$*" >&2
  exit 1
}

if [[ "${1:-}" == "--source-only" ]]; then
  SOURCE_ONLY=1
elif [[ $# -ne 0 ]]; then
  fail "unexpected argument: $1"
fi

KEEP_WORK="${SOUNIO_DUPLICATE_MODULE_NAME_GATE_KEEP:-0}"

[[ -f "$FRONTEND" ]] || fail "missing module frontend: $FRONTEND"
[[ -f "$SOURCE" ]] || fail "missing main fixture: $SOURCE"
[[ -f "$DEPENDENCY" ]] || fail "missing dependency fixture: $DEPENDENCY"

python3 - "$FRONTEND" <<'PY' || exit 1
import re
import sys
from pathlib import Path

source = Path(sys.argv[1]).read_text(encoding="utf-8")

def function_body(name: str) -> str:
    match = re.search(r"(?:pub\s+)?fn\s+" + re.escape(name) + r"\s*\(", source)
    if match is None:
        raise AssertionError(f"missing_function_{name}")
    start = source.find("{", match.end())
    depth = 0
    for pos in range(start, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[start : pos + 1]
    raise AssertionError(f"unterminated_function_{name}")

try:
    logical_index = function_body("module_frontend_closure_logical_node_index")
    assert "str_len(logical_path) <= 0" in logical_index, "empty_logical_path_not_legacy"
    collector = function_body("module_frontend_collect_ast_closure_programs_into")
    collision = collector.index("module_frontend_closure_logical_node_index(")
    publication = collector.index("(*programs)[dependency_id] = *dependency_program", collision)
    marker = collector.index("module_closure: duplicate_logical_identity logical=", collision)
    assert collision < marker < publication, "collision_not_rejected_before_publication"
    assert "(*out).ambiguity_count = ambiguity_index + 1" in collector[marker:publication], "collision_not_classified"
except (AssertionError, ValueError) as exc:
    print(f"[duplicate-module-name] FAIL: source_contract_{exc}", file=sys.stderr)
    raise SystemExit(1)
PY

if [[ "$SOURCE_ONLY" -eq 1 ]]; then
  printf '%s\n' 'DUPLICATE_MODULE_NAME_SOURCE_RECEIPT status=pass logical=nonempty classification=ambiguity publication=refused marker=instrumented empty_logical=legacy'
  exit 0
fi

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]] ||
   [[ ! "$(uname -m 2>/dev/null || echo unknown)" =~ ^(x86_64|amd64)$ ]]; then
  printf '[duplicate-module-name] SKIP: Linux x86-64 compiler gate\n'
  exit 0
fi

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
sounio_require_madaros

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
if ! grep -Eq '^module_closure: duplicate_logical_identity logical=same first=.+ duplicate=.+$' "$CHECK_LOG"; then
  printf '[duplicate-module-name] FAIL: check log lacks duplicate logical identity marker\n' >&2
  contract_failed=1
fi
if ! grep -Eq '^module_closure: duplicate_logical_identity logical=same first=.+ duplicate=.+$' "$COMPILE_LOG"; then
  printf '[duplicate-module-name] FAIL: compile log lacks duplicate logical identity marker\n' >&2
  contract_failed=1
fi

if [[ "$contract_failed" -ne 0 ]]; then
  printf '%s\n' '--- check.log ---' >&2
  cat "$CHECK_LOG" >&2 || true
  printf '%s\n' '--- compile.log ---' >&2
  cat "$COMPILE_LOG" >&2 || true
  exit 1
fi

printf '[duplicate-module-name] PASS: duplicate logical module declarations reject with rc=1, causal marker, and no ELF\n'
