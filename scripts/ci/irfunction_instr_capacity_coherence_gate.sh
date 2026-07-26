#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${IRFUNCTION_INSTR_CAPACITY_SOURCE:-$ROOT/self-hosted/ir/ir.sio}"
EXPECTED_CAPACITY=4096

fail() {
  printf 'IRFUNCTION_INSTR_CAPACITY_FAIL reason=%s\n' "$1" >&2
  exit 1
}

[[ -f "$SOURCE" ]] || fail source_missing
command -v python3 >/dev/null 2>&1 || fail python3_missing

python3 - "$SOURCE" "$EXPECTED_CAPACITY" <<'PY' || exit $?
import pathlib
import re
import sys


source = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
text = source.read_text(encoding="utf-8")


def fail(reason: str) -> None:
    print(f"IRFUNCTION_INSTR_CAPACITY_FAIL reason={reason}", file=sys.stderr)
    raise SystemExit(1)


def unique(pattern: str, subject: str, reason: str, flags: int = 0) -> re.Match[str]:
    matches = list(re.finditer(pattern, subject, flags))
    if len(matches) != 1:
        fail(f"{reason}_count_{len(matches)}")
    return matches[0]


cap_match = unique(
    r"^pub let IR_MAX_INSTRS: i64 = ([0-9]+)\s*$",
    text,
    "ir_max_instrs",
    re.MULTILINE,
)

struct_match = unique(
    r"^pub struct IrFunction\s*\{(?P<body>.*?)^\}\s*$",
    text,
    "irfunction_struct",
    re.MULTILINE | re.DOTALL,
)
type_match = unique(
    r"^\s*pub instrs:\s*\[IrInstr;\s*([0-9]+)\],\s*$",
    struct_match.group("body"),
    "irfunction_instrs_field",
    re.MULTILINE,
)

function_starts = list(re.finditer(r"^pub fn ir_empty_function\(\) -> IrFunction\s*\{", text, re.MULTILINE))
if len(function_starts) != 1:
    fail(f"ir_empty_function_count_{len(function_starts)}")
module_start = text.find("\npub fn ir_empty_module()", function_starts[0].end())
if module_start < 0:
    fail("ir_empty_function_end_missing")
function_body = text[function_starts[0].end():module_start]
initializer_match = unique(
    r"^\s*instrs:\s*\[ir_nop\(\);\s*([0-9]+)\],\s*$",
    function_body,
    "ir_empty_function_instrs_initializer",
    re.MULTILINE,
)

cap = int(cap_match.group(1))
field = int(type_match.group(1))
initializer = int(initializer_match.group(1))

if cap != expected:
    fail(f"ir_max_instrs_expected_{expected}_got_{cap}")
if field != expected:
    fail(f"irfunction_field_expected_{expected}_got_{field}")
if initializer != expected:
    fail(f"initializer_expected_{expected}_got_{initializer}")
if not (cap == field == initializer):
    fail(f"divergent_cap_{cap}_field_{field}_initializer_{initializer}")

print(
    "IRFUNCTION_INSTR_CAPACITY_CHECK "
    f"cap={cap} field={field} initializer={initializer} coherent=pass"
)
PY

source_sha256="$(sha256sum "$SOURCE" | awk '{print $1}')"
head_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf not_available)"
tree_sha="$(git -C "$ROOT" rev-parse 'HEAD^{tree}' 2>/dev/null || printf not_available)"
worktree_state=not_available
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=all)" ]]; then
    worktree_state=clean
  else
    worktree_state=dirty
  fi
fi

printf '%s\n' \
  'IRFUNCTION_INSTR_CAPACITY_BOUNDARY storage_coherence=proved runtime_readback=not_claimed soir_v6=unchanged legacy=preserved'
printf 'IRFUNCTION_INSTR_CAPACITY_PASS source=%s source_sha256=%s expected=4096 head=%s tree=%s worktree=%s\n' \
  "$SOURCE" "$source_sha256" "$head_sha" "$tree_sha" "$worktree_state"
