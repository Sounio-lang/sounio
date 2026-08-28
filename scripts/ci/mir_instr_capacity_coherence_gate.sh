#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${MIR_INSTR_CAPACITY_SOURCE:-$ROOT/self-hosted/native/machine_ir.sio}"
EXPECTED_CAPACITY=4096

fail() {
  printf 'MIR_INSTR_CAPACITY_FAIL reason=%s\n' "$1" >&2
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
    print(f"MIR_INSTR_CAPACITY_FAIL reason={reason}", file=sys.stderr)
    raise SystemExit(1)


def unique(pattern: str, subject: str, reason: str, flags: int = 0) -> re.Match:
    matches = list(re.finditer(pattern, subject, flags))
    if len(matches) != 1:
        fail(f"{reason}_count_{len(matches)}")
    return matches[0]


cap_match = unique(
    r"^pub let MIR_MAX_INSTRS: i64 = ([0-9]+)\s*$",
    text,
    "mir_max_instrs",
    re.MULTILINE,
)

struct_match = unique(
    r"^pub struct MachineBlock\s*\{(?P<body>.*?)^\}\s*$",
    text,
    "machineblock_struct",
    re.MULTILINE | re.DOTALL,
)
type_match = unique(
    r"^\s*pub instrs:\s*\[MachineInstr;\s*([0-9]+)\],\s*$",
    struct_match.group("body"),
    "machineblock_instrs_field",
    re.MULTILINE,
)

fn_starts = list(re.finditer(r"^pub fn machine_block_new\(\) -> MachineBlock(?:\s+with[^{]+)?\s*\{", text, re.MULTILINE))
if len(fn_starts) != 1:
    fail(f"machine_block_new_count_{len(fn_starts)}")
next_fn = text.find("\npub fn machine_function_new()", fn_starts[0].end())
if next_fn < 0:
    fail("machine_block_new_end_missing")
body = text[fn_starts[0].end():next_fn]
init_match = unique(
    r"^\s*instrs:\s*\[machine_empty_instr\(\);\s*([0-9]+)\],\s*$",
    body,
    "machine_block_new_instrs_initializer",
    re.MULTILINE,
)

# Fail-closed overflow sites must mention mir_instr_capacity
fail_closed = len(re.findall(r'mir_instr_capacity', text))
if fail_closed < 3:
    fail(f"fail_closed_sites_expected_ge_3_got_{fail_closed}")

cap = int(cap_match.group(1))
field = int(type_match.group(1))
initializer = int(init_match.group(1))

if cap != expected:
    fail(f"mir_max_instrs_expected_{expected}_got_{cap}")
if field != expected:
    fail(f"machineblock_field_expected_{expected}_got_{field}")
if initializer != expected:
    fail(f"initializer_expected_{expected}_got_{initializer}")
if not (cap == field == initializer):
    fail(f"divergent_cap_{cap}_field_{field}_initializer_{initializer}")

print(
    "MIR_INSTR_CAPACITY_CHECK "
    f"cap={cap} field={field} initializer={initializer} "
    f"fail_closed_sites={fail_closed} coherent=pass"
)
PY

source_sha256="$(sha256sum "$SOURCE" | awk '{print $1}')"
head_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf not_available)"
printf 'MIR_INSTR_CAPACITY_PASS source=%s source_sha256=%s expected=%s head=%s\n' \
  "$SOURCE" "$source_sha256" "$EXPECTED_CAPACITY" "$head_sha"
