#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${FLOAT_SLOT_CAPACITY_SOURCE:-$ROOT/self-hosted/native/machine_ir.sio}"
EXPECTED_CAPACITY=2048

fail() {
  printf 'FLOAT_SLOT_CAPACITY_FAIL reason=%s\n' "$1" >&2
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
    print(f"FLOAT_SLOT_CAPACITY_FAIL reason={reason}", file=sys.stderr)
    raise SystemExit(1)


def unique(pattern: str, subject: str, reason: str, flags: int = 0) -> re.Match:
    matches = list(re.finditer(pattern, subject, flags))
    if len(matches) != 1:
        fail(f"{reason}_count_{len(matches)}")
    return matches[0]


cap_match = unique(
    r"^pub let MIR_MAX_FLOAT_SLOTS: i64 = ([0-9]+)\s*$",
    text,
    "mir_max_float_slots",
    re.MULTILINE,
)

struct_match = unique(
    r"^pub struct MachineFunction\s*\{(?P<body>.*?)^\}\s*$",
    text,
    "machinefunction_struct",
    re.MULTILINE | re.DOTALL,
)
type_match = unique(
    r"^\s*pub is_float_slot:\s*\[i64;\s*([0-9]+)\],\s*$",
    struct_match.group("body"),
    "machinefunction_is_float_slot_field",
    re.MULTILINE,
)

fn_starts = list(
    re.finditer(
        r"^pub fn machine_function_new\(\) -> MachineFunction(?:\s+with[^{]+)?\s*\{",
        text,
        re.MULTILINE,
    )
)
if len(fn_starts) != 1:
    fail(f"machine_function_new_count_{len(fn_starts)}")
next_fn = text.find("\npub fn native_v2_legalize_summary_new()", fn_starts[0].end())
if next_fn < 0:
    fail("machine_function_new_end_missing")
body = text[fn_starts[0].end() : next_fn]
init_match = unique(
    r"^\s*is_float_slot:\s*\[0;\s*([0-9]+)\],\s*$",
    body,
    "machine_function_new_is_float_slot_initializer",
    re.MULTILINE,
)

# Local summary/legalize scratch tables must match capacity too.
local_tables = re.findall(
    r"var is_float_slot:\s*\[i64;\s*([0-9]+)\]\s*=\s*\[0;\s*([0-9]+)\]",
    text,
)
if len(local_tables) < 2:
    fail(f"local_is_float_slot_tables_expected_ge_2_got_{len(local_tables)}")
for field, init in local_tables:
    if int(field) != expected or int(init) != expected:
        fail(f"local_table_divergent_field_{field}_init_{init}_expected_{expected}")

# Bounds must use the named constant (no leftover hardcoded 256 float-slot wall).
hardcoded = []
for i, line in enumerate(text.splitlines(), 1):
    stripped = line.lstrip()
    if stripped.startswith("//"):
        continue
    if "is_float_slot" in line and re.search(r"\b256\b", line):
        hardcoded.append(i)
    if re.search(r"<\s*256\b", line) and any(
        k in line
        for k in (
            "is_float_slot",
            "src_slot",
            "dst_slot",
            "dslot",
            "machine_instr_dst_value",
            "instr.dst",
            "ir_instr.dst",
        )
    ):
        hardcoded.append(i)
if hardcoded:
    fail(f"hardcoded_256_float_slot_lines_{hardcoded[:8]}")

# Fail-closed overflow sites must mention float_slot_capacity
fail_closed = len(re.findall(r"float_slot_capacity", text))
if fail_closed < 3:
    fail(f"fail_closed_sites_expected_ge_3_got_{fail_closed}")

# Reset loop must scan the full table
if not re.search(r"while i < MIR_MAX_FLOAT_SLOTS", text):
    fail("reset_loop_missing_mir_max_float_slots")

cap = int(cap_match.group(1))
field = int(type_match.group(1))
initializer = int(init_match.group(1))

if cap != expected:
    fail(f"mir_max_float_slots_expected_{expected}_got_{cap}")
if field != expected:
    fail(f"machinefunction_field_expected_{expected}_got_{field}")
if initializer != expected:
    fail(f"initializer_expected_{expected}_got_{initializer}")
if not (cap == field == initializer):
    fail(f"divergent_cap_{cap}_field_{field}_initializer_{initializer}")

# Bound sites that consult the table must use the constant name
bound_uses = len(re.findall(r"<\s*MIR_MAX_FLOAT_SLOTS", text))
if bound_uses < 20:
    fail(f"bound_uses_expected_ge_20_got_{bound_uses}")

print(
    "FLOAT_SLOT_CAPACITY_CHECK "
    f"cap={cap} field={field} initializer={initializer} "
    f"local_tables={len(local_tables)} bound_uses={bound_uses} "
    f"fail_closed_sites={fail_closed} coherent=pass"
)
PY

source_sha256="$(sha256sum "$SOURCE" | awk '{print $1}')"
head_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf not_available)"
printf 'FLOAT_SLOT_CAPACITY_PASS source=%s source_sha256=%s expected=%s head=%s\n' \
  "$SOURCE" "$source_sha256" "$EXPECTED_CAPACITY" "$head_sha"
