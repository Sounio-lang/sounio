#!/usr/bin/env bash
# irfunction_instr_capacity_coherence_gate.sh — pin what IR_MAX_INSTRS now means.
#
# This gate used to assert that IR_MAX_INSTRS, the `pub instrs: [IrInstr; N]`
# field and its `[ir_nop(); N]` initialiser all carried the SAME literal, because
# Sounio array sizes must be literals and could silently diverge from the
# constant.
#
# #1649 removed that field: instruction storage is a variable-size arena region,
# so there is no literal left to keep coherent, and the old gate fails with
# `irfunction_instrs_field_count_0`. The invariant worth pinning changed shape.
#
# WHAT CAN GO WRONG NOW
# ---------------------
# Raising IR_MAX_INSTRS is cheap in storage and dangerous in analysis. Several
# passes populate a FIXED per-instruction context and stop at their own cap:
#
#     while i < func.instr_count && i < DCE_MAX_INSTRS { ... }
#
# A truncated liveness or constant-propagation analysis is not a weaker analysis,
# it is a WRONG one -- a use past the cap is never recorded, so its definition
# looks dead and the sweep deletes live code. Silently, at rc=0. That is the same
# family as #1586 / #1570 / #1577.
#
# So each such pass must REFUSE a function it cannot hold, and this gate proves
# the refusal is present rather than trusting that someone remembered.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${IRFUNCTION_INSTR_CAPACITY_SOURCE:-$ROOT/self-hosted/ir/ir.sio}"
EXPECTED_CAPACITY="${IRFUNCTION_INSTR_CAPACITY_EXPECTED:-16384}"

fail() {
  printf 'IRFUNCTION_INSTR_CAPACITY_FAIL reason=%s\n' "$1" >&2
  exit 1
}

[[ -f "$SOURCE" ]] || fail source_missing
command -v python3 >/dev/null 2>&1 || fail python3_missing

python3 - "$SOURCE" "$EXPECTED_CAPACITY" "$ROOT" <<'PY' || exit $?
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
root = pathlib.Path(sys.argv[3])
text = source.read_text(encoding="utf-8")


def fail(reason: str) -> None:
    print(f"IRFUNCTION_INSTR_CAPACITY_FAIL reason={reason}", file=sys.stderr)
    raise SystemExit(1)


def unique(pattern, subject, reason, flags=0):
    matches = list(re.finditer(pattern, subject, flags))
    if len(matches) != 1:
        fail(f"{reason}_count_{len(matches)}")
    return matches[0]


cap = int(
    unique(
        r"^pub let IR_MAX_INSTRS: i64 = ([0-9]+)\s*$", text, "ir_max_instrs", re.M
    ).group(1)
)
if cap != expected:
    fail(f"ir_max_instrs_expected_{expected}_got_{cap}")

max_funcs = int(
    unique(
        r"^pub let IR_MAX_FUNCS: i64 = ([0-9]+)\s*$", text, "ir_max_funcs", re.M
    ).group(1)
)
max_strings = int(
    unique(
        r"^pub let IR_MAX_STRINGS: i64 = ([0-9]+)\s*$", text, "ir_max_strings", re.M
    ).group(1)
)

# The inline array must stay GONE. Its return would restore the ~2 GB per-module
# reservation and, under #1655, a global array of aggregates is a silent no-op.
struct_body = unique(
    r"^pub struct IrFunction\s*\{(?P<body>.*?)^\}\s*$", text, "irfunction_struct",
    re.M | re.S,
).group("body")
if re.search(r"^\s*pub instrs:\s*\[IrInstr;", struct_body, re.M):
    fail("inline_instrs_field_reintroduced")
if not re.search(r"^\s*pub region:\s*IrInstrRegion,\s*$", struct_body, re.M):
    fail("irfunction_region_handle_missing")

# Analysis passes whose fixed context is smaller than IR_MAX_INSTRS must refuse
# rather than truncate. Each entry: file, cap name, function that must guard.
GUARDED = [
    ("self-hosted/ir/dce.sio", "DCE_MAX_INSTRS", "dce_run_impl"),
    ("self-hosted/ir/const_prop.sio", "CP_MAX_INSTRS", "cp_run_impl"),
]

receipts = []
for rel, capname, fnname in GUARDED:
    path = root / rel
    if not path.is_file():
        fail(f"guarded_source_missing_{capname}")
    body = path.read_text(encoding="utf-8")
    m = re.search(rf"^let {capname}: i64 = ([0-9]+)\s*$", body, re.M)
    if not m:
        fail(f"{capname.lower()}_declaration_missing")
    pass_cap = int(m.group(1))

    fn = re.search(rf"^fn {fnname}\(.*?^\}}", body, re.M | re.S)
    if not fn:
        fail(f"{fnname}_not_found")
    guarded = re.search(
        rf"instr_count\s*>\s*{capname}\s*\{{", fn.group(0)
    ) is not None

    if pass_cap < cap and not guarded:
        fail(f"{fnname}_truncates_at_{pass_cap}_without_refusing_cap_{cap}")
    receipts.append(f"{capname}={pass_cap}:{'refuses' if guarded else 'covers'}")

# IrModule.functions and the SOIR deserializer function table are compile-time
# literals. A handwritten 16384 that merely equals IR_MAX_FUNCS today is the
# same defect class as the old [IrFunction; 1024]: it will not rise next time.
# This gate is the bind — Sounio cannot write [IrFunction; IR_MAX_FUNCS].
mod_fn = unique(
    r"pub functions:\s*\[IrFunction;\s*([0-9]+)\]", text, "irmodule_functions_field"
)
if int(mod_fn.group(1)) != max_funcs:
    fail(f"irmodule_functions_literal_{mod_fn.group(1)}_ne_ir_max_funcs_{max_funcs}")

ser_path = root / "self-hosted/ir/serialize.sio"
if not ser_path.is_file():
    fail("serialize_source_missing")
ser = ser_path.read_text(encoding="utf-8")
ser_fn = unique(
    r"var functions:\s*\[IrFunction;\s*([0-9]+)\]", ser, "serialize_functions_array"
)
if int(ser_fn.group(1)) != max_funcs:
    fail(f"serialize_functions_literal_{ser_fn.group(1)}_ne_ir_max_funcs_{max_funcs}")
ser_str = unique(
    r"var string_table:\s*\[Name;\s*([0-9]+)\]", ser, "serialize_string_table"
)
if int(ser_str.group(1)) != max_strings:
    fail(f"serialize_string_table_literal_{ser_str.group(1)}_ne_ir_max_strings_{max_strings}")

# Capacity refuse must be DETECTABLE (counter), not a silent empty-module clamp.
if not re.search(r"^pub var SOIR_DESER_REFUSAL_COUNT:\s*i64\s*=\s*0\s*$", ser, re.M):
    fail("soir_deser_refusal_count_missing")
if ser.count("soir_note_deser_refusal()") < 3:
    fail(f"soir_note_deser_refusal_call_count_{ser.count('soir_note_deser_refusal()')}")
if "fn_count < 0 || fn_count > IR_MAX_FUNCS" not in ser:
    fail("serialize_fn_count_not_checked_against_ir_max_funcs")

# The same failure class exists at REGISTER granularity in opt_cleanup, whose
# peels carry register-indexed [_; 256] state, and it is NOT guarded here on
# purpose. Refusing was tried twice and made things worse, measured: skipping the
# whole cleanup SIGSEGVs (the structural peels are not optional for codegen), and
# skipping only the register-indexed peels returns a WRONG answer (a 60-function
# stress program went 56 -> 95) because the peels are a pipeline, not a menu.
# The real fix is widening the 256-wide state, and it is its own change --
# tracked by tests/known_failures/opt_cleanup_wide_register_file.sio.

print(
    "IRFUNCTION_INSTR_CAPACITY_CHECK "
    f"cap={cap} max_funcs={max_funcs} max_strings={max_strings} "
    "storage=arena_region inline_field=absent "
    f"serialize_functions={max_funcs} serialize_strings={max_strings} "
    "soir_deser_refusal=detectable "
    + " ".join(receipts)
    + " coherent=pass"
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
  'IRFUNCTION_INSTR_CAPACITY_BOUNDARY storage_coherence=proved analysis_refusal=proved runtime_readback=not_claimed soir_v6=unchanged legacy=preserved'
printf 'IRFUNCTION_INSTR_CAPACITY_PASS source=%s source_sha256=%s expected=%s head=%s tree=%s worktree=%s\n' \
  "$SOURCE" "$source_sha256" "$EXPECTED_CAPACITY" "$head_sha" "$tree_sha" "$worktree_state"
