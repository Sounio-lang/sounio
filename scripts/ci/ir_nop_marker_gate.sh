#!/usr/bin/env bash
# ir_nop_marker_gate.sh — an IrNop carrying IR_FLOAT_REG_MARKER_FLAG is not dead.
#
# #1669: ocp_mfi_compact_nops removed every IrNop, marker included. The marker is
# how lowering tells codegen that a register holds f64, so with it gone the
# register was read as an integer and EVERY f64 returned from a call came back as
# i64 MAX. Five lines reproduced it, on origin/main as well as on the #1649
# branch, and nothing anywhere stated the rule it violated.
#
# WHY THIS IS NOT "grep for a marker check"
# -----------------------------------------
# `!= IrOpcode::IrNop` is not by itself a bug. Most uses are analysis asking "is
# this a real instruction", and they are fine. The dangerous shape is a survivor
# test that gates a COPY TO A COMPACTION WRITE INDEX, and that distinction is
# semantic -- no grep separates it from the harmless ones. So this gate does two
# things instead:
#
#   1. pins the invariants that were actually discovered, including one that is
#      load-bearing BY ACCIDENT, and
#   2. censuses the survivor tests, so a NEW one cannot appear without somebody
#      classifying it and updating the number here on purpose.
#
# See scripts/dev/ir-arena/NOP_MARKER_AUDIT.md for the full classification of all
# 93 IrNop mentions.
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SRC_ROOT="${IR_NOP_MARKER_SRC_ROOT:-$ROOT_DIR}"
VACUITY="${IR_NOP_MARKER_VACUITY:-0}"

fail() { printf 'IR_NOP_MARKER_FAIL reason=%s\n' "$1" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || fail python3_missing

check() {
python3 - "$1" <<'PY'
import pathlib, re, sys

root = pathlib.Path(sys.argv[1])

def fail(reason):
    print(f"IR_NOP_MARKER_FAIL reason={reason}", file=sys.stderr)
    raise SystemExit(1)

def read(rel):
    p = root / rel
    if not p.is_file():
        fail(f"missing_{rel.replace('/','_')}")
    return p.read_text(encoding="utf-8")

ir = read("self-hosted/ir/ir.sio")
ocp_raw = read("self-hosted/ir/opt_cleanup.sio")
cg_raw = read("self-hosted/native/codegen_x86_linux.sio")
mir_raw = read("self-hosted/native/machine_ir.sio")

# --- 1. the marker itself -------------------------------------------------
m = re.search(r"^pub let IR_FLOAT_REG_MARKER_FLAG: i64 = (\d+)\s*$", ir, re.M)
if not m:
    fail("marker_constant_missing")
if m.group(1) != "64064":
    fail(f"marker_constant_changed_to_{m.group(1)}")

def strip_comments(text):
    # The first version of this gate was satisfied by its own COMMENTS: the
    # explanatory prose in each peel names IR_FLOAT_REG_MARKER_FLAG, so the
    # check passed with the actual condition deleted. Its vacuity run caught it.
    return "\n".join(line.split("//")[0] for line in text.splitlines())

def body(text, fn):
    mm = re.search(rf"^fn {fn}\(.*?^\}}", text, re.M | re.S)
    if not mm:
        fail(f"{fn}_not_found")
    return strip_comments(mm.group(0))

# --- 2. both nop deleters must consult the marker -------------------------
# These are the ONLY two sites in the tree that delete a nop. Everything else
# skips over them or compares opcodes; see the audit.
for fn in ("ocp_mfi_compact_nops", "ocp_compact_nops"):
    b = body(ocp_raw, fn)
    if "IrOpcode::IrNop" not in b:
        fail(f"{fn}_no_longer_filters_nops")
    if "IR_FLOAT_REG_MARKER_FLAG" not in b:
        fail(f"{fn}_deletes_nops_without_checking_the_marker")

# --- 3. the ACCIDENTAL safety ---------------------------------------------
# ocp_mfi_dce_once overwrites dead instructions with ir_nop(), which would erase
# a marker. It is saved only because ocp_has_dst has no IrNop arm. Adding one
# looks harmless -- a marker nop does have a dst -- and reintroduces #1669
# through a different door.
hd = body(ocp_raw, "ocp_has_dst")
if re.search(r"IrOpcode::IrNop\s*=>\s*true", hd):
    fail("ocp_has_dst_now_claims_IrNop_has_a_dst_see_1669")

# --- 4. the consumers must still be reading it ----------------------------
# Four readers in two subsystems. If they all vanish the marker is dead weight
# and this gate should be retired deliberately, not silently.
consumers = 0
cg = strip_comments(cg_raw)
mir = strip_comments(mir_raw)
ocp = strip_comments(ocp_raw)
if re.search(r"IrOpcode::IrNop\s*&&\s*IR_A_IMM_FLAGS\[.*?IR_FLOAT_REG_MARKER_FLAG", cg, re.S):
    consumers += 1
consumers += len(re.findall(r"imm_flags == IR_FLOAT_REG_MARKER_FLAG", mir))
if re.search(r"imm_flags & IR_FLOAT_REG_MARKER_FLAG", ocp):
    consumers += 1
if consumers < 4:
    fail(f"marker_consumers_dropped_to_{consumers}_of_4")

# --- 5. census of survivor tests ------------------------------------------
# A new `!= IrOpcode::IrNop` is not necessarily wrong, but it has to be looked
# at. Bumping these numbers is the act of having looked.
EXPECTED = {
    "self-hosted/compiler/main.sio": 3,
    "self-hosted/ir/opt_cleanup.sio": 6,
}
actual = {}
for p in sorted(root.glob("self-hosted/**/*.sio")):
    n = p.read_text(encoding="utf-8", errors="replace").count("!= IrOpcode::IrNop")
    if n:
        actual[str(p.relative_to(root))] = n
if actual != EXPECTED:
    added = {k: v for k, v in actual.items() if EXPECTED.get(k) != v}
    gone = {k: v for k, v in EXPECTED.items() if actual.get(k) != v}
    fail(f"survivor_test_census_changed added_or_changed={added} expected={gone}")

print(
    "IR_NOP_MARKER_CHECK marker=64064 deleters=2:both_guarded "
    f"consumers={consumers} survivor_tests={sum(actual.values())} coherent=pass"
)
PY
}

if [ "$VACUITY" = "1" ]; then
  # The gate must be able to FAIL. Strip the marker check from a scratch copy of
  # the peel that caused #1669 and confirm it is rejected.
  TMP="$(mktemp -d "${TMPDIR:-/tmp}/sounio-nop-marker.XXXXXX")"
  trap 'rm -rf "$TMP"' EXIT
  mkdir -p "$TMP/self-hosted/ir" "$TMP/self-hosted/native"
  cp -r self-hosted/ir/*.sio "$TMP/self-hosted/ir/"
  cp -r self-hosted/native/*.sio "$TMP/self-hosted/native/"
  mkdir -p "$TMP/self-hosted/compiler"
  cp self-hosted/compiler/main.sio "$TMP/self-hosted/compiler/"
  sed -i 's/|| marker == IR_FLOAT_REG_MARKER_FLAG//' "$TMP/self-hosted/ir/opt_cleanup.sio"
  if check "$TMP" >/dev/null 2>&1; then
    fail "vacuous_gate_passes_with_the_marker_check_removed"
  fi
  printf 'IR_NOP_MARKER_VACUITY_OK rejected_the_unguarded_deleter\n'
fi

check "$SRC_ROOT" || exit $?

head_sha="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || printf not_available)"
printf 'IR_NOP_MARKER_BOUNDARY rule=a_nop_with_the_float_marker_is_not_dead audit=scripts/dev/ir-arena/NOP_MARKER_AUDIT.md\n'
printf 'IR_NOP_MARKER_PASS head=%s\n' "$head_sha"
