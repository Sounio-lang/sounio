#!/usr/bin/env bash
# scripts/ci/madaros_ir_capacity_probe.sh
#
# STAGE 1 of the "Madaros compiles Madaros" line: size the codegen wall WITHOUT
# compiling anything.
#
# The question this answers is "is Madaros one bug away from compiling itself,
# or fifty?" — and the cheapest instrument for it is not a compile, it is a
# count. Walk main.sio's real import closure with --science-boundary-closure
# (parse only, seconds, no build lock), sum the `fn` declarations over exactly
# those files, and put the total beside IR_MAX_FUNCS.
#
# Measured 2026-08-04 against origin/main 40116b661d, before any of this
# line's work:
#
#     closure nodes                97
#     fn declarations in closure   9404
#     IR_MAX_FUNCS                 2048
#     ratio                        4.59x
#
# So the answer was neither "one bug" nor "fifty bugs": it was one architectural
# constraint that no amount of typecheck work touches. Same day, after interning
# IrInstr.name and sweeping the backend for hardcoded 2048s:
#
#     closure nodes                120     (the import cap fix admitted 23 more)
#     fn declarations in closure   10705
#     IR_MAX_FUNCS                 8192
#
# STILL RED, and correctly so — but the number that decides self-compilation is
# not the declaration count, it is the REACHABLE count, and that one now fits:
# a name-based reachability census from `main` leaves 5997 declarations live,
# under 8192. Declared-but-unreachable code is what cross-module DCE exists to
# drop (see scripts/ci/madaros_dce_reachability_gate.sh). This gate stays on the
# declaration count because it is the number you can get without running
# anything, and because DCE only runs on the specialized-collapse path.
#
# SOUNIO_IR_CAPACITY_PROBE_REPORT_ONLY=1 prints the census and exits 0. Use it
# to record movement between stages without a red gate; CI uses the default.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

. "$ROOT_DIR/scripts/lib/gate_assert.sh"
gate_name "madaros_ir_capacity"

ENTRY="${SOUNIO_IR_CAPACITY_ENTRY:-self-hosted/compiler/main.sio}"
MADAROS="${MADAROS_BIN:-${SOUC_BIN:-}}"

if [[ -z "$MADAROS" ]]; then
  echo "MADAROS_IR_CAPACITY_SKIP: set MADAROS_BIN to a raw Madaros ELF" >&2
  exit 0
fi

require_executable "$MADAROS"

# The same refusal madaros_self_parse_gate.sh makes: a wrapper script resolves
# to whatever compiler happens to be installed, so its verdict is
# unattributable to the tree under test.
if head -c2 "$MADAROS" 2>/dev/null | grep -q '#!'; then
  gate_fail "$MADAROS is a wrapper script, not a raw ELF — point MADAROS_BIN at the binary itself"
fi

require_file "$ENTRY"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/madaros-ir-capacity.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
CLOSURE="$WORK/closure.txt"

# module_frontend.sio:334 prefers $SOUNIO_STDLIB_PATH over the tree's own
# stdlib/, so an inherited value silently pulls closure nodes out of ANOTHER
# CHECKOUT — measured on 2026-08-04: seven of 120 nodes came from
# /workspace/sounio/stdlib while the tree under test was a worktree on a
# different branch. The two agreed that day. Nothing in the report says which
# tree a node came from, so a disagreement would have been invisible.
if [[ -d "$ROOT_DIR/stdlib" ]]; then
  export SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib"
fi

# Madaros recurses in the parser; CI's default 16 MiB soft stack is not enough.
( ulimit -s 524288 2>/dev/null || true
  "$MADAROS" --science-boundary-closure "$ENTRY" ) >"$CLOSURE" 2>&1 || true

require_nonempty_file "$CLOSURE" "the closure walk produced no output at all"
require_text "SOUNIO_BOUNDARY_CLOSURE_V1" "$CLOSURE"

# A truncated or failed closure makes every count below an undercount, and an
# undercount here reads as headroom that does not exist.
VERDICT="$ROOT_DIR/scripts/lib/boundary_closure_verdict.sh"
require_executable "$VERDICT"
if ! VERDICT_OUT="$(bash "$VERDICT" "$CLOSURE" 2>&1)"; then
  gate_fail "the closure for $ENTRY is not clean, so the function census below would be an undercount:
$VERDICT_OUT"
fi

NODES="$(grep -c '^node' "$CLOSURE" || true)"
require_nonempty "$NODES" "node count"
require_min_count "$NODES" 2 "closure nodes"

IR_MAX_FUNCS="$(grep -E '^pub let IR_MAX_FUNCS: i64 = [0-9]+' self-hosted/ir/ir.sio \
                 | grep -oE '[0-9]+$' | sed -n 1p)"
require_nonempty "$IR_MAX_FUNCS" "IR_MAX_FUNCS — it is no longer declared where this gate looks"

FN_RE='^[[:space:]]*(pub[[:space:]]+|pub\(crate\)[[:space:]]+)?fn[[:space:]]'

TOTAL=0
MISSING=0
CENSUS="$WORK/census.tsv"
: >"$CENSUS"
while read -r _tag path; do
  if [[ ! -f "$path" ]]; then
    MISSING=$((MISSING + 1))
    continue
  fi
  n="$(count_matches "$FN_RE" "$path")"
  printf '%s\t%s\n' "$n" "$path" >>"$CENSUS"
  TOTAL=$((TOTAL + n))
done < <(grep '^node' "$CLOSURE")

if [[ "$MISSING" -gt 0 ]]; then
  gate_fail "$MISSING of $NODES closure nodes are not files on disk — the census is not over the closure it claims"
fi

# The other half of the same guard: even with SOUNIO_STDLIB_PATH pinned above, a
# node resolving outside this tree means the count is partly about a different
# checkout, and the gate must say so rather than average the two.
FOREIGN="$(grep '^node' "$CLOSURE" | awk '{print $2}' | grep '^/' | grep -v "^$ROOT_DIR/" || true)"
if [[ -n "$FOREIGN" ]]; then
  gate_fail "closure nodes resolve outside the tree under test ($ROOT_DIR); this census would be partly about another checkout:
$(printf '%s' "$FOREIGN" | sed -n '1,5p')"
fi

# A census that finds almost no functions is broken, not empty. main.sio alone
# declares over a thousand.
require_min_count "$TOTAL" 500 "fn declarations in the closure"

echo "MADAROS_IR_CAPACITY_V1"
echo "entry            $ENTRY"
echo "closure_nodes    $NODES"
echo "fn_declarations  $TOTAL"
echo "ir_max_funcs     $IR_MAX_FUNCS"
echo "headroom         $((IR_MAX_FUNCS - TOTAL))"
echo "top_contributors"
# No early-exiting reader in a pipefail pipeline: `| head -10` closes the
# pipe after ten lines and `sort` can die on the flush (observed in CI on
# 2026-08-17: "sort: fflush failed: Broken pipe" killed this REPORT_ONLY
# probe before its exit 0). sed -n '1,10{...p}' reads everything and prints
# ten. Guarded by scripts/ci/sigpipe_hygiene_gate.sh.
sort -rn "$CENSUS" | sed -n '1,10{s/^/  /p}'

if [[ "${SOUNIO_IR_CAPACITY_PROBE_REPORT_ONLY:-0}" == "1" ]]; then
  echo "MADAROS_IR_CAPACITY_REPORT_ONLY: not enforcing the cap"
  exit 0
fi

if [[ "$TOTAL" -gt "$IR_MAX_FUNCS" ]]; then
  gate_fail "$ENTRY's import closure declares $TOTAL functions against IR_MAX_FUNCS=$IR_MAX_FUNCS.
Every one that survives to lowering needs a slot in a single merged IrModule.

This limit is REPORTED, not silent — verified 2026-08-04 with a 9002-function
witness: 'too many functions: shared IR module capacity exceeded', rc=1, no ELF.
So the wall is honest; it is just a wall.

IR_MAX_FUNCS went 2048 -> 4096 -> 8192 on 2026-08-04, after interning
IrInstr.name and after fixing two backend guards that read the literal 2048
while the arrays they guarded were sized from the constant. Raising it alone had
made a compiler that silently dropped symbols from exactly 2046 functions
upward. Verified correct at 4090, 4096, 5000, 6000 and 7900 functions; still
refuses loudly at 9002. The boundary is pinned by tests/multimodule/ir_capacity,
run from scripts/ci/madaros_dce_reachability_gate.sh. Anyone raising it again
must sweep for literals FIRST and then raise that fixture past the old ceiling,
or the gate goes on passing for the wrong reason.

Cross-module DCE (spec_dce_unreachable_item_fns) does not clear it either. A
name-based reachability census from main over this closure leaves 5997
declarations live, and 3988 even with the whole test suite cut out of the graph.
Both are over 2048.

The memory arithmetic, corrected by measurement. IrInstr WAS ~248 bytes, of
which 136 was an inline Name most instructions never used; it is now ~120 with
the name interned. IrFunction.instrs is [IrInstr; 4096]. IrModule.functions is
[IrFunction; 8192] — and the shipped ELF's BSS is 3428082840 bytes at 2048, at
4096 and at 8192, byte for byte, because the IrModule is allocated at RUNTIME
and not stored in the executable. The claim that '2.1 GB of the 3.40 GB BSS is
the functions array', which is why this constant went untouched for so long, was
never true.

AND THE ARENA IS NOT THE ANSWER — this was tested, not assumed. Shrinking
IrFunction.instrs from 3072 entries to 64, i.e. making IrFunction 48x smaller
(504 KB -> 11 KB), moved the per-function cost of a compile by 93 KB:

    IrFunction.instrs = 3072   895.4 KB per function
    IrFunction.instrs =   64   801.8 KB per function

So an arena over IrFunction.instrs — 4518 `.instrs[` call sites, in the file
where a wrong edit is a silent miscompile — buys 10%, and 802 KB per function is
not IrFunction at all. Do not build it. The per-function cost that remains has
not been identified; find it the way the others were found (smaps for the
mapping, lower_live fn_begin/fn_done for the phase, then read the live path),
not by reasoning about struct sizes."
fi

gate_pass "$TOTAL fn declarations across $NODES closure nodes fit IR_MAX_FUNCS=$IR_MAX_FUNCS"
