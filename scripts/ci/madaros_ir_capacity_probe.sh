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
# Measured 2026-08-04 against origin/main 40116b661d:
#
#     closure nodes                97
#     fn declarations in closure   9404
#     IR_MAX_FUNCS                 2048
#     ratio                        4.59x
#
# So the answer is neither "one bug" nor "fifty bugs": it is one architectural
# constraint that no amount of typecheck work touches. Every function in the
# closure that survives to lowering needs a slot in a single merged IrModule
# whose backing array holds 2048, and `ir_merge_modules_into`
# (compiler/module_frontend.sio:1254) stops copying at the cap — silently.
#
# THIS GATE IS EXPECTED TO FAIL TODAY, loudly, with a number. That is its job:
# it converts an unknown into a denominator, and it turns green only when the
# closure actually fits. Do not "fix" it by raising IR_MAX_FUNCS —
# IrFunction.instrs is [IrInstr; 4096] at roughly 1 MB per function and the
# shipped ELF's RW segment is already 3.40 GB of BSS.
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
                 | grep -oE '[0-9]+$' | head -1)"
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
$(printf '%s' "$FOREIGN" | head -5)"
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
sort -rn "$CENSUS" | head -10 | sed 's/^/  /'

if [[ "${SOUNIO_IR_CAPACITY_PROBE_REPORT_ONLY:-0}" == "1" ]]; then
  echo "MADAROS_IR_CAPACITY_REPORT_ONLY: not enforcing the cap"
  exit 0
fi

if [[ "$TOTAL" -gt "$IR_MAX_FUNCS" ]]; then
  gate_fail "$ENTRY's import closure declares $TOTAL functions against IR_MAX_FUNCS=$IR_MAX_FUNCS.
Every one that survives to lowering needs a slot in a single merged IrModule.

This limit is REPORTED, not silent — verified 2026-08-04 with a 9002-function
witness: 'too many functions: shared IR module capacity exceeded (max 2048
slots)', rc=1, no ELF. So the wall is honest; it is just a wall.

Cross-module DCE (spec_dce_unreachable_item_fns) does not clear it either. A
name-based reachability census from main over this closure leaves 5997
declarations live, and 3988 even with the whole test suite cut out of the graph.
Both are over 2048.

The memory arithmetic behind the constant: IrInstr is ~248 bytes, of which 136
is an inline Name that most instructions never use; IrFunction.instrs is
[IrInstr; 4096] ~= 0.97 MB; IrModule.functions is [IrFunction; 2048] ~= 2.1 GB
of the shipped ELF's 3.40 GB BSS. Interning IrInstr.name would halve the
instruction and buy 4096 slots at today's memory — that is the change this
number is asking for, not a bigger constant on the same layout."
fi

gate_pass "$TOTAL fn declarations across $NODES closure nodes fit IR_MAX_FUNCS=$IR_MAX_FUNCS"
