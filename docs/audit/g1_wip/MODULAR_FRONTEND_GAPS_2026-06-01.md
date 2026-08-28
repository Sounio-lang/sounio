<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.modular-frontend-gaps-2026-06-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.modular-frontend-gaps-2026-06-01
-->

# Modular compiler (mc.elf --check) feature-gap diagnosis — 2026-06-01

Source: 8-agent diagnosis workflow on the post-fix build (mc.elf `daaa5758`, after the
bare-pattern qualification fix). ~481 example programs still crash/fail; this maps why.

## Unifying root cause: the *mut `--check` checker spine is HALF-CONVERTED
The move-codegen *mut spine handles leaf exprs in-place but **bridges every non-leaf
expr back to the OLD by-value `check_expr`** (`checker_check_expr_mut` →
`(*c).check_expr(e)`, check.sio:1146-1149) and **stubs declaration collection**. That
by-value path is the one with the large-struct/SRET frame disease. So:

- **A. Declarations not collected** — `checker_collect_item_inplace` (check.sio:2267)
  handles only ItemFn (+no-op Use/Session); `_ => {}` at **check.sio:2278** SKIPS
  ItemStruct/ItemEnum/ItemImpl/ItemTypeAlias (documented "CORRECTNESS GAP… 8MB SRET
  frame overflowed even `fn main(){}`… collectors pending"). The by-value collector
  `collect_struct_def` (check.sio:11225) works but is bypassed. → `self.structs.find`
  misses → **E015 "unknown struct type"** for ALL struct/enum usage (check.sio:16266).
- **B. Non-leaf exprs recurse by-value** → frame blowup or state loss:
  - `if/else`: else arm calls `c.check_expr(*else_e)` by value (check.sio:15987) →
    **rc=139** (3/3; `if` without else is rc=0). check_if_expr at 15959.
  - `match` with an arm BLOCK containing a STATEMENT: by-value `c.check_expr(arm.body)`
    (check.sio:16041) → **rc=139** (pure-expr arms are fine).
  - `return <expr>`: `current_return_type` reads as TyUnit at
    checker_check_return_expr_inplace (check.sio:2489) → spurious mismatch.
  - `with Epistemic` `.value` gate: `current_effects` ([i64;8]) is empty at the by-value
    gate (check.sio:13198) though set in `*c` (check.sio:2387) — the by-value
    materialization of `*c` into `check_expr`'s `self` (bridge check.sio:1147) drops the
    fixed-size effect array → **spurious E170**.

## Independent parser/handler gaps (not the spine)
- **methods/impl**: ALL probes die at PARSE time (parser/lexer bridge in the running
  mc.elf) though source looks correct → type-checker never runs.
- **enum tuple-variant decl** `Some(i64)`: unparseable — parse_enum_item
  (parser/items.sio:559-580) only handles struct-style `Variant{f:T}`. (Pattern parser
  DOES handle `Some(x)`.)
- **enum payload construction**: even struct-style `Opt::Some{val:42}` → E015 because
  check_struct_lit (check.sio:16266) looks up self.structs, never self.enums.
- **slice `&a[0..2]`**: handler check_slice_borrow_expr (check.sio:13067) is ORPHANED —
  reached only when the `&` operand is itself an ExprRange, but the body requires
  ExprIndex (mutually exclusive); guard tests the wrong nesting level (check.sio:13072).

## Works
- let / var / `x=e` / compound assign / shadowing — fully correct. (`let mut` is
  unsupported BY DESIGN; Sounio uses `var`.) The G1 `let x=1` crash does NOT reproduce
  on this binary.

## Ranked next-fixes (highest leverage first)
1. **Implement `collect_struct_def_inplace` + enum/impl/typealias collectors; wire at
   check.sio:2278.** Unblocks the largest class (structs, enums, generics). This is the
   move-codegen spine's original purpose — finish it.
2. **Move `if/else` and `match` into the *mut spine** (handle inline, no by-value
   recurse) → kills the if-else and match-statement crashes.
3. **Fix `*c` state across the by-value bridge** (propagate `current_effects` /
   `current_return_type`), or move return/field-access into the *mut spine → fixes
   spurious E170 / return-type errors (the headline epistemic features).
4. **Parser**: methods/impl parse failure; enum tuple-variant decl (items.sio:559);
   slice-borrow guard (check.sio:13072).

NOTE: this is SEPARATE from the bare-pattern qualification fix (committed on
g1/qualify-bare-patterns) — that was a bin/souc codegen bug; these are modular-source
completeness gaps. Both needed for the modular compiler to actually work.

## CORRECTION (verified): the synthesis's "FIX #0 — just rebuild mc.elf" is FALSIFIED
mc.elf `daaa5758` was freshly rebuilt 6× THIS session from current source (bin/souc
`e35ef063` unchanged). On that fresh binary, `loop`+`break`, `impl`/`self`, and
`type X = Y` STILL fail with parse errors (rc=1, "parse error: expected token"). So
these are NOT mc.elf staleness. Most likely: **bin/souc miscompiles the modular parser
source** for these constructs (another bin/souc codegen bug, same family spirit as the
bare-pattern one), or a genuine modular-parser gap. "Rebuild" will not fix them — they
need real investigation (diff what the modular parser SOURCE does vs what the built
parser accepts; suspect a bin/souc miscompile of parse_param/self / parse_impl_item /
loop / type-alias). This slots ABOVE FIX #1 only if methods/impl are a priority;
otherwise FIX #1 (decl collection) remains the highest-leverage single fix.
