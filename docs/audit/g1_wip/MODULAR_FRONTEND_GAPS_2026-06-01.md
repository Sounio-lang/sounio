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
