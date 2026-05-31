# *mut Checker refactor — execution plan (the montanha)

**Decision (2026-05-31):** after the move-codegen wedge plan was overturned (Wedge A a verified
no-op; frames pervasive 6–19MB; setrlimit dead — see memory `project-move-codegen-premise-overturned`),
the user committed to the **`*mut` Checker refactor** as the structural cure. This converts the
by-value `fn f(self) -> (Checker,X)` check spine to `fn f_mut(c: *mut Checker) -> X` with in-place
`(*c).field` access — collapsing the multi-MB frames to KB.

**SCOPE OF THE CLAIM (honest, per advisor):** `*mut` **fixes the FRAME OVERFLOW** (solid, by
mechanism). Whether it **fixes the correctness garbage** (E063 flood / "bitwise-on-addition") is a
SEPARATE axis — and the current partial `*mut` is contradicted on it (onestruct.sio floods E063).
ROOT CAUSE FOUND: `checker_collect_item_inplace` (check.sio:2267) routes ONLY `ItemFn` to `*mut`;
`ItemStruct`/`ItemEnum`/etc fall to `_ => checker_store_from_value(c, (*c).collect_item(item))` — the
BY-VALUE path → state corruption → E063. So correctness is fixable BUT requires COMPLETING the
item-kind routing (and the full check spine), not just the frame-shrink. Until the routing is
complete, do NOT claim "closes the work / reaches E2E" — claim "fixes the crash; correctness lands
as the routing completes."

Branch `modular/move-codegen` (worktree `/workspace/sounio-move-codegen`, forked from
`modular/mut-checker-refactor` @ `0e19f261a`). **Worktree is CLEAN — no edits made this session.**

## ✅ SESSION FINDINGS — CURE EMPIRICALLY VALIDATED (2026-05-31, "beat the montanha")

Live experiments on freshly-built mc.elf (bin/souc compiling main.sio, 2.5min each):

1. **The cure works.** Replacing collect_item_inplace's `_ => checker_store_from_value(c,(*c).collect_item(item))`
   (by-value, copies 164KB Checker) with `_ => {}` makes **`fn main(){}` and `struct P{a}`+main
   reach `check: OK`, rc=0 at ulimit 8MB** — the FIRST time --check returns success. Stable across
   all whitespace variants (v0..v4 all rc=0). gdb confirmed the crash was the probe faulting on
   collect_item's exact 8,063,776-byte (8.06MB) by-value frame.
2. **The universal crash root cause:** EVERY program (even empty `fn main(){}`) carries non-ItemFn
   items that hit collect_item_inplace's by-value `_` arm. The by-value Checker copy both (a)
   overflows the 8MB frame AND (b) CORRUPTS the items-list cursor → the loop reads phantom items.
   Marker counts: clean build = 1 non-fn item; debug-bloated build (15-arm match) = 15 phantom
   items, 2× for onestruct. The corruption magnitude is **mc.elf-code-layout-sensitive** (matches
   [[project_modular_span_sensitive_crash]]) — BUT a SMALL/clean collect path is stable. Since *mut
   SHRINKS frames, it moves AWAY from the corruption threshold. Keep converted methods lean.
3. **Programs with a BODY still crash in the CHECK spine:** `fn main(){let x=1}` (min_let),
   two_fn, hello → rc=139 even with collect fixed. The let path bridges StmtLet →
   checker_check_stmt_bridge → by-value check_stmt → by-value check_let_stmt → by-value check_expr
   (12.3MB frame). Converting check_let_stmt to *mut (routing inits via check_expr_inplace, which
   handles IntLit inline) is the next milestone.
4. **The spine is MORE converted than the memory implied:** collect_fn_def_inplace, check_item_inplace,
   check_fn_item_inplace, check_block_inplace, check_stmts_inplace are ALREADY true *mut. The entry
   (check_items_verdict_boot4, mod.sio:494) ALREADY heap-allocs the Checker (8MB) + uses *mut. The
   ONLY remaining by-value leaks on the empty-program path: collect_item_inplace `_` arm (collect)
   and check_item_inplace ItemImpl arm (check). On the body path: the check_stmt/expr subtree.

REMAINING WORK (precise): (a) collect_item_inplace `_` arm — route the ~15 non-fn kinds to *mut
collectors (convert collect_struct_def/enum_def/etc, OR confirm they're skippable prelude); (b)
check_item_inplace ItemImpl → *mut check_impl_item; (c) the check_stmt subtree: check_let_stmt,
check_var_stmt, check_assign_stmt → *mut; (d) the expr subtree behind check_expr_mut: check_binary_expr,
check_call_expr, check_field_access, check_method_call → *mut. Gate each with `--check` at 8MB on the
input that exercises only the converted path (min_empty → min_let → two_fn → hello).

## The mechanical pattern (validated against existing code)

**Bridge (the leak — to be REMOVED):** `check.sio:2416`
```
fn checker_check_stmt_bridge(c: *mut Checker, s: Stmt) -> TypeEntry {
    let pair = (*c).check_stmt(s)        // by-value call: copies 164KB Checker IN + OUT (SRET)
    checker_store_from_value(c, pair.0)  // copies the result back: a THIRD copy
    pair.1
}
```
Every bridge = 3 Checker copies. ~13 bridges on the hello path → the 6–19MB frames.

**True conversion (the target):** `check.sio:2450 checker_check_expr_inplace`
```
fn checker_check_X_inplace(c: *mut Checker, args) -> TypeEntry {
    ... (*c).field reads/writes directly ...           // no copy
    let t = checker_check_Y_inplace(c, ...)            // delegate to *mut child, not (*c).check_Y()
    ... return just the TypeEntry ...
}
```

**Per-method transform:**
1. Signature: `fn check_X(self, a) -> (Checker, TypeEntry)` → `fn checker_check_X_inplace(c: *mut Checker, a) -> TypeEntry`.
2. Body: delete the `var c = self` / `c = pair.0` threading. Replace `self.field`/`c.field` → `(*c).field`.
   Replace `let p = c.method(x); c = p.0; ...p.1` → `let r = checker_method_inplace(c, x); ...r`.
3. Return the bare `TypeEntry` (drop the `(c, ...)` tuple).
4. Rewire callers: `let p = X.check_X(a)` → `let r = checker_check_X_inplace(c, a)`.
5. At the moving boundary, keep a temporary `*_bridge` for any not-yet-converted callee.

## Conversion order (BOTTOM-UP — leaves first, so each step only delegates to already-`*mut` children)

Convert in this order; after EACH coherent group, run the gate (below). Group = one self-contained
self-reproducing increment.

- **Gc (collect routing — cheap, do FIRST, fixes struct/enum correctness):** in
  `checker_collect_item_inplace` (2267), replace the by-value `_` arm with explicit `*mut` arms for
  `ItemStruct`/`ItemEnum`/`ItemTrait`/`ItemImpl`/etc (convert each `collect_*` they need, or bridge).
  This removes the E063 corruption for non-fn items. Gate: `--check onestruct.sio` E063 flood gone.
- **G0 (leaf helpers, mostly done):** `lower_opt_type` → needs `checker_lower_opt_type_inplace`
  (only `checker_lower_type_expr_mut` @1031 exists; wrap it). `check_refinement_literal`,
  `check_binding_init_expr` → `*_inplace`. These have NO Checker-returning children except via
  report_*, which already have `_inplace`.
- **G1 statements:** `check_let_stmt`, `check_var_stmt`, `check_assign_stmt` → `*_inplace`; then
  flip `checker_check_stmt_inplace` (2422) arms StmtLet/StmtVar/StmtAssign off `_bridge` onto them.
- **G2 expr leaves up:** `check_field_access`, `check_binary_expr` → `*_inplace` (delegate to
  `checker_check_expr_inplace` for sub-exprs).
- **G3 calls:** `check_call_expr`, `check_method_call` → `*_inplace` (the two biggest: 14.1MB /
  15.0MB). These have the most `self.method()` calls — convert their callees or bridge.
- **G4 expr dispatch:** rewrite `checker_check_expr_mut` (2472 fallback) to dispatch the remaining
  `e.kind` arms to the new `*_inplace` methods instead of `(*c).check_expr(e)`. Removes the last
  heavy bridge.
- **G5 block/stmts/item:** `check_block`, `check_stmts`, `check_item`, `check_block_expr` →
  `*_inplace`; flip `checker_check_block_inplace`/`_stmts_inplace`/`_item_inplace` off their bridges.

Target component = the **38** crash-path `check_*` methods (entry `check_program` mod.sio:369).
69 `*mut`/`_inplace` methods already exist; 299 by-value remain (most off the --check path — do NOT
touch those; scope strictly to the connected component reachable from check_program on hello.sio).

## Gate cadence (after EACH group — check.sio is the MODULAR compiler, NOT bin/souc)

1. `bin/souc self-hosted/compiler/main.sio /tmp/mc.elf` → rc=0, **0 errors** (bin/souc is canonical
   and UNAFFECTED by check.sio edits — it's the lean_single fixed point; it just must still compile
   main.sio cleanly).
2. `chmod +x /tmp/mc.elf` then `ulimit -s 8192; /tmp/mc.elf --check <input>` → **rc=0** (not 139), ×5.
   Use the input that exercises ONLY the converted path so far: `fn main(){}` then `fn main(){let x=1}`
   after the spine; `hello.sio` (has x+2, print) only after G2–G4. (All three crash rc=139 today.)
   This is the cure landing: frames small enough to fit 8MB AND no E063 corruption flood.
3. Frame-shrink diagnostic: recompile main.sio, confirm the converted methods' named frames dropped
   (e.g. check_call_expr 14.1MB → KB). `grep 'stack frame too large' | wc -l` should fall below 86.
4. Multi-item program (fn+struct+enum) `--check` → rc=0 (the `_`-arm also by-value-bridges).
5. `scripts/ci/run_pass_output_gate.sh` (compare) — guards against a conversion miscompile
   corrupting output. (Golden `tests/run-pass/move_inplace_aliasing.sio` value 47531047531 already
   built + baked, reusable.)

⚠️ The decisive proof is **step 2 at 8MB ulimit** — frames small enough to not overflow a normal
stack. Self-reproduction of bin/souc is NOT the gate here (bin/souc isn't edited); the gate is
mc.elf's behavior.

## ⚠️ ORDERING REALITY (the crash is in the SPINE ENTRY, not the leaves)

Measured: `fn main(){}` (EMPTY, no expr) ALREADY crashes rc=139 at 8MB with zero error output. So
the overflowing frame is in the spine ENTRY (collect_fn_def / check_item / check_block / check_fn_item),
which in a pure bottom-up order convert LAST. Consequence: **no `--check` gate passes until the spine
entry is converted** — a leaf-only increment compiles clean but still crashes. Two intermediate
gates DO move before then: (a) `bin/souc main.sio` rc=0/0-err, (b) the frame-shrink count. Use those
for the early groups; the `--check rc=0` gate only becomes meaningful once check_item/check_block/
collect_fn_def are `*mut`. Practical order: do **Gc + the SPINE (G5 check_item/check_block/check_stmt
+ collect_fn_def) EARLY** so the entry frames shrink and `--check` can return, bridging DOWN to
still-by-value leaves; then convert leaves G0–G4 to remove the remaining bridges/frames. (Bridges
DOWN to unconverted leaves are safe — they copy, but the ENTRY frame is what overflows first.)

## First increment for the fresh session (start here, on FULL context)

Gc + spine-entry: complete `checker_collect_item_inplace` routing (Gc), then make
`collect_fn_def`/`check_item`/`check_block`/`check_fn_item` true `*mut` (they currently bridge —
`checker_collect_fn_def_inplace` etc. value-thread internally; make them `(*c).field`). Bridge DOWN
to by-value `check_stmt`/`check_expr` for now. **Valid before/after test: `fn main(){}` and
`fn main(){ let x = 1 }` — BOTH crash rc=139 today (confirmed); success = rc=0 at 8MB.** Do NOT use
`hello.sio` as the G1 gate — its `x+2` (check_binary_expr) and `print(y)` (check_call_expr) stay on
the by-value bridge until G2–G4, so it would still crash and mislead. Once the entry returns rc=0,
proceed to convert the leaves and watch the frame count fall and hello.sio start passing.

⚠️ DO NOT start on low context. Each group is one self-reproduce + 5-gate cycle. A half-converted
spine (bridge removed before callee converted) bricks mc.elf — keep bridges until the callee is live.
