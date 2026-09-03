<!-- docs:meta
topic_id: repo.docs.audit.mut-refactor-execution-plan-2026-05-31
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.mut-refactor-execution-plan-2026-05-31
-->

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

## ✅✅✅ G1 LANDED + COMMITTED (2026-05-31, next session) = `e1f99c7d` — `let x=1` --check rc=0

The StmtLet path is now true `*mut`. **`fn main(){let x=1}` reaches `check: OK` rc=0 at 8MB (was
rc=139).** One file (`self-hosted/check/check.sio`, +184), bin/souc UNTOUCHED, worktree clean.
- New `*mut` fns (faithful transcriptions): `checker_check_let_stmt_inplace` (mirrors check_let_stmt
  @12208; NO inline by-value Checker call — every helper is *mut or an isolated leaf);
  `checker_check_binding_init_expr_inplace` + `_bridge` (simple inits → *mut expr spine which handles
  IntLit inline & sets last_literal_*; the 4 epistemic special cases — Contest/lift_knowledge/
  AuditAttach/Path-TyNamed — route through ONE isolated by-value leaf calling the EXISTING
  check_binding_init_expr method = a single 8MB SRET site kept off the hot frame, never hit by `let x=1`);
  `checker_lower_opt_type_inplace`; `checker_check_refinement_literal_inplace` (replicates the final
  last_literal_kind=0 reset @12338); `checker_eval_const_int_{expr,opt_expr}_mut` (avoid eval_const_int_*'s
  by-value Checker param via checker_const_int_lookup_mut); `checker_const_int_bind_value_inplace`.
- `checker_check_stmt_inplace` StmtLet arm flipped `bridge` → `let_stmt_inplace`. StmtVar/StmtAssign
  STAY on the bridge (out of G1 scope).
- GATE (`ulimit -s 8192; mc.elf --check`): `fn main(){}` rc=0 ×5; `fn main(){let x=1}` **rc=139→rc=0
  check: OK ×5**; NON-VACUITY PROVEN — `let x: bool = 1` → `error[E001] expected bool, got i64`
  check:FAIL rc=1 (the spine genuinely type-checks); `let x: i64 = 1` → rc=0. bin/souc main.sio rc=0/0-err.
- ⚠️ DRIVER IS `self-hosted/compiler/main.sio` (an Explore agent hallucinated `mc_main.sio`).
- NEXT = G2 expr spine (Layer 2 of the recipe): ExprIdent/ExprBinary/ExprFieldAccess leaves, gating on
  TAIL-position exprs (let-RHS + StmtVar/Assign stay vacuous behind the by-value bridge until converted).

## ✅✅ INCREMENT 1 LANDED + COMMITTED (2026-05-31, fresh session) — first `--check rc=0` BANKED

The Gc+spine-entry frame-isolation increment is **done, gated, and committed** (no longer just a draft).
Edits to `self-hosted/check/check.sio` (one file, bin/souc untouched):
1. `checker_collect_item_inplace` `_` arm → `_ => {}` (was the 8MB by-value `(*c).collect_item`).
   **Documented correctness gap:** user struct/enum/impl/policy decls no longer collected → programs
   that USE them are not yet checked. Gc *mut collectors are the next correctness step.
2. `checker_check_item_inplace` ItemImpl → new leaf `checker_check_impl_item_bridge` (isolates the
   8MB check_impl_item SRET out of the hot frame; impl-free programs no longer carry it).
3. multitest E042: `(*c).report_multitest_correction_required(...)` by-value Checker copy → new *mut
   leaf `checker_report_multitest_correction_required_inplace` (mutates (*c) scalars + prints; no copy).
4. The two `(*c).fn_sigs.get(sig_id)` reads in `checker_check_fn_item_inplace` are LEFT as proven
   by-value `.find`/`.get` — they copy only the ~38KB FnSigTable, never the 8MB Checker, so they're
   immaterial against the 8MB budget. (FnSig is ~600B, NOT the 1.7MB earlier notes assumed.) A direct
   `(*c).fn_sigs.entries[idx].field` read variant (mc_v1) was built and is behaviorally IDENTICAL on
   every runnable input, but could NOT be VALIDATED this session: the only error path that consumes
   the read (a real return-type mismatch, `fn g()->bool{5}` or `return 5`) crashes rc=139 on the
   unconverted expr/Return spine on BOTH variants. So the conservative proven path ships; revisit
   direct-read WITH a real return-mismatch gate once the expr spine lands.

**GATE RESULTS (mc.elf = bin/souc compiling main.sio, then `ulimit -s 8192; mc.elf --check`):**
- `fn main(){}` → **`check: OK` rc=0 ×5** (was rc=139). **First --check success ever on this branch.**
- `struct P{a}`+main → **`check: OK` rc=0 ×5** (vacuous: P unused; collection skipped per gap above).
- `fn main(){let x=1}` → rc=139 (UNCHANGED — expr/let spine not yet converted; next increment).
- `run_pass_output_gate.sh` → PASS, no regression (88 PASS; bin/souc unaffected).
- bin/souc compiles main.sio rc=0, **0 errors**. Frame-warning count 86 (the empty-main *path*
  frames dropped below 8MB; total count unchanged because off-path by-value fns still warn).

**Read-style settled (the gates were insensitive to it):** the passing gates (`fn main(){}`,
`struct_p`) yield `check: OK` under EVERY read-failure mode — empty/unit bodies never reach the one
error path that consumes the return-type read, so they validate the frame fix but NOT the read value.
A sensitive test (`return 5` / `fn g()->bool{5}`) crashes rc=139 on BOTH the direct-read (mc_v1) and
`.get()` (mc_v2) builds (its body routes through the unconverted expr/Return spine) → the read is
**untestable this session**. Conservative call for the canonical compiler: ship the proven `.get()`
version. `fn f()->i64{}`→`check: OK` is identical on both builds — consistent with (but not proof of)
pre-existing empty-body leniency. ⚠️ The real return-mismatch gate (`fn g()->bool{5}`) becomes
available once the expr/Return spine lands; use it to validate direct-read before adopting it.

**NEXT INCREMENT:** G1 + expr spine so `fn main(){let x=1}` reaches rc=0 (see "NEXT BLOCKER" below).

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

## ⚠️ CUMULATIVE-FRAME REALITY + more inline-bridge sites (2026-05-31, tested hands-on)
Converting ExprIdent (clean *mut, committed pattern below) + collect `_ => {}` makes min_empty pass
but a TYPED FUNCTION WITH A BODY (`fn idf(x:i64)->i64{x}`) STILL crashes rc=139 at 8MB. gdb: the
faulting frame is **3,536,176 B (~3.5MB)** — BELOW the 4MB warn threshold, so it never appears in
the frame-warning map; it overflows CUMULATIVELY once the body-check chain runs deep. Root: the
hot-path *mut functions STILL contain inline by-value sites that each compile-allocate a Checker-sized
buffer:
  - check_fn_item_inplace (2358-2367): `(*c).report_multitest_correction_required(...)` inline
    [FIXED this session — extracted to checker_report_multitest_correction_bridge leaf], AND two
    `let sig = (*c).fn_sigs.get(sig_id)` / `let sig2 = (*c).fn_sigs.get(sig_id)` FnSig-BY-VALUE copies
    (2347/2360) — FnSig is large (~1.7MB each → ~3.5MB), the dominant remaining frame. FIX: add a
    *mut field accessor (read sig.return_type/effects/effect_count without copying the whole FnSig),
    or a `checker_fn_sig_get_*` helper returning only the needed scalars.
So the frame-isolation rule applies not just to Checker-returning calls but to ANY large-aggregate
by-value materialization (FnSig, TypeEnv, StructInfo) inline in a hot *mut function. Sweep every
hot-path *mut function for inline `let x = (*c).bigfield.get(...)` / `(*c).method() -> BigAggregate`
and either isolate to a leaf or read fields directly. min_empty passes because empty-body main keeps
the chain shallow; depth is what tips the cumulative sum over 8MB.

VALIDATED PATTERN (paste-ready, correct *mut, re-derivable):
  checker_check_ident_expr_inplace(c, e) -> TypeEntry: ident_ty=(*c).env.lookup(e.name); if
  checker_is_linear_type_inplace(c,ident_ty) && (*c).suppress_linear_consume_depth==0 { if
  (*c).borrows.is_borrowed(e.name){checker_report_error_at_inplace(c,e.span,38,0,0,0)} else { let
  cr=(*c).borrows.consume_linear(e.name);(*c).borrows=cr.0; if cr.1!=0
  {checker_report_error_at_inplace(c,e.span,cr.1,0,0,0)} } } ; ident_ty
  + dispatch arm `ExprIdent => checker_check_ident_expr_inplace(c, e)` in check_expr_inplace (2471).

## ⚠️ EMPIRICAL CORRECTION to the spec-workflow recipe (2026-05-31, tested)
The recipe (MUT_EXPR_SPINE_RECIPE_2026-05-31.json) claims extracting collect_item_inplace's `_`
arm into a leaf bridge is "necessary-and-sufficient for the collect-side frame." **FALSE, tested:**
with the bridge extracted, `fn main(){}` STILL crashes rc=139 at 8MB — because the prelude carries
non-fn items that hit the bridge, and the bridge's own `(*c).collect_item(item)` by-value call has
the 8MB frame (just relocated, not removed). The ONLY validated collect states for `fn main(){}`:
(1) `_ => {}` (skip) → PASSES (but skips real user struct/enum collection); (2) full *mut collectors
for EVERY prelude kind → would pass + be correct. Bridge-extraction-alone is WORSE than skip (turns
min_empty from pass→crash). OPEN QUESTION for next session: is the per-program prelude (one item of
~13 non-fn kinds) REDUNDANT with checker_init_in_place's register_builtin_* (in which case `_ => {}`
is correct), or real declarations needing collection? Resolve this BEFORE choosing skip vs full
convert. (The "15 phantom items" earlier was layout-sensitive corruption from by-value bloat, not 15
real items — a clean minimal collect path showed 1.)

## ⚠️ CRITICAL CONVERSION RULE (learned the hard way, 2026-05-31)

**A by-value Checker-returning call allocates its 8MB SRET buffer in the ENCLOSING function's
frame at COMPILE time — regardless of whether its runtime branch executes.** So an `if rare { let p
= (*c).by_value_method(...); ... }` bridge sitting INLINE in a hot *mut function gives that hot
function an 8MB frame and it overflows when called, even though the rare branch never runs.
**Therefore: every remaining by-value bridge MUST live in its OWN dedicated leaf function**, never
inline in a converted hot-path function. (This is exactly why collect_item_inplace's `_` arm crashes:
its `(*c).collect_item(item)` by-value call puts collect_item's 8MB frame into collect_item_inplace.)
Verified: `fn main(){}` passes only when the `_` arm has NO inline by-value call (`_ => {}`).

## NEXT BLOCKER (precise, gdb-confirmed): checker_check_expr_inplace is INCOMPLETE
checker_check_expr_inplace (check.sio:2450) handles ONLY {IntLit, FloatLit, BoolTrue/False,
StringLit, CharLit, Return} inline; EVERYTHING else (`_ => checker_check_expr_mut(c,e)` →
by-value `(*c).check_expr` = the 12.29MB frame). So ANY real init/expr (ident, binary, call,
struct-lit, …) still routes to the by-value 12.29MB check_expr. `min_let` (`let x=1`) traced:
LET_INPLACE_ENTRY → BINDING_INPLACE → **EXPR_MUT_BYVAL** then the E048/E005 corruption cascade
(the by-value check_expr runs on already-corrupted state). So the EXPR SPINE is the real mass:
check_expr_inplace must dispatch ExprIdent/ExprBinary/ExprCall/ExprPath/ExprField/ExprMethodCall/
ExprStructLit/… to NEW *mut handlers (checker_check_binary_expr_inplace, _call_expr_inplace,
_field_access_inplace, _method_call_inplace, _path_expr_inplace, …), each obeying the
isolated-bridge rule above. Until the expr spine is converted, only literal-only bodies pass.

## MILESTONES ACHIEVED THIS SESSION (committed at ce30d1220 = findings; let-spine drafted)
- `fn main(){}` and `struct P{a}`+main → **`check: OK` rc=0 at 8MB** (first --check successes ever).
- check_let_stmt → *mut spine DRAFTED + compiles + self-reproduces (checker_check_let_stmt_inplace,
  _binding_init_expr_inplace + isolated special bridge, eval_const_int_opt_expr_mut,
  _const_int_lookup_mut, _const_int_bind_value_inplace, _check_refinement_literal_inplace). Correct
  *mut code; blocked only by the check_expr_inplace incompleteness above (so not yet wired-in as a
  pass). These functions are the template for the rest of the spine.

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
