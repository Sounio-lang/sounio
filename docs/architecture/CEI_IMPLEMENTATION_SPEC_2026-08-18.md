<!-- docs:meta
topic_id: repo.docs.architecture.cei-implementation-spec-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.cei-implementation-spec-2026-08-18
-->

# CEI — Consolidated Implementation Spec (workflow-synthesised, adversarially verified)

_Produced by the `cei-program-advance` workflow (fan-out survey → opus adversarial-verify → opus synthesis), run `wf_12e329df-b89`, 2026-08-18. Model routing: sonnet survey / opus verify+synthesis; WS-C re-routed to opus/high after a StructuredOutput failure. Read-only design; no source edited._

**Verdicts:** `P0-lowering`=SOUND (high), `WS-B-memory`=FLAWED (high), `WS-C-nonassoc`=FLAWED (medium), `WS-D-lean`=FLAWED (high)

---

Confirmed the decisive integration fact: on this tree (`/workspace/.wt/fable-1`, `lower.sio` = 19461 lines, `struct Lowerer` @593) the GPU precedent the P0 design mirrors returns **zero** hits. The P0 line-references are valid only on the `lower-array` tree. That branch-lock leads the spec. Here is the consolidated deliverable.

---

```markdown
# CEI — Consolidated Implementation Spec (4-Workstream Integration)

*Synthesised 2026-08-18 from four independently adversarially-verified workstream designs
(P0-lowering = SOUND; WS-B-memory = FLAWED-but-rescued; WS-C-nonassoc = FLAWED-but-rescued;
WS-D-lean = STUB, no design to integrate). Read-only design; no files edited. Every line/function
coordinate below is branch-conditional — see §0.*

## 0. BRANCH-LOCK (blocking precondition for the entire spec)

**The single most important fact across all four workstreams.** Every line number in the P0-lowering
design — and its whole "mirror `expr_is_gpu_sync_call_ref`" strategy — is valid **only** on the worktree
`/workspace/.wt/lower-array` (branch `lane/fable-1/handle-table-182-dispatch`, `lower.sio` = 17819 lines,
`struct Lowerer` @723). **This was empirically confirmed false on the live checkout:**

- Current worktree `/workspace/.wt/fable-1` (branch `lane/fable-1/p0f-ffi-takeover`):
  `lower.sio` = **19461 lines**, `struct Lowerer` @**593**, and
  `grep -c 'expr_is_gpu_sync_call_ref\|gpu_launch_kernel_name_ref' lower.sio` = **0**.
- The canonical tree `/workspace/sounio` has the same absence.

**Consequence:** if P0 must land on `main` (or on `p0f-ffi-takeover`), **there is no GPU precedent to mirror**
and no `check_method_call` GPU bypass to parallel. The design must be **re-derived against the target tree**
before any coordinate is trusted. Do not implement P0 from the coordinates as written without first
re-grepping on the actual integration base.

**Action item 0 (do before anything else):** pick and record the integration base. Two options:
(A) integrate onto `lane/fable-1/handle-table-182-dispatch` where the GPU precedent exists, then merge
forward; or (B) integrate onto `main`/current tree and re-derive the perform-hook and checker-bypass from
first principles (there is no GPU no-op arm there to copy). **UNCERTAIN which is intended — confirm with the
operator.** All coordinates in §1 below are annotated `[lower-array-only]` where they depend on the GPU tree.

---

## WS-A / P0 — ExprHandle lowering + perform-call inlining  (verdict: SOUND)

**Thesis role:** minimal tail-resumptive handler mechanism. **Honest scoping (verifier correction):** P0 does
**not** wire the orphaned 2991-line CPS runtime `self-hosted/effects/handlers.sio`; it is a parallel minimal
mechanism. Do **not** describe P0 as "wiring the CPS engine" or as validating
handlers-as-proof-carrying-interpreters — multi-shot / abortive / non-tail-resume are out of scope.

### Corrected verified step list

1. **Clause form (grammar, zero parser change).** `with { ... }` is parsed by the generic
   `parse_block`/`parse_stmt` (`parser/stmts.sio:166`, `:254`), which accepts only `StmtLet`/`StmtVar`/expr-
   statements. The **only** clause shape that parses today is a **let-bound closure**:
   `let <op> = |<params>| { <body> }` → `Stmt{StmtLet, name:<op>, expr:Some(ExprClosure)}`.
   The `op(params) => body` syntax in `examples/effects.sio` is **commented-out aspirational, NOT implemented** —
   do not treat it as working.
2. **Callee shape.** `perform Epistemic.add(x,y)` drops `perform` at parse and becomes a **single**
   `ExprMethodCall{name:"add", left:Some(ExprIdent{"Epistemic"}), args:[x,y]}` — **not** the curried double-call
   of `GPU.launch`. Precedent to mirror is `expr_is_gpu_sync_call_ref` `[lower-array-only]`, **not**
   `gpu_launch_kernel_name_ref`.
3. **Lowerer state.** Add to `struct Lowerer`: `handler_effect_stack: [Name;4]`,
   `handler_block_stack: [Option<Box<Block>>;4]`, `handler_depth: i64`, following the `loop_labels`/`loop_depth`
   idiom. **Do NOT** introduce a `HandlerFrame`/`HandlerClause` struct-of-arrays — that is the exact nested-by-value
   aggregate shape with a documented miscompile history in this codebase (struct-array elem dispatch / nested
   field-store). Store raw `e.name` + `e.handler_block` pointer per depth; re-walk the `StmtList` at each perform
   site to find the matching clause.
4. **`lower_handle_expr_ref`.** Push (cap 4, error via `report_error_at` beyond, mirroring `bind_local`'s 4096
   guard); lower body via `lower_block_ref` (ExprHandle stores its body in the same `.block:Option<Box<Block>>`
   field as ExprBlock); pop `handler_depth` on **every** exit path. **Never** lower `e.handler_block` itself
   through `lower_block_ref` — that would run each `let op = |...|{}` through the ordinary closure path
   (`lower_closure_expr_ref`) and can trip "capturing closure cannot be used as a value". Walk its `StmtList`
   structurally only.
5. **Dispatch arm.** Add `else if e.kind == ExprKind::ExprHandle { self.lower_handle_expr_ref(e) }` immediately
   before the catch-all in `lower_expr_non_binary_ref`. **Characterisation fix (verifier):** ExprHandle does not
   silently "evaporate" — today it hits the catch-all `report_error_at` (a compile-error path), not a no-op.
6. **Perform recognition + inline arm** inside `lower_method_call_expr_ref`, positioned as early as the GPU no-op
   `[lower-array-only]` so no ordinary method-call machinery ever sees the perform call. Shape-check mirrors
   `expr_is_gpu_sync_call_ref`; walk `handler_depth-1 → 0` (innermost-first, correct shadowing); find `StmtLet`
   whose `.name == op` and `.expr` is `ExprClosure`. **On no clause / arity mismatch: refuse hard**
   (`lowerer_mark_error; return IR_INVALID_REG`) — never fall through, or codegen mints a body-less mangled
   `Epistemic_add` and SIGSEGVs (the documented GPU.sync() pre-fix failure class).
7. **Clause inline (tail-resumptive).** `scope_depth += 1; saved = locals.count`; for each arg,
   `lower_expr_ref` then `bind_local(param_name[i], reg, false)`. **CRITICAL:** `bind_local` zeroes
   `scalar_kind[idx]=0` by design → an unclassified f64 param hits the `println` kind-0 char*-printer SIGSEGV
   (project memory `project_println_bool_scalarkind_segv`). **Immediately re-bind kind** from `ClosureParam.ty`
   (f64→kind 2, i64→kind 1, struct→`bind_local_struct_type` per the `Some(x)`-payload idiom). Lower clause body
   (ExprBlock→`lower_block_ref`, else `lower_expr_ref`). Restore `scope_depth -= 1; locals.count = saved`. The
   clause body's result register **is** the perform value — resume-once-at-the-tail, no CPS, no `resume()`.
8. **Check EDIT 1 — `check_handle_expr` restructure.** Push `e.name`+`e.handler_block` onto new by-value
   `Checker` fields (`handler_effect_stack`/`handler_block_stack`/`handler_depth`) **before** checking the body,
   so perform sites inside the body see the clauses; pop after; keep the existing `check_block(*hb)` (checks each
   clause closure standalone — cheap, retain).
9. **Check EDIT 2 — the load-bearing missing piece (top risk).** `check_method_call` types the **receiver first**
   via `check_opt_expr(e.left)`. For `Epistemic.add(x,y)` the bare `ExprIdent{"Epistemic"}` is not a bound local
   → **fails as an undeclared-identifier (E137-class) error in `check` before any lowering runs**. Add a
   GPU-style early-return bypass `[lower-array-only precedent]` **before** the `check_opt_expr(e.left)` call:
   predicate `checker_expr_is_active_handler_perform` matches AST shape *(bare-Ident method-call receiver)*
   against the **live handler-effect stack** (never against the struct table — so effect-id `Epistemic` and
   stdlib struct `Epistemic` never collide). On receiver-match + clause-match: type-check args against
   `ClosureParam` types (reuse `check_closure_expr`'s param-bind+body-infer subroutine) and **return the clause
   body's inferred type** — this is what lets the *same* `Epistemic.mul(x,y)` source type as `Epistemic` under a
   GUM handler and `MCResult` under an MC handler. On receiver-match + no clause / arity mismatch: emit the new
   **"unhandled effect operation / clause arity mismatch"** diagnostic (this fuses the "clause completeness +
   arity" check into the call site — no separate collection pass). On no receiver-match: fall through unchanged.
10. **Signature nit (verifier):** the predicate lives in the **by-value** `check_method_call` whose checker is
    `self: Checker` (by value) — read the stacks off `self`, **not** a `*mut Checker`. EDIT 1 threads the same
    by-value `Checker` functionally through `check_opt_block`; fields are visible to the by-value perform site.
11. **Third-edit guard (verifier, must confirm):** there is a **parallel `*mut` inplace checker**
    `checker_check_method_call_inplace` carrying its own GPU bypass. EDIT 2 (by-value) suffices **only** because
    handle bodies are checked entirely in the by-value spine (inplace `else` → `checker_check_expr_mut` →
    by-value `check_handle_expr` → by-value `check_block`/`check_method_call`, no re-entry to inplace inside the
    subtree). **Confirm no handle-body path re-enters `checker_check_method_call_inplace`;** if any does, a
    THIRD parallel bypass edit there is required, or `Epistemic.<op>` returns E137.
12. **Diagnostic code (open):** grep the `report_error_at`/`print_error_message` table for the next free E-code
    before adding the "unhandled effect operation" diagnostic — **do not guess a number** (collision risk).
13. **Scalar-kind classifier of the inlined result (open, gates the smoke test).** `expr_result_scalar_kind_ref`
    (~`lower.sio:12690-12820`, consulted when lowering `let z = <call>`) has **not** been verified to classify
    an **inlined clause-body** result (vs a real call's declared return). **Read it before implementing.** If it
    returns `scalar_kind 0` for the handle result, `println(r)` in the smoke test SIGSEGVs via the char*-printer.
    **The `expect-stdout: 5` claim is NOT yet demonstrable** until this is resolved. Extend it the same way the
    bool-`println` fix extended `println_dispatch_name`/`bind_local` kind-rebinding.
14. **`check/effects.sio`:** no change — `Epistemic=8` resolves via `effect_name_to_id`. Reuse as-is.
15. **P0 smoke (int-only).** `examples/effect_uncertainty_smoke.sio`, new: one trivial handler,
    `handle<Epistemic>{ Epistemic.add(2,3) } with { let add = |a:i64,b:i64| { a+b } }; println(r)`,
    `//@ run-pass` / `//@ expect-stdout: 5`. Plain `i64` isolates the lowering mechanism from any stdlib adapter
    risk. Verify with `souc check` then a **from-source** Madaros build (`bash scripts/ci/build_modular_madaros.sh`,
    never the prebuilt wrapper) then `souc run` — **gated on step 13**.
16. **P1 demonstrator (not part of the P0 arm).** `examples/effect_uncertainty_gum_vs_mc.sio`: thin adapters over
    `stdlib/epistemic/knowledge.sio` (`ep_measured`/`ep_add`/`ep_mul` @76/94/112) and
    `stdlib/epistemic/montecarlo.sio` (`mc_input_normal`/`mc_add`/`mc_mul` @198/298/341); same handled body under
    two `with{}` blocks. **Confirm cross-module visibility of the non-`pub` `mc_*` fns with `souc check` first**
    (project visibility model: `pub` is advisory here).

### Exact files
- `self-hosted/ir/lower.sio` — struct fields; `lower_handle_expr_ref` + perform hook; dispatch arm; inline logic.
- `self-hosted/check/check.sio` — `check_handle_expr` restructure; `check_method_call` bypass (EDIT 2);
  new `Checker` fields; new diagnostic code; **possibly** `checker_check_method_call_inplace` (step 11).
- `self-hosted/check/effects.sio` — read-only (`Epistemic=8`).
- `examples/effect_uncertainty_smoke.sio` (new, P0), `examples/effect_uncertainty_gum_vs_mc.sio` (new, P1).
- Read-only: `parser/exprs.sio`, `parser/stmts.sio`, `stdlib/epistemic/knowledge.sio`,
  `stdlib/epistemic/montecarlo.sio`, `examples/effects.sio` (aspirational syntax — do not mistake for working).

### Single most important next action
**Resolve §0 branch-lock (which tree has the GPU precedent), then implement + `souc check`-verify EDIT 2
(`check_method_call` receiver bypass) on a from-source build — because without EDIT 2, `Epistemic.add(x,y)`
dies in `check` and P0 never reaches the lowering arm at all.**

---

## WS-B — Effect-scoped memory reclamation  (verdict: FLAWED → rescued; architecture SURVIVES)

**Verifier's fatal correction (must propagate everywhere):** the design's central "the probe is a stub
(`emit_xor_rax_rax` then early return, real scan dead below)" claim is **FALSE**. Both
`native_v2_emit_current_frame_live_probe` definitions (`codegen.sio:1527`, `codegen_x86_linux.sio:2911`) are
**byte-identical FULL conservative scans** (loop over machine + spill slots, `test`/`jnz` → live, else
`xor rax,rax` = not-live *after* the scan). No stub, no early return, no dead scan. They are **dead only in the
sense of being uncalled** in `codegen_x86_linux.sio`.

**Downstream verdict inversions (apply):**
- **Item-5-alone (wire the existing probe+reset) is SOUND but INEFFECTIVE**, not a "fabricated pass."
  `nonzero=live` is a *safe over-approximation*. It fails on the discard-per-iteration witness for the **same**
  `nonzero≠dead` reason as item-4: the stale handle slot stays nonzero until overwritten, so the scan reads it
  as live and the reset never fires. The item-4 vs item-5 distinction collapses — **both fail for one identical
  reason; neither is unsound.**
- The genuine fabricated-pass risk appears **only** if someone weakens the scan to always-not-live — which is
  **not** the current code. Route A's real remaining work is **wiring the existing scan into the slow path**,
  not "replacing a stub."

**Line-number corrections (verifier, ~70-line offset in `codegen_x86_linux.sio`):** slow-path `exit(182)` ≈ **6534**
(heap-limit `exit(181)` sibling ≈ 6528); probe **2911** (not 2842); reset **2955** (not 2895);
`own_context_new` @**951/1052** (not 1019/1122); escape-return code-91 reject at `check.sio:4597` (also 4315,
7860); literal `error[E091]` at `check.sio:11559` (not 12327).

**Structural claims that survived refutation (verified true):** `ownership.sio` is instantiated-not-driven
(zero `own_declare_var`/`own_check_move`/`own_compute_drops` calls); `escape.sio` fns are all module-private
(need `pub`); `regalloc.sio` `LiveInterval`/`liveness_build_intervals` present (do not re-derive liveness);
`stack_maps` `root_kind_mask` only ever `mixed`; **no rbp-chain / stack-walk anywhere in `self-hosted/native`**
(so the process-global top-risk holds); `codegen.sio` is the sole wired probe→reset+`pin_count` site.

**Two soundness holes the architecture must close (both real):**
- **`nonzero ≠ dead`:** a per-slot root *bitmap* (plan item 4) alone moves nothing — a stale handle from a prior
  iteration reads as live until overwritten. The fix is a **per-safepoint LIVE-root fact**, sourced by **nulling
  the slot at a certified drop point** (cheap extra store) so the existing conservative scan becomes *accurate*.
- **Unboxed ≤16B values** (`native_v2_handle_raw_ptr_threshold`) consume **no** handle slot but **do** consume
  heap-cursor bytes. A certificate over handle roots only is blind to them → resetting `heap_cursor` while an
  unboxed value is live is a silent use-after-free. The classification (Step 4) **must** flag raw-unboxed-pointer
  temps as managed roots.

### Corrected verified step list

0. **(blocking) E091 code collision.** `check.sio:4597` raises code 91 (frame-local escape) and
   `check.sio:11559` prints literal `error[E091]` for an unrelated "audit counterexample incomplete" diagnostic.
   Give the escape reject a distinct code (or disambiguating detail text) **before** writing any falsifier that
   asserts on E091 text. *(Step-0 substance CONFIRMED correct by verifier; only its coordinates were wrong.)*
1. **(architecture decision, needs operator sign-off) Route A vs Route B.**
   - **Route A** (reactive backstop): wire the **existing conservative scan** (`codegen_x86_linux.sio:2911`) +
     reset (`:2955`) into the `handle_slow_path` (~6534), mirroring `codegen.sio:3760-3830` exactly (retry-once →
     `exit(182)`/`exit(181)`, **preserve the `pin_count` gate**). This is now understood as *"wire an existing
     sound conservative scan,"* not *"replace a stub."*
   - **Route B** (primary, thesis-aligned): **scope-exit arena-pop** — restore `runtime_context.handle_count`
     and `heap_cursor` to a watermark snapshot taken at scope entry (~15-line emitter reusing
     `emit_load/store_runtime_context_field` @2897-2900). Pops a **local** watermark rather than resetting the
     whole table → sidesteps Route A's whole-process-liveness problem, and directly realises "effect-scoped
     certified reclamation." **Recommend B primary, A retained as pressure-triggered backstop.**
2. **Home the certifier in `check/borrow.sio`, NOT `ownership.sio`.** `ownership.sio` (2837-line
   `OwnContext`/`own_compute_drops`) is fully unwired — driving it means building a whole new drop-point layer.
   `borrow.sio`'s `BorrowEnv` is already driven on every checked function and backs `tests/compile-fail/
   ownership_*.sio`. Add per-entry `alloc-site id` + `certified-dead-at-scope-N` bit to `BorrowEntry`.
3. **Certifier ("3-lite").** Generalise `checker_expr_provenance` / `checker_place_root_local` from the single
   return-escape call site to a **scope-exit sweep** at `checker_check_block_inplace`'s `pop_scope` point: for
   every `BorrowEntry` in the closing scope, emit a "provably-scope-local, free-at-exit" **fact** (persist into a
   small IR side-table keyed by alloc/HIR-node id) when: never returned, never stored into a first-class place,
   and — until Steps 6/7 — **never passed by reference into any call** (a call = unconditional escape).
4. **Managed-root classification into the MIR temp table.** Liveness intervals already exist (`regalloc.sio`,
   do not re-derive). What is missing is **per-vreg type tagging** (handle / scalar / raw-unboxed-pointer) — no
   `temp_types`/`vreg_type`/`is_handle_type` table exists in `native/` or `ir/`. Thread the checker's `TypeEntry`
   through `ir/lower.sio` into the MIR temp table; the stack-map recorder consults it to set `root_kind` per slot.
   **Must also flag raw-unboxed-pointer temps** (closes the ≤16B hole).
5. **Emit the per-safepoint LIVE-root bitmap in `stack_maps.sio`** (today `root_temp_count`/`root_spill_count`
   are set equal to `temp_count`/`spill_count` — a no-op alias). Bit = 1 only when the slot is **both**
   handle/raw-pointer-typed (Step 4) **and** not-yet-certified-dead (Step 3). Null certified-dead slots at their
   drop point so the conservative scan becomes accurate (the `nonzero≠dead` fix).
6. **Wire `escape.sio` as the consumer of Step-3 facts (general case).** Add `pub` to the ~15 fns
   (`esc_add_node`/`esc_add_edge`/`esc_mark_*`/`esc_propagate`/`esc_analyze`); feed the graph from the **same**
   provenance walk (do not duplicate taint logic). Its fixed-point subsumes the current informal `if/match/
   struct-lit` union.
7. **Defer `ir/memory_analysis.sio`** (Andersen points-to, 2686 lines) — it is interprocedural **callee escape
   summaries** (plan item 2), unneeded by the *call-free* discard witness. Wire only once a **witness with calls**
   is the acceptance target. **Correctness caveat (top risk):** callee summaries stop being "optional later phase"
   and become a **precondition** the instant the certified region contains any call.
8. **Implement the chosen mechanism, gated strictly by the Step-3 certified fact** (never a bare
   nonzero/liveness heuristic). Route B: watermark-restore emitter at the certified scope-exit. Route A: call the
   Step-5-bitmap-driven scan + reset from the slow path exactly as `codegen.sio:3799-3823`, preserving `pin_count`
   and the `exit(182)`/`exit(181)` fallthrough as a hard backstop — **do not remove it.**
9. **Falsifier suite — TWO controls (a single "must-refuse" proves nothing about no-use-after-reclaim):**
   - **C1 (conservatism):** twin of the witness that returns/stores the struct out of scope on some iteration →
     certifier must **refuse** to mark dead → no pop/reset for that alloc (program still hits 182/181, same as
     today).
   - **C2 (positive use-after-reclaim detector):** wire `native_v2_handle_entry_field_epoch` (written @~6372,
     bumped by the reset's `collector_epoch` increment @~2904-2908) into an epoch check inside
     `native_v2_resolve_handle_to_object_base_rax` (~2919-2946) under a debug gate; force a pop while a handle is
     live and confirm the **trap fires** (the mechanical form of the P0-F "positive zero-detector" audit bar).
10. **P5 acceptance artifact:** discard-per-iteration witness (loop; alloc >16B struct/iter; use only own fields;
    no call passing it out; no return; no global store) runs past **2²²** iterations unbounded, alongside C1/C2 in
    the same commit.

### Exact files
`check/borrow.sio`, `check/check.sio`, `check/ownership.sio` (read-only, do not drive),
`analysis/escape.sio`, `ir/memory_analysis.sio` (deferred), `ir/lower.sio`, `native/regalloc.sio` (read-only),
`native/stack_maps.sio`, `native/frame.sio`, `native/codegen.sio` (the wired precedent),
`native/codegen_x86_linux.sio` (the LIVE backend), `native/gc.sio`, `native/runtime_context.sio`,
`scripts/ci/precise_stack_maps_honesty_gate.sh`, `tests/compile-fail/ownership_use_after_move.sio`,
`tests/compile-fail/ownership_move_while_borrowed.sio`.

### Single most important next action
**Get operator sign-off on Route A vs Route B (they carry different Lean N2-capstone obligation shapes and
different generalisation costs), then re-verify the ~70-line coordinate offsets on the target tree — because the
design's coordinates and its "stub" framing were both wrong, and the executor must not re-inherit either.**

---

## WS-C — Non-associative uncertainty as a NonAssoc handler  (verdict: FLAWED → rescued by 3 corrections)

**Design intent (verified sound):** order dependence rides the **carrier**, not the vocabulary. New module
`stdlib/epistemic/affine_octonion.sio` with carrier `AffOct` (field-wise struct of `PnOct`) whose `aff_mul`
preserves octonion order. `mul(mul(x,y),z)` vs `mul(x,mul(y,z))` → **different** variance under NonAssoc,
**identical** under GUM (`ep_mul@knowledge.sio:112` is order-blind — CONFIRMED). No `mul3` op added to the WS-A
interface. Build entirely on `product_nonassoc.sio`'s field-wise return-by-value `PnOct`/`pn_oct_mul` (execution-
verified under **default Madaros**: Fano 0.25 / non-Fano 4.25) — **NOT** the `&![f64;8]` out-param
`algebra::octonion::oct_mul` idiom, which **segfaults under default Madaros** (both affine demos pin their
executable claim to lean_single).

### Corrections applied (from the FLAWED verdict — all three rescue the design)

1. **Missing `pn_add` (fatal-as-written).** `aff_mul`'s `d_k` uses `pn_add(...)` but `product_nonassoc.sio`
   exports **no `pn_add`** (only `pn_oct_mul`/`pn_oct_norm_sq`/`pn_associator`/`pn_oct_basis`/`associator_norm_sq`/
   `product_nonassoc_augment`). **Fix:** define a local field-wise `pn_add(a:&PnOct,b:&PnOct)->PnOct` (and any
   `pn_scale` `aff_scale` needs) inside the new module; add them to the import/reuse lists.
2. **Wrong line anchors (functions exist; cites wrong).** `perform` parse = `parse_perform_expr`
   **`exprs.sio:640-644`** (not 597); match `FatArrow` expect **`exprs.sio:906`** (match @877, not 860);
   `check_handle_expr` **`check.sio:23050`** (not 24078; eff via `effect_name_to_id` @~23063 — its "discharges
   `Epistemic=8`" behavioural claim is CORRECT, only the coordinate was wrong).
3. **Under-specified headline math (must be derivable, not retrofitted — CLAUDE.md §6).** The load-bearing
   assertion "spread of the ε0-coefficient between `((x*y)*z)` and `(x*(y*z))` == `associator_norm_sq(x,y,z)`
   machine-exactly" holds **only for a specific `aff_leaf` coeff choice** the design never pins. **Fix:** pin the
   `aff_leaf` coeff convention in the witness (state that x/y/z carry `d0=coeff` chosen so the first-order symbol
   row reproduces the associator) and **show the derivation**, not just the expected number.

### Corrected verified step list

1. Create `stdlib/epistemic/affine_octonion.sio` (library, no `fn main`). Header fixes **NSYM=4** (noise symbols
   ε0..ε3) vs **N=factors** ("order-safe iff N≤3" = factors); states the idiom decision (canonical math cited
   against `algebra::octonion::{oct_mul,oct_associator,oct_norm_sq}` but re-implemented field-wise for
   default-Madaros safety).
2. Import from `product_nonassoc.sio`:
   `{PnOct, pn_oct_mul, pn_oct_norm_sq, pn_associator, associator_norm_sq, pn_oct_basis, product_nonassoc_augment}`.
   **Do NOT import `algebra::octonion`** (named-import E137 risk under Madaros). *(Note: these `pn_*` are declared
   without `pub`, but Sounio visibility is advisory — importing is fine; acknowledge rather than imply they are
   exported.)* **Add local `pn_add` (+ `pn_scale` if needed) per Correction 1.**
3. Carrier: `struct AffOct { v: PnOct, d0: PnOct, d1: PnOct, d2: PnOct, d3: PnOct }` (central + 4 noise-symbol
   coeffs). Field-wise struct-of-PnOct — no `[f64;32]` exclusive-ref arrays.
4. Constructors/accessors (helpers before callers, no unary minus): `aff_zero`, `aff_leaf(v,k,coeff)` **[coeff
   convention pinned per Correction 3]**, `aff_central`, `aff_variance` = Σ `pn_oct_norm_sq(d_k)`.
5. Arithmetic return-by-value: `aff_add`/`aff_sub`/`aff_scale`/`aff_shift`; then
   `aff_mul`: central = `pn_oct_mul(&a.v,&b.v)`; `d_k = pn_add(pn_oct_mul(&a.v,&b.d_k), pn_oct_mul(&a.d_k,&b.v))`.
   **`aff_mul` preserves octonion order — the whole discriminator.**
6. `aff_pentagon_variance(w,x,y,z)` reusing `associator_field.sio:pentagon_variance:154` math on symbol
   coefficients (all 5 parenthesizations via `aff_mul`, 8 comps × 5 vertices, 5-vertex population variance).
   Document: N≤3 → one associator suffices; N=4 → all 5 vertices needed.
7. NonAssoc handler adapters realising the WS-A op interface over `AffOct`: `na_lift`, `na_add=aff_add`,
   `na_sub=aff_sub`, `na_mul=aff_mul` (**no new op**), `na_scale`, `na_shift`, `na_observe`,
   `na_collapse(u,k_sigma)->(val,lo,hi)`. **Pin collapse scalarization** (which component/norm → `val`, `k_sigma`
   value); label output **"order-spread-augmented variance, NOT a certified enclosure"** (that belongs to
   interval/p-box).
8. `na_mul3(a,b,c,kappa)` convenience folding `kappa*associator_norm_sq` into the budget — **document it is NOT
   load-bearing** (order dependence already emerges from `aff_mul`); vocabulary stays
   {lift,add,sub,mul,div,scale,shift,observe,collapse}.
9. Handler-dispatch design note (WS-A not yet built): construct `handle<Epistemic>{body} with {NonAssoc clauses}`
   discharges `Epistemic(8)` at `check.sio:23050`; NonAssoc(14) is the calculus identity. **Open:** DISCHARGE vs
   PROPAGATE NonAssoc — leans propagate-upward; `check_handle_expr` currently discharges the handled effect only.
10. **Tier-(a) witness** `examples/epistemic/affine_octonion_order_witness.sio` — **runnable TODAY under default
    Madaros** via direct `na_*`/`aff_*` calls (no perform/handle). `//@ expect-stdout` asserts FOUR: (1) ε0-coeff
    spread `((x*y)*z)` vs `(x*(y*z))` == `associator_norm_sq(x,y,z)` (<1e-9) **[with the pinned-coeff derivation
    shown]**; (2) Fano e1,e2,e3 → 0.0 (N≤3 null control); (3) non-Fano e1,e2,e4 → 4.0; (4) GUM control:
    `ep_mul` associativity-blind (variances EQUAL) while `na_collapse` of the two parenthesizations differ by
    exactly `kappa*4.0`.
11. **Tier-(b) witness** `examples/epistemic/effect_uncertainty_nonassoc.sio` — same triple-product source under
    `with{GUM}` (identical variance) vs `with{NonAssoc}` (order-sensitive). **Blocked until WS-A P0 ExprHandle arm
    lands.**
12. Verify: `souc check` both files against a from-source build; `souc run` tier-(a); diff vs `expect-stdout`;
    **label every receipt with its engine** (tier-a → default Madaros; N=3/N=4 affine demo receipts → lean_single).

### Exact files
New: `stdlib/epistemic/affine_octonion.sio`, `examples/epistemic/affine_octonion_order_witness.sio` (tier-a),
`examples/epistemic/effect_uncertainty_nonassoc.sio` (tier-b, blocked on P0).
Reuse (read-only): `stdlib/epistemic/product_nonassoc.sio`, `.../knowledge.sio`, `.../uncertain_octonion.sio`,
`stdlib/algebra/octonion.sio` (cited only), `.../associator_field.sio`, `examples/epistemic/affine_nonassoc_demo.sio`,
`.../affine_nonassoc_n4_demo.sio`, `self-hosted/ir/lower.sio`, `.../check/check.sio`, `.../check/effects.sio`.

### Single most important next action
**Write `affine_octonion.sio` with the local `pn_add` and the pinned `aff_leaf` coeff convention, then land
tier-(a) — because it is the ONLY workstream deliverable runnable end-to-end on default Madaros today, and it
proves the expressive-superiority claim (order-blind GUM vs order-sensitive NonAssoc) with zero dependency on the
un-landed P0 ExprHandle arm.**

---

## WS-D — Lean proof obligation  (verdict: FLAWED — STUB, nothing to integrate)

**Status: NO DESIGN EXISTS.** The submitted JSON is a placeholder (`summary:"test"`, `steps:["a","b"]`,
`files:["x","y"]`, `top_risk:"risk"`). Files `x`/`y` do not exist. **Nothing to certify, nothing to integrate.**
This workstream is **UNSTARTED** and must be authored from scratch. Requirements the real design must meet
(from the verifier + CEI ground truth):

1. Name the **actual proof obligation**: which handler soundly realises which effect (the "handler-as-proof-
   carrying-interpreter" mapping), with real file paths — `self-hosted/effects/handlers.sio`,
   `self-hosted/analysis/escape.sio`, `self-hosted/ir/memory_analysis.sio`, and the specific `formal/*.lean`
   obligation files.
2. **Target the UNCONDITIONAL certified result — the p-box / interval enclosure — NOT the conditional
   (curvature-dependent) GUM soundness.** Over-claiming global GUM soundness is a fatal over-claim.
3. **Acknowledge the corpus is not globally sorry-free (~64 real sorries)** and scope the machine-checked claim
   to the specific lemma being added, not the corpus.
4. Anchor verification on a **from-source `souc`** build (`bin/souc` is prebuilt).

### Single most important next action
**Author a real WS-D design (or dispatch it as a fresh workstream). It is the thesis capstone —
"the compiler machine-checks that a handler soundly realises an effect" — and it currently does not exist.
Until it does, the CEI PL-theory contribution rests on WS-A/B/C mechanism without the Lean certificate that is
the paper's headline. This is the largest open gap in the program.**

---

## Cross-workstream SEQUENCING (what unblocks what)

```
[0] BRANCH-LOCK RESOLUTION (§0)  ──unblocks──▶  ALL of WS-A (every coordinate is branch-conditional)
                                                 └─ if base = main/current tree: WS-A must be RE-DERIVED
                                                    (no GPU precedent to mirror — confirmed 0 hits here)

WS-C tier-(a)  ── independent, runnable on default Madaros TODAY ──▶  ships FIRST, unblocked by nothing
                 (proves expressive-superiority without P0)

WS-A P0 (ExprHandle lowering + EDIT 1/EDIT 2)
     │  gate: EDIT 2 (check_method_call bypass) MUST land & souc-check before lowering is trusted
     │  gate: step-13 scalar-kind classifier MUST be verified before the int smoke can print
     ├──unblocks──▶ WS-A P1 (GUM-vs-MC same-source demonstrator)
     └──unblocks──▶ WS-C tier-(b) (perform/handle NonAssoc-vs-GUM headline demo)

WS-B Route decision (A vs B)  ── independent of WS-A/C ──▶  gates all WS-B implementation
     │  Step-0 (E091 collision) blocks any WS-B falsifier test
     │  Steps 3→4→5 are a strict chain (certifier fact → MIR type tag → live-root bitmap)
     └──item-7 (memory_analysis / callee summaries): DEFERRED, but becomes a
        CORRECTNESS PRECONDITION the moment any certified region contains a call

WS-D (Lean)  ── UNSTARTED ──▶  consumes the soundness facts WS-B Step-3 produces AND the
                                handler-realises-effect mapping WS-A defines; cannot be
                                authored meaningfully until at least one of WS-A/WS-B has a
                                concrete handler+certificate to prove about.
```

**Critical-path summary:**
- **Ship-now, no dependencies:** WS-C tier-(a). Do this first — it is the only default-Madaros end-to-end proof.
- **Unblocks the most:** §0 branch-lock → WS-A. WS-A then unblocks P1 **and** WS-C tier-(b).
- **Parallelisable:** WS-B is independent of WS-A/C once its Route is chosen; can proceed concurrently.
- **Terminal dependency:** WS-D (Lean) needs a concrete handler+certificate from WS-A/WS-B before it can be
  authored; it is currently a stub and is the program's largest open gap.

---

## Consolidated RISK + FALSIFIABILITY register

| # | Workstream | Risk | Severity | Falsifier / how we'd know | Status |
|---|---|---|---|---|---|
| R1 | §0 / WS-A | GPU precedent absent on the actual integration base (0 grep hits on current + canonical trees) — the whole "mirror `expr_is_gpu_sync_call_ref`" strategy has no precedent to copy | **CRITICAL** | `grep -c expr_is_gpu_sync_call_ref lower.sio` on the target tree = 0 | **CONFIRMED true on `/workspace/.wt/fable-1` and `/workspace/sounio`; design was written against `/workspace/.wt/lower-array` only** |
| R2 | WS-A | EDIT 2 missing → `Epistemic.add(x,y)` dies as E137 in `check` before lowering runs | High | Build from source; `souc check` the smoke file without EDIT 2 → undeclared-identifier error | Design correctly identifies it; **must build-verify** |
| R3 | WS-A | Inplace method-call checker (`checker_check_method_call_inplace`) re-entered inside a handle body → EDIT 2 bypassed → E137 | Med | Trace any handle-body path that re-enters inplace; if found, a 3rd bypass is needed | **UNCERTAIN — must confirm no re-entry** |
| R4 | WS-A | `expr_result_scalar_kind_ref` misclassifies inlined clause-body result → `println(r)` SIGSEGV via char*-printer; `expect-stdout:5` NOT demonstrable | High | `souc run` the int smoke; a SIGSEGV / garbage print falsifies | **UNCERTAIN — classifier not yet read; gates P0 done** |
| R5 | WS-A | Nested-aggregate `HandlerFrame` struct miscompiles (known codebase failure family) | Med | Design avoids it (raw `[Name;4]` arrays); regression if a struct-of-clauses is introduced | Mitigated by design |
| R6 | WS-A | `Epistemic` is both effect-id and stdlib struct name → a missed bypass case mints a body-less mangled fn → codegen SIGSEGV | Med | Any perform site that falls through partially; hard-refuse on no-clause-match prevents it | Mitigated by hard-refuse |
| R7 | WS-B | Handle/heap state is **process-global** but the only obtainable certificate is per-scope; no rbp-chain walk exists → Route A unsound with any live caller-held handle; Route B unsound once a callee stashes a handle from the popped scope | **CRITICAL** | C1 conservatism control + a hand-built caller-held-handle case; use-after-reclaim under C2 trap | **CONFIRMED architectural (no stack-walk in `native/`); callee summaries become a precondition once calls appear** |
| R8 | WS-B | ≤16B unboxed values consume heap-cursor but no handle slot → certificate over handle roots is blind → silent use-after-free on `heap_cursor` reset | High | C2 epoch trap forced while an unboxed value is live | Design closes it via Step-4 raw-pointer flagging; **must implement** |
| R9 | WS-B | `nonzero ≠ dead` — a root bitmap/probe alone never fires the reset on the discard witness | High | Run the witness with only item-4 or only item-5 wired → still `exit(182)` at 2²² | **CONFIRMED; fix = null-slot-at-drop-point (Step 5)** |
| R10 | WS-B | MIR per-vreg type tagging (Step 4) perturbs the self-host fixed point (gen2==gen3) | Med | `make build` fixed-point check after the plumbing lands | **UNCERTAIN — not build-verified** |
| R11 | WS-B | E091 code collision → falsifier can't tell which diagnostic fired | Low (blocking) | Two E091 emitters at `check.sio:4597` + `:11559` | **CONFIRMED (coordinates corrected)** |
| R12 | WS-C | `&![f64;8]` out-param octonion idiom segfaults on default Madaros → witness downgrades to lean_single-only | High | `souc run` tier-(a) on default Madaros | Mitigated: build field-wise on `PnOct` (verified Fano 0.25/non-Fano 4.25) |
| R13 | WS-C | `pn_add` does not exist → `aff_mul` doesn't compile | High (was fatal) | `souc check affine_octonion.sio` | **Fix identified: add local `pn_add`** |
| R14 | WS-C | Headline identity (ε0-spread == associator_norm_sq) holds only for a specific unpinned `aff_leaf` coeff → risk of retrofitting a tolerance (CLAUDE.md §6 violation) | Med | `expect-stdout` <1e-9 with the derivation shown, not just the number | **Fix: pin coeff convention + show derivation** |
| R15 | WS-C | DISCHARGE vs PROPAGATE NonAssoc(14) undecided | Low | check_handle_expr currently discharges handled effect only | **UNCERTAIN — design leans propagate; confirm** |
| R16 | WS-D | No design exists (stub); the Lean capstone — the paper's headline machine-checked claim — is unstarted | **CRITICAL** | The submitted JSON has no real content | **CONFIRMED stub; must author** |
| R17 | WS-D | Over-claiming global GUM soundness instead of the unconditional p-box/interval enclosure | High (if authored naively) | Any Lean statement asserting unconditional GUM correctness | Guardrail stated; not yet at risk (nothing written) |
| R18 | all | `bin/souc` is prebuilt — behaviour claims from the wrapper are invalid | Med | Build from source (`build_modular_madaros.sh`) before any receipt | Standing discipline |

**Falsifiability posture:** WS-A and WS-C are falsifiable **by execution** (`souc run` + `expect-stdout`), WS-B
by **two controls** (C1 conservatism = must-refuse; C2 = positive use-after-reclaim epoch trap — a single control
proves nothing), WS-D by **Lean check** once authored. The strongest immediately-falsifiable deliverable is
**WS-C tier-(a)** (runs on default Madaros today). The weakest-supported claim is **WS-A `expect-stdout:5`**
(gated on the unverified scalar-kind classifier, R4) and **WS-D** (does not exist, R16).

## UNCERTAIN — open items requiring resolution before implementation
1. **Integration base** (R1): `lower-array` (precedent exists) vs `main`/current (must re-derive). **Blocks all WS-A.**
2. **WS-A R3:** inplace-checker re-entry inside handle bodies — confirm or add a 3rd bypass.
3. **WS-A R4:** `expr_result_scalar_kind_ref` classification of inlined results — read `lower.sio:~12690-12820`.
4. **WS-A diagnostic code:** next free E-code — grep before adding.
5. **WS-B Route A vs B:** operator sign-off (different Lean N2 obligation shapes).
6. **WS-B R10:** fixed-point stability of MIR type-tag plumbing — build-verify.
7. **WS-C R15:** discharge vs propagate NonAssoc(14).
8. **WS-C R14:** exact `aff_leaf` coeff + collapse scalarization (component index, k_sigma).
9. **WS-D:** entire design — author from scratch, targeting the unconditional p-box/interval enclosure.
```

---

Notes on the deliverable above (not part of the appendable Markdown):

- **The one decisive integration finding** is R1 / §0: I empirically confirmed on the live checkout that the GPU precedent P0 depends on has **zero hits** here (`lower.sio` = 19461 lines vs the design's 17819; `struct Lowerer` @593 vs @723). The P0 design's own verdict flagged this as its top correction; it is now confirmed, not just asserted.
- WS-D contributed nothing (stub); I integrated it as an explicit UNSTARTED gap rather than inventing content.
- WS-B's "stub probe" framing was fatally wrong per its verifier; the spec propagates the inversion (item-5 is sound-but-ineffective, not a fabricated pass) and carries the ~70-line coordinate corrections.
- WS-C's three fixable flaws (missing `pn_add`, wrong anchors, unpinned coeff) are all folded in as corrections that rescue the design.

---

## P0 — turnkey implementation state (Option A, re-anchored on lane/fable-1/handle-table-182-dispatch)

Base validated + set up this session (branch `lane/fable-1/cei-p0-handler-lowering` @ its HEAD; P0 files
claimed on lane `fable-1-cei-p0-handler`). GPU precedent confirmed present AS NAMED. All coordinates below are
re-anchored to `lower.sio`=17819 / `check.sio` on this tree — trust names, re-grep before editing.

**Resolved design decision (settled the two workflow runs' disagreement):** `loop_labels` is
`Box<LowerLoopLabels>` with only `loop_depth: i64` inline (lower.sio:723 struct). => **BOX the handler stack.**
Add `struct LowerHandlerStack { effects: [Name;4], blocks: [Option<Box<Block>>;4] }`, and to `struct Lowerer`
add `handler_stack: Box<LowerHandlerStack>` + `handler_depth: i64` (do NOT add inline `[Name;4]` — the Lowerer
is by-value-copied and multi-MiB per the @913 comment). Mirror on `struct Checker`.

**Exact edit sites (re-anchored):**
- `lower.sio` struct Lowerer @723; **~5 `Lowerer{}` init literals** to extend: @991 (`lowerer_new`), @1455,
  @1519, @1587, @2015 — each gets `handler_stack: Box::new(LowerHandlerStack{ effects:[empty_name();4],
  blocks:[None;4] }), handler_depth: 0`.
- `lower.sio` **ExprHandle dispatch arm**: insert `else if e.kind == ExprKind::ExprHandle {
  self.lower_handle_expr_ref(e) }` in the chain at ~16496 (beside `ExprBlock`→`lower_block_expr_ref`).
  New `lower_handle_expr_ref`: push effect(`e.name`)+block(`e.handler_block`) at `handler_depth` (cap 4,
  `report_error_at` beyond), lower `e.block` via `lower_block_expr_ref` shape, pop on every exit.
- `lower.sio` **perform hook** in `lower_method_call_expr_ref` @15677: add `expr_is_active_handler_perform_ref`
  (mirror `expr_is_gpu_sync_call_ref` @15664 shape: ExprMethodCall + `left` ExprIdent==active effect; walk
  `handler_depth-1..0`) FIRST, before `.len()`. On match: find `StmtLet{name==op, expr:ExprClosure}` in the
  block; scope-push; per-arg `lower_expr_ref`+`bind_local(param,reg)` then **re-bind scalar-kind from
  `ClosureParam.ty`** (f64→2,i64→1) [avoids the println kind-0 char* SIGSEGV]; lower clause body; restore scope;
  result reg = body reg. No clause/arity: `lowerer_mark_error`+`IR_INVALID_REG` — never fall through.
- `check.sio` **check_method_call bypass** @20438 (the load-bearing edit): add an arm parallel to the existing
  `checker_expr_is_gpu_sync_call(e)` bypass @20439 — `checker_expr_is_active_handler_perform` matches the bare-
  Ident-receiver shape against the live handler-effect stack BEFORE `check_opt_expr(e.left)` (which otherwise
  fails Epistemic as undeclared-identifier). Type args vs clause `ClosureParam` types; return the clause body's
  type. Needs Checker handler-stack fields + push/pop in `check_handle_expr` @24078 (push effect+block before
  checking body, pop after; keep the existing standalone `check_block(handler_block)`).

**Smoke test written:** `examples/effect_uncertainty_smoke.sio` (int-only, `//@ expect-stdout: SMOKE 5`).
**Next increment:** make the struct + 4 code-site edits, build from source (needs an unthrottled slot), run the
smoke test, then the P1 GUM-vs-MC demonstrator. Honesty gate unchanged (GUM conditional; p-box/interval the
unconditional certified result).

---

## P0 design REVISION — module globals instead of struct fields (de-risked, 2026-08-18)

Investigation result that supersedes the "box the handler stack on struct Lowerer/Checker" step: **Sounio
module-level `var` globals CAN hold boxed AST** — proven by `var LAST_EXPR: Option<Box<Expr>> = None`
(`parser/parser.sio:26`). Therefore the handler stack is a set of **module globals**, not struct fields:

- In `self-hosted/ir/lower.sio` (top-level, near the other `var IR_LOWER_*` globals):
  `var LOWER_HANDLER_DEPTH: i64 = 0`, `var LOWER_HANDLER_EFFECT: [Name; 4] = [empty_name(); 4]`, and the handler
  blocks as boxed globals mirroring LAST_EXPR — a `var LOWER_HANDLER_BLOCKS: [Option<Box<Block>>; 4]` if the
  fixed-array-of-Option-Box lowers cleanly, else four scalar `LOWER_HANDLER_BLOCK_0..3: Option<Box<Block>>`.
- In `self-hosted/check/check.sio`: the same as `CHECK_HANDLER_*` (separate pass, separate globals).

**Why this is strictly better:** it removes the by-value struct-field blast radius (struct Lowerer @723 + ~5
`Lowerer{}` init literals @991/1455/1519/1587/2015; struct Checker + its inits) — the exact nested-aggregate-by-
value shape with a documented miscompile history here. Push/pop of a global stack is precisely the discipline
this needs, and it is the pattern already used for P0-F extern-block parsing.

**Revised P0 edit set (small):**
1. lower.sio: declare the 3 globals; `lower_handle_expr_ref` = push `e.name`+`e.handler_block` at
   LOWER_HANDLER_DEPTH (cap 4), lower `e.block` (via lower_block_expr_ref shape), pop; `else if ExprHandle`
   dispatch arm @~16496; perform hook `expr_is_active_handler_perform_ref` in lower_method_call_expr_ref @15677
   (mirror expr_is_gpu_sync_call_ref shape; on match find `StmtLet{name==op, ExprClosure}` in the active block,
   inline tail-resumptively with the scalar-kind re-bind).
2. check.sio: declare CHECK_HANDLER_* globals; push/pop in check_handle_expr @24078; the load-bearing
   check_method_call bypass @20438 (mirror the checker_expr_is_gpu_sync_call arm @20439) matching bare-Ident
   receiver against CHECK_HANDLER_EFFECT before check_opt_expr(e.left).
3. Verify loop: `souc check self-hosted/compiler/main.sio` after each file (cheap, no 4-min build); final
   from-source build + run `examples/effect_uncertainty_smoke.sio` (expect `SMOKE 5`).

---

## P0 CORRECTION — the clause-storage step in the spec is UNSOUND; use op→fn_id dispatch (2026-08-18)

Implementation attempt uncovered a design hole BOTH workflow runs missed: the spec (and my module-global
revision) says "store `e.handler_block` (Option<Box<Block>>) per depth and re-walk it at each perform site."
**This does not typecheck.** In `lower_expr_ref(self, e: &Expr)` the handler block is **borrowed from the AST**;
it cannot be moved into a module global (globals need ownership, as `parser_store_expr_box` takes an owned
`Box<Expr>`) nor stored as a borrow in the by-value `Lowerer`, and Sounio exposes no address-of-reference→i64
to stash a raw pointer (TypeRawPtr exists as a type but there is no clean `&Block`→i64→`&Block` round trip in
lower.sio). No AST-clone/rewrite infra exists for blocks either.

**Correct approach (owned IR, copyable dispatch map):**
1. In `lower_handle_expr_ref`, BEFORE lowering the body, iterate the handler block's `StmtList`; for each clause
   `Stmt{StmtLet, name:op, expr:Some(ExprClosure{closure_params, closure_body})}`, **lower the clause closure to
   an owned IR function** (reuse the closure-lowering path — verify non-capturing closures reduce to a plain
   callable fn_id) and record `(op_name: Name, fn_id: i64)` in a **module-global scalar map**:
   `var LOWER_HANDLER_OP_NAMES: [Name;16]`, `var LOWER_HANDLER_OP_FNIDS: [i64;16]`,
   `var LOWER_HANDLER_OP_COUNT: i64`, with a `[i64;4] LOWER_HANDLER_SCOPE_MARK` for push/pop by depth. Names +
   i64 are copyable — no borrow, global-safe. (Verify `[Name;16]` as a *global* lowers; it works as a Lowerer
   FIELD `fo_bind_names:[Name;128]`, so likely fine.)
2. Lower the body. The perform hook (`expr_is_active_handler_perform_ref`, still mirroring the
   `expr_is_gpu_sync_call_ref` bare-Ident-receiver shape while `LOWER_HANDLER_OP_COUNT>0`) resolves `op` to its
   `fn_id` and emits an **ordinary `ir_call(dst, fn_id, op_name, args)`** — no clause-body re-walk, no borrow.
   The scalar-kind re-bind concern disappears (args are real call arguments through the normal call path).
3. Pop the scope mark on every exit.
4. Checker side: the load-bearing `check_method_call` bypass @20438 still stands (type args against the clause
   closure's `ClosureParam` types, return the clause body type), and `check_handle_expr` @24078 registers clause
   op-names before checking the body — but it stores op NAMES only (copyable), never the borrowed block.

**Open verification (first thing in the code pass):** confirm a non-capturing `ExprClosure` lowers to a directly
callable `fn_id` via the existing closure path, and that a global `[Name;16]` lowers. This is the crux the two
workflow runs did not reach; it is genuine compiler design, not mechanical wiring.
