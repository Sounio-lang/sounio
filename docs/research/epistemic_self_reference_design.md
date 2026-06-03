# Design: genuine epistemic self-reference in the bootstrap compiler

**Status:** design spec (2026-06-03). Goal chosen by the owner: make §6's
self-application headline **true** rather than retract it — give the
bootstrapping compiler `lean_single.sio` a *genuinely uncertain* decision,
annotate it with `Knowledge<T>`, so the self-compile certifies real uncertainty
and the `gen2==gen3` fixed-point becomes epistemically meaningful.

Context: the audit (`docs/audit/EPISTEMIC_SELF_APPLICATION_AUDIT_2026-06-03.md`)
showed `lean_single.sio` is deterministic — `EXPR_CONF = 1000` everywhere, so
"100% certain / 0 guarded" is vacuous. The genuine uncertainty in the codebase
lives in the *modular* compiler's optimizer (`ir/egraph.sio` etc.), which does
not bootstrap. To make *self-reference* real, the uncertainty must live **in the
bundle that bootstraps**.

## The honesty constraint (non-negotiable)

The added uncertainty must be **genuine**, not a number sprinkled on a
deterministic operation. Lexing, parsing, type-checking, and codegen are exact —
annotating them would fabricate uncertainty (the same overclaim the audit
caught). A decision is legitimately epistemic only if the compiler **lacks the
information to be certain** and must *estimate*.

## The genuine uncertainty: static branch-likelihood

A self-hosted compiler with **no profile data** that wants to lay out basic
blocks (place the likely successor as fall-through) must *guess* which branch is
hotter. Standard static heuristics (Ball–Larus; "loops are usually taken",
"branches to `panic`/error paths are usually not taken") are genuinely
uncertain — they are *estimates*, wrong a measurable fraction of the time. This
is real epistemic uncertainty the compiler actually has.

Elegantly, the compiler that *implements* `Knowledge<T>` can **dogfood** it for
its own heuristic — and `lean_single.sio` already type-checks `Knowledge<T>`,
`measure`, and the `with Epistemic` effect, so it can compile such code in its
own source.

## ⚠️ Reachability finding (2026-06-03): the consumer does not exist

A grep of `lean_single.sio` for any uncertain-decision site — block layout,
fall-through/branch ordering, jump threading, register spill, inlining heuristic
— comes back **empty**. The compiler is a pure deterministic single-pass emitter
(emits blocks in source order; `local_bss_spill_bytes` is a fixed `524288`; the
"inline" hits are C-string emission and Option-representation, not optimization).

**Consequence (scope doubles):** there is no existing decision to annotate. To do
genuine self-reference one must **first add a real optimization pass** (a basic-
block layout pass that chooses fall-through using the static branch-likelihood
heuristic) to the bootstrap compiler, *then* annotate its estimate. This is
defensible — block layout is a real, useful optimization that genuinely has no
profile data, so the heuristic is honestly uncertain (not a decision invented
solely to be uncertain) — but it is a **substantial codegen feature on the
bootstrap path**, not a small annotation. The plan below therefore has a Step 0.

## Concrete plan (minimal, reversible)

**Step 0 (the newly-revealed prerequisite): add a layout decision to consume the
estimate.** A block-ordering pass over the emitter's basic blocks that, for each
conditional, places the estimated-likely successor as fall-through. Layout-only
(see neutrality below): it changes block *order*, never program meaning.

1. **A genuinely-uncertain estimator in `lean_single.sio`'s own source:**
   ```
   fn branch_likelihood(is_loop_backedge: bool, leads_to_panic: bool)
       -> Knowledge<i64> with Epistemic {
     // static heuristic, honestly uncertain — no profile data:
     //   loop back-edge  → ~90% taken  (conf 900)
     //   path to panic   → ~10% taken  (conf 100, i.e. likely-not)
     //   otherwise       → 50/50       (conf 500)
     if is_loop_backedge { measure(90, uncertainty: 10) }
     else if leads_to_panic { measure(10, uncertainty: 10) }
     else { measure(50, uncertainty: 25) }
   }
   ```
   These confidences are **below `GATE_THRESHOLD = 950` by construction** — the
   estimate is honestly a guess.

2. **A consumer that acts on it** (block-layout fall-through choice). Reading the
   point estimate requires `.value` → forces `with Epistemic` on the consuming
   codegen function and routes the confidence through the epistemic pass.

3. **What the self-compile then shows (non-vacuous):**
   - the epistemic pass over `lean_single.sio`'s *own* HIR now computes
     `EXPR_CONF < 950` at the branch-layout sites;
   - those sites emit the real `66 90` marker → `gates[direct=N guarded=M>0]`
     when the compiler compiles itself — the first non-zero guarded count on the
     bootstrap;
   - the fixed-point `gen2==gen3` now witnesses **epistemic stability**: the
     compiler's confidence in its own heuristic is identical across generations
     (the original §6.5 claim, now with actual content).

## Verification (what "done" means)

- `bin/souc` (committed binary `6374e52f`) compiles the edited `lean_single.sio`
  → new binary; `gates[…guarded=M]` with **M > 0** (non-vacuous self-uncertainty).
- Re-bootstrap through the **global build lock** (`scripts/dev/souc-build-lock.sh`,
  per CLAUDE.md §4): `gen_k == gen_{k+1}` bit-identical → fixed-point preserved
  *with* the epistemic annotations live.
- The branch-layout heuristic is *functionally* sound (compiler still correct:
  full self-host gate green) — the Knowledge annotation must not change emitted
  semantics, only add the (zero-cost) marker + confidence tracking.

## Risk + discipline (this touches the bootstrap)

- **Brick risk.** Edits to `lean_single.sio` can break self-compilation. Mitigate:
  keep the patch **small and reversible**; build against the **committed** binary
  (the working-tree `souc` `9d4ef541` miscompiles `cd_mul` — must not be used);
  verify `souc check` before any full self-compile.
- **CPU / pod stability.** Full self-compile MUST go through the build lock
  (§4 — pod evicted under CPU saturation before). One build at a time.
- **Scope honesty.** This is multi-session. The first milestone is *the estimator
  + consumer compile and the self-compile shows guarded>0*; the fixed-point
  re-verification is the second; the paper §6 rewrite (now truthful) is the third.
- **Functional neutrality.** The heuristic must be layout-only (fall-through
  choice) so a wrong guess never changes program meaning — only block order.
  Otherwise we'd trade an honesty problem for a correctness one.

## Advisor caveats to honor during implementation

- **Fixed-point hash will change, and that's fine.** Block reorder ⇒ different
  jump offsets ⇒ different bytes. "Fixed-point preserved" means
  `gen_k == gen_{k+1}` for the *new* code — **re-pin from scratch**; do not expect
  the documented `54327028`.
- **Heuristic must be a pure function of the IR.** If `is_loop_backedge` /
  `leads_to_panic` read anything that varies run-to-run, the layout (and bytes)
  won't be stable and the fixed-point won't close.
- **Success = `guarded > 0` AND re-bootstrap closure — a *small* number is still
  success.** The pass may propagate sub-950 confidence only to the immediate
  consumer (e.g. `guarded=3`), leaving the rest at 1000. That is genuine
  non-vacuous self-uncertainty and is the win. **Do not inflate the annotation
  surface to manufacture a bigger count** — one honest guarded site beats fifty
  contrived ones.

## Why this is the honest headline, not theater

The compiler genuinely cannot know which branch is hotter without a profile;
it estimates; the estimate is wrong sometimes; that uncertainty is now a
first-class `Knowledge<i64>` in the compiler's own source, certified and gated
by the same epistemic pass it applies to user code. "The compiler applies its
epistemic type system to its own source" becomes a mechanical fact with real
content — and the fixed-point proves the self-confidence is stable.
