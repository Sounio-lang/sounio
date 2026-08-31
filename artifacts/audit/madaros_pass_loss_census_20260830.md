# Madaros pass loss census — 2026-08-30

**Lineage:** next executable bridge of Garden seed
[`docs/internal/garden/seeds/2026-08-29-pharos-loss-ledger.md`](../docs/internal/garden/seeds/2026-08-29-pharos-loss-ledger.md)
(Pharos — a compiler that publishes its own erasures).

**Evidence label:** `implemented` as a document; pass classifications are
**static readings of the source**, not executions. Nothing here was compiled
or run (the fleet build lock was deadlocked at census time; this census needed
no build by design). Reading surface: `/workspace/sounio` @
`lane/cursor-1/20260826` (`d031627a4fa6`). Companion table:
`madaros_pass_loss_census_20260830.tsv` (same directory).

**Scope rule:** class (a) means "no loss observed by static reading", not
"proven loss-free". Class (c) means "loss uncharacterized", not "bug".

## Method

Enumerate every optimization/lowering stage reachable from the Madaros
pipeline call sites, read each one's source, and classify what semantic
information it may discard:

- **(a) loss-free by construction** — pure/structural transforms, or transforms
  that refuse inexact cases;
- **(b) loss with known bound** — documented truncation/quantization/trade-off;
- **(c) loss uncharacterized** — the transform can discard or alter semantic
  information (order, precision, provenance, epistemic metadata, observability)
  with no record, no bound, and no audit.

## Headline numbers

48 stages enumerated: **19 (a), 8 (b), 20 (c)**, 1 reject-only gate, 1 inert
pass. Of the 20 class-(c) stages, **13 sit on production paths**.

## Findings

1. **There is no pass manager.** Pipelines are hardcoded call sequences at four
   sites: `self-hosted/compiler/main.sio:5326-5397`,
   `self-hosted/compiler/module_loader.sio:2923-2937`,
   `self-hosted/compiler/module_frontend.sio:6148,6206`, and the by-value spine
   `self-hosted/ir/opt_cleanup.sio:8944-8974`. A loss ledger would have no
   single interception point today — it would have to be built at these four
   call sequences or the pipeline would need to become data.

2. **Exactly one stage carries a machine-readable loss audit** — the rebracket
   authority in `opt_cleanup.sio:52-274`, which restricts itself to exact
   bitwise AND/OR/XOR reassociation, refuses float arithmetic, requires a flow
   certificate, and emits `OcpExactBitwiseRebracketProbeReceipt`. It is the
   accidental embryo of Pharos. It also **does not run on the production
   multi-module route**: `opt_cleanup_function_mfi` (opt_cleanup.sio:9686-9702)
   is a field-wise subset, and the comment at 9687-9690 states the rebracket
   authority, full e-graph, mul3_sr, LICM and epistemic rewrites stay on the
   by-value self-test path. The most audited machinery guards the route that
   self-tests use, not the route users get.

3. **DCE treats epistemic opcodes as dead-eliminable.**
   `ocp_has_side_effect` (opt_cleanup.sio:1494-1510) omits `IrMeasure`,
   `IrContest`, `IrLiftKnowledge`, `IrDebugGuard` and certificate ops from the
   side-effect set, so `ocp_dce_once` may delete them when their destination
   register is unobserved. The test-only DCE in `ir/optimize.sio:345-359` has
   the same shape. Partial mitigation exists only on the e-graph path
   (`ocp_collect_observed_regs` 1647; `eg observe_barriers`,
   egraph.sio:1340-1345), not in DCE.

4. **Epistemic structure is erased at native codegen.**
   `IrMeasure` lowers to a bare MOV of the value (native/lower_ir.sio:296-298);
   `IrLiftKnowledge`, `IrPlanRecourse`, `IrProposeAlternatives`,
   `IrCommitAlternative` likewise become MOV (285-295); `IrDebugGuard` becomes
   a NOP stub (299-301). Uncertainty and provenance exist in the IR and vanish
   at the last millimetre. Possibly intentional; currently unrecorded anywhere.

5. **Profile counts can flip a function's semantic contract.**
   `sprof_apply_promotion` (ir/profile.sio:132) mutates `compile_strategy`
   from runtime profile counts, and can promote a function into
   `AGGRESSIVE` — the strategy under which float reassociation is allowed
   (ir.sio:692). A semantic gate is thus controllable by measurement noise
   from a previous run.

6. **Latent constant collision:** `IR_STRATEGY_PRECISION_PRESERVING_CHAOTIC`
   and `IR_STRATEGY_EXTERN` are both `6` (ir.sio:697-698). An `extern "C"`
   function is indistinguishable from a chaotic-integration function to the
   e-graph epistemic gate (opt_cleanup.sio:8153-8154). Static reading only —
   not runtime-verified.

7. **Float folding happens on the host in more than one place.**
   SCCP folds float binops in host f64 at compile time
   (const_prop.sio:671-688, refusing only division by 0.0). The GPU optimizer
   folds f32 on the host (gpu/opt/optimizer.sio:400-408) where `div_ff` panics
   while the device would produce `inf` — a host/device semantics divergence
   frozen into a fold.

8. **The dormant pass is the most dangerous one.**
   `avec_is_associative` (auto_vectorize.sio:963-970) treats Add/Mul as
   associative with no int/float distinction — a float reduction reordering
   hazard — but the vectorizer has no production caller today. If it is ever
   wired in, it enters as class (c).

9. **The LLVM route is a delegated black box.** `llvm/passes.sio:24-83` hands
   `O0`-`Oz` pipeline strings to `LLVMRunPasses`; from Sounio's side the whole
   LLVM middle-end is one uncharacterized loss surface with no fast-math audit
   visible at the call site.

10. **What is already done right:** the e-graph float-reassociation machinery
    is fail-closed by default (`allow_inexact_reassoc` defaults false,
    "FORBIDDEN" for IEEE-754, egraph.sio:1324-1331), gated by `chaotic` and by
    the algebra axis `reassoc_strategy` with the 168-theorem Fano predicate
    (`ir_can_reassociate_triple`, egraph.sio:1786). The epistemic e-graph pass
    documents its own trade ("deliberately trades bits for a lower-uncertainty
    evaluation order", egraph.sio:1328-1330) — a class-(b) with the bound
    written down. The danger is the opt-in constructor `eg_small_init_algebra`
    (egraph.sio:1369-1395), which sets `allow_inexact_reassoc: true`.

## What this census is not

- Not a claim that any class-(c) stage has ever produced a wrong result.
- Not runtime evidence; nothing was executed (see evidence label).
- Not a complete stage list of every experimental path (GPU-IR analyses and
  test-only pipelines are included only where cited).
- Not a promotion of the Pharos seed past `Garden`; this measures the silence
  surface, it does not instrument it.

## Next bridges (one at a time)

1. Promote finding 3 or 4 into a witness: a `tests/` program whose epistemic
   opcode is observable, compiled with `-O`, asserting the opcode survives.
   That converts a static (c) into a measured verdict — and if the opcode is
   eliminated, into a concrete bug with a reproducer.
2. Resolve finding 6 (one-line constant fix or a comment proving harmlessness)
   — smallest differentiating change in the whole census.
3. Only then: decide whether a Pharos-style ledger hooks the four pipeline
   call sites or replaces them with a data-driven pass list.
