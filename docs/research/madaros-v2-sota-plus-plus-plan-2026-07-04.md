<!-- docs:meta
topic_id: repo.docs.research.madaros-v2-sota-plus-plus-plan-2026-07-04
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.madaros-v2-sota-plus-plus-plan-2026-07-04
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Madaros v2 SOTA+++ architecture plan

Status: architecture plan and novelty thesis, not a public claim.

Date: 2026-07-04
Branch observed: `gpu/epistemic-tensor-core-next`

## Executive decision

Stop treating the current Madaros repair loop as the architecture. Keep the
current compiler lane alive as the S0 safety oracle, then build Madaros v2 as a
receipt-carrying, multi-stage compiler whose intermediate states are stable,
hashable, and independently validated.

The intended novelty is not "Sounio uses MLIR ideas", "Sounio uses e-graphs",
or "Sounio has a KAN example". Those are ingredients. The novelty candidate is:

> a receipt-carrying epistemic compiler architecture where typed scientific
> uncertainty, equality saturation, and E-KAN symbolic surrogates are first-class
> compiler objects, and every source-to-binary stage can be validated by
> bit-identical receipts, translation-validation witnesses, and exact fallback
> semantics.

This is deliberately more ambitious than the current tactical bug lane. It also
keeps the tactical lane honest: a compiler is not "fixed" until it can preserve
the gates below, not merely pass a narrow local witness.

## SOTA anchors reviewed

These are anchors, not claims of equivalence.

- MLIR shows the mainstream shape for reusable, extensible multi-level compiler
  infrastructure across abstraction levels, hardware targets, and DSLs:
  <https://arxiv.org/abs/2002.11054>.
- TensorIR shows a hardware-specialized tensor abstraction that makes tensor
  primitives first-class and can target specialized accelerators competitively:
  <https://arxiv.org/abs/2207.04296>.
- `egg` shows equality saturation with e-class analyses as a practical optimizer
  and synthesis substrate:
  <https://arxiv.org/abs/2004.03082>.
- Persistent e-graphs in compiler IR are now an active SOTA direction:
  native e-graph representation inside MLIR/xDSL flows, rather than throwing
  equalities away after one optimization phase:
  <https://arxiv.org/abs/2602.16707>.
- KAN 2.0 frames KANs as a bridge between neural fitting and scientific
  symbolic discovery, including feature discovery, modular structure discovery,
  formula discovery, `kanpiler`, and tree conversion:
  <https://arxiv.org/abs/2408.10205>.
- FastKAN/RBF work gives an implementation-friendly route: approximate KAN
  B-spline bases with Gaussian radial basis functions when a faster compiler or
  runtime representation is needed:
  <https://arxiv.org/abs/2405.06721>.
- LLM-assisted translation validation is useful as a guardrail pattern, but not
  as a proof substitute: formal tools first, LLM prediction when formal tools
  cannot confirm, then fuzzing/counterexamples:
  <https://arxiv.org/abs/2401.16797>.

## Repo-local facts this plan must respect

- `docs/MADAROS_STATUS.md` says Madaros is the default `bin/souc` compiler, but
  this branch has a caveat: `gpu/epistemic-tensor-core-next` may contain status
  text before the actual protected compiler-route files and `bin/madaros-relocgate`
  are present. Verify locally before claiming the safe route.
- `scripts/dev/madaros_two_gate.sh` exists in this branch and is the cheap
  two-gate shape: Gate A = imported SMT 6/6, Gate B = dissertation-facing probe.
- `self-hosted/hlir/ir.sio` already contains a high-level SSA-ish IR with
  epistemic types and compile strategies. It is a starting point, not yet a full
  v2 contract.
- `self-hosted/gpu/kernel_ir.sio` is a direct PTX-oriented GPU IR. It is useful
  for target lowering, but explicitly lacks SSA/phi structure and should not
  become the general mid-level optimizer.
- `docs/research/ekan_native_bridge_status_2026-07-04.md` proves a real but
  partial E-KAN native frontier. Current E-KAN is a witness/runtime surface; v2
  needs to lift it into typed compiler IR with receipts.
- `docs/MADAROS_STATUS.md` records the GPU/PTX module-combination TODO:
  `docs/audit/MADAROS_GPU_KERNEL_IR_LOWER_TO_PTX_PTX_MODULE_COMBINATION_2026-07-02.md`,
  commit `8ea996fe3`, with the reproducible empty-main import failure
  (`335 E175 + 18 E177 + 5 E046`). v2 must gate this combination explicitly.

## Seven-stage gradual compiler

Each stage has one job, one canonical artifact, and one acceptance gate. Stages
may be implemented incrementally, but the public claim only advances when the
stage receipt is stable.

### S0 - Current compiler oracle

Purpose: preserve a known-good compiler route while v2 is built.

Canonical artifact:
- receipt-gated Madaros binary or checked known-green fallback.

Gate:
- `make madaros-full-gate` when heavy validation is available.
- Cheap local guard: `scripts/dev/madaros_two_gate.sh <elf>`.
- Imported SMT remains 6/6.

Rule:
- S0 is a safety oracle, not the architecture. Do not keep adding semantic
  features to S0 unless they unblock v2 receipts or protect the user path.

### S1 - Canonical source, AST, and module graph

Purpose: produce a deterministic, canonical representation of parsed source and
module dependencies before type checking or optimization.

Canonical artifact:
- `MadarosV2S1Receipt` with source hash, include/import graph, canonical AST
  hash, parser version, diagnostic summary, and module path table.

Gate:
- parser surface gate over representative self-hosted modules.
- module-combination gate for GPU/PTX imports, including the
  `lower_to_ptx.sio` + `ptx.sio` combination that `mod.sio` previously missed.
- no IR_MAX_INSTRS escape hatch as a "fix"; if caps fire, the receipt must say
  which phase and which module caused it.

Non-goal:
- no type checking, no E-KAN optimization, no ABI decisions.

### S2 - Typed HIR / THIR with effects and epistemic declarations

Purpose: make types, effects, imports, public/private visibility, and epistemic
metadata explicit before lowering.

Canonical artifact:
- typed HIR hash, effect table, public symbol table, refinement table, epistemic
  declaration table.

Gate:
- deterministic HIR roundtrip hash.
- public/import audit for every `self-hosted/gpu/` and compiler module.
- type/effect diagnostics must be structured, not crash or bulk E175/E177/E046
  spew.

E-KAN status:
- declaration surface only: basis family, domains, uncertainty model,
  approximation intent. No optimizer rewrite yet.

### S3 - HLIR SSA and ownership/effect normalization

Purpose: normalize control/data flow into a durable SSA-like compiler IR.

Canonical artifact:
- canonical HLIR module hash with normalized IDs, blocks, values, effects,
  ownership marks, and type layout placeholders.

Gate:
- HLIR serialization roundtrip.
- source -> HIR -> HLIR deterministic hash.
- no duplicate enum definitions or accidental semantic aliases in core type
  systems. `self-hosted/hlir/ir.sio` currently has duplicated
  `HlirTypeContest` / `HlirTypeRobust` enum entries that should be audited
  before this stage is considered contractual.

Rule:
- HLIR owns high-level optimization legality. GPU KernelIR does not.

### S4 - Eq/E-KAN optimization IR

Purpose: introduce non-destructive optimization and scientific symbolic
surrogate reasoning without losing exact semantics.

Canonical artifact:
- persistent e-graph/equality receipt plus E-KAN receipts for any learned or
  approximated law.

Gate:
- equality-saturation extraction must carry proof obligations or translation
  validation for every selected rewrite.
- E-KAN proposals must include domain bounds, basis family, coefficients,
  training/provenance hashes, approximation error bound, GUM/covariance
  assumptions, and exact fallback expression.
- no E-KAN rewrite may replace exact semantics unless interval/SMT/translation
  validation succeeds for the declared domain.

Novelty center:
- compiler-native, receipt-carrying E-KAN/e-graph optimizer for epistemic
  scientific programs.
- E-KAN is not a runtime model bolted on the side. It is a pass family that can:
  identify symbolic features, propose modular scientific structure, suggest
  rewrites/cost models, and emit receipts proving what it did and where it is
  allowed.

Implementation sketch:
- start with KAN-like univariate edge functions.
- support FastKAN/RBF-compatible basis lowering for speed.
- keep exact fallback expressions for all approximations.
- represent uncertainty propagation and GUM metadata in the IR, not in comments.

### S5 - MIR, ABI, and numeric tower

Purpose: commit to low-level program semantics: calls, layouts, returns,
register classes, stack, aggregate passing, and exact numeric widths.

Canonical artifact:
- MIR hash plus ABI receipt.

Gate:
- ABI witness suite: scalar, struct, tuple, array, SRET, imported function,
  f64/f128, i128/u128, i256/u256, vector/tensor registers.
- differential native-v2 vs interpreter/lean_single where applicable.
- f64 print/return/call witnesses must pass before f128 or i256 are promoted.

Rule:
- f128/i256 are not "types in the parser" milestones. They become real only
  when layout, operations, ABI, diagnostics, and fallback semantics are gated.

### S6 - Target IRs and accelerator lowering

Purpose: lower MIR/HLIR into target-specific forms.

Targets:
- CPU native.
- LLVM path when useful.
- WASM.
- GPU KernelIR/PTX.
- TensorIR-like tensor-kernel abstraction for tensor core and matrix kernels.

Gate:
- per-target canonical hash and target receipt.
- CPU/GPU/WASM differential tests for shared semantics.
- PTX module-combination CI gate, including the current `lower_to_ptx.sio`
  missing-import failure shape.
- hardware receipts for Blackwell tensor-core paths when claims mention GB10,
  sm_121, tensor cores, or CUDA/PTX behavior.

Rule:
- `self-hosted/gpu/kernel_ir.sio` remains the target-facing GPU IR. It should
  not absorb S3/S4 responsibilities.

### S7 - Self-hosted fixed point

Purpose: make the compiler prove itself through deterministic self-hosting.

Canonical artifact:
- source tree hash -> S1..S6 receipt chain -> binary hash.

Gate:
- stage N compiler builds stage N+1 compiler.
- independent rebuild yields bit-identical receipts and, where required,
  bit-identical binaries.
- cross-target differential validation.
- translation validation for all optimizer-selected transforms.

Claim threshold:
- only S7 plus external benchmark/proof evidence can support broad "compiler
  novelty" claims.

## E-KAN inside the compiler

Working name: E-KAN = Epistemic Kolmogorov-Arnold Network compiler pass family.

It has four compiler jobs:

1. Scientific feature discovery: identify candidate variables, separable terms,
   conserved quantities, monotonicity surfaces, and modular laws from program
   traces or user-declared domains.
2. Rewrite proposal: propose algebraic or numerical rewrites through an
   equality/e-graph surface, not by mutating IR directly.
3. Cost and schedule modeling: help choose tensor/core/GPU lowering strategies
   under explicit error/performance constraints.
4. Uncertainty propagation: preserve or improve GUM/Knowledge metadata through
   transformations and approximations.

Every E-KAN artifact must carry:

- source/input data hash;
- basis family (`spline`, `RBF`, polynomial, exact symbolic);
- coefficient table hash;
- approximation domain;
- error bound and method used to establish it;
- uncertainty/covariance assumptions;
- exact fallback expression or interpreter path;
- validation result and counterexample set, if any.

Hard rule:

- E-KAN can suggest. Receipts decide. Exact semantics win.

## Validation ladder

Use this as the acceptance ladder for architecture work:

| Level | Name | Evidence |
|---|---|---|
| L0 | Sketch | doc-only plan, no implementation claim |
| L1 | Local witness | one stage receipt on a tiny source |
| L2 | Repo gate | stage gate over curated repo modules |
| L3 | Cross-stage proof | source -> stage N deterministic receipts |
| L4 | Translation validated | selected transforms validated or rejected |
| L5 | Differential | CPU/GPU/WASM/interpreter agree on witnesses |
| L6 | Self-host fixed point | compiler rebuilds itself with stable receipts |
| L7 | Publishable novelty | external benchmarks, ablations, negative results, and independent review |

The current document is L0.

## Migration from current Madaros

1. Freeze current compiler work as S0 safety. Continue only repairs that protect
   default users or unblock S1 receipt generation.
2. Create S1 receipt and parser/module graph artifacts first. This is the
   highest leverage move because it separates import/path/diagnostic truth from
   later lowering crashes.
3. Add the missing GPU/PTX module-combination gate before changing GPU lowering.
4. Promote E-KAN current witnesses into a staged plan:
   - current: runtime/witness examples and native-v2 frontier;
   - next: S2 declarations for domains/basis/uncertainty;
   - then: S4 optimizer proposals with receipts.
5. Audit HLIR before treating it as contractual:
   - duplicate epistemic enum entries;
   - fixed array caps;
   - effect metadata shape;
   - serialization/canonical hashing.
6. Keep GPU KernelIR as S6 target IR and build a tensor-kernel bridge rather
   than turning KernelIR into the optimizer.
7. Only after S1-S3 receipts are stable, introduce S4 e-graph/E-KAN rewriting.
   This prevents neural/symbolic novelty from hiding compiler correctness bugs.

## Swarm workflow

When subagent capacity is available, run the work in bounded waves. Nobody edits
the same file in the same wave.

Wave A - read-only synthesis:

- Compiler architecture scout (`gpt-5.4-mini`, medium): map repo IR surfaces,
  gates, and hazards.
- Literature scout (`gpt-5.4-mini`, medium): update SOTA anchors and extract
  safe claim boundaries.
- E-KAN scout (`gpt-5.4`, high): map current E-KAN witnesses to S2/S4 compiler
  concepts.
- Validation scout (`gpt-5.4`, high): design receipts, translation validation,
  differential testing, and counterexample gates.

Wave B - implementation, disjoint write scopes:

- S1 worker owns only `self-hosted/compiler/*s1*`, S1 docs, and S1 gates.
- HLIR worker owns only `self-hosted/hlir/*` and HLIR receipt tests.
- E-KAN worker owns only E-KAN declaration/receipt files and tests.
- GPU gate worker owns only GPU/PTX module-combination gates and fixtures.

Current session note:

- A subagent spawn was attempted on 2026-07-04 and failed with
  `agent thread limit reached`. This plan is therefore written locally, but its
  workflow is swarm-ready.

## Claim discipline

Allowed now:

- "Madaros v2 has a SOTA-informed architecture plan."
- "The novelty target is receipt-carrying epistemic compilation with E-KAN and
  persistent equality reasoning as first-class compiler objects."

Not allowed yet:

- "Madaros v2 is implemented."
- "Sounio has a proven novel compiler architecture."
- "E-KAN compiler optimization is sound."
- "f128/i256 are supported" without ABI/layout/codegen/diagnostic receipts.

## Immediate next lane

Do this before more broad compiler repair:

1. Add an S1 receipt skeleton for canonical source + module graph.
2. Add a tiny S1 gate over:
   - `tests/run-pass/hello.sio`;
   - one imported stdlib module;
   - the GPU/PTX `lower_to_ptx.sio` + `ptx.sio` combination that currently has
     a known documentation/audit record.
3. Make the S1 receipt stable and hashable.
4. Only then resume the current lowering crash/debug lane with a clean receipt
   showing exactly which module and body crosses the boundary.

This gives the project a spine. The tactical compiler repair can then become a
stage-specific bug, instead of the whole story.
