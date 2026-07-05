<!-- docs:meta
topic_id: repo.docs.research.madaros-v2-s4-egraph-ekan-receipts-2026-07-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.madaros-v2-s4-egraph-ekan-receipts-2026-07-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Madaros v2 S4 e-graph/E-KAN receipts

Status: the S4 conservative e-graph/E-KAN receipt boundary plus receipt-only
extraction/cost-model boundary is implemented and gated. The S3 HLIR binary
operand provenance blocker found on 2026-07-05 is now closed by the S3 lowering
fix and operand-fidelity gate: S4 accepts exact constant-fold candidates, a
non-constant neutral-element symbolic identity subset, and a param/block-param
reflexive-comparison subset (`x == x`, `x != x`, `x <= x`, `x >= x`,
`x < x`, `x > x`) plus same-SSA symbolic subtraction (`x - x -> 0`) over
params/block params and local leaf call results, keeps counterexample-backed
rejected proposals out of extraction, and selects no blocked rewrites for
extraction. The current fixture set has three classified producer-evaluation
blockers, all excluded from extraction. Global S4
optimization is not complete:
equality saturation, approximate learned E-KAN proposals, broad counterexample
search, and downstream optimizer integration remain future work. S5 has an
executable preflight from the S4 extraction receipts and now reports
`status = pass` with `s5_input_contract_ready = true`. S5 now also has an
input-boundary MIR/ABI classification receipt for the current selected exact S4
subset, but real MIR serialization and ABI layout/call/return receipts remain
future work.

## Implemented surface

- `bin/madaros s4-receipt <source> [--out-dir OUT]`
- `scripts/dev/madaros_v2_s4_receipt.py`
- `scripts/dev/madaros_v2_s4_gate.sh`
- `tests/madaros/v2_s4/manifest.tsv`
- `tests/madaros/v2_s4/exact_identity.sio`
- `tests/madaros/v2_s4/extract_cost_chain_i64.sio`
- `tests/madaros/v2_s4/symbolic_identity_i64.sio`
- `tests/madaros/v2_s4/symbolic_reflexive_cmp_i64.sio`
- `tests/madaros/v2_s4/symbolic_reflexive_cmp_pure_call_i64.sio`
- `tests/madaros/v2_s4/symbolic_sub_self_i64.sio`
- `tests/madaros/v2_s4/reject_distinct_symbolic_cmp_i64.sio`
- `tests/madaros/v2_s4/reject_call_result_self_cmp_i64.sio`
- `tests/madaros/v2_s4/reject_distinct_symbolic_sub_i64.sio`
- `tests/madaros/v2_s4/reject_call_result_sub_self_i64.sio`
- `tests/madaros/v2_s4/reject_div_self_zero.sio`
- `tests/madaros/v2_s4/reject_div_self_mixed_with_accepted.sio`
- `scripts/dev/madaros_v2_s5_preflight_gate.sh`

S4 consumes S3 HLIR receipts rather than source text directly. For each case it
emits:

- `<case>.s4.egraph.json`
- `<case>.s4.rewrites.json`
- `<case>.s4.extraction.json`
- `<case>.s4.receipt.json`

The rewrite receipt schema is `madaros.v2.ekan.rewrite/0.1`, matching the S4
plan's E-KAN receipt boundary.

## What S4 Accepts And Rejects

The current S4 pass is intentionally conservative. It accepts only candidates
that survive both operand-provenance guards and semantic validation. The current
accepted subset is:

- `constant_fold_i64` rewrites
- `symbolic_identity_i64` rewrites for `x + 0`, `0 + x`, `x * 1`, `1 * x`,
  and `x - 0`
- `symbolic_reflexive_cmp_i64` rewrites for param/block-param and local leaf
  call-result same-SSA comparisons: `x == x -> true`, `x != x -> false`,
  `x <= x -> true`, `x >= x -> true`, `x < x -> false`, and
  `x > x -> false`
- `symbolic_sub_self_i64` rewrites for param/block-param and local leaf
  call-result same-SSA subtraction: `x - x -> 0`
- `basis_family = exact_symbolic`
- `validator = translation-validation`
- `error_bound = 0`
- non-empty exact fallback expression hashes
- original and rewritten e-node hashes

For symbolic identities, the proposed/rewrite enode is a `value_ref` to the
existing S3 SSA producer, not a copied expression or a constant. The gate checks
`identity_kind`, neutral side/constant, non-constant symbolic producer,
`domain = all-i64-values-with-neutral-element`, and `validator_attempted`
contains both `translation-validation` and `neutral-element-proof`.

For reflexive comparisons, the proposed/rewrite enode is the exact bool const
for the comparison kind. The gate checks `same_operand_id = true`,
`result_const`, producer policy, `domain =
all-i64-values-with-reflexive-equality-and-order`, and `validator_attempted`
contains `translation-validation`, `reflexive-comparison-proof`, and
`producer-evaluation-preservation-proof`. Param/block-param producers use
`producer_is_param_or_block_param_no_effectful_eval`. Call-result producers are
accepted only when the callee is local leaf pure in the current HLIR
(`call_summary.purity_reason = local_leaf_no_call_direct`) and the extraction
decision carries
`replace_binary_predicate_expr_with_const_bool_keep_producer_evaluated`.
Call-result self-comparisons whose callees contain `call_direct` are blocked
with `producer_evaluation_not_proven` and excluded from extraction.

For sub-self arithmetic, the proposed/rewrite enode is the exact int const
`0`. The gate checks `subtraction_kind = sub_self_zero`,
`same_operand_id = true`, `result_const = ["int", 0]`, producer policy,
`domain = all-i64-values-with-same-ssa-subtraction`, and
`validator_attempted` contains `translation-validation`,
`same-ssa-subtraction-proof`, and `producer-evaluation-preservation-proof`.
Local leaf call producers use
`replace_binary_sub_self_expr_with_const_i64_zero_keep_producer_evaluated`;
non-leaf/effectful call-result sub-self rewrites are blocked with
`producer_evaluation_not_proven` and excluded from extraction.

It also emits deterministic rejected proposal receipts for:

- `x_div_x_to_one`
- `distinct_symbolic_sub_to_zero` (`x - y -> 0`)
- `proposal_kind = algebraic_identity` for division, `rejected_symbolic_sub_self`
  for distinct symbolic subtraction
- `validator = rejected`
- `rejection_reason_code = counterexample_found`
- counterexample `x = 0`, where the original expression traps on division by
  zero but the proposed rewrite returns `1`
- counterexample `x = 1, y = 2`, where distinct symbolic subtraction returns
  `-1` but the proposed rewrite returns `0`
- `selected_for_extraction = false`
- `ir_mutation_allowed = false`

It can also emit deterministic blocked proposal receipts for:

- `operand_provenance_blocked`
- `validator = blocked`
- `rejection_reason_code = operand_provenance_ambiguous`
- duplicated or otherwise ambiguous S3 HLIR binary operand provenance where the
  source expression cannot be proven from the emitted HLIR
- `selected_for_extraction = false`
- `ir_mutation_allowed = false`

It also emits producer-evaluation blockers for same-SSA reflexive comparisons
and same-SSA subtraction whose producer would otherwise be removed without a
proven evaluation-preserving policy:

- `rewrite_kind = symbolic_reflexive_cmp_i64` or `symbolic_sub_self_i64`
- `proposal_kind = blocked_symbolic_reflexive_comparison` or
  `blocked_symbolic_sub_self`
- `ekan_receipt_kind = ekan_blocked_producer_evaluation`
- `rejection_reason_code = producer_evaluation_not_proven`
- `producer_evaluation_policy = blocked: producer evaluation is not proven`
- `selected_for_extraction = false`
- `ir_mutation_allowed = false`

The current local gate has `blocked=3`: operand-fidelity blockers remain
available but are not triggered by the current fixtures; the current blockers
come from non-leaf/effectful call-result self-comparisons and sub-self
arithmetic where producer evaluation is not yet proven.

This is a real receipt boundary for one exact optimizer subset, not an
approximation claim and not global S4 completion. It builds a persistent
e-graph/equality artifact and records accepted, rejected, and blocked rewrites,
but it does not yet apply approximate learned E-KAN laws, run equality
saturation, or mutate downstream code.

## S4 Full-Slice Contract

An S4 rewrite family is not accepted just because one fixture turns green. Each
family must land as a full slice:

- positive fixtures for the whole declared exact domain;
- rejected counterexample fixtures for tempting but invalid sibling laws;
- blocked fixtures for ambiguous provenance, unsafe producer evaluation, effects,
  overflow/trap uncertainty, or unsupported semantics;
- receipt fields for domain, validator attempts, fallback hashes, original and
  rewritten e-node hashes, producer policy, error bound, and extraction status;
- gate assertions that accepted, rejected, and blocked records are disjoint;
- extraction receipts proving selected IDs exactly equal accepted IDs;
- S5 preflight compatibility proving the rewrite has no hidden MIR/ABI impact;
- docs and offload review when semantics are mathematical or externally claimed.

If any part is missing, the result is a partial witness, not an S4 family.

## Extraction Boundary

S4 now emits `madaros.v2.s4.extraction/0.1` receipts. This is receipt-only
extraction over the current S4 boundary artifact, not full compiler e-graph
extraction and not downstream IR mutation. For every rewrite, the extraction
receipt records:

- input HLIR/e-graph/rewrite hashes
- deterministic extraction policy and cost-model/config hashes
- one decision per rewrite
- selected accepted rewrite IDs
- rejected rewrite IDs
- blocked rewrite IDs
- cost before/after/delta and cost components
- validator log, exact fallback, coefficient, domain, and basis evidence
- `extraction_applied_to_ir = false`
- `ir_mutation_allowed = false`

The S4 gate proves that selected IDs exactly equal accepted IDs, rejected and
blocked IDs are excluded from extraction, accepted decisions remain exact
translation-validated zero-error decisions when present, and the extraction
receipt is byte-deterministic across duplicate emission.

## Gate

```bash
bash scripts/dev/madaros_v2_s4_gate.sh
```

The gate pins `SOUNIO_STDLIB_PATH` to this checkout's `stdlib/` and, when no
`MADAROS_RAW_BIN` is supplied, proves `artifacts/self-hosted/madaros` through
`scripts/ci/madaros_full_gate.sh` before consuming HLIR.

Observed local result on 2026-07-05 after the S3 operand-fidelity fix,
reflexive-comparison slice, and sub-self arithmetic slice:

```text
[madaros-v2-s4] ok receipt=exact_identity.s4.receipt.json accepted=3 rejected=0 blocked=0 selected=3 egraph_sha=1b5c2790c49d extraction_sha=342e8492421f
[madaros-v2-s4] ok receipt=extract_cost_chain_i64.s4.receipt.json accepted=6 rejected=0 blocked=0 selected=6 egraph_sha=667d12c57234 extraction_sha=a85cc30506a4
[madaros-v2-s4] ok receipt=symbolic_identity_i64.s4.receipt.json accepted=8 rejected=0 blocked=0 selected=8 egraph_sha=6ed268e74dde extraction_sha=3ec34adcde81
[madaros-v2-s4] ok receipt=symbolic_reflexive_cmp_i64.s4.receipt.json accepted=6 rejected=0 blocked=0 selected=6 egraph_sha=77655d13e1c3 extraction_sha=e692c0eefd87
[madaros-v2-s4] ok receipt=symbolic_reflexive_cmp_pure_call_i64.s4.receipt.json accepted=2 rejected=0 blocked=0 selected=2 egraph_sha=8ae383c0b9ff extraction_sha=c632174764cf
[madaros-v2-s4] ok receipt=symbolic_sub_self_i64.s4.receipt.json accepted=2 rejected=0 blocked=0 selected=2 egraph_sha=e56621dd4c50 extraction_sha=1ca05ce54edf
[madaros-v2-s4] ok receipt=reject_distinct_symbolic_cmp_i64.s4.receipt.json accepted=0 rejected=0 blocked=0 selected=0 egraph_sha=d600c866af79 extraction_sha=43d73c0a8bec
[madaros-v2-s4] ok receipt=reject_call_result_self_cmp_i64.s4.receipt.json accepted=0 rejected=0 blocked=2 selected=0 egraph_sha=2064c04e1415 extraction_sha=a366624a9561
[madaros-v2-s4] ok receipt=reject_distinct_symbolic_sub_i64.s4.receipt.json accepted=0 rejected=1 blocked=0 selected=0 egraph_sha=209d7fe5794e extraction_sha=859286dba635
[madaros-v2-s4] ok receipt=reject_call_result_sub_self_i64.s4.receipt.json accepted=0 rejected=0 blocked=1 selected=0 egraph_sha=b59dfffefce7 extraction_sha=f3192d747032
[madaros-v2-s4] ok receipt=recursion_fact.s4.receipt.json accepted=0 rejected=0 blocked=0 selected=0 egraph_sha=31532e5b09e4 extraction_sha=496e7ccf7342
[madaros-v2-s4] ok receipt=reject_div_self_zero.s4.receipt.json accepted=0 rejected=1 blocked=0 selected=0 egraph_sha=34a7c2129d00 extraction_sha=54e83fe3c2d0
[madaros-v2-s4] ok receipt=reject_div_self_mixed_with_accepted.s4.receipt.json accepted=1 rejected=1 blocked=0 selected=1 egraph_sha=08a67da48631 extraction_sha=a34f4ab3e0b8
[madaros-v2-s4] summary_sha=7f6eeb116689 accepted=28 rejected=3 blocked=3 selected=28
[madaros-v2-s4] PASS: S4 boundary receipts are deterministic and validated (S4 FULL remains blocked by listed obligations)
```

## S5 Boundary

S5 needs MIR hashes and ABI/layout/call/return receipts. The current S4
boundary now provides a ready input subset for the next S5 work item:
translation-validated accepted rewrites are selected, rejected rewrites are
explicitly excluded, and blocked rewrites are classified and excluded from
extraction.

The S5 input-contract preflight is executable through:

```bash
bash scripts/dev/madaros_v2_s5_preflight_gate.sh
```

Observed local result on 2026-07-05 after the sub-self arithmetic slice:

```text
[madaros-v2-s5-preflight] ok cases=13 rewrites=28 blocked=3 status=pass sha=ea7f1ffc4eeb
[madaros-v2-s5-preflight] PASS: S5 preflight classified current S4 extraction input without overclaiming readiness
```

The preflight receipt uses schema `madaros.v2.s5.preflight/0.1` and records
`s5_input_contract_ready = true`, `s5_ready = false`, and
`s5_implemented = false`.

The S5 MIR/ABI input-boundary gate is executable through:

```bash
bash scripts/dev/madaros_v2_s5_mir_abi_gate.sh
```

Observed local result on 2026-07-05:

```text
[madaros-v2-s5-mir-abi] ok rewrites=28 blocked=3 abi_classes=scalar_bool,scalar_i64 sha=bc24ae6a1bda
[madaros-v2-s5-mir-abi] PASS: S5 MIR/ABI input-boundary receipts classify the current S4 selected subset without claiming real MIR/ABI or S5 FULL
```

The input-boundary receipt uses schema
`madaros.v2.s5.mir_abi_input_boundary/0.1` and records
`s5_mir_abi_input_boundary_complete = true`,
`s5_mir_abi_boundary_complete = false`, `real_mir_emitted = false`,
`real_abi_layout_emitted = false`, `s5_ready = false`,
`s5_implemented = false`, and `s5_full_complete = false`. It classifies the
current selected S4 subset as scalar input only (`scalar_i64`/`scalar_bool`) and
records no call-signature, stack, SRET, aggregate-layout, or ABI impact.

The S5 scalar MIR-effect and direct-call/return slice is executable through:

```bash
bash scripts/dev/madaros_v2_s5_mir_effect_gate.sh
```

Observed local result on 2026-07-05:

```text
[madaros-v2-s5-mir-effect] ok effects=28 native_witnesses=3 opcodes=mir.alias.i64,mir.const.bool,mir.const.i64 sha=0c53f7750ec4
[madaros-v2-s5-mir-effect] PASS: S5 MIR-effect module roundtrips for the current selected subset without claiming full program MIR/ABI or S5 FULL
```

The MIR-effect receipt uses schema `madaros.v2.s5.mir_effect_roundtrip/0.1` and
records `s5_mir_effect_roundtrip_complete = true`,
`s5_scalar_i64_bool_direct_call_return_slice_complete = true`,
`real_mir_effects_serialized = true`, `real_program_mir_emitted = false`,
`real_abi_layout_emitted = false`, `s5_ready = false`,
`s5_implemented = false`, and `s5_full_complete = false`. The manifest is
exact-cardinality gated to the three scalar native-v2 witnesses in
`tests/madaros/v2_s5/`: i64 literal return, i64 direct-call return, and bool
direct-call return. The gate compiles and runs those witnesses, verifies their
exit codes, hashes their ELFs/logs, and inspects the executable ELF segment for
the expected internal-call shape before recording the Machine-IR contract
`ARG_MOVE,CALL,CAPTURE_RET,STORE_STACK,RET`. Aggregate, SRET, imported-call,
stack-arg, f64, f128, and i256 promotion remain blocked.

That is deliberately not a claim that S5 is implemented. Under the S-FULL rule,
S5 completion requires MIR hashes, ABI/layout/call/return receipts, numeric tower
witnesses, diagnostics, fallback semantics, and cross-stage differential gates.
