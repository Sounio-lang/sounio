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

Status: the S4 conservative e-graph/E-KAN accepted/rejected receipt boundary
plus receipt-only extraction/cost-model boundary is implemented and gated.
Global S4 optimization is not complete: equality saturation, approximate
learned E-KAN proposals, broad counterexample search, and downstream optimizer
integration remain future work. S5 has a gated input-contract preflight from the
S4 extraction receipts, but MIR/ABI lowering is not implemented here.

## Implemented surface

- `bin/madaros s4-receipt <source> [--out-dir OUT]`
- `scripts/dev/madaros_v2_s4_receipt.py`
- `scripts/dev/madaros_v2_s4_gate.sh`
- `tests/madaros/v2_s4/manifest.tsv`
- `tests/madaros/v2_s4/exact_identity.sio`
- `tests/madaros/v2_s4/extract_cost_chain_i64.sio`
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

The current S4 pass is intentionally conservative. It accepts only:

- `constant_fold_i64` rewrites
- `basis_family = exact_symbolic`
- `validator = translation-validation`
- `error_bound = 0`
- non-empty exact fallback expression hashes
- original and rewritten e-node hashes

It also emits deterministic rejected proposal receipts for:

- `x_div_x_to_one`
- `proposal_kind = algebraic_identity`
- `validator = rejected`
- `rejection_reason_code = counterexample_found`
- counterexample `x = 0`, where the original expression traps on division by
  zero but the proposed rewrite returns `1`
- `selected_for_extraction = false`
- `ir_mutation_allowed = false`

This is a real receipt boundary for one exact optimizer subset, not an
approximation claim and not global S4 completion. It builds a persistent
e-graph/equality artifact and records accepted and rejected rewrites, but it
does not yet apply approximate learned E-KAN laws, run equality saturation, or
mutate downstream code.

## Extraction Boundary

S4 now emits `madaros.v2.s4.extraction/0.1` receipts. This is receipt-only
extraction over the current S4 boundary artifact, not full compiler e-graph
extraction and not downstream IR mutation. For every rewrite, the extraction
receipt records:

- input HLIR/e-graph/rewrite hashes
- deterministic extraction policy and cost-model/config hashes
- one decision per rewrite
- selected accepted rewrite IDs
- blocked rejected rewrite IDs
- cost before/after/delta and cost components
- validator log, exact fallback, coefficient, domain, and basis evidence
- `extraction_applied_to_ir = false`
- `ir_mutation_allowed = false`

The S4 gate proves that selected IDs exactly equal accepted IDs, rejected IDs
are blocked from extraction, accepted decisions remain exact
translation-validated zero-error decisions, and the extraction receipt is
byte-deterministic across duplicate emission.

## Gate

```bash
bash scripts/dev/madaros_v2_s4_gate.sh
```

The gate pins `SOUNIO_STDLIB_PATH` to this checkout's `stdlib/` and, when no
`MADAROS_RAW_BIN` is supplied, proves `artifacts/self-hosted/madaros` through
`scripts/ci/madaros_full_gate.sh` before consuming HLIR.

Observed local result on 2026-07-05:

```text
[madaros-v2-s4] ok receipt=exact_identity.s4.receipt.json accepted=3 rejected=0 egraph_sha=f99cee31ce4a
[madaros-v2-s4] ok receipt=extract_cost_chain_i64.s4.receipt.json accepted=6 rejected=0 selected=6 egraph_sha=8ad9a151df78 extraction_sha=b4a196f11453
[madaros-v2-s4] ok receipt=recursion_fact.s4.receipt.json accepted=2 rejected=0 selected=2 egraph_sha=c74170fdb0af extraction_sha=b30cbed09425
[madaros-v2-s4] ok receipt=reject_div_self_zero.s4.receipt.json accepted=0 rejected=1 selected=0 egraph_sha=34a7c2129d00 extraction_sha=f75250c7acbf
[madaros-v2-s4] ok receipt=reject_div_self_mixed_with_accepted.s4.receipt.json accepted=1 rejected=1 selected=1 egraph_sha=96b78aff4c35 extraction_sha=52b882d72384
[madaros-v2-s4] summary_sha=305f730a5ab9 accepted=12 rejected=2 selected=12
[madaros-v2-s4] PASS: conservative e-graph/E-KAN rewrite and extraction receipts are deterministic and validated
```

## S5 Boundary

S5 needs MIR hashes and ABI/layout/call/return receipts. The current S4
boundary provides a validated input subset for that lane: extraction-selected
accepted rewrites are exact i64 constant folds that do not change function
signatures, calls, control flow, aggregate layout, or numeric width semantics.
Rejected rewrites are explicitly present in the S4 extraction receipt as blocked
decisions and are not consumed by S5.

The S5 input-contract preflight is executable through:

```bash
bash scripts/dev/madaros_v2_s5_preflight_gate.sh
```

Observed local result on 2026-07-05:

```text
[madaros-v2-s5-preflight] ok cases=5 rewrites=12 sha=40ca97219b50
[madaros-v2-s5-preflight] PASS: current S4 boundary receipts are MIR/ABI-safe S5 inputs; S4 global completion and S5 remain future work
```

The preflight receipt uses schema `madaros.v2.s5.preflight/0.1` and records
`s5_input_contract_ready = true`, `s5_ready = false`, and
`s5_implemented = false`.
