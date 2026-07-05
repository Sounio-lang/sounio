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

Status: the S4 conservative e-graph/E-KAN rejected/blocked receipt boundary
plus receipt-only extraction/cost-model boundary is implemented and gated. The
latest operand-provenance guard blocks previously accepted constant-fold
candidates when S3 HLIR duplicates binary operand IDs and therefore cannot prove
source operand fidelity. Global S4 optimization is not complete: S3 operand
provenance repair/proof, equality saturation, approximate learned E-KAN
proposals, broad counterexample search, and downstream optimizer integration
remain future work. S5 has an executable preflight from the S4 extraction
receipts, but it now reports `status = blocked` and
`s5_input_contract_ready = false` until S4 has accepted extraction-selected
rewrites with trustworthy operand provenance.

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

The current S4 pass is intentionally conservative. It accepts only candidates
that survive both operand-provenance guards and semantic validation. In the
current worktree, the operand-provenance guard blocks all former constant-fold
candidates, so the accepted count is currently zero. The accepted contract, when
the S3 operand provenance blocker is closed, is:

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

It also emits deterministic blocked proposal receipts for:

- `operand_provenance_blocked`
- `validator = blocked`
- `rejection_reason_code = operand_provenance_ambiguous`
- duplicated S3 HLIR binary operand IDs such as `lhs == rhs` where the source
  expression had distinct operands
- `selected_for_extraction = false`
- `ir_mutation_allowed = false`

This is a real receipt boundary for one exact optimizer subset, not an
approximation claim and not global S4 completion. It builds a persistent
e-graph/equality artifact and records accepted, rejected, and blocked rewrites,
but it does not yet apply approximate learned E-KAN laws, run equality
saturation, or mutate downstream code.

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

Observed local result on 2026-07-05:

```text
[madaros-v2-s4] ok receipt=exact_identity.s4.receipt.json accepted=0 rejected=0 blocked=3 selected=0 egraph_sha=01d589ee3943 extraction_sha=bd9d4124cf20
[madaros-v2-s4] ok receipt=extract_cost_chain_i64.s4.receipt.json accepted=0 rejected=0 blocked=6 selected=0 egraph_sha=24332471e8ae extraction_sha=2e66e4d31d7b
[madaros-v2-s4] ok receipt=recursion_fact.s4.receipt.json accepted=0 rejected=0 blocked=2 selected=0 egraph_sha=bd9d49d517d2 extraction_sha=f95f1fb1bd74
[madaros-v2-s4] ok receipt=reject_div_self_zero.s4.receipt.json accepted=0 rejected=1 blocked=0 selected=0 egraph_sha=34a7c2129d00 extraction_sha=94ec87029881
[madaros-v2-s4] ok receipt=reject_div_self_mixed_with_accepted.s4.receipt.json accepted=0 rejected=1 blocked=1 selected=0 egraph_sha=cc85d34d44b0 extraction_sha=2794ce5e477b
[madaros-v2-s4] summary_sha=3190e03cd733 accepted=0 rejected=2 blocked=12 selected=0
[madaros-v2-s4] PASS: conservative e-graph/E-KAN rewrite and extraction receipts are deterministic and validated
```

## S5 Boundary

S5 needs MIR hashes and ABI/layout/call/return receipts. The current S4
boundary does **not** provide a ready input subset: after the operand-provenance
guard, there are no extraction-selected accepted rewrites. Rejected and blocked
rewrites are explicitly present in the S4 extraction receipt as excluded
decisions and are not consumed by S5.

The S5 input-contract preflight is executable through:

```bash
bash scripts/dev/madaros_v2_s5_preflight_gate.sh
```

Observed local result on 2026-07-05:

```text
[madaros-v2-s5-preflight] ok cases=5 rewrites=0 blocked=12 status=blocked sha=1e0fe617fc87
[madaros-v2-s5-preflight] PASS: S5 preflight classified current S4 extraction input without overclaiming readiness
```

The preflight receipt uses schema `madaros.v2.s5.preflight/0.1` and records
`s5_input_contract_ready = false`, `s5_ready = false`, and
`s5_implemented = false`.
