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

Status: the S4 conservative e-graph/E-KAN receipt boundary plus extraction,
S4->S5 application-plan, and applied-S5-input bundle are implemented and gated.
The S3 HLIR binary
operand provenance blocker found on 2026-07-05 is now closed by the S3 lowering
fix and operand-fidelity gate: S4 accepts exact constant-fold candidates, a
non-constant neutral-element symbolic identity subset, and a param/block-param
reflexive-comparison subset (`x == x`, `x != x`, `x <= x`, `x >= x`,
`x < x`, `x > x`) plus same-SSA symbolic subtraction (`x - x -> 0`) over
params/block params and local leaf call results, keeps counterexample-backed
rejected proposals out of extraction, and selects no blocked rewrites for
extraction. The symbolic identity family now also gates the full five-rule
neutral-element matrix over both param/block-param producers and local-leaf
pure-call producers. The current fixture set has nine classified counterexample
rejections (including a six-case distinct-comparison sibling-law matrix) and
seven classified producer-evaluation blockers, all excluded from extraction and
from the applied S5-input bundle. The reflexive-comparison family now gates the
full six-operator matrix for param/block-param positives, local-leaf pure-call
positives, distinct-operand negatives, and effectful-call blockers. Global S4
optimization is not complete:
equality saturation, approximate learned E-KAN proposals, broad counterexample
search beyond the current family matrix, and real compiler-IR mutation remain
future work. S5 has an executable preflight from the S4 extraction receipts and
the applied bundle and now reports `status = pass` with
`s5_input_contract_ready = true`. S5 now also has an
input-boundary MIR/ABI classification receipt for the current selected exact S4
subset, and the applied-extraction bundle hash is propagated through S5
preflight, MIR/ABI input-boundary, MIR-effect, and final program-MIR/ABI
receipts. Real compiler-IR mutation by S4 and full S5 remain future work.

## Implemented surface

- `bin/madaros s4-receipt <source> [--out-dir OUT]`
- `scripts/dev/madaros_v2_s4_receipt.py`
- `scripts/dev/madaros_v2_s4_gate.sh`
- `tests/madaros/v2_s4/manifest.tsv`
- `tests/madaros/v2_s4/exact_identity.sio`
- `tests/madaros/v2_s4/extract_cost_chain_i64.sio`
- `tests/madaros/v2_s4/symbolic_identity_i64.sio`
- `tests/madaros/v2_s4/symbolic_identity_pure_call_i64.sio`
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

The current local gate has `blocked=7`: operand-fidelity blockers remain
available but are not triggered by the current fixtures; the current blockers
come from the six-operator non-leaf/effectful call-result self-comparison
matrix and sub-self
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

The S4 gate also emits `madaros.v2.s4.applied_extraction/0.1`. This is a
deterministic applied bundle for S5 input, not a claim that compiler IR has
already been mutated. It records selected/rejected/blocked action buckets,
per-effect pre/post hashes, `application_applied_to_s5_input = true`,
`application_applied_to_compiler_ir = false`, and the same MIR/ABI-safe/no-ABI-
impact invariants consumed by S5 preflight.

## Gate

```bash
bash scripts/dev/madaros_v2_s4_gate.sh
```

The gate pins `SOUNIO_STDLIB_PATH` to this checkout's `stdlib/` and, when no
`MADAROS_RAW_BIN` is supplied, proves `artifacts/self-hosted/madaros` through
`scripts/ci/madaros_full_gate.sh` before consuming HLIR.

Observed local result on 2026-07-06 after the S3 operand-fidelity fix,
reflexive-comparison slice, sub-self arithmetic slice, distinct-comparison
counterexample matrix, and applied-extraction bundle:

```text
[madaros-v2-s4] ok receipt=exact_identity.s4.receipt.json accepted=3 rejected=0 blocked=0 selected=3 egraph_sha=1b5c2790c49d extraction_sha=342e8492421f
[madaros-v2-s4] ok receipt=extract_cost_chain_i64.s4.receipt.json accepted=6 rejected=0 blocked=0 selected=6 egraph_sha=667d12c57234 extraction_sha=a85cc30506a4
[madaros-v2-s4] ok receipt=symbolic_identity_i64.s4.receipt.json accepted=8 rejected=0 blocked=0 selected=8 egraph_sha=6ed268e74dde extraction_sha=3ec34adcde81
[madaros-v2-s4] ok receipt=symbolic_identity_pure_call_i64.s4.receipt.json accepted=5 rejected=0 blocked=0 selected=5 egraph_sha=1dc914524301 extraction_sha=d889c7687e63
[madaros-v2-s4] ok receipt=symbolic_reflexive_cmp_i64.s4.receipt.json accepted=6 rejected=0 blocked=0 selected=6 egraph_sha=77655d13e1c3 extraction_sha=e692c0eefd87
[madaros-v2-s4] ok receipt=symbolic_reflexive_cmp_pure_call_i64.s4.receipt.json accepted=6 rejected=0 blocked=0 selected=6 egraph_sha=f7b7b085706d extraction_sha=bd369f4cff40
[madaros-v2-s4] ok receipt=symbolic_sub_self_i64.s4.receipt.json accepted=2 rejected=0 blocked=0 selected=2 egraph_sha=e56621dd4c50 extraction_sha=1ca05ce54edf
[madaros-v2-s4] ok receipt=reject_distinct_symbolic_cmp_i64.s4.receipt.json accepted=0 rejected=6 blocked=0 selected=0 egraph_sha=6e20317c0bb8 extraction_sha=2fffd6c4414e
[madaros-v2-s4] ok receipt=reject_call_result_self_cmp_i64.s4.receipt.json accepted=0 rejected=0 blocked=6 selected=0 egraph_sha=e213f9bc70ba extraction_sha=cf12ef393ec9
[madaros-v2-s4] ok receipt=reject_distinct_symbolic_sub_i64.s4.receipt.json accepted=0 rejected=1 blocked=0 selected=0 egraph_sha=209d7fe5794e extraction_sha=859286dba635
[madaros-v2-s4] ok receipt=reject_call_result_sub_self_i64.s4.receipt.json accepted=0 rejected=0 blocked=1 selected=0 egraph_sha=824c38d53989 extraction_sha=9c9e1d092790
[madaros-v2-s4] ok receipt=recursion_fact.s4.receipt.json accepted=0 rejected=0 blocked=0 selected=0 egraph_sha=31532e5b09e4 extraction_sha=496e7ccf7342
[madaros-v2-s4] ok receipt=reject_div_self_zero.s4.receipt.json accepted=0 rejected=1 blocked=0 selected=0 egraph_sha=34a7c2129d00 extraction_sha=54e83fe3c2d0
[madaros-v2-s4] ok receipt=reject_div_self_mixed_with_accepted.s4.receipt.json accepted=1 rejected=1 blocked=0 selected=1 egraph_sha=08a67da48631 extraction_sha=a34f4ab3e0b8
[madaros-v2-s4] summary_sha=7456e261ca42 accepted=37 rejected=9 blocked=7 selected=37 app_plan=0af0de21c232 applied=4527e90ac399
[madaros-v2-s4] PASS: S4 boundary receipts are deterministic and validated (S4 FULL remains blocked by listed obligations)
[madaros-v2-s4] PASS: S4->S5 application plan emitted for selected exact rewrites without mutating IR
[madaros-v2-s4] PASS: S4 applied extraction materialized as deterministic S5 input effects without mutating compiler IR
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

Observed local result on 2026-07-06 after the applied-extraction bundle,
five-rule pure-call symbolic identity matrix, and six-operator reflexive-
comparison producer matrix:

```text
[madaros-v2-s5-preflight] ok cases=14 rewrites=37 blocked=7 status=pass sha=8a415c8e6c9c
[madaros-v2-s5-preflight] PASS: S5 preflight classified current S4 extraction input without overclaiming readiness
```

The preflight receipt uses schema `madaros.v2.s5.preflight/0.1` and records
`s5_input_contract_ready = true`, `s5_ready = false`,
`s5_implemented = false`, and
`input_applied_extraction_contract = madaros.v2.s4.applied_extraction/0.1`.
It verifies that every accepted S4 rewrite has a matching applied S5-input
effect and that rejected/blocked rewrites remain outside the materialized input.
The applied bundle hash for this run is
`4527e90ac399de5ce6f746d328a95dfd8c85d819d908aaf2f5bfb94a81b35f8d`.

The S5 MIR/ABI input-boundary gate is executable through:

```bash
bash scripts/dev/madaros_v2_s5_mir_abi_gate.sh
```

Observed local result on 2026-07-06:

```text
[madaros-v2-s5-mir-abi] ok rewrites=37 blocked=7 abi_classes=scalar_bool,scalar_i64 sha=039163c2df53
[madaros-v2-s5-mir-abi] PASS: S5 MIR/ABI input-boundary receipts classify the current S4 selected subset without claiming real MIR/ABI or S5 FULL
```

The input-boundary receipt uses schema
`madaros.v2.s5.mir_abi_input_boundary/0.1` and records
`s5_mir_abi_input_boundary_complete = true`,
`s5_mir_abi_boundary_complete = false`, `real_mir_emitted = false`,
`real_abi_layout_emitted = false`, `s5_ready = false`,
`s5_implemented = false`, and `s5_full_complete = false`. It classifies the
current selected S4 subset as scalar input only (`scalar_i64`/`scalar_bool`) and
records no call-signature, stack, SRET, aggregate-layout, or ABI impact. Each
input-boundary witness carries the S4 applied-extraction hash and its source
`applied_effect_sha256`.

The S5 scalar MIR-effect and direct-call/return slice is executable through:

```bash
bash scripts/dev/madaros_v2_s5_mir_effect_gate.sh
```

Observed local result on 2026-07-06:

```text
[madaros-v2-s5-mir-effect] ok effects=37 native_witnesses=3 opcodes=mir.alias.i64,mir.const.bool,mir.const.i64 sha=13c3ba069d30
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
`ARG_MOVE,CALL,CAPTURE_RET,STORE_STACK,RET`. Full S5 remains blocked by the
explicit program-gate obligations, especially f128 generic helper/execution
differentials and broader full-program MIR/ABI proof beyond the promoted
receipt surfaces.
Each MIR-effect carries `input_applied_extraction_sha256`,
`source_applied_effect_sha256`, and the post-apply S5-input hashes propagated
from the MIR/ABI boundary.

The scalar compiler-MachineModule + ABI-shadow gate is executable through:

```bash
bash scripts/dev/madaros_v2_s5_program_mir_abi_gate.sh
```

Each scalar source also has a canonical compiler-facing receipt path:

```bash
./bin/madaros s5-receipt tests/madaros/v2_s5/scalar_i64_direct_call_return_42.sio \
  --expected-exit 42 \
  --case-id scalar_i64_direct_call_return_42
```

Observed deterministic local result on 2026-07-06:

```text
[madaros-v2-s5-program-mir-abi] ok programs=3 target=x86_64-linux sha=d0b0baa689a4
[madaros-v2-s5-program-mir-abi] PASS: scalar i64/bool + SRET + f64/XMM0 + wide-int + local+imported i256/u256 wide ABI call-return + generic aggregate + f128 value-contract binary128 compiler MachineModule ABI receipts are deterministic without claiming S5 FULL
```

That receipt uses schema `madaros.v2.s5.program_mir_abi_scalar_shadow/0.1`.
It checks the three scalar programs at the program boundary and now requires
compiler-exported `madaros.v2.s5.machine_module/0.1` JSON from
`--native-v2-compile <src> -o <elf> --machine-module-json <json>`. The gate
checks merged-IR function counts (`1,2,2`), MachineModule function/instruction
totals, ELF internal-call counts (`1,2,2`), scalar ABI signatures (`rdi` for the
single i64 parameter where present, `rax` for i64/bool returns), and the legal
MIR call-return contract anchored to `self-hosted/native/machine_ir.sio` and
`self-hosted/native/codegen_x86_linux.sio`.
It also records S4 semantic negatives and producer-evaluation blockers as
not-selected/not-promoted controls, so rejected or blocked rewrites cannot leak
into the scalar S5 slice.
The S5.6 f128 arithmetic value-contract surface is now intentionally a finite
decimal-tenths matrix rather than a singleton: exact positive finite cases cover
add, multiply, divide, and a one-chain `add -> sub` metadata propagation witness,
while unsupported fractional products and out-of-matrix sums remain fail-closed
with `f128_arithmetic_pending`. This still does not promote generic IEEE f128
helpers, arbitrary decimal materialization, NaN/Inf behavior, external SysV
f128 ABI, f128 call ABI, or f128 return ABI.
The rebuilt S-next compiler on 2026-07-06 produced S5.6 receipt sha
`349eedae3f35ecf4969a13699d98e43427bd1ec30c0ad912e7f0091144f05bd1`
(`case_count=7`, `positive_case_count=5`, `negative_case_count=2`) and
aggregate S5 program MIR/ABI receipt sha
`feb8a05c2d11f07db24c58fae0488b3178728c3ea7cc957272ba3c188380f629`.
The same rebuilt `artifacts/self-hosted/madaros` also passed
`scripts/ci/madaros_full_gate.sh` (including imported-SMT `6/6`) and
`scripts/ci/madaros_source_to_elf_gate.sh`.
The final program receipt now records `input_applied_extraction_sha256 =
4527e90ac399de5ce6f746d328a95dfd8c85d819d908aaf2f5bfb94a81b35f8d` and
`s4_applied_extraction_consumed = true`, matching the preflight, MIR/ABI
boundary, and MIR-effect receipts.
The gate now also calls `madaros s5-receipt` for each scalar witness and
requires three `madaros.v2.s5.receipt/0.1` per-source receipts to match the
aggregate witness shape (`canonical_s5_source_receipt_count = 3`,
`canonical_s5_source_receipts_present = true`).
It deliberately records `compiler_machine_module_exported = true`,
`real_program_mir_emitted = true`, `real_abi_layout_emitted = false`,
`s5_ready = false`, and `s5_full_complete = false`.

That is deliberately not a claim that S5 is implemented. Under the S-FULL rule,
S5 completion requires wider MachineModule coverage, ABI/layout/call/return
receipts, numeric tower witnesses, diagnostics, fallback semantics, and
cross-stage differential gates.
