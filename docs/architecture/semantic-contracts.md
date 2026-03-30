<!-- docs:meta
topic_id: repo.docs.architecture.semantic-contracts
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.semantic-contracts
-->

# Ontology Semantic Contracts

## Summary

This note defines the current semantic split for the ontology subsystem without
changing behavior. It exists to prepare legacy extraction planning now that the
rebuilt ontology gate is operational and the remaining failures are semantic,
not wrapper-plumbing failures.

OWL-like inference and compile-time rejection are separate responsibilities:

- the ontology kernel answers semantic questions
- the validation layer enforces ontology declaration constraints
- the boundary engine decides how compiler type and call boundaries consult ontology semantics
- the summary API exposes only scalar validation status to rebuilt-driver code

Non-goals for this phase:

- no ABox features
- no class-expression reasoning
- no legacy removal
- no fixture expectation changes
- no change to rebuilt/default/diff behavior

Related docs:

- [truth-frontier.md](./truth-frontier.md)
- [compiler-maturity-blueprint.md](./compiler-maturity-blueprint.md)
- [truth-layers.md](./truth-layers.md)

## Layer Contract

### 1. Inference Kernel

Owns:

- ontology class/property/axiom materialization
- subclass and disjointness queries
- ontology kernel profile materialization used by semantic checks

Must not own:

- user-facing compile-time boundary policy
- rebuilt-driver status reporting

Current anchors in `self-hosted/check/check.sio`:

- `ontology_inference_collect_kernel(...)`
- `collect_ontology_kernel_classes(...)`
- `collect_ontology_kernel_property_axioms_from_legacy(...)`
- `collect_ontology_kernel_disjoints(...)`
- `collect_ontology_roles(...)`
- `ontology_inference_query_subclass(...)`
- `ontology_inference_query_disjoint(...)`

### 2. Validation Layer

Owns:

- declaration-time ontology consistency checks
- inverse-role consistency checks
- strengthening and profile-oriented validation passes
- duplicate-role and missing target diagnostics as they currently surface during ontology collection

Must not own:

- call-site type substitution policy
- rebuilt-driver scalar summaries

Current anchors in `self-hosted/check/check.sio`:

- `ontology_validation_apply_decl_constraints(...)`
- `validate_ontology_kernel_inverse_consistency(...)`
- `validate_ontology_kernel_strengthening(...)`
- declaration diagnostics still emitted through `collect_ontology_roles(...)`

Current evidence for this layer is primarily passing fixtures, not unique failing ones:

- `ontology_inverse_role_diagnostics.sio`
- `ontology_role_duplicate_decl.sio`
- `ontology_role_bad_domain_or_range.sio`

### 3. Boundary Engine

Owns:

- binding ontology class names into compiler-visible types
- compiler call-site ontology compatibility checks
- mapping ontology kernel answers into compile-time diagnostics

Must not own:

- axiom materialization
- declaration/profile validation
- rebuilt-driver status reporting

Current anchors in `self-hosted/check/check.sio`:

- `ontology_boundary_bind_named_types(...)`
- `ontology_boundary_check_call_arg_contract(...)`
- `check_call_arg_ontology_boundary(...)`

### 4. Summary API

Owns:

- rebuilt-driver-facing scalar status only
- post-check state access that must stay inside `check.sio`

Must not own:

- ontology semantics
- authoritative debug counts for semantic interpretation

Current anchors in `self-hosted/check/check.sio`:

- `checker_check_items_inner_store_return_had_error_mut(...)`
- `checker_error_count_scalar(...)`
- `checker_warning_count_scalar(...)`
- `checker_fn_sig_count_scalar(...)`

The summary API is operationally useful for rebuilt validation, but its debug
counts are not yet authoritative semantic data.

## Current Failure Map

| Fixture | Layer | Reason |
|---|---|---|
| `ontology_direct_disjoint_reject.sio` | inference/kernel issue | direct disjointness rejection still false-passes |
| `ontology_inherited_disjoint_reject.sio` | inference/kernel issue | inherited disjointness rejection still false-passes |
| `ontology_role_domain_reject.sio` | inference/kernel issue | domain-side ontology subsumption rejection still false-passes |
| `ontology_role_range_reject.sio` | inference/kernel issue | range-side ontology subsumption rejection still false-passes |
| `ontology_subclass_reject.sio` | inference/kernel issue | negative subclass/subsumption rejection still false-passes |
| `ontology_type_mismatch.sio` | boundary/type-mapping issue | compiler boundary accepts an ontology-typed call that should be rejected |
| `test_ontology.sio` | baseline checker noise | shared rebuilt/default run failure, not isolated to ontology-kernel semantics |

At the current baseline, no remaining failing fixture is uniquely attributable to
the validation layer.

## Extraction Readiness Rule

Legacy extraction planning may proceed only if:

- rebuilt and default gates remain operational
- the failing set remains stably mapped by semantic layer
- wrapper boundaries remain behavior-preserving

Actual legacy removal stays out of scope until a later phase resolves the
kernel and boundary failures above.
