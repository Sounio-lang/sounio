<!-- docs:meta
topic_id: repo.docs.evidence.call-arg-release-stack.implicit-native-followup
authority: repo_only
audience: users
last_validated: 2026-11-01
validated_by: Slurm 20996.4 targeted stack/borrow gate; see recorded per-test results
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.call-arg-release-stack.implicit-native-followup
-->

# Independent native implicit-reference defect

This is a pre-existing lowering defect, outside the call-argument release
traversal change. It remains open; no native implicit-reference success is
claimed by the stack repair.

Minimal witness:
`tests/fixtures/call_arg_release_stack/implicit_native_known_failure.sio`.

Expected: `read(x)` for `read(x: &i64)` preserves the checker-accepted implicit
borrow and returns 7; the program exits 0.
Actual: both baseline 9b45b9fc... and candidate a08c5660... check and build
successfully, then the generated native executable exits 139 at 8 MiB.
Changing only the call to `read(&x)` makes both execute with exit 0.
A 64-reference explicit-call control also executes with exit 0 in both.
See `implicit-native-comparison.txt`.

The emitted implicit-one executable loads the scalar stack slot into RDI at
0x4010c2 and calls read at 0x4010c9. read dereferences it at 0x40103a. The scalar
value 7 is therefore used as an address.

Localisation: `self-hosted/ir/lower.sio::lower_expr_args_masked_borrow_ref`
receives `ref_struct_param_mask` from the call path and synthesises OpRef only
when `lower_arg_is_struct_value_to_borrow_ref` accepts the argument. That
predicate intentionally recognises struct values, not scalar locals. The
checker accepts scalar implicit loans, while this native path does not
materialise the required scalar address.

Reason for separate follow-up under AGENTS.md: repairing scalar implicit-reference
lowering changes a different subsystem and its ABI/value-address rules; it
clearly exceeds this narrow traversal patch. lower.sio also has an active claim
in another lane. No changes to that file are made here.

The acceptance gate checks implicit shared/exclusive calls (including 64 distinct
places and consecutive calls) in checker mode. Native execution coverage uses
explicit references. The original over-broad gate failure and its source remain
preserved under `results/gate` in the worker directory. It was investigated,
not reported as a passing runtime test.

The separate 65-argument diagnostic follow-up also remains open: check succeeds,
build returns 1 with a generic IR lowering refusal. Stack safety is not a claim
that this diagnostic has been improved.
