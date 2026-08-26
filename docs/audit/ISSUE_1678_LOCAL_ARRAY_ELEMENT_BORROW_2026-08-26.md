<!-- docs:meta
topic_id: repo.docs.audit.issue-1678-local-array-element-borrow-2026-08-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.issue-1678-local-array-element-borrow-2026-08-26
-->

# #1678 local array element borrows

## Scope

This wave fixes the still-live `&array[index]` parser/codegen gap for local
arrays in `lean_single` on x86-64 and AArch64. It does not claim the broader
boxed-field problem described by the earlier read-only handoff: those boxed
repros already pass on the current source-built baseline.

Base measured: `origin/main` at
`f7924106dd7d35ff62e1aceb7c3c90693f226893`.

## Live control and treatment

The source-built pre-patch compiler
(`/tmp/souc-1678-current`, SHA-256
`81c4929823460a807762abd4a878ca9adbd053313e7ac6662fa489456269a4a3`)
misparsed a closed single index as the start of a slice. The local aggregate
repro emitted both `slice borrow requires a..b range` and
`slice borrow missing closing ]`, then refused the call with `E001`.

The treatment recognizes `]` immediately after the compiled index as a
single-element borrow. It:

- resolves local arrays, array references, slice references, and global arrays;
- checks `index < len` with an unsigned comparison;
- enforces mutable provenance for `&!`;
- addresses inline scalar cells while loading aggregate pointer slots; and
- returns the element reference hash rather than a slice reference hash.

The existing range path remains selected by `..` or `..=`. The same semantic
split and representation rules are emitted by both backends.

## Causal gate

`scripts/ci/lean_single_local_array_element_borrow_gate.sh` verifies:

- x86-64 positive execution for immutable and exclusive aggregate borrows;
- scalar element borrowing from a local array and a fixed-length array ref;
- out-of-bounds refusal;
- refusal of `&!` from an immutable local array;
- preservation of the fat-slice range path plus a single-element borrow from a
  slice with runtime length;
- AArch64 ELF codegen, bounds comparison, aggregate slot load, and scalar cell
  address calculation; and
- same-source sabotage compilers with only the new x86 or AArch64 rule disabled.

Both sabotage compilers recover the exact pre-treatment `E001`, proving that
the new branch, rather than an incidental rule, admits the witness.

Gate result:

```text
lean-single-local-array-element-borrow: PASS fixed_point_sha256=455365f19b6c96506991cfac5fed3d86ca655a324567d71bc9309ae5cd2aa759 x86=EXECUTED aarch64=CODEGEN_VERIFIED aggregate_slot=LOADED scalar_cell=ADDRESSED bounds=REFUSED mutability=REFUSED slice=PRESERVED slice_element=EXECUTED sabotage_x86=E001 sabotage_a64=E001
```

The canonical harness also passed all 15 tests selected by `--filter borrow`,
including the new run-pass and compile-fail witnesses. Targeted regressions
`slice_fat_pointers.sio` and `array_elem_field_store.sio` pass.

## Seed provenance

The attempted supported Slurm refresh stopped before job submission because
this pod does not mount `/orangefs`. The Beagle profile surface independently
returned HTTP 401, so no remote job or Slurm provenance is claimed.

The documented `--local-locked` fallback completed under
`souc-build-lock.sh`. Receipt
`bin/souc-lean-single-x86_64.SeedReceipt.json` records:

- placement: `local-locked`;
- fixed point: `g1 == g2`;
- MD5: `25c8e30c247774fadafcd5e1c8a6f9e2`;
- SHA-256:
  `455365f19b6c96506991cfac5fed3d86ca655a324567d71bc9309ae5cd2aa759`;
- canonical compiler gate: pass; and
- seed verification: pass.

The forced seed-surface provenance gate accepted source, ELF, and receipt. Its
wrong-source mutant control was refused. Diverse double compilation also
converged from the independent baseline seed
`81c4929823460a807762abd4a878ca9adbd053313e7ac6662fa489456269a4a3`
to the committed fixed point.

## Orthogonal review

Grok's first read-only audit found that `type_is_array_ref` also recognizes
slices, so testing it first supplied a constant length of zero. That audit was
a genuine blocker: isolated slice element borrows aborted with rc 1. Resolving
`type_is_slice_ref` first on both backends fixed the defect.

Final read-only reruns by `grok/fleet-grok-cli2` and
`cursor/fleet-cursor-1` both passed on seed
`455365f19b6c96506991cfac5fed3d86ca655a324567d71bc9309ae5cd2aa759`.
Grok additionally checked isolated `&middle[0]` and `&middle[1]`, the forced
receipt gate, and DDC convergence. Neither reviewer edited the worktree.

## Evidence boundaries

- AArch64 was compiled and inspected, not executed on AArch64 hardware.
- Aggregate repeat initialization remains a separate surface. The positive
  witness assigns each aggregate slot before borrowing it.
- Scalar `(*p) = value` store behavior is separate; the exclusive aggregate
  witness mutates a field through `&!Cell`.
- The boxed repros from the earlier `f8af3e7bdd` handoff were green on the live
  pre-treatment baseline and are not claimed as fixed by this wave.
