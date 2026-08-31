# Garden: Sovereign Change Kernel V1

Action: 9043

Concept-ID: `SOUNIO-LOOM-SOVEREIGN-CHANGE-KERNEL`

## Question

Can LOOM authorize and apply one exact source mutation without exporting write
authority, bind the resulting Git commit byte-for-byte to that mutation, and
let CI admit the commit by consuming a Sounio decision rather than recreating
its meaning?

## Ordered Experiment

`GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> OPERATIONAL_ATTACHMENT -> PRODUCT_GATE`

The Sounio executable and its expected decisions must exist before the OCaml
kernel receives any ChangeGrant logic. The operational layer may measure hashes
and enforce a frozen decision. It may not invent a successful result or promote
a receipt produced by another language to semantic authority.

## ChangeIntent

One canonical intent binds all of the following before a mutation:

- structured operation kind: `Write`, `Edit`, or `apply_patch`;
- exact raw hook-event digest and exact patch digest;
- worktree identity, `HEAD`, index tree, and pre-change worktree digest;
- canonical non-empty set of repository-relative target paths;
- authenticated peer identity inherited from action 9042;
- frozen action-9042 semantic and product hashes.

No wildcard, directory prefix, transport token, file descriptor, or textual
grant identifier is write authority.

## Ordered States

1. `CHANGE_PREPARED`: an in-memory, non-bearer, single-use ChangeGrant exists
   for one canonical ChangeIntent. No file has changed.
2. `CHANGE_CONSUMED`: the post-image exactly matches the authorized mutation,
   no ungranted path changed, and the grant was consumed atomically once.
3. `COMMIT_ADMIT`: the Git index, tree, parent, message, and complete consumed
   change set match the authorized post-image. A receipt is issued.
4. `CI_ADMIT`: CI verifies the receipt, commit, tree, action-9043 semantics, and
   authority binding. CI does not recompute or reinterpret the policy result.
5. `PRODUCTION_GATE_READY`: Write, Edit, apply_patch, commit, and CI are all
   attached and `claim_ready=true` is justified by the preceding states.

Later-state facts are forbidden in earlier modes. A prepared grant cannot be
presented as consumed, a consumed grant cannot be presented as committed, and
a commit receipt cannot be presented as CI admission.

## Causal Controls

- Change one byte of the patch after preparation: refuse before commit.
- Change one target path while preserving the patch digest field: refuse.
- Modify an ungranted file between preparation and consumption: refuse.
- Replay a consumed ChangeIntent: refuse without a second filesystem effect.
- Change the index after consumption but before commit: refuse.
- Change the commit tree, parent, or message after receipt construction: refuse.
- Present a valid receipt for a different commit or semantic freeze: CI refuses.
- Make CI recompute an ALLOW from raw facts instead of consuming the receipt:
  refuse the control witness.
- Flip only the Sounio no-reinterpretation rule: the unchanged negative CI
  witness must become admitted, proving the rule is load-bearing.
- Python and Rust oracle traps must remain unexecuted.

## Receipt Boundary

The receipt binds the Sounio source and frozen semantics, producing language
and role, toolchain, hardware, command, ChangeIntent, patch, file set, pre-state,
post-state, commit, tree, parent, message, result, and receipt authority. CI may
verify these bindings and the receipt authority. It may not substitute a local
policy engine or infer a different expected result.

## Exit Claim

The experiment may set `write_attached=true`, `commit_attached=true`,
`ci_attached=true`, and `claim_ready=true` only after the installed product path
passes all treatment, replay, path-confusion, post-image, commit-drift, receipt,
CI-reinterpretation, Python-oracle, and Rust-oracle controls.
