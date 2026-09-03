<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-source-removal-execution
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-source-removal-execution
-->

# Physical Extraction Source-Removal Execution

Status: executable R3 policy-bound local execution interface; canonical repository cutover is not executed.

## Scope

The execution interface consumes one fully reconstructed
`sounio.physical-extraction-source-removal-authorization.v1` and a separately
authored `sounio.physical-extraction-source-removal-execution-policy.v1`. It
executes the exact authorized removals and repairs against one explicitly
marked local repository tree, reruns the authorized post-removal gates,
verifies the materialized copies, and emits one deterministic
`sounio.physical-extraction-source-removal-execution.v1` receipt.

The accepted receipt states exactly:

```text
execution_type = policy-bound-local-source-removal
execution_status = executed-and-verified
source_removal_status = executed
assurance_level = identity-only
```

This interface performs real filesystem removal in its bound execution root.
Its fixture gate therefore uses disposable temporary repositories. The Sounio
canonical repository has no production execution policy or receipt and is not
an execution target of this work.

## Required Authority

Execution requires all of the following:

1. The original inventory, materialization, removal policy and source tree
   still reconstruct the supplied authorization exactly before mutation.
2. Every materialized destination still contains the exact planned regular
   files and marker bound by that authorization.
3. A closed execution policy binds the authorization file and identity,
   materialization file and identity, inventory identity, exact pre-execution
   tree, exact authorized post-execution tree and complete removal scope.
4. The execution root contains a retained byte-bound approval marker and at
   least one retained operator-approval evidence file.
5. The policy records exact authorization, scope and pre-execution-tree
   confirmations. Distinct labels and files are identity evidence only; they
   do not prove a person's identity or organizational authority.
6. The CLI repeats the authorization identity, scope identity, policy identity
   and pre-execution tree identity exactly. A missing or incorrect confirmation
   refuses before creating a transaction backup.
7. The external transaction workspace is preexisting, separate from sources
   and destinations, and on the same filesystem as the execution root. The
   final receipt is absent and outside the execution root, destinations, and
   transaction workspace.
8. The operator has quiesced nonparticipating writers. Executors using this
   interface serialize on the execution-root inode, and the root is checked
   again after backup, but unrelated processes do not participate in that lock.

The execution policy identity is a deterministic change detector, not a
signature or production approval. A fixture policy cannot be widened into
canonical authority by copying or renaming it.

## Execute

```bash
python3 tools/science_boundary/source_removal_executor.py execute \
  --repo-root /explicitly-approved/repository-tree \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --inventory /approved/extraction-inventory.json \
  --destination-policy /approved/destination-policy.json \
  --destinations-root /approved/destinations \
  --materialization-receipt /approved/materialization-receipt.json \
  --removal-policy /approved/source-removal-policy.json \
  --authorization-receipt /approved/source-removal-authorization.json \
  --execution-policy /approved/source-removal-execution-policy.json \
  --workspace-root /same-filesystem/transaction-workspace \
  --execution-receipt /approved/source-removal-execution.json \
  --confirm-authorization-identity <authorization-sha256> \
  --confirm-scope-identity <scope-sha256> \
  --confirm-policy-identity <execution-policy-sha256> \
  --confirm-pre-execution-tree <source-tree-sha256>
```

There is no implicit default execution root, policy, receipt or confirmation.

## Transaction Semantics

Before the first removal, the executor locks the execution-root inode, copies
every regular source file outside `.git` into a transaction directory, verifies
that full backup against the authorized pre-execution tree, and checks the root
again. It then:

1. removes only the authorized root directories;
2. writes only the authorized repair bytes, using the verified backup as the
   replacement source;
3. compares the complete regular-file tree with the authorized candidate;
4. runs the authorization's exact post-removal command and output contracts;
5. revalidates destinations, authorization, materialization, execution policy,
   marker and operator evidence;
6. promotes the deterministic execution receipt last, with hardlink creation as
   the commit point; later staging cleanup or directory-sync errors are reported
   as committed-state warnings and do not trigger rollback.

Any ordinary failure before receipt promotion restores the complete
pre-execution regular-file tree from backup and verifies the rollback. A
rollback failure retains the transaction path and refuses with
`E-SRB-EXEC-007` for manual recovery.

Several removals, repairs and receipt promotion cannot form one
crash-atomic filesystem transaction. Interruption without a final valid receipt
is not accepted execution; the retained transaction workspace is the recovery
basis. The receipt deliberately does not claim preservation of empty
directories, modes, timestamps, ACLs, extended attributes or other metadata
outside the regular-file identity model.

## Verify

After execution, the removed source files are intentionally unavailable for a
full authorization replay. Verification therefore checks the immutable
authorization and execution-policy identities, the exact retained execution
tree, the separately materialized planned files and markers, retained approval
evidence, absent removal roots, repair bytes, post-removal gates and the exact
execution receipt. Gates run on a full disposable copy in the verification
workspace; a mutating verification gate refuses without modifying the executed
tree.

```bash
python3 tools/science_boundary/source_removal_executor.py verify \
  --repo-root /executed/repository-tree \
  --destinations-root /approved/destinations \
  --materialization-receipt /approved/materialization-receipt.json \
  --authorization-receipt /approved/source-removal-authorization.json \
  --execution-policy /approved/source-removal-execution-policy.json \
  --workspace-root /same-filesystem/verification-workspace \
  --execution-receipt /approved/source-removal-execution.json \
  --confirm-authorization-identity <authorization-sha256> \
  --confirm-scope-identity <scope-sha256> \
  --confirm-policy-identity <execution-policy-sha256> \
  --confirm-pre-execution-tree <source-tree-sha256>
```

Added, removed, reintroduced, mutated or symlinked source members; changed
materialized content; changed approval evidence; failing gates; and forged or
rehashed receipts refuse verification.

## Evidence Boundary

A passing execution receipt establishes that one exact local repository tree
under one exact execution policy reached the authorized removed-and-repaired
tree, retained exact materialized copies, and passed the declared gates.

It does not establish:

- that the executed tree is the canonical Sounio production repository;
- remote repository creation, commit, push, default-branch or publication
  state;
- ownership, maintainership, legal or organizational authority;
- complete environment capture, crash atomicity or independent replay;
- scientific truth, clinical validation or clinical authority.

The focused gate is
`scripts/ci/physical_extraction_source_removal_execution_gate.py`. It executes
only disposable temporary roots, proves deterministic receipts across two
equivalent roots, root-inode serialization, isolated verification gates, and
exact rollback after an execution-only gate mutation. Its composed shell gate
first passes R0-R2, R2.5, R2.6, inventory, materialization and authorization
against the same source and current-source Madaros witness.

The current composed fixture witness is Slurm job `6558` on
`gpuorangefs-r770-proxmox`. One archived source snapshot and one current-source
Madaros passed 178 R0-R2, 65 R2.5, 82 R2.6, 141 inventory, 167 materialization,
527 authorization, and 164 execution checks in 45 seconds. The execution
identity is
`682791965ae6f553f87faf3a77fea395af71c804dfd42ca76c073ea828b803ba`;
the policy identity is
`913a5d53e4b1a5061216b00c9d8a8810534004d5888b2cf8b0cbd21cf177955b`.
These are deterministic disposable-fixture identities, not a canonical
cutover approval or receipt.

## Canonical Repository Status

No production execution-root marker, operator approval, execution policy or
execution receipt exists for the canonical Sounio repository. All canonical
scientific-package and research source roots remain present. The executable
fixture is not permission to remove them.

## Downstream Interfaces

`r3-physical-extraction-canonical-cutover-approval` now binds exact source and
destination Git worktrees and remote branch refs, reviewed repairs and gates,
operator evidence, and a recovery procedure while rehearsing only on a
disposable copy. Its receipt remains `approved-not-executed`.

`r3-physical-extraction-canonical-cutover-execution` now consumes that approval
plus a separate policy that pre-binds the exact Git tree and commit. It performs
real file removal and local/remote ref updates only in disposable acceptance
fixtures. Neither interface creates production evidence or executes the Sounio
canonical repository.
