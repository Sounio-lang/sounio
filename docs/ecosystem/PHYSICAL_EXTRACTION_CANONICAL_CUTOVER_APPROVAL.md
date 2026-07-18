<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-canonical-cutover-approval
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-canonical-cutover-approval
-->

# Physical Extraction Canonical Cutover Approval

Status: executable R3 Git-state and rehearsal approval interface; canonical repository cutover is not executed.

`tools/science_boundary/canonical_cutover_authorizer.py` consumes the complete
R3 inventory, materialization, removal policy and source-removal authorization
chain plus a separately authored
`sounio.physical-extraction-canonical-cutover-policy.v1`. It emits a
deterministic
`sounio.physical-extraction-canonical-cutover-approval.v1` receipt only after
the exact source and destination repository states and a reversible cutover
rehearsal pass.

This interface authorizes one bound state. It has no operation that removes a
canonical source root. Its exact statuses are:

```text
canonical_cutover_approval_status = approved-not-executed
canonical_cutover_execution_status = not-executed
source_removal_status = not-executed
assurance_level = identity-plus-git-remote-ref
```

## Bound Evidence

The cutover policy binds all of the following:

1. Exact authorization and materialization file and receipt identities.
2. Inventory, pre-cutover tree, authorized post-cutover tree, removal scope,
   repair-set and gate-set identities.
3. The canonical worktree repository ID, retained marker, branch, `HEAD`,
   configured remote URL and observed remote branch object ID.
4. One exact destination worktree for every materialized unit, including
   target ownership metadata, complete regular-file tree, branch, `HEAD`,
   configured remote URL, observed remote branch object ID and retained owner
   approval evidence.
5. A repository-retained recovery plan that names the required full backup,
   external same-filesystem transaction workspace, receipt commit point and
   recovery behavior before and after receipt promotion.
6. Operator evidence and exact confirmations for authorization, scope,
   pre/post trees, destination set, recovery plan, repairs and gates.
7. Four repeated CLI confirmations for the authorization, scope, cutover
   policy and pre-cutover tree identities.

Both the canonical root and each destination must be the root of a clean,
standalone Git worktree with its own `.git` directory and repository format 0;
linked worktrees, submodules, bare repositories and shared common directories
refuse. The local `HEAD` must equal the policy object ID and the object ID
returned by `git ls-remote --heads` for the bound branch. The configured remote
URL must match the policy exactly. Destination repository IDs and checkout
paths are unique, the destination repository root has no extra members, and
every destination file must equal the materialization receipt.

Remote-ref observation establishes only that the configured Git transport
reported the bound branch and object ID at evaluation time. It does not prove
hosting administration, namespace ownership, maintainership or organizational
authority.

## Authorize

```bash
python3 tools/science_boundary/canonical_cutover_authorizer.py authorize \
  --repo-root /exact/canonical-worktree \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --inventory /approved/extraction-inventory.json \
  --destination-policy /approved/destination-policy.json \
  --destinations-root /approved/materialized-copies \
  --materialization-receipt /approved/materialization.json \
  --removal-policy /approved/source-removal-policy.json \
  --authorization-receipt /approved/source-removal-authorization.json \
  --repositories-root /approved/destination-worktrees \
  --cutover-policy /approved/canonical-cutover-policy.json \
  --workspace-root /external/rehearsal-workspace \
  --cutover-approval-receipt /external/canonical-cutover-approval.json \
  --confirm-authorization-identity <authorization-sha256> \
  --confirm-scope-identity <scope-sha256> \
  --confirm-policy-identity <cutover-policy-sha256> \
  --confirm-pre-cutover-tree <source-tree-sha256>
```

There is no default canonical root, repository set, policy, workspace,
receipt or confirmation. The receipt must remain outside the source,
materialized-copy, destination-repository and rehearsal roots.

## Rehearsal And Promotion

The authorizer locks the canonical root directory inode and reconstructs the
complete source-removal authorization while the source still exists. It then:

1. revalidates every materialized copy;
2. verifies the exact canonical and destination Git states and retained
   evidence;
3. copies every regular canonical source file into both a disposable candidate
   and a full rehearsal backup;
4. removes only the authorized roots from the candidate;
5. applies only the byte-bound repairs and runs the exact approved gates;
6. verifies the candidate against the authorized post-cutover tree;
7. restores the candidate from the full backup and verifies the exact
   pre-cutover tree;
8. reconstructs the source chain and revalidates the policy, Git refs,
   destinations and retained evidence again;
9. promotes the deterministic approval receipt by hardlink while the canonical
   root lock is still held.

The receipt records the four CLI values actually supplied to the successful
evaluation and marks their comparison `matched`; verification reconstructs the
same object from its own required CLI values.

All removal, repair and gate effects occur under the disposable rehearsal
directory. Failure deletes that directory and leaves the canonical root and
destination worktrees unchanged. An occupied output is never overwritten.

## Verify

Verification replays the same reconstruction, Git checks, destination checks,
rehearsal, restoration and four CLI confirmations because the source remains
present:

```bash
python3 tools/science_boundary/canonical_cutover_authorizer.py verify \
  --repo-root /exact/canonical-worktree \
  --rings science-rings.tsv \
  --ownership docs/ecosystem/science-physical-extraction-ownership.tsv \
  --inventory /approved/extraction-inventory.json \
  --destination-policy /approved/destination-policy.json \
  --destinations-root /approved/materialized-copies \
  --materialization-receipt /approved/materialization.json \
  --removal-policy /approved/source-removal-policy.json \
  --authorization-receipt /approved/source-removal-authorization.json \
  --repositories-root /approved/destination-worktrees \
  --cutover-policy /approved/canonical-cutover-policy.json \
  --workspace-root /external/rehearsal-workspace \
  --cutover-approval-receipt /external/canonical-cutover-approval.json \
  --confirm-authorization-identity <authorization-sha256> \
  --confirm-scope-identity <scope-sha256> \
  --confirm-policy-identity <cutover-policy-sha256> \
  --confirm-pre-cutover-tree <source-tree-sha256>
```

Any changed source receipt, tree, policy, marker, approval record, recovery
plan, destination content, worktree state, local `HEAD`, configured remote URL
or observed remote branch refuses. A receipt remains invalid when a forged
field is followed by recomputing its JSON identity.

## Evidence Boundary

A passing receipt establishes that, for one exact policy evaluation:

- the full earlier R3 authorization chain reconstructed;
- the bound canonical and destination worktrees were clean;
- their local `HEAD` values equaled their observed bound remote branch refs;
- destination content equaled the materialized units;
- the approved removal, repairs and gates passed on a disposable copy; and
- that copy restored exactly to the pre-cutover regular-file tree.

It does not establish or perform:

- canonical cutover execution or canonical source removal;
- production approval when `approval_context = disposable-fixture`;
- hosting administration, repository namespace ownership or transferred
  maintainership;
- human identity, organizational authority or reviewer independence;
- complete production environment capture or crash-atomic multi-file cutover;
- an independent signature or independent replay;
- scientific truth, clinical validation or clinical authority.

The v1 Git binding accepts the 40-hex object IDs observed from SHA-1 Git
repositories. Those IDs and all SHA-256 JSON/file identities are deterministic
change detectors, not independent signatures.

## Acceptance Gate

The focused gate is
`scripts/ci/physical_extraction_canonical_cutover_approval_gate.py`. It builds
bare remotes and complete source and destination worktrees only under temporary
fixture roots. It proves byte-identical policies and receipts across two
equivalent physical roots, exact Git/ref and content refusal, occupied-output
preservation, receipt reconstruction, and confinement of a mutating rehearsal
gate. The composed shell gate first runs R0-R2, R2.5, R2.6, inventory,
materialization, authorization and local execution gates with the same
current-source Madaros witness.

The focused fixture currently passes 172 assertions. Its deterministic
approval identity is
`15e8b3ad7b0b01a95c5a3ad717176d8901f5941740d9c9c52680a70e293074a9`;
its policy identity is
`b12ae97b10691cc7ef8b77c3ec03b620304b8be4be8414f2af40f0d3ae6da6be`.
These identify a disposable Git fixture, not the canonical Sounio repository.

The composed current-source witness is Slurm job `6602` on
`gpuorangefs-r770-proxmox`. Commit
`851c9eba290294135c4921ae9b2475ade889ab79`, compressed source archive
`c9556d94e8c50ef200cb6fcb9ea60fbc26a45e091a8de4960718fb140e0c9273`
and current-source Madaros
`6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88`
passed 178 R0-R2, 65 R2.5, 82 R2.6, 141 inventory, 167 materialization,
527 authorization, 164 local execution and 172 cutover approval checks in 44
seconds with `MaxRSS=1389440K`. Stdout SHA-256 is
`a8985fd6e21bd47428f109b6f61895c4108371417e8c191d9c9d67378c0b15f6`;
stderr is empty.

The source, compiler, fixture workspaces and logs used node-local `/tmp`
because OrangeFS was full. Job `6600` stopped before all gates because the
worker image lacked `/usr/bin/time`; job `6601` passed the complete earlier
stack and stopped before this focused gate because the worker image lacked
Git. Job `6602` used the same immutable archive and compiler after provisioning
Git 2.43.0 in that ephemeral worker and passed. Neither earlier failure entered
the cutover approval tool or required an implementation fallback.

## Canonical Repository Status

No `canonical-production` cutover policy or approval receipt was created for
the Sounio repository. No production destination repository set, operator
decision or recovery plan is asserted by the disposable fixture. Every
canonical scientific-package and research source root remains present.

## Downstream Interface

`r3-physical-extraction-canonical-cutover-execution` is implemented in
`tools/science_boundary/canonical_cutover_executor.py`. It consumes this
approval plus a separate execution policy, revalidates every bound Git and
source state, pre-binds the exact Git tree and commit, and can execute the
cutover under the bound recovery procedure. Its acceptance gate performs real
removal and remote-ref updates only in disposable fixtures.

This approval receipt is still not execution permission. Sounio has no
`canonical-production` approval, execution policy, destination repository set,
human execution decision, or canonical execution receipt.
