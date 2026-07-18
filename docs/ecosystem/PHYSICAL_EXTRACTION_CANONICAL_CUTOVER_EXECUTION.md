<!-- docs:meta
topic_id: repo.docs.ecosystem.physical-extraction-canonical-cutover-execution
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ecosystem.physical-extraction-canonical-cutover-execution
-->

# Physical Extraction Canonical Cutover Execution

Status: executable R3 policy-bound canonical Git cutover interface; exercised only in disposable fixtures for this repository.

`tools/science_boundary/canonical_cutover_executor.py` consumes one fully
reconstructable `canonical-cutover-approval` receipt and a separately authored
`sounio.physical-extraction-canonical-cutover-execution-policy.v1`. The policy
pre-binds the exact post-cutover Git tree and commit. The executor removes only
the approved source roots, applies only approved repairs, creates that exact
commit, advances the bound local branch with compare-and-swap, publishes the
same commit to the bound remote branch with an exact old-ref lease, and emits a
separate deterministic execution receipt.

The accepted receipt states:

```text
canonical_cutover_approval_status = consumed
canonical_cutover_execution_status = executed-and-verified
source_removal_status = executed
assurance_level = identity-plus-git-remote-ref-and-published-commit
```

This interface performs real file removal and Git ref updates in its bound
repository. The acceptance gate therefore uses only temporary standalone Git
repositories and local bare remotes. No `canonical-production` execution
policy, execution receipt, destination repository set, or human cutover
decision exists for the Sounio canonical repository.

## Separate Execution Decision

The earlier approval receipt remains `approved-not-executed`. It is necessary
but is not sufficient to execute. The execution policy separately binds:

1. Exact approval receipt bytes and identity, cutover-policy bytes and
   identity, authorization and materialization identities, pre/post regular
   file trees, removal scope, destination set, and recovery-plan identity.
2. The approval context, which must remain exactly `disposable-fixture` or
   `canonical-production`; one context cannot be renamed into the other.
3. Canonical repository ID, remote name and URL, branch, pre-cutover local and
   remote object IDs, expected post-cutover Git tree, and expected commit.
4. Deterministic commit author, committer, dates, message, local compare-and-
   swap strategy, and exact remote lease strategy.
5. Retained execution-decision evidence plus confirmations of the approval,
   old local and remote refs, authorized post tree, expected commit,
   destination set, and recovery plan.
6. Five new CLI confirmations for the cutover approval, execution policy,
   pre-cutover head, expected commit, and execution context. The four approval
   confirmations are repeated independently to reconstruct the earlier
   receipt.

The expected Git transition is recomputed before mutation in an isolated bare
object repository and alternate worktree/index. It preserves the actual modes
of retained files and replacement files, so the policy binds the exact Git
tree in addition to the regular-file content tree.

## Execute

```bash
python3 tools/science_boundary/canonical_cutover_executor.py execute \
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
  --cutover-approval-receipt /approved/canonical-cutover-approval.json \
  --execution-policy /approved/canonical-cutover-execution-policy.json \
  --workspace-root /same-filesystem/transaction-workspace \
  --execution-receipt /external/canonical-cutover-execution.json \
  --confirm-authorization-identity <authorization-sha256> \
  --confirm-scope-identity <scope-sha256> \
  --confirm-policy-identity <cutover-policy-sha256> \
  --confirm-pre-cutover-tree <regular-file-tree-sha256> \
  --confirm-cutover-approval-identity <approval-sha256> \
  --confirm-execution-policy-identity <execution-policy-sha256> \
  --confirm-pre-cutover-head <git-object-id> \
  --confirm-expected-cutover-commit <git-object-id> \
  --confirm-execution-context canonical-production
```

There is no default canonical repository, repository set, policy, transaction
workspace, receipt, context, or confirmation. The final receipt must be absent
and outside every source, materialization, destination-repository, and
transaction root.

## Transaction And Git Semantics

The executor locks the canonical-root directory inode and reconstructs the
complete approval, including its disposable rehearsal, while all source files
still exist. It then:

1. validates the separate execution policy and recomputes its exact Git tree
   and commit in isolation;
2. verifies retained marker, operator, destination-owner, recovery, and
   execution-decision evidence;
3. creates a full regular-file backup with modes in a preexisting external
   same-filesystem transaction workspace;
4. reconstructs every approval and execution input again after the backup;
5. removes only the approved roots and applies only approved replacement bytes;
6. compares the complete regular-file tree and reruns the exact approved gates;
7. revalidates destination copies and standalone destination Git worktrees;
8. stages the deterministic final receipt without promoting it;
9. creates the pre-bound commit with Git plumbing and advances the local ref by
   compare-and-swap;
10. advances the remote branch from the exact old object to the exact new object
    with `--force-with-lease`, then verifies local `HEAD`, remote ref, commit
    tree, parent, worktree cleanliness, destinations, and retained evidence;
11. promotes the receipt by hardlink as the final accepted state.

Git hooks are bypassed for the ref publication. The command uses no implicit
push target and no unconstrained force update.

## Failure And Recovery

An ordinary failure before receipt promotion is not accepted execution. The
executor observes the actual remote and local refs rather than trusting whether
the Git client reported success. When a ref equals the expected new commit, it
first attempts an exact-lease remote rollback, then restores the local ref,
regular files, modes, and index to the bound pre-cutover state and verifies the
complete standalone Git repository again.

If the remote is neither the old nor expected new object, if the exact lease
rollback fails, or if the tree cannot be restored exactly, the transaction is
retained and the command refuses with `E-SRB-CUTOVER-EXEC-007` for manual
recovery. It does not overwrite an occupied receipt.

The remote ref update and local hardlink receipt promotion cannot be one atomic
transaction. A process or host crash after the remote accepts the commit but
before receipt promotion may leave the remote at the new object without a
valid receipt. The bound recovery plan, transaction workspace, old/new object
IDs, and manual review are required in that state. The receipt does not claim
otherwise.

## Verify

After execution, the removed source files are intentionally unavailable for a
full approval replay. Verification instead checks the immutable approval,
cutover-policy, authorization, materialization, and execution-policy bytes and
identities; exact post-cutover Git commit and remote ref; retained evidence;
destination repository states and content; absent removal roots; repair bytes;
and the complete regular-file tree. Post-removal gates run only on a disposable
copy.

```bash
python3 tools/science_boundary/canonical_cutover_executor.py verify \
  --repo-root /executed/canonical-worktree \
  --destinations-root /approved/materialized-copies \
  --materialization-receipt /approved/materialization.json \
  --authorization-receipt /approved/source-removal-authorization.json \
  --repositories-root /approved/destination-worktrees \
  --cutover-policy /approved/canonical-cutover-policy.json \
  --cutover-approval-receipt /approved/canonical-cutover-approval.json \
  --execution-policy /approved/canonical-cutover-execution-policy.json \
  --workspace-root /same-filesystem/verification-workspace \
  --execution-receipt /external/canonical-cutover-execution.json \
  --confirm-authorization-identity <authorization-sha256> \
  --confirm-scope-identity <scope-sha256> \
  --confirm-policy-identity <cutover-policy-sha256> \
  --confirm-pre-cutover-tree <regular-file-tree-sha256> \
  --confirm-cutover-approval-identity <approval-sha256> \
  --confirm-execution-policy-identity <execution-policy-sha256> \
  --confirm-pre-cutover-head <git-object-id> \
  --confirm-expected-cutover-commit <git-object-id> \
  --confirm-execution-context canonical-production
```

Changed or reintroduced source content, a dirty worktree, changed local or
remote ref, changed destination content/ref, altered retained evidence,
incorrect confirmation, mutating verification gate, and forged or rehashed
receipt all refuse.

## Evidence Boundary

A passing execution receipt establishes that one exact approved Git transition
was applied, committed, observed at the bound remote branch, and verified with
the bound destination set and post-removal gates.

It does not establish:

- hosting administration, namespace ownership, transferred maintainership, or
  a default-branch setting outside the bound ref;
- that a `disposable-fixture` execution is a production cutover;
- human identity, organizational authority, or reviewer independence;
- distributed crash atomicity or independent replay;
- complete filesystem metadata preservation outside regular-file content,
  retained/replacement modes, and the Git tree;
- scientific truth, clinical validation, or clinical authority.

## Acceptance Gate

The focused gate is
`scripts/ci/physical_extraction_canonical_cutover_execution_gate.py`. It builds
standalone source/destination worktrees and bare remotes only under temporary
directories. Two equivalent physical roots produce byte-identical policies,
approvals, expected commits, and execution receipts. The gate exercises real
root removal, deterministic Git commit creation, local and remote ref updates,
post-execution verification, stale/dirty/tampered-state refusal, rollback after
a canonical-only mutating gate, and rollback of both remote and local refs when
a concurrent process occupies the receipt after staging and push.

The focused disposable fixture currently passes 81 assertions. Its execution
identity is
`f7ea56e8028f1a21f6f23afd316bfb93e6a27416067442fdf0da63f28e064d21`,
policy identity is
`e55de2ce7f6d82e57c7408cb1fc95948deb7fdc2c36da611ce6ebba1455a406c`,
and expected commit is `789611457cc681226baa2885391d7bbbd29a5fa7`.
These identify a disposable fixture, not the Sounio canonical repository.

The composed shell gate first runs R0-R2, R2.5, R2.6, inventory,
materialization, source-removal authorization, local execution, and canonical
cutover approval gates with one current-source Madaros witness.

The composed current-source witness is Slurm job `6613` on
`gpuorangefs-r770-proxmox`. Commit
`002d5f2277da8f9510b37f8e4d0ac8e9e994a06f`, compressed source archive
`16a7cfdddd120cbd47a0b471506126fe3724fd6b1e9e27b772ce9ab73245c642`
(339676469 bytes), current-source Madaros
`6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88`,
and Git 2.43.0 passed 178 R0-R2, 65 R2.5, 82 R2.6, 141 inventory, 167
materialization, 527 authorization, 164 local execution, 172 cutover approval,
and 81 cutover execution checks in 53 seconds with `MaxRSS=1390324K`.
Stdout is 3065 bytes with SHA-256
`06e16b40fd59d757c734e289388202afc1b55a3e786505839c302e41f3d25a53`;
stderr is empty with SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The immutable inputs, extraction, transaction, fixture repositories, bare
remotes, and logs used node-local `/tmp` because OrangeFS was full. No
implementation fallback ran, and the only changed refs and removed roots were
inside the disposable fixtures created by the gate.

## Canonical Repository Status

No `canonical-production` cutover approval or execution policy was created for
Sounio. No real destination repository set, expected production commit,
operator execution decision, or execution receipt is asserted. Every canonical
scientific-package and research source root remains present. The remaining
permission-bearing step is a separately authored production evidence set and
an explicit human decision; this fixture interface is not that decision.

The non-authorizing next step is documented in
`PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT.md`. It observes the
current repository catalog and exact planned targets without creating a mapping
proposal, production policy, approval, decision, repository, or ref update.
