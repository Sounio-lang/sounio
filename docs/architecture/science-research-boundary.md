<!-- docs:meta
topic_id: repo.docs.architecture.science-research-boundary
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.science-research-boundary
-->

# Science and Research Boundary

Status: executable R0-R2 boundary with an R2.5 local package-release boundary,
an R2.6 local registry-attestation policy contract, and R3 physical extraction
inventory, local exact-copy materialization, temporary-copy source-removal
authorization, policy-bound local source-removal execution, and exact Git-state
canonical-cutover approval and execution interfaces; the Sounio canonical
repository cutover remains not executed. Promotion is
bound to the transitive raw-AST witness in
`scripts/ci/science_boundary_gate.sh`, the package gate in
`scripts/ci/package_boundary_release_gate.sh`, the attestation gate in
`scripts/ci/registry_attestation_spec_gate.sh`, and a current-source Madaros.
R3 means moving scientific-package and research sources into separately owned
repositories or distributions. The inventory binds proposed ownership and
exact file identity. The materializer can copy approved units to preexisting
local destinations and verify them, but the canonical repository has no
approved production destination policy, materialization receipt, removal
policy, authorization receipt, execution policy, or execution receipt and
moves no source file. R2.6 launches no registry service.

The enforced dependency direction is:

```text
Sounio Research -> Scientific Packages -> Sounio PL Core
```

The direction is a software authority boundary. It does not assert scientific
truth, clinical validity, regulatory readiness, a security sandbox, or public
registry status.

## Rings

The conclusive rings are `pl-core`, `scientific-package`, and `research`.
`scientific-package-candidate`, `mixed-unresolved`, and `unclassified` remain
visible in receipts but produce `UNKNOWN`. Strict mode refuses `UNKNOWN`.

Allowed dependencies are:

```text
pl-core            -> pl-core
scientific-package -> scientific-package | pl-core
research           -> research | scientific-package | pl-core
```

A `public` caller cannot depend on a `protected` or `embargoed` module.

The graph is the transitive module-import closure rooted at the compiled
artifact. Each source module receives the most-specific matching policy path;
every resolved import is an edge. A cycle is finite when all participating
modules have already been visited. Saturation means the collector reached its
declared node or edge capacity before closure completed. Root escape means a
source, policy row, claim, or evidence path resolves outside the manifest root.
Unresolved imports, saturation, parser failure, and root escape are incomplete
authority and therefore cannot produce `OK`.

## CLI

The public `souc` and Madaros launchers accept:

```text
--science-boundary off|advisory|strict
--science-manifest <path>
--claim-contract <path>
--emit-boundary-receipt <path>
--release-bundle <path>
```

Without a scientific declaration the effective mode is `off`. Discovery of a
`sounio.toml` `[science]` declaration selects `advisory`. Only an explicit flag
selects `strict`.

In strict native builds the launcher compiles to a temporary file. It promotes
the ELF only after an `OK` preflight, unchanged source/policy/compiler hashes,
and successful receipt finalization. Refusal, `UNKNOWN`, closure saturation,
root escape, or hashing failure leaves the requested final ELF absent.

An `OK` verdict requires `SOUNIO_BOUNDARY_CLOSURE_V1` from the current-source
raw Madaros AST collector. The launcher detects and invokes that interface
before lowering. A stale raw ELF without the interface cannot silently fall
back: the host syntax audit is recorded as non-authoritative and yields
`UNKNOWN`. Rebuild with `make build-madaros` or point `MADAROS_RAW_BIN` at a
current-source ELF before using strict mode.

## R2.5 Package Release Bundle

`souc pkg build <project> --science-boundary strict --claim-contract <path>`
is the opt-in package release path. It requires the package's own
`sounio.toml` as the policy root and emits a deterministic
`sounio.package-release-bundle.v1` directory. The default destination is
`target/release/<name>-<version>.sio-release`; `--release-bundle` can select a
different final directory.

The launcher first builds into a sibling staging directory. Promotion requires
all of the following:

1. strict receipt verdict `OK`;
2. `madaros-raw-ast-v1` closure with no saturation or unresolved imports;
3. a claim contract authorized by the root ring and bound to verified content;
4. revalidated source, policy, claim, compiler, and native artifact hashes;
5. an exact bundle inventory and a valid bundle identity hash.

Only then is the staging directory renamed to its final path. Refusal,
`UNKNOWN`, compilation failure, mutation, tamper, or an occupied destination
leaves the requested final bundle absent and never overwrites an existing
bundle. The bundle contains the native artifact, boundary receipt, exact claim
contract copy, and `package-release.json`.

`souc pkg verify <bundle> --root <project>` repeats the bundle checks and the
full receipt verification. The original sources, policy, claim path, and
compiler are required. The bundle deliberately does not claim environment
capture, independent replay, registry publication, scientific truth, or
clinical authority.

## R2.6 Registry Attestation Policy

`tools/science_boundary/registry_attestation.py` consumes a fully verified
R2.5 bundle, its original verification inputs, and a separate
`sounio.registry-attestation-policy.v1`. It emits a deterministic
`sounio.registry-attestation.v1` only when the bundle's conclusive ring,
visibility, requested claim class, identity-only assurance, strict mode, and
`OK` verdict match the local policy.

The attestation type is `unsigned-local-policy-evaluation`, its decision is
`POLICY_MATCH`, its authority scope is `local-catalog-index`, and its
publication status is `disabled`. Verification reconstructs the entire
attestation from the bundle, source tree, package policy, compiler, and
registry policy. A forged field remains invalid even if its JSON identity hash
is recomputed.

R2.6 binds a local catalog decision to exact content. It does not assert public
registry status, upload, namespace ownership, issuer identity, remote
signature, independent replay, scientific truth, or clinical authority. The
complete contract is `docs/ecosystem/REGISTRY_ATTESTATION_SPEC.md`.

## R3 Physical Extraction Inventory

`tools/science_boundary/physical_extraction_inventory.py` binds every root in
`science-rings.tsv` to exactly one row in
`docs/ecosystem/science-physical-extraction-ownership.tsv`. It recursively
records every regular file, byte size, SHA-256 digest, per-unit tree identity,
and the root's proposed ownership disposition.

The v1 rules retain `pl-core` in the root repository, plan separately named
distributions for conclusive `scientific-package` and `research` roots, and
block candidate or unresolved roots without assigning a destination. Roots
must be repository-relative and non-overlapping; symbolic links and incomplete
coverage refuse.

Every emitted artifact has type `physical-extraction-planning-snapshot`,
status `not-executed`, and `identity-only` assurance. It proves neither source
movement nor destination existence or ownership transfer. Those claims require
the separate materialization interface. The complete inventory contract is
`docs/ecosystem/PHYSICAL_EXTRACTION_INVENTORY.md`.

## R3 Physical Extraction Materialization

`tools/science_boundary/physical_extraction_materializer.py` consumes a fully
reverified inventory and an explicit
`sounio.physical-extraction-destination-policy.v1`. Every planned target must
have a unique approved policy row, repository-local approval evidence bound by
size and SHA-256, and a preexisting local destination carrying an exact marker
bound to the same inventory.

The tool stages every regular-file copy and verifies its byte identity before
promoting any final content path. Unit promotion uses same-filesystem directory
renames; the deterministic receipt is promoted last. Verification reconstructs
the source inventory, policy bindings, markers, exact destination trees, and
receipt. Source or destination mutation, incomplete approval, symlinks,
occupied output, malformed inputs, and forged or rehashed receipts refuse.

The materialization receipt has type `verified-local-exact-copy`, status
`copied-and-verified`, `identity-only` assurance, and source-removal status
`not-authorized`. It proves local byte-copy identity only. It does not establish
a remote repository, commit or push, ownership transfer, publication,
independent replay, scientific truth, or clinical authority. Multiple unit
renames cannot be crash-atomic as one filesystem transaction; a valid final
receipt is required to accept the complete operation.

No approved production destination policy or receipt currently exists for the
canonical Sounio roots, so this executable interface does not claim that the
repository migration occurred. The complete contract is
`docs/ecosystem/PHYSICAL_EXTRACTION_MATERIALIZATION.md`.

## R3 Source-Removal Authorization

`tools/science_boundary/source_removal_authorizer.py` consumes a fully
reverified materialization and one exact approved removal policy. The policy
must bind the complete planned scope, at least two distinct review evidence
records, one or more byte-exact repairs, and one or more post-removal gate
commands with expected exit and output identities.

The tool snapshots every regular source file into an external temporary copy,
removes only the `extract-planned` roots from that copy, applies the declared
repairs, and runs the gates directly from their argument vectors. The complete
candidate must remain equal to the original snapshot minus the planned files
plus the exact repairs. Original sources, inventory, materialization,
destinations, policy and evidence are reverified before receipt promotion; the
temporary candidate is then discarded.

The deterministic receipt has type
`verified-post-removal-candidate-authorization`, status
`authorized-not-executed`, source execution status `not-executed`, and
`identity-only` assurance. It does not prove reviewer independence, production
migration, source deletion, remote repository state, ownership, publication,
independent replay, scientific truth, or clinical authority. The authorizer
contains no canonical removal operation. The full contract is
`docs/ecosystem/PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION.md`.

## R3 Source-Removal Execution

`tools/science_boundary/source_removal_executor.py` consumes an authorization
that is fully reconstructable before mutation plus a separate execution policy
bound to the exact authorization, materialization, inventory, pre-execution
tree, post-execution tree, removal scope, retained root marker, and operator
approval evidence. Four exact CLI confirmations repeat the authorization,
scope, policy, and pre-execution tree identities.

Before modifying the bound root, the tool locks its directory inode, creates
and verifies a full regular-file backup in an external same-filesystem
transaction workspace, and checks the root again. It then removes only
authorized roots, applies only authorized repair bytes, reruns the authorized
gates, verifies the complete resulting tree and materialized copies, and
promotes the execution receipt last. Ordinary failure before receipt promotion
restores and verifies the exact pre-execution file tree. Post-execution
verification gates run only on a disposable copy. Crash atomicity across
multiple filesystem operations is not claimed.

The deterministic receipt has type `policy-bound-local-source-removal`, status
`executed-and-verified`, source-removal status `executed`, and `identity-only`
assurance. The focused gate performs real removal only in disposable fixture
roots. No production execution policy or receipt exists for the canonical
Sounio repository, so its source roots remain unchanged. The complete contract
is `docs/ecosystem/PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION.md`.

## R3 Canonical Cutover Approval

`tools/science_boundary/canonical_cutover_authorizer.py` reconstructs the full
source-removal authorization while the original source remains present and
consumes a separate exact cutover policy. The policy binds the authorization,
materialization, inventory, pre/post trees, removal scope, repair and gate sets,
the clean canonical Git worktree, and one clean content-exact destination Git
worktree for every materialized unit. Each Git binding includes repository ID,
branch, local `HEAD`, configured remote URL, and the branch object ID observed
through `git ls-remote`.

The authorizer also binds a retained canonical marker, destination-owner
evidence, operator confirmations, and a structured recovery plan. It rehearses
the exact removals, repairs and gates on a disposable full-file copy, restores
that copy from a full backup, and revalidates every source, policy, evidence,
Git and remote-ref binding before promoting the receipt while holding the
canonical-root lock. It never removes a canonical source file.

The deterministic receipt reports `approved-not-executed`, cutover execution
`not-executed`, source removal `not-executed`, and
`identity-plus-git-remote-ref` assurance. A `disposable-fixture` receipt is not
production approval; Git remote-ref observation does not prove hosting
administration, ownership or organizational authority. No production cutover
policy or receipt exists for the canonical Sounio repository. The complete
contract is
`docs/ecosystem/PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL.md`.

## R3 Canonical Cutover Execution

`tools/science_boundary/canonical_cutover_executor.py` consumes the complete
approval plus a separately authored execution policy. The policy binds the
approval bytes and identity, exact source/destination/recovery evidence,
execution context, pre-cutover local and remote refs, deterministic commit
metadata, expected post-cutover Git tree, and expected commit. Five new CLI
confirmations repeat the approval, execution-policy, old-head, expected-commit,
and execution-context values.

Before mutation, the executor recomputes the planned Git tree and commit in an
isolated bare object repository and alternate worktree/index. Under the
canonical-root lock it creates a full regular-file-and-mode backup, reconstructs
the complete approval again, applies only authorized removals and repairs, runs
the exact gates, stages the receipt, advances the local ref with compare-and-
swap, and publishes the exact commit with a lease on the exact old remote ref.
It promotes the receipt only after local/remote refs, commit topology, complete
regular-file tree, destination repositories, and retained evidence reverify.

An ordinary pre-receipt failure observes the actual refs, rolls the remote back
under an exact lease when necessary, restores the local ref/tree/modes/index,
and verifies the pre-cutover Git state. The remote update and receipt promotion
are not a distributed atomic transaction; crash recovery may require the bound
manual procedure. The deterministic receipt reports approval `consumed`,
cutover `executed-and-verified`, source removal `executed`, and
`identity-plus-git-remote-ref-and-published-commit` assurance.

The focused gate executes only disposable standalone repositories and local
bare remotes. No `canonical-production` execution policy or receipt exists for
Sounio, so every canonical source root remains present. The complete contract
is `docs/ecosystem/PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION.md`.

## Declarations

The repository inventory is `science-rings.tsv`. Its columns are fixed by the
v1 contract. A package can instead declare:

```toml
[science]
schema = "sounio.science-manifest.v1"
ring = "scientific-package"
evidence-status = "passes-gate"
context-of-use = "bounded research software"
visibility = "public"
allowed-claim-classes = ["compile", "runtime"]
evidence-refs = ["gate:package-boundary-receipt"]
```

Scientific `[[example]]` entries require `maturity`, `context-of-use`, and
`evidence-refs`. `calibrated` or `validated` examples require typed `dataset`,
`split`, `diagnostics`, `gate`, and `review` references.

Legacy `[epistemic]` metadata remains readable for compatibility and emits
`W-SRB-LEGACY-001`. `score`, `regulatory-ready`, `provenance-level`,
`gum-compliant`, and `validation-coverage` confer no ring, claim, promotion, or
release authority. There is no automatic translation.

## Claims and Receipts

`sounio.claim-contract.v1` requires an explicit claim ID, requested class,
context of use, root artifact, and typed evidence. Every evidence item carries
a lowercase SHA-256 digest. Source, package policy, and compiler bindings are
checked against the build identities; other evidence refs must be root-local
files and are re-hashed during receipt verification. Claim classes are never
inferred from names, directories, scores, or package metadata. A GUM claim must
name both `method` and `witness` evidence; legacy `gum-compliant` is never a
substitute.

Those hashes establish file identity, existence, policy matching, and change detection only.
They do not establish that a dataset is genuine, a method is correct, a review
is independent, or evidence supports scientific or clinical validity. External
hosted registry authority, remote signatures, attested execution, independent replay,
`ClinicalAuthority`, and `ClinicalRelease` remain outside R0-R2.

`sounio.package-boundary-receipt.v1` records the ternary verdict, mode, graph,
diagnostics, source/policy/claim/compiler/ELF hashes, engine identity, claim
summary, assurance level, and limitations. Identity-only receipts contain no
timestamp or absolute path and serialize deterministically.
They are cryptographic build manifests, not standalone reproducibility audits;
environment capture and independent replay are deliberately not implied.

`E-SRB-001` rejects ring inversion, `E-SRB-002` rejects visibility leakage,
`E-SRB-003` rejects a conclusive ring without named evidence references,
`E-SRB-004` rejects incompatible evidence or context, `E-SRB-005` rejects an
empirical/clinical claim supported only by compile/runtime evidence,
`E-SRB-006` rejects missing provenance bindings, and `E-SRB-007` rejects a
claim class not authorized by the root ring. `E-SRB-000` identifies invalid or
incomplete authority in strict mode.

Receipts identify `madaros-raw-ast-v1` or `sounio-host-syntax-v1` explicitly.
Only the raw AST collector can produce `OK`; the host collector is retained for
advisory inventory and always contributes `E-SRB-000`/`UNKNOWN`.

## Current Acceptance Status

The host attestor, CLI flags, ternary policy, deterministic receipt, claim
contract, legacy quarantine, strict temporary-ELF flow, atomic R2.5 package
release bundle, deterministic R2.6 local registry-policy attestation, R3
physical extraction inventory, R3 local materialization, R3 temporary-copy
source-removal authorization, R3 policy-bound local source-removal execution,
and R3 Git-state canonical-cutover approval and execution interfaces are
implemented.
The current-source raw collector follows the established per-node AST reload
loop and resolves the real `hello_pkg` closure to `main.sio`, `greet.sio`, and
their import edge. The named gate passes 178 assertions, including runnable
strict ELFs, deterministic receipts, refusal/`UNKNOWN`, tamper detection, and
absence of a final ELF after strict refusal. The R2.5 gate composes those 178
assertions with 65 assertions for bundle determinism, round-trip verification,
refusal without a final bundle, exact-inventory enforcement, and component
tamper detection. The R2.6 gate adds 82 assertions for policy matching,
deterministic attestation identity, output promotion, full input revalidation,
and forged or rehashed attestation refusal.

The composed current-source acceptance witness is Slurm job `6394`: the same
Madaros ELF passed all 178 R0-R2 assertions, 65 R2.5 assertions, and 82 R2.6
assertions. This establishes the named software boundary only; it does not
promote any package or claim to a stronger evidence class.

The focused R3 gate adds 141 assertions for exact ring and ownership coverage,
deterministic file identity, retained/planned/blocked dispositions, occupied
output preservation, source mutation detection, and forged or rehashed
inventory refusal. Its current repository witness covers seven ownership units
and more than 3,000 regular files while keeping extraction status
`not-executed`.

The composed current-source R3 acceptance witness is Slurm job `6434` on
`gpuorangefs-r770-proxmox`: the same Madaros ELF passed all 178 R0-R2, 65 R2.5,
82 R2.6, and 141 R3 assertions. The emitted repository snapshot contains 3,277
regular files across seven ownership units and verifies with extraction status
`not-executed`. This is evidence for the inventory boundary only, not physical
materialization or ownership transfer.

The focused materialization gate adds 167 assertions for exact approval
coverage, destination-marker binding, deterministic receipts across physical
roots, source preservation, occupied-output preservation, and source,
destination, policy, inventory, marker, and receipt tamper refusal. It
materializes two approved fixture units while retaining one core unit and
leaving one unresolved unit blocked. A production witness for the canonical
five planned targets remains absent because no production destination policy
has been approved.

The composed current-source materialization witness is Slurm job `6478` on
`gpuorangefs-r770-proxmox`: the same Madaros ELF passed all 178 R0-R2, 65 R2.5,
82 R2.6, 141 R3 inventory, and 167 materialization assertions. The deterministic
fixture receipt covers two approved units and three files, reports
`copied-and-verified`, and keeps source removal `not-authorized`. The first job
attempt, `6477`, placed R2.5 temporary promotion state on OrangeFS and received
`EINVAL` from a directory `fsync`; job `6478` corrected the harness to use
node-local temporary storage while retaining the exact source archive,
compiler, gate, and OrangeFS evidence logs.

The focused source-removal authorization gate adds 527 checks for exact
materialization binding, complete removal scope, distinct review records,
byte-bound repairs, deterministic candidate receipts, source preservation,
post-removal command evidence, candidate mutation refusal, and forged or
rehashed policy and authorization refusal. It authorizes two fixture units and
three fixture files as `authorized-not-executed`; all original fixture roots
remain present. No production policy or canonical authorization exists.

The composed current-source authorization witness is Slurm job `6527` on
`gpuorangefs-r770-proxmox`: one archived source snapshot and the unchanged
source-fresh Madaros passed all 178 R0-R2, 65 R2.5, 82 R2.6, 141 inventory,
167 materialization, and 527 authorization checks in 42 seconds. The final
fixture receipt identities match the local witness, reports
`authorized-not-executed` and `not-executed`, and leaves every original source
root present. Node-local temporary storage was used for every promotion and
candidate workspace; no fallback implementation path ran.

The focused execution gate adds checks for exact authorization replay,
execution-policy binding, explicit CLI confirmations, deterministic receipts
across equivalent roots, exact planned-root removal, retained and blocked-root
preservation, repair and gate evidence, root-inode serialization,
ordinary-failure rollback, isolated verification-gate mutation, occupied output
preservation, post-execution verification, and forged or rehashed policy and
receipt refusal. It executes planned units only inside disposable fixture
roots. No canonical source root is an execution target.

The composed current-source execution witness is Slurm job `6558` on
`gpuorangefs-r770-proxmox`: archive
`31df0a309a7fac2cf7703bda5931f093b216137c9072cccd9aa5033465313323`
and Madaros
`6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88`
passed all 178 R0-R2, 65 R2.5, 82 R2.6, 141 inventory, 167 materialization,
527 authorization, and 164 execution checks in 45 seconds with
`MaxRSS=1422288K`. The final execution receipt identity matches the local
witness. All real removal occurred only in disposable node-local fixture roots;
no implementation fallback ran and no canonical source root changed.

The focused canonical-cutover approval gate adds checks for complete
authorization replay, exact standalone Git worktree layout, clean source and
destination trees, local and remote branch object equality, exact destination
content and coverage, marker/owner/operator/recovery evidence, matched CLI
confirmations, disposable removal/repair/gate rehearsal, exact backup
restoration, deterministic receipts, occupied-output preservation, mutating
rehearsal confinement, and forged or rehashed policy and receipt refusal. Its
fixture receipt reports `approved-not-executed`, cutover execution
`not-executed`, and source removal `not-executed`.

The composed current-source approval witness is Slurm job `6602` on
`gpuorangefs-r770-proxmox`. Commit `851c9eba2`, compressed archive
`c9556d94e8c50ef200cb6fcb9ea60fbc26a45e091a8de4960718fb140e0c9273`
and Madaros
`6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88`
passed all 178 R0-R2, 65 R2.5, 82 R2.6, 141 inventory, 167 materialization,
527 authorization, 164 local execution, and 172 approval checks in 44 seconds
with `MaxRSS=1389440K`. Node-local temporary storage was used because OrangeFS
was full; after two pre-gate worker-image dependency failures, the same archive
and compiler passed with Git 2.43.0 provisioned in the ephemeral worker. No
implementation fallback ran and no canonical source root changed.

The focused canonical-cutover execution gate adds 81 assertions for exact
approval replay, separate execution-policy and CLI confirmation binding,
isolated recomputation of the expected Git tree and commit, deterministic
execution receipts across equivalent roots, real root removal, local ref
compare-and-swap, exact-lease remote publication, destination preservation,
post-execution verification, stale/dirty/tampered-state refusal, canonical-only
gate-mutation rollback, and exact remote/local/tree rollback when receipt
promotion is raced after push. Its disposable receipt reports approval
`consumed`, execution `executed-and-verified`, and source removal `executed`.

The local fixture execution identity is
`f7ea56e8028f1a21f6f23afd316bfb93e6a27416067442fdf0da63f28e064d21`;
the policy identity is
`e55de2ce7f6d82e57c7408cb1fc95948deb7fdc2c36da611ce6ebba1455a406c`;
and the expected commit is `789611457cc681226baa2885391d7bbbd29a5fa7`.
These are disposable fixture identities. No Sounio production policy, commit,
ref update, source removal, or execution receipt was created.

The composed current-source execution witness is Slurm job `6613` on
`gpuorangefs-r770-proxmox`. Commit `002d5f227`, archive
`16a7cfdddd120cbd47a0b471506126fe3724fd6b1e9e27b772ce9ab73245c642`,
and Madaros
`6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88`
passed the complete prior stack and all 81 execution checks in 53 seconds with
`MaxRSS=1390324K`. The run used Git 2.43.0 and node-local disposable workspaces;
no implementation fallback ran and no Sounio canonical ref or source changed.

The canonical-production gap assessor then separates observable repository
prerequisites from permission. Its v1 schema fixes execution authority to
`none` and execution status to `not-executed`; even a complete fixture mapping
can reach only `production-evidence-and-human-decision-required`. The current
Sounio observation binds five planned targets, zero mappings, zero observed
mapped destinations, and eight missing prerequisites under assessment identity
`0fe82728ea24520af7792d4b5cf45c6c20e62c47a09138d0c4b81207e998e816`.
It does not infer that `sounio-examples` is the planned research distribution,
and it records that the reviewed stack is not yet the cataloged `main` head.

The composed current-source gap witness is Slurm job `6615` on
`gpuorangefs-r770-proxmox`. Commit `4dc8749a7`, archive
`484fde3c4881905d60fbf601d8d42a7ba5d389cb6392b9657022c2267a791ede`,
and Madaros
`6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88`
passed the complete prior stack and all 90 gap checks in 55 seconds with
`MaxRSS=1390896K`. The final fixture status remained
`production-evidence-and-human-decision-required` with authority `none`; no
implementation fallback, real catalog mutation, canonical source change, or
remote ref update occurred.
