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
an R2.6 local registry-attestation policy contract, and an R3 physical
extraction inventory; physical materialization remains deferred. Promotion is
bound to the transitive raw-AST witness in
`scripts/ci/science_boundary_gate.sh`, the package gate in
`scripts/ci/package_boundary_release_gate.sh`, the attestation gate in
`scripts/ci/registry_attestation_spec_gate.sh`, and a current-source Madaros.
R3 means moving scientific-package and research sources into separately owned
repositories or distributions. The current R3 inventory binds proposed
ownership and exact file identity but moves no source file. R2.6 launches no
registry service.

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
movement nor destination existence or ownership transfer. Full materialization
is the separate `r3-physical-extraction-materialization` interface. The
complete inventory contract is
`docs/ecosystem/PHYSICAL_EXTRACTION_INVENTORY.md`.

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
release bundle, deterministic R2.6 local registry-policy attestation, and R3
physical extraction inventory are implemented.
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
