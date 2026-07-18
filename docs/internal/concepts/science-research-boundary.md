<!-- docs:meta
topic_id: repo.docs.internal.concepts.science-research-boundary
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.science-research-boundary
-->

# Science Research Boundary

Concept-ID: `SOUNIO-SCIENCE-RESEARCH-BOUNDARY`

## Founder Intent

Research must remain easy to express without allowing compilation success,
package location, scalar scores, or provenance metadata to silently become
scientific or clinical authority.

## Canonical Distinctions

```text
programming-language core != scientific package != research artifact
compile evidence          != empirical evidence
provenance                != assurance
package maturity          != claim authorization
receipt identity          != scientific truth
validated research        != clinical validation
```

## Evidence Status

Status: `executable`

The R0-R2 host attestor, compiler integration, R2.5 package release boundary,
R2.6 local registry attestation, R3 physical extraction inventory, R3 local
exact-copy materialization, R3 temporary-copy source-removal authorization,
and R3 policy-bound local source-removal execution interfaces are executable.
The named gates
prove pass, refuse, `UNKNOWN`, deterministic receipt identity, source
sensitivity, evidence and receipt tamper refusal, absence of a final ELF after
strict refusal, and a real transitive raw-AST import witness. The current-source
Madaros was built through the canonical source-tracking bootstrap path on Slurm.

R2.5 adds no new scientific claim class. It makes the existing receipt a
promotion prerequisite for one local, opt-in release bundle and preserves the
identity-versus-assurance distinction.

R2.6 binds that bundle to a local catalog policy without publication. R3 binds
the declared roots to an exact-file ownership plan, permits a separately
approved local copy whose receipt keeps source removal `not-authorized`, and
can authorize an exact removed-and-repaired temporary candidate while keeping
execution `not-executed`. A separate Git-bound executor can consume that
approval, pre-bind an exact commit, remove/repair the bound tree, and publish an
exact leased ref update. Its acceptance evidence exists only for disposable
fixtures. None of these interfaces promotes scientific authority, and no
canonical production destination, materialization, removal policy,
authorization receipt, approval, execution policy, or execution receipt
currently exists.

## Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R0-R2-20260715
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: research freedom remains available while claim-bearing builds must cross an explicit evidence and dependency boundary
Transformation: make the research to package to PL-core direction executable with ternary verdicts and deterministic receipts
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish that the configured software boundary accepted, rejected, or could not classify one build graph
Claims-Forbidden: scientific truth, clinical validity, ClinicalAuthority, ClinicalRelease, security sandboxing, public registry status, attested execution, independent replay, or R3 physical extraction
Assumptions: source imports use the supported Sounio module grammar and every authoritative ring or claim is explicit
Write-Set: tools/science_boundary/attestor.py; bin/madaros; bin/souc; self-hosted/compiler/{main,module_frontend,module_parse}.sio; self-hosted/compiler/pkg/{cli,lock,manifest,registry_client,scorer}.sio; science-rings.tsv; schemas/sounio.*boundary*; package sounio.toml declarations; scripts/dev/registry_serve.py; docs/{architecture,ecosystem,internal/concepts}/*science*; scripts/ci/science_boundary_gate.{py,sh}
Read-Set: FOUNDER_INTENT.md; AGENTS.md; .claude/PARALLEL_BLOCKER_CONTRACT.md; self-hosted/compiler/module_frontend.sio; package manifests
Positive-Witness: strict allowed graph emits a runnable ELF and deterministic OK receipt
Negative-Witness: inverted rings, visibility leaks, unresolved closure, unauthorized claims, and tampered receipts refuse with structured diagnostics
Acceptance-Gate: bash scripts/ci/science_boundary_gate.sh
Integration-Target: origin/main after review
Authoritative-Only-If: the gate passes with a current-source raw Madaros that emits SOUNIO_BOUNDARY_CLOSURE_V1; host syntax audit remains advisory-only
```

## Pending Interface

`r3-physical-extraction-canonical-production-policy-and-human-decision`

## R3 Canonical Production Mapping Decision Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R3-CANONICAL-PRODUCTION-MAPPING-DECISION-20260718
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: an explicit human repository selection may prepare a reviewable proposal without becoming repository-creation permission, production approval, cutover authority, or scientific authority
Transformation: validate a reviewed transcription against the exact governed targets, point-in-time catalog, and clean canonical Git observation; classify reuse, creation request, and target revision; emit a proposed-not-approved mapping only when every target reuses one exact available observed repository
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish that one exact transcribed mapping selection is complete for proposal review, requires destination provisioning and reconfirmation, or requires ownership-policy revision while keeping execution authority none
Claims-Forbidden: authenticated responder identity, organizational authority, inferred mapping by repository name, repository creation or modification, destination-owner consent, materialization, source removal, Git ref update, canonical production approval, cutover approval or execution, independent replay, scientific truth, clinical validity, ClinicalAuthority, or ClinicalRelease
Assumptions: the linked response is independently reviewed before transcription; supplied catalog metadata is a point-in-time observation; a catalog or canonical-source change invalidates the selection binding; issue #1122 has no response at the time of this lane
Write-Set: tools/science_boundary/canonical_production_mapping_decision_processor.py; schemas/sounio.physical-extraction-canonical-production-mapping-decision{,-receipt}.v1.schema.json; scripts/ci/physical_extraction_canonical_production_mapping_decision_gate.{py,sh}; scripts/ci/sounio_package_support_gate.sh; docs/ecosystem/{PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION.md,PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT.md,ECOSYSTEM_ROADMAP_2026.md}; docs/internal/concepts/science-research-boundary.md; generated docs governance artifacts; .claude/llm_offload_log.md
Read-Set: FOUNDER_INTENT.md; AGENTS.md; science-rings.tsv; docs/ecosystem/science-physical-extraction-ownership.tsv; tools/science_boundary/{physical_extraction_inventory.py,canonical_production_gap_assessor.py}; canonical production catalog and a separately reviewed human response transcription
Positive-Witness: equivalent standalone fixture roots emit byte-identical all-reuse decision receipts and proposed-not-approved mappings; the existing production-gap assessor consumes the mapping and still reports production-evidence-and-human-decision-required with execution authority none
Negative-Witness: missing, extra, duplicate, unsorted, stale, unavailable, action-inconsistent, colliding, occupied, changed, forged, or rehashed selection, catalog, source, receipt, and proposal states refuse without changing source files, repositories, or refs
Acceptance-Gate: SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/physical_extraction_canonical_production_mapping_decision_gate.sh
Integration-Target: codex/physical-extraction-canonical-production-readiness-r3-20260717, then origin/main after the prior stack lands
Authoritative-Only-If: the complete prior R0-R3 stack and focused mapping-decision gate pass on one archived source snapshot with one current-source Madaros; processing a real response additionally requires an exact reviewed transcription and fresh bound catalog/canonical observation
```

## R3 Physical Extraction Inventory Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R3-PHYSICAL-EXTRACTION-INVENTORY-20260717
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: scientific-package and research sources acquire explicit future ownership without allowing repository location, catalog membership, or file identity to become scientific authority
Transformation: bind every science-rings.tsv root to one explicit retain, extract, or blocked disposition and one deterministic exact-file snapshot
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish that the current repository snapshot has exact ownership-plan coverage and content identity for every declared science ring root
Claims-Forbidden: completed source movement, target repository existence, transferred ownership, public publication, independent replay, scientific truth, clinical validity, ClinicalAuthority, or ClinicalRelease
Assumptions: source roots are repository-relative non-overlapping directories; extraction snapshots contain regular files only and refuse symbolic links
Write-Set: tools/science_boundary/physical_extraction_inventory.py; schemas/sounio.physical-extraction-inventory.v1.schema.json; scripts/ci/physical_extraction_inventory_gate.{py,sh}; scripts/ci/sounio_package_support_gate.sh; docs/ecosystem/{PHYSICAL_EXTRACTION_INVENTORY.md,science-physical-extraction-ownership.tsv,SOUNIO_TOML_SPEC.md,ECOSYSTEM_ROADMAP_2026.md}; docs/{architecture,internal/concepts}/science-research-boundary.md; docs/internal/concepts/registry.tsv; docs/governance/{topic-registry.v1.json,DOCS_ACCEPTANCE_REPORT.md,DOCS_AUTHORITY_MATRIX.md}; .claude/llm_offload_log.md
Read-Set: science-rings.tsv; tools/science_boundary/{attestor.py,package_release.py,registry_attestation.py}; schemas/sounio.registry-attestation.v1.schema.json; package manifests
Positive-Witness: exact ring coverage emits a deterministic identity-only snapshot whose file inventory round-trips against the same repository and ownership policy
Negative-Witness: missing, duplicate, overlapping, escaping, symlinked, ring-mismatched, disposition-invalid, mutated, added, deleted, forged, or rehashed inputs refuse
Acceptance-Gate: SOUNIO_PHYSICAL_EXTRACTION_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/physical_extraction_inventory_gate.sh
Integration-Target: codex/registry-attestation-r26-20260717, then origin/main after the R2.6 stack lands
Authoritative-Only-If: R0-R2, R2.5, R2.6, and the R3 inventory gate pass on one source snapshot, with the first three using the same current-source raw Madaros AST collector
```

## R3 Physical Extraction Materialization Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R3-PHYSICAL-EXTRACTION-MATERIALIZATION-20260717
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: scientific packages and research artifacts can acquire independently verifiable physical copies without allowing location, copy success, or a destination label to become scientific authority
Transformation: require one verified R3 inventory and one exact approved-destination policy before copying every planned regular-file byte into preexisting separate local destinations and emitting a deterministic verification receipt
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish that approved local fixture destinations contain byte-identical copies of every extract-planned inventory unit while retained and blocked units remain outside the transfer
Claims-Forbidden: canonical repository extraction, source deletion authority, remote repository creation or push, transferred ownership or maintainership, publication, independent replay, scientific truth, clinical validity, ClinicalAuthority, or ClinicalRelease
Assumptions: destinations are preexisting direct children of one external local root on the same filesystem; the inventory intentionally covers regular-file bytes rather than uninventoried filesystem metadata
Write-Set: tools/science_boundary/physical_extraction_materializer.py; schemas/sounio.physical-extraction-{destination-policy,materialization}.v1.schema.json; scripts/ci/physical_extraction_materialization_gate.{py,sh}; scripts/ci/sounio_package_support_gate.sh; docs/ecosystem/{PHYSICAL_EXTRACTION_INVENTORY.md,PHYSICAL_EXTRACTION_MATERIALIZATION.md,SOUNIO_TOML_SPEC.md,ECOSYSTEM_ROADMAP_2026.md}; docs/{architecture,internal/concepts}/science-research-boundary.md; docs/internal/concepts/registry.tsv; docs/governance/{topic-registry.v1.json,DOCS_ACCEPTANCE_REPORT.md,DOCS_AUTHORITY_MATRIX.md}; .claude/llm_offload_log.md
Read-Set: science-rings.tsv; docs/ecosystem/science-physical-extraction-ownership.tsv; tools/science_boundary/physical_extraction_inventory.py; schemas/sounio.physical-extraction-inventory.v1.schema.json
Positive-Witness: two approved planned fixture units are staged, byte-verified, promoted, and round-trip verified with an identical receipt across two different physical destination roots while one core unit is retained and one candidate remains blocked
Negative-Witness: incomplete approval, missing or altered destination markers, source mutation, destination mutation, symlinks, occupied outputs, policy or inventory mismatch, and forged or rehashed receipts refuse without authorizing source removal
Acceptance-Gate: SOUNIO_PHYSICAL_EXTRACTION_MATERIALIZATION_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/physical_extraction_materialization_gate.sh
Integration-Target: codex/physical-extraction-inventory-r3-20260717, then origin/main after the R3 inventory stack lands
Authoritative-Only-If: R0-R2, R2.5, R2.6, the R3 inventory gate, and the R3 materialization gate pass on one source snapshot; acceptance proves the interface behavior but not a canonical production migration without a separately approved production policy and receipt
```

## R3 Source-Removal Authorization Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R3-SOURCE-REMOVAL-AUTHORIZATION-20260717
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: planned scientific-package and research roots may become removable only after exact materialization, review, repair, and post-removal evidence without converting location or gate success into scientific authority
Transformation: reconstruct one removed-and-repaired candidate in an external temporary copy, run policy-bound post-removal gates, reverify the untouched original, and emit authorized-not-executed evidence
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish that one exact planned scope produced one exact temporary candidate whose declared repairs and post-removal commands passed while the original source snapshot remained unchanged
Claims-Forbidden: actual or canonical source deletion, unlisted-path authority, production migration, reviewer independence, remote repository creation or push, transferred ownership or maintainership, publication, independent replay, scientific truth, clinical validity, ClinicalAuthority, or ClinicalRelease
Assumptions: the materialization remains fully verifiable; review labels identify distinct local evidence records only; post-removal commands and expected outputs are explicitly approved by policy
Write-Set: tools/science_boundary/source_removal_authorizer.py; schemas/sounio.physical-extraction-source-removal-{policy,authorization}.v1.schema.json; scripts/ci/physical_extraction_source_removal_authorization_gate.{py,sh}; scripts/ci/sounio_package_support_gate.sh; docs/ecosystem/{PHYSICAL_EXTRACTION_MATERIALIZATION.md,PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION.md,ECOSYSTEM_ROADMAP_2026.md}; docs/{architecture,internal/concepts}/science-research-boundary.md; docs/internal/concepts/registry.tsv; docs/governance/{topic-registry.v1.json,DOCS_ACCEPTANCE_REPORT.md,DOCS_AUTHORITY_MATRIX.md}; .claude/llm_offload_log.md
Read-Set: science-rings.tsv; docs/ecosystem/science-physical-extraction-ownership.tsv; tools/science_boundary/{physical_extraction_inventory.py,physical_extraction_materializer.py}; schemas/sounio.physical-extraction-{inventory,destination-policy,materialization}.v1.schema.json
Positive-Witness: an exact two-unit fixture scope produces identical authorized-not-executed receipts across distinct temporary workspace roots and round-trip verification while all original roots remain present
Negative-Witness: stale materialization or destination content, incomplete or forged scope, missing or duplicate review evidence, invalid repair identity, failing or mutating gates, unsafe workspace, occupied output, source mutation, and forged or rehashed authorization all refuse
Acceptance-Gate: SOUNIO_PHYSICAL_EXTRACTION_SOURCE_REMOVAL_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/physical_extraction_source_removal_authorization_gate.sh
Integration-Target: codex/physical-extraction-materialization-r3-20260717, then origin/main after the R3 materialization stack lands
Authoritative-Only-If: R0-R2, R2.5, R2.6, R3 inventory, R3 materialization, and source-removal authorization gates pass on one archived source snapshot with one current-source raw Madaros witness; canonical execution additionally requires a separate human-approved execution interface
```

## R3 Source-Removal Execution Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R3-SOURCE-REMOVAL-EXECUTION-20260717
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: approved scientific-package and research roots may be removed only from one exact explicitly marked local tree without converting removal success, location, or operator labels into scientific authority
Transformation: reconstruct authorization before mutation, bind a separate execution policy and four CLI confirmations, back up the complete regular-file tree, execute only authorized removals and repairs, rerun gates, and promote a deterministic receipt last
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish that an explicitly marked disposable local root reached the exact authorized post-removal tree and that ordinary execution failure restored the exact pre-execution regular-file tree
Claims-Forbidden: canonical production cutover, unlisted-path removal, crash atomicity, operator identity or organizational authority, remote repository creation or push, transferred ownership or maintainership, publication, independent replay, scientific truth, clinical validity, ClinicalAuthority, or ClinicalRelease
Assumptions: the pre-execution authorization remains fully reconstructable; the transaction workspace is external and on the same filesystem; nonparticipating writers are quiesced; the regular-file identity model intentionally excludes filesystem metadata
Write-Set: tools/science_boundary/source_removal_executor.py; schemas/sounio.physical-extraction-source-removal-execution{,-policy}.v1.schema.json; scripts/ci/physical_extraction_source_removal_execution_gate.{py,sh}; scripts/ci/sounio_package_support_gate.sh; docs/ecosystem/{PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION.md,PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION.md,ECOSYSTEM_ROADMAP_2026.md}; docs/{architecture,internal/concepts}/science-research-boundary.md; docs/internal/concepts/registry.tsv; docs/governance/{topic-registry.v1.json,DOCS_ACCEPTANCE_REPORT.md,DOCS_AUTHORITY_MATRIX.md}; .claude/llm_offload_log.md
Read-Set: tools/science_boundary/{physical_extraction_inventory.py,physical_extraction_materializer.py,source_removal_authorizer.py}; schemas/sounio.physical-extraction-{inventory,destination-policy,materialization,source-removal-policy,source-removal-authorization}.v1.schema.json; science-rings.tsv; physical extraction ownership policy
Positive-Witness: two equivalent disposable roots execute identical two-unit removal and one repair, emit byte-identical receipts, retain core and blocked roots, and round-trip verify against exact materialized copies
Negative-Witness: stale authorization or materialization, wrong policy binding, missing approval evidence, incorrect CLI confirmation, changed root marker, occupied or destination-contained receipt, locked or mutated source root, execution-only gate mutation, verification-only gate mutation, post-execution mutation, and forged or rehashed receipt all refuse; execution-only mutation restores the pre-execution root exactly and verification-only mutation remains confined to a disposable copy
Acceptance-Gate: SOUNIO_PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/physical_extraction_source_removal_execution_gate.sh
Integration-Target: codex/physical-extraction-source-removal-auth-r3-20260717, then origin/main after the authorization stack lands
Authoritative-Only-If: R0-R2, R2.5, R2.6, inventory, materialization, authorization, and execution gates pass on one archived snapshot with one current-source Madaros; canonical cutover additionally requires a separate production approval and real destination evidence
```

## R3 Canonical Cutover Approval Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R3-CANONICAL-CUTOVER-APPROVAL-20260717
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: a canonical scientific-package or research cutover may proceed only after exact source, destination, repair, gate, operator, and recovery evidence is bound without converting repository location or Git state into scientific authority
Transformation: reconstruct the complete authorization, bind clean canonical and destination Git worktrees plus observed remote branch refs, rehearse removal/repair/gates/restoration on a disposable copy, and promote an approved-not-executed receipt while the canonical-root lock remains held
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish that one exact Git-bound fixture repository set was approved but not executed after a complete disposable cutover and restoration rehearsal
Claims-Forbidden: canonical cutover execution, source removal, production approval from fixture evidence, hosting administration or namespace ownership, transferred maintainership, human identity or organizational authority, crash-atomic multi-file execution, independent replay, scientific truth, clinical validity, ClinicalAuthority, or ClinicalRelease
Assumptions: the complete authorization remains reconstructable; each bound worktree is clean; local HEAD equals the exact observed remote branch ref; the v1 Git repositories use 40-hex SHA-1 object IDs; nonparticipating writers remain quiesced
Write-Set: tools/science_boundary/canonical_cutover_authorizer.py; schemas/sounio.physical-extraction-canonical-cutover-{policy,approval}.v1.schema.json; scripts/ci/physical_extraction_canonical_cutover_approval_gate.{py,sh}; scripts/ci/sounio_package_support_gate.sh; docs/ecosystem/{PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION.md,PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL.md,ECOSYSTEM_ROADMAP_2026.md}; docs/{architecture,internal/concepts}/science-research-boundary.md; docs/internal/concepts/{registry,bindings}.tsv; docs/governance/{topic-registry.v1.json,DOCS_ACCEPTANCE_REPORT.md,DOCS_AUTHORITY_MATRIX.md}; .claude/llm_offload_log.md
Read-Set: tools/science_boundary/{physical_extraction_inventory.py,physical_extraction_materializer.py,source_removal_authorizer.py,source_removal_executor.py}; schemas/sounio.physical-extraction-{inventory,destination-policy,materialization,source-removal-policy,source-removal-authorization,source-removal-execution-policy,source-removal-execution}.v1.schema.json; science-rings.tsv; physical extraction ownership policy
Positive-Witness: two equivalent disposable source and destination Git repository sets bind identical local and remote branch object IDs, pass complete cutover and restoration rehearsals, and emit byte-identical approved-not-executed receipts while every source root remains present
Negative-Witness: stale source binding, dirty worktree, changed branch, HEAD, remote URL or remote branch ref, missing or extra destination, destination content mismatch, duplicated repository or owner evidence, changed marker or recovery plan, incorrect CLI confirmation, occupied output, mutating rehearsal gate, and forged or rehashed receipt all refuse without changing source or destinations
Acceptance-Gate: SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/physical_extraction_canonical_cutover_approval_gate.sh
Integration-Target: codex/physical-extraction-source-removal-execution-r3-20260717, then origin/main after the execution stack lands
Authoritative-Only-If: the complete R0-R3 stack and focused canonical-cutover approval gate pass on one archived source snapshot with one current-source Madaros; canonical production execution still requires a separate production policy, human decision, and execution interface
```

## R3 Canonical Cutover Execution Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R3-CANONICAL-CUTOVER-EXECUTION-20260717
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: physical separation may become canonical only through an exact reversible evidence chain without turning repository location, Git publication, or successful removal into scientific authority
Transformation: consume one reconstructed cutover approval plus a separately authored execution policy, pre-bind the exact Git tree and commit, execute approved removal/repair/gates, publish the exact remote ref under lease, and promote a deterministic receipt after complete post-state verification
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish that one exact disposable Git repository transition removed the approved roots, created and published the pre-bound commit, preserved destinations, and emitted a verified execution receipt
Claims-Forbidden: Sounio canonical production cutover from fixture evidence, production permission without a canonical-production policy and human decision, hosting administration or namespace ownership, transferred maintainership, human identity or organizational authority, distributed crash atomicity, independent replay, scientific truth, clinical validity, ClinicalAuthority, or ClinicalRelease
Assumptions: the full approval remains reconstructable before mutation; source and destination repositories are standalone and clean; local and remote pre-cutover refs are equal and quiescent; the remote accepts an exact leased update and, before receipt promotion failure, an exact leased rollback; v1 Git object IDs are 40-hex SHA-1
Write-Set: tools/science_boundary/canonical_cutover_executor.py; schemas/sounio.physical-extraction-canonical-cutover-execution{-policy,}.v1.schema.json; scripts/ci/physical_extraction_canonical_cutover_execution_gate.{py,sh}; scripts/ci/sounio_package_support_gate.sh; docs/ecosystem/{PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION.md,PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL.md,PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION.md,ECOSYSTEM_ROADMAP_2026.md}; docs/{architecture,internal/concepts}/science-research-boundary.md; docs/internal/concepts/registry.tsv; docs/governance/{topic-registry.v1.json,DOCS_ACCEPTANCE_REPORT.md,DOCS_AUTHORITY_MATRIX.md}; .claude/llm_offload_log.md
Read-Set: tools/science_boundary/{physical_extraction_inventory.py,physical_extraction_materializer.py,source_removal_authorizer.py,source_removal_executor.py,canonical_cutover_authorizer.py}; schemas/sounio.physical-extraction-*.v1.schema.json; science-rings.tsv; physical extraction ownership policy
Positive-Witness: two equivalent disposable standalone source/destination repository sets produce identical policies, approval receipts, planned Git trees/commits, executed local and remote refs, and final execution receipts while all destination repositories remain unchanged
Negative-Witness: invalid context, stale or dirty Git state, changed evidence, incorrect CLI confirmation, wrong planned commit, occupied output, canonical-only gate mutation, receipt race after remote push, reintroduced source, changed destination, remote drift, mutating verification, and forged or rehashed receipt all refuse; pre-receipt failures restore the exact old local/remote refs and source tree
Acceptance-Gate: SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/physical_extraction_canonical_cutover_execution_gate.sh
Integration-Target: codex/physical-extraction-canonical-cutover-approval-r3-20260717, then origin/main after the approval stack lands
Authoritative-Only-If: the complete R0-R3 stack and focused canonical-cutover execution gate pass on one archived source snapshot with one current-source Madaros; Sounio production execution additionally requires a separately authored canonical-production evidence set and explicit human decision
```

## R3 Canonical Production Gap Assessment Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R3-CANONICAL-PRODUCTION-GAP-ASSESSMENT-20260717
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: progress toward physical separation must expose missing production prerequisites without allowing repository availability, access labels, proposals, or continuation commands to become cutover permission or scientific authority
Transformation: rebuild the exact R3 inventory, compare one local Git worktree with a supplied point-in-time repository catalog, optionally validate a complete proposed-not-approved target mapping, and emit a deterministic non-authorizing gap assessment
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish which repository and evidence prerequisites are observed or missing for one exact snapshot while fixing execution authority to none and execution status to not-executed
Claims-Forbidden: production mapping approval, destination-owner consent, repository creation, materialization, source removal, Git ref update, canonical production approval, human decision, hosting administration or namespace ownership, transferred maintainership, independent replay, scientific truth, clinical validity, ClinicalAuthority, or ClinicalRelease
Assumptions: the supplied catalog is a point-in-time metadata observation rather than live hosting attestation; target mappings are never inferred by name; the canonical repository and every extract-planned target remain explicit
Write-Set: tools/science_boundary/canonical_production_gap_assessor.py; schemas/sounio.physical-extraction-canonical-production-{repository-catalog,mapping-proposal,gap-assessment}.v1.schema.json; scripts/ci/physical_extraction_canonical_production_gap_gate.{py,sh}; scripts/ci/sounio_package_support_gate.sh; docs/ecosystem/{PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION.md,PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT.md,ECOSYSTEM_ROADMAP_2026.md}; docs/{architecture,internal/concepts}/science-research-boundary.md; docs/governance/{topic-registry.v1.json,DOCS_ACCEPTANCE_REPORT.md,DOCS_AUTHORITY_MATRIX.md}; .claude/llm_offload_log.md
Read-Set: science-rings.tsv; docs/ecosystem/science-physical-extraction-ownership.tsv; tools/science_boundary/{physical_extraction_inventory.py,canonical_cutover_authorizer.py,canonical_cutover_executor.py}; schemas/sounio.physical-extraction-*.v1.schema.json; supplied repository catalog and optional mapping proposal
Positive-Witness: equivalent standalone fixture roots produce byte-identical assessments both without a proposal and with all repositories observed; the latter stops at production-evidence-and-human-decision-required with execution authority none
Negative-Witness: invalid, stale, duplicate, unsorted, incomplete, approved, authorized, reused, archived, missing, dirty, mutated, occupied, forged, or rehashed catalog, proposal, source, and assessment states refuse or remain explicit prerequisite gaps without mutating any repository or ref
Acceptance-Gate: SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/physical_extraction_canonical_production_gap_gate.sh
Integration-Target: codex/physical-extraction-canonical-cutover-execution-r3-20260717, then origin/main after the execution stack lands
Authoritative-Only-If: the complete R0-R3 stack and focused gap gate pass on one archived source snapshot with one current-source Madaros; a point-in-time real catalog assessment may identify gaps but cannot grant permission or substitute for the still-pending canonical-production policy and explicit human decision
```

## R2.5 Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R2.5-RELEASE-20260717
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: research remains unrestricted while a claim-bearing local package release requires explicit authorization and content identity
Transformation: make package-boundary-receipt an atomic promotion boundary for opt-in local release bundles
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish that one local bundle exactly binds an OK receipt, claim contract, compiler, sources, policy, and native artifact
Claims-Forbidden: scientific truth, clinical validity, ClinicalAuthority, ClinicalRelease, public registry status, remote signature authority, attested execution, independent replay, or R3 physical extraction
Assumptions: the package has one resolvable native entrypoint and its sounio.toml is the release policy root
Write-Set: tools/science_boundary/package_release.py; schemas/sounio.package-release-bundle.v1.schema.json; bin/{madaros,souc}; scripts/ci/package_boundary_release_gate.{py,sh}; docs/ecosystem/{CURATED_PACKAGES.md,SOUNIO_TOML_SPEC.md,ECOSYSTEM_ROADMAP_2026.md,curated-package-release-inventory.tsv}; docs/{architecture,internal/concepts}/science-research-boundary.md; docs/internal/concepts/registry.tsv; .claude/llm_offload_log.md
Read-Set: tools/science_boundary/attestor.py; schemas/sounio.{claim-contract,package-boundary-receipt}.v1.schema.json; package manifests; science-rings.tsv
Positive-Witness: strict package build emits a deterministic runnable bundle that passes full round-trip verification
Negative-Witness: UNKNOWN, unauthorized claim, tamper, mutation, missing claim, and occupied destination refuse without promoting or overwriting a final bundle
Acceptance-Gate: SOUNIO_PACKAGE_BOUNDARY_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/package_boundary_release_gate.sh
Integration-Target: origin/main after review
Authoritative-Only-If: both the 178-assertion R0-R2 gate and the R2.5 gate pass with the same current-source raw Madaros AST collector
```

## R2.6 Semantic Lane

```text
Semantic-Lane-ID: SCIENCE-BOUNDARY-R2.6-REGISTRY-ATTESTATION-20260717
Owner: Codex
Concept-IDs: SOUNIO-SCIENCE-RESEARCH-BOUNDARY
Intent-Preserved: package discovery may advance without converting catalog presence, metadata, or a local policy decision into scientific authority
Transformation: bind one fully verified R2.5 release bundle to one explicit local registry policy through a deterministic unsigned attestation
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: a named gate can establish that one exact R2.5 bundle matches one exact local catalog policy
Claims-Forbidden: scientific truth, clinical validity, ClinicalAuthority, ClinicalRelease, public registry publication, namespace ownership, issuer identity, remote signature authority, attested execution, independent replay, or R3 physical extraction
Assumptions: the original R2.5 bundle inputs remain available and registry policy v1 keeps publication disabled
Write-Set: tools/science_boundary/registry_attestation.py; schemas/sounio.registry-attestation*.v1.schema.json; scripts/ci/registry_attestation_spec_gate.{py,sh}; scripts/ci/sounio_package_support_gate.sh; docs/ecosystem/{REGISTRY_ATTESTATION_SPEC.md,registry-attestation-policy.example.toml,REGISTRY_ARCHITECTURE.md,SOUNIO_TOML_SPEC.md,ECOSYSTEM_ROADMAP_2026.md}; docs/architecture/science-research-boundary.md; docs/internal/concepts/{science-research-boundary.md,registry.tsv}; docs/governance/topic-registry.v1.json; tools/registry/README.md; .claude/llm_offload_log.md
Read-Set: tools/science_boundary/{attestor.py,package_release.py}; schemas/sounio.package-release-bundle.v1.schema.json; package manifests and claim contracts
Positive-Witness: verified R2.5 bundle emits a deterministic POLICY_MATCH attestation that round-trips against the same sources, policies, and compiler
Negative-Witness: denied ring, visibility, claim, malformed policy, mutated input, occupied output, and forged or rehashed attestation all refuse
Acceptance-Gate: SOUNIO_REGISTRY_ATTESTATION_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/registry_attestation_spec_gate.sh
Integration-Target: codex/package-boundary-release-r25-20260717, then origin/main after R2.5 lands
Authoritative-Only-If: the R0-R2, R2.5, and R2.6 gates pass with the same current-source raw Madaros AST collector
```

## Initial Integration Receipt

```text
Semantic-Outcome: executable R0-R2 package boundary
Concept-Status-Before: hypothesis
Concept-Status-After: executable
Distinctions-Added: ring dependency authority; claim-class authorization; receipt identity versus assurance
Distinctions-Preserved: compile success versus runtime parity; formal model versus empirical claim; computational provenance versus physical causality
Distinctions-Erased: none
Evidence-Run: SOUNIO_SCIENCE_BOUNDARY_MADAROS_BIN=artifacts/self-hosted/madaros bash scripts/ci/science_boundary_gate.sh -> SOUNIO_SCIENCE_BOUNDARY_GATE_PASS (178 assertions)
Fallback-Path: host syntax closure is advisory-only and always yields UNKNOWN without a raw AST report
Legacy-Kept: [epistemic] parser retained for compatibility with W-SRB-LEGACY-001 and zero claim authority
Conflicting-Lanes: none in the declared write-set at lane creation
Next-Semantic-Interface: package-boundary-receipt
```

## R2.5 Integration Receipt

```text
Semantic-Outcome: executable atomic boundary for opt-in local package release bundles
Concept-Status-Before: executable R0-R2 boundary with package-boundary-receipt pending
Concept-Status-After: executable R0-R2 plus R2.5 release promotion boundary
Distinctions-Added: receipt finalization versus bundle promotion; bundle identity versus registry attestation
Distinctions-Preserved: compilation versus scientific validity; claim authorization versus claim truth; local release versus publication; identity versus independent replay
Distinctions-Erased: none
Compiler-Source-Snapshot: git archive from 03674dd3a4e3fed160016cfd2da4640ce704f360 over current compiler/build inputs, sha256=85d92d49ae89e3f8bbe3095792a0de49ef640437af3bde240d3cac8d2e07ab7f
Evidence-Build: Slurm job 6299 on gpuorangefs-r770-proxmox -> COMPLETED 0:0 in 00:03:11; source-tracking build emitted 98756167-byte Madaros sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
Evidence-Closure: SOUNIO_BOUNDARY_CLOSURE_V1 status=complete nodes=main.sio|greet.sio edge=main.sio->greet.sio saturated=false parse_failed=false
Evidence-Run: SOUNIO_PACKAGE_BOUNDARY_MADAROS_BIN=<source-fresh-elf> bash scripts/ci/package_boundary_release_gate.sh -> SOUNIO_SCIENCE_BOUNDARY_GATE_PASS (178 assertions) and SOUNIO_PACKAGE_BOUNDARY_RELEASE_GATE_PASS (65 assertions)
Supporting-Gate: bash scripts/ci/sounio_package_support_gate.sh -> SOUNIO_PACKAGE_SUPPORT_GATE_PASS
Fallback-Path: none authorized; stale Madaros or host syntax closure yields UNKNOWN/refusal and no final bundle
Legacy-Kept: non-strict pkg build and [epistemic] compatibility remain unchanged; neither gains release authority
Curated-Inventory: five Phase 1 candidates inventoried; zero inferred release-eligible
Conflicting-Lanes: upstream rebracketing registry update preserved during rebase; no overlapping compiler/IR files edited by R2.5
LLM-Offload: xai/grok-4.3 review; package claim preflight and internal artifact-label scope addressed; fsync and schema-scope false positives documented in .claude/llm_offload_log.md
Remaining-Blockers: none for the local R2.5 boundary; public registry and remote attestation remain deliberately unspecified
Next-Semantic-Interface: registry-attestation-spec
```

## R2.6 Integration Receipt

```text
Semantic-Outcome: executable deterministic local registry-policy attestation for verified R2.5 bundles
Concept-Status-Before: executable R0-R2 boundary plus R2.5 local package release; registry attestation unspecified
Concept-Status-After: executable R0-R2 plus R2.5 release and R2.6 unsigned local policy-attestation boundary
Distinctions-Added: bundle identity versus catalog policy match; local policy match versus publication; attestation identity versus issuer identity
Distinctions-Preserved: compilation versus scientific validity; claim authorization versus claim truth; provenance versus assurance; local release versus public registry status
Distinctions-Erased: none
Evidence-Source-Snapshot: git archive of 028ae28d85e3260cf603b1b77b6c7fb645eaf1ab, sha256=d12cfe296a12e22eef1cb3e162c58d2d80a9a957436d58362161ba66bb35f000
Evidence-Compiler: unchanged R2.5 source-fresh Madaros, 98756167 bytes, sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
Evidence-Run: Slurm job 6394 on gpuorangefs-r770-proxmox -> COMPLETED 0:0 in 00:00:30; R0-R2 178 assertions, R2.5 65 assertions, R2.6 82 assertions all PASS
Supporting-Gates: registry_attestation_spec_gate.py -> PASS 82; sounio_package_support_gate.sh -> PASS; check_docs_registry.sh -> PASS; check_docs_consistency.sh -> PASS
Fallback-Path: none authorized; any R2.5 verification failure, policy mismatch, malformed input, or forged attestation refuses
Legacy-Kept: R2.5 bundles remain unchanged and scripts/dev/registry_serve.py continues returning 501 for publication
Conflicting-Lanes: none reported for SOUNIO-SCIENCE-RESEARCH-BOUNDARY when the R2.6 lane opened
LLM-Offload: xai/grok-4.3 review; predicate-derived check recording, rehashed claim/compiler binding adversaries, and source-root wording addressed; raw=/tmp/llm-offload-6VaZiR/
Remaining-Blockers: none for R2.6; hosted publication, namespace/issuer authority, remote signatures, independent replay, and physical extraction remain separate future interfaces
Next-Semantic-Interface: r3-physical-extraction-inventory
```

## R3 Physical Extraction Inventory Integration Receipt

```text
Semantic-Outcome: executable deterministic ownership and exact-file inventory for every science-rings.tsv root
Concept-Status-Before: executable R0-R2 plus R2.5 release and R2.6 local registry attestation; physical extraction ownership inventory pending
Concept-Status-After: executable R0-R3 inventory boundary with physical materialization explicitly not executed
Distinctions-Added: source root versus future destination; extraction plan versus completed transfer; repository file identity versus scientific authority
Distinctions-Preserved: programming-language core versus scientific package versus research artifact; catalog policy match versus publication; claim authorization versus claim truth; identity versus independent replay
Distinctions-Erased: none
Evidence-Source-Snapshot: git archive of f2e5a7f7b6f7c57929a7fdda2ba7ce5904a3b6ac, 339588982 bytes, sha256=bb5ae9a1db43bce90b00ae49b513621ffd3bd0986e067cfb4d5e6260b10037ac
Evidence-Compiler: unchanged source-fresh Madaros, sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
Evidence-Run: Slurm job 6434 on gpuorangefs-r770-proxmox -> COMPLETED 0:0 in 00:00:33, MaxRSS=1286488K; R0-R2 178 assertions, R2.5 65 assertions, R2.6 82 assertions, and R3 141 assertions all PASS
Evidence-Inventory: file sha256=be939c881942fda319d815065a11fc8a1efc7749f092481570609442466a1a2f; identity sha256=7c26219994df2364b8599586437588060d16587883d0e48bb9583c412126e91e; units=7 files=3277 bytes=53701973 retained=1 planned=5 blocked=1 status=not-executed
Supporting-Gates: physical_extraction_inventory_gate.py -> PASS 141; sounio_package_support_gate.sh -> PASS; check_docs_registry.sh -> PASS; check_docs_consistency.sh -> PASS
Fallback-Path: none authorized; incomplete coverage, invalid disposition, non-regular content, mutation, occupied output, or forged/rehashed inventory refuses
Legacy-Kept: every inventoried source remains in place; stdlib remains blocked-classification; R2.5 bundles and R2.6 attestations remain unchanged; registry publication remains disabled
Conflicting-Lanes: semantic scanner reported zero dirty bindings for SOUNIO-SCIENCE-RESEARCH-BOUNDARY; no compiler, IR, stdlib, package, example, or self-hosted source file was edited by R3
LLM-Offload: xai/grok-4.3 review -> PASS with no requested-severity issue; raw=/tmp/llm-offload-9v40Di/
Remaining-Blockers: none for the R3 inventory boundary; stdlib classification, approved destination existence, exact copy verification, and source-removal authorization are prerequisites for the separate materialization interface
Next-Semantic-Interface: r3-physical-extraction-materialization
```

## R3 Physical Extraction Materialization Integration Receipt

```text
Semantic-Outcome: executable approved-destination exact-copy and verification boundary for R3 planned units
Concept-Status-Before: executable R0-R3 ownership and exact-file inventory with materialization explicitly not executed
Concept-Status-After: executable R0-R3 local materialization interface; canonical production extraction and source removal remain not executed
Distinctions-Added: destination approval versus destination label; local byte-copy completion versus remote repository state; materialization receipt versus source-removal authority
Distinctions-Preserved: programming-language core versus scientific package versus research artifact; location versus scientific authority; file identity versus independent replay; local transfer versus ownership or publication
Distinctions-Erased: none
Evidence-Source-Snapshot: git archive of ceb242832cac525f1619dbb6935ab9a82924ebdb, 339608500 bytes, sha256=1d3a814a916d06daf34ed8c6c0e89052bef9e392630c1abfd49266646fc06cef
Evidence-Compiler: unchanged source-fresh Madaros, 98756167 bytes, sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
Evidence-Run: Slurm job 6478 on gpuorangefs-r770-proxmox -> COMPLETED 0:0 in 00:00:40, MaxRSS=1422308K; R0-R2 178 assertions, R2.5 65 assertions, R2.6 82 assertions, R3 inventory 141 assertions, and R3 materialization 167 assertions all PASS
Evidence-Materialization: receipt identity=61efbfb32b4dc74e8bea2bed82d67ff564580f73ad4eb5b8a194da94fc3ae950; policy identity=25f37fcf3b00d4842ff4d2b64960b49d5cf50374aa65e327cbfa2322e367020a; units=2 files=3 bytes=66 status=copied-and-verified source-removal=not-authorized
Supporting-Gates: physical_extraction_materialization_gate.py -> PASS 167; sounio_package_support_gate.sh -> PASS; check_docs_registry.sh plus selftest -> PASS; check_docs_consistency.sh -> PASS; check_offload_policy.sh -> PASS
Harness-Routing: Slurm job 6477 put R2.5 temporary promotion state on OrangeFS and failed with EINVAL during directory fsync after R0-R2 passed; job 6478 used node-local temporary storage with the exact same source archive, compiler, and gate and passed, so no implementation fallback was used
Fallback-Path: none authorized; incomplete approval, unsafe or occupied destinations, any source or destination mismatch, or forged/rehashed inputs refuse
Legacy-Kept: all canonical source roots remain in place; no production destination policy or materialization receipt was created; stdlib remains blocked-classification; R2.5 bundles, R2.6 attestations, and R3 inventories remain unchanged
Conflicting-Lanes: semantic scanner reported zero dirty bindings for SOUNIO-SCIENCE-RESEARCH-BOUNDARY when the materialization lane opened; no compiler, IR, stdlib, package, example, or self-hosted source file was edited
LLM-Offload: xai/grok-4.3 review completed with six documented scope/code-reading disagreements and no accepted bug; raw=/tmp/llm-offload-h5t453/
Remaining-Blockers: none for the executable local materialization interface; real destination approval and provisioning are prerequisites for canonical migration, and any source removal requires the separate authorization interface
Next-Semantic-Interface: r3-physical-extraction-source-removal-authorization
```

## R3 Source-Removal Authorization Integration Receipt

```text
Semantic-Outcome: executable authorization-only boundary for an exact removed, repaired, and gate-passing R3 temporary candidate
Concept-Status-Before: executable R0-R3 local materialization interface with source removal not authorized
Concept-Status-After: executable R0-R3 source-removal authorization interface; canonical production extraction and removal remain not executed
Distinctions-Added: candidate removal versus original-source deletion; distinct review records versus reviewer independence; authorization versus execution
Distinctions-Preserved: programming-language core versus scientific package versus research artifact; location versus scientific authority; file identity versus independent replay; local evidence versus ownership or publication
Distinctions-Erased: none
Evidence-Source-Snapshot: git archive of 4435ee73c7eac2a0742cb8522c34ec9a4bfe6bfe, 339627188 bytes, sha256=6b9253d534f7211f9728374fe8b264213506a6c1bd7b251d9967dc29228c6a44
Evidence-Compiler: unchanged source-fresh Madaros, 98756167 bytes, sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
Evidence-Run: Slurm job 6527 on gpuorangefs-r770-proxmox -> COMPLETED 0:0 in 00:00:42, MaxRSS=1422484K; R0-R2 178 assertions, R2.5 65 assertions, R2.6 82 assertions, R3 inventory 141 assertions, R3 materialization 167 assertions, and source-removal authorization 527 assertions all PASS
Evidence-Authorization: authorization identity=84f864551bcbb2265006fab62d7a19895c3deb59163a1967d380c0e027a90a28; policy identity=efb3071bb11d220a25e4e279bd54323a8db406a1a324191486fe6001100f80a0; scope identity=879ad3a7b3508335154f657907373800963ba05bbcd250adfb62f9e75cd07c2c; units=2 files=3 status=authorized-not-executed execution=not-executed
Evidence-Logs: gate sha256=ecbab58b65d92f19c8f8899732c0f5d44b122371fbd7ff48e6e45941eb4543f4; stdout sha256=f7a4369225b7ce882acce35969e81918094de84c567eaaa743da3ab7b628f83c; stderr empty sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
Supporting-Gates: physical_extraction_source_removal_authorization_gate.py -> PASS 527; sounio_package_support_gate.sh -> PASS; check_docs_registry.sh plus selftest -> PASS; check_docs_consistency.sh -> PASS; check_offload_policy.sh -> PASS
Harness-Routing: the source-removal Madaros variable is statically required to forward into the materialization stack; Slurm extraction, promotion, and candidate workspaces used node-local /tmp while durable inputs and logs remained on OrangeFS
Fallback-Path: none authorized or used; any stale materialization, incomplete review or scope, invalid repair, failing or mutating post-removal gate, source mutation, or forged/rehashed receipt refuses
Legacy-Kept: all canonical source roots remain in place; no production destination, materialization, removal policy, or authorization receipt was created; stdlib remains blocked-classification; all R2.5, R2.6, inventory, and materialization interfaces remain unchanged
Conflicting-Lanes: semantic scanner reported zero dirty bindings for SOUNIO-SCIENCE-RESEARCH-BOUNDARY; no compiler, IR, stdlib, package, example, or self-hosted source file was edited
LLM-Offload: xai/grok-4.3 review found no BLOCKER, MAJOR, or MINOR issue in the authorization-only implementation and contract; raw=/tmp/llm-offload-cfWgYf/
Remaining-Blockers: none for the executable authorization-only interface; real destination approval and materialization, approved production repairs and post-removal gates, and explicit human permission are prerequisites for any canonical execution
Next-Semantic-Interface: r3-physical-extraction-source-removal-execution
```

## R3 Source-Removal Execution Integration Receipt

```text
Semantic-Outcome: executable policy-bound removal, repair, rollback, and post-execution verification boundary for one exact explicitly marked local R3 tree
Concept-Status-Before: executable R0-R3 source-removal authorization interface with execution explicitly not performed
Concept-Status-After: executable R0-R3 local source-removal execution interface; canonical production cutover remains not approved or executed
Distinctions-Added: authorization versus execution policy; pre-commit failure versus promoted receipt; execution gate versus isolated verification gate; disposable local root versus canonical source tree
Distinctions-Preserved: programming-language core versus scientific package versus research artifact; location versus scientific authority; deterministic identity versus independent replay; local operator label versus organizational authority
Distinctions-Erased: none
Evidence-Source-Snapshot: git archive of 9ab7d49a1e7d24e6baf456a7e7a490f438958e35, 339644024 bytes, sha256=31df0a309a7fac2cf7703bda5931f093b216137c9072cccd9aa5033465313323
Evidence-Compiler: unchanged source-fresh Madaros, 98756167 bytes, sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
Evidence-Run: Slurm job 6558 on gpuorangefs-r770-proxmox -> COMPLETED 0:0 in 00:00:45, MaxRSS=1422288K; R0-R2 178 assertions, R2.5 65 assertions, R2.6 82 assertions, R3 inventory 141 assertions, R3 materialization 167 assertions, source-removal authorization 527 assertions, and source-removal execution 164 assertions all PASS
Evidence-Execution: execution identity=682791965ae6f553f87faf3a77fea395af71c804dfd42ca76c073ea828b803ba; policy identity=913a5d53e4b1a5061216b00c9d8a8810534004d5888b2cf8b0cbd21cf177955b; authorization identity=ccd1d6e24d573879b85ea031c3755371700da70c3e3fb83b9a2679c9747f1fa3; units=2 files=3 status=executed-and-verified source-removal=executed assurance=identity-only
Evidence-Logs: gate sha256=fc72bc6f1a8eb5992491766f7fb7b8d034cc7a8d2c3faec1e3146e673225ee41; stdout sha256=8a80922806fb8281309047574da9a9d017faa383c3f09a32b9d46c6827514d3b; stderr empty sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
Supporting-Gates: physical_extraction_source_removal_execution_gate.py -> PASS 164; sounio_package_support_gate.sh -> PASS; check_docs_registry.sh plus selftest -> PASS; check_docs_consistency.sh -> PASS; check_offload_policy.sh -> PASS
Harness-Routing: the execution Madaros variable forwarded into the complete authorization stack; Slurm extraction, promotion, transaction, candidate, and verification workspaces used node-local /tmp while immutable inputs and durable logs remained on OrangeFS
Fallback-Path: none authorized or used; invalid identity or confirmation, occupied or destination-contained receipt, locked or stale root, failing or mutating execution gate, mutated verification copy, rollback mismatch, or forged/rehashed receipt refuses
Legacy-Kept: all canonical scientific-package and research source roots remain present; no production destination, materialization, removal, authorization, execution policy, or execution receipt was created; all earlier R2.5, R2.6, inventory, materialization, and authorization interfaces remain unchanged
Conflicting-Lanes: semantic scanner reported zero dirty bindings for SOUNIO-SCIENCE-RESEARCH-BOUNDARY; no compiler, IR, stdlib, package, example, self-hosted, or canonical scientific source file was edited
LLM-Offload: xai/grok-4.3 reviewed executor, adversarial gate, schemas, and docs in three bounded inputs and found no BLOCKER or MAJOR issue; raw=/tmp/llm-offload-sWBX3O/,/tmp/llm-offload-zSzogY/,/tmp/llm-offload-UgMEQv/
Remaining-Blockers: none for the executable policy-bound local interface; a real canonical root, real destination state, production repairs and gates, operator approval, and recovery procedure remain prerequisites for canonical cutover
Next-Semantic-Interface: r3-physical-extraction-canonical-cutover-approval
```

## R3 Canonical Cutover Approval Integration Receipt

```text
Semantic-Outcome: executable approval-only boundary for one exact canonical and destination Git repository set after complete disposable cutover and restoration rehearsal
Concept-Status-Before: executable R0-R3 local source-removal execution interface with canonical production cutover not approved or executed
Concept-Status-After: executable R0-R3 canonical-cutover approval interface; canonical cutover execution and source removal remain not executed
Distinctions-Added: local Git HEAD versus observed remote branch ref; fixture approval versus canonical-production approval; approval receipt versus execution receipt; rehearsal backup restoration versus production crash atomicity
Distinctions-Preserved: programming-language core versus scientific package versus research artifact; location versus scientific authority; deterministic identity versus independent signature or replay; operator label versus human or organizational authority
Distinctions-Erased: none
Evidence-Source-Snapshot: git archive of 851c9eba290294135c4921ae9b2475ade889ab79, gzip size 363264985 bytes, sha256=c9556d94e8c50ef200cb6fcb9ea60fbc26a45e091a8de4960718fb140e0c9273
Evidence-Compiler: unchanged current-source Madaros, 98756167 bytes, sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
Evidence-Run: Slurm job 6602 on gpuorangefs-r770-proxmox -> COMPLETED 0:0 in 00:00:44, MaxRSS=1389440K; R0-R2 178 assertions, R2.5 65 assertions, R2.6 82 assertions, inventory 141 assertions, materialization 167 assertions, authorization 527 assertions, local execution 164 assertions, and canonical-cutover approval 172 assertions all PASS
Evidence-Approval: approval identity=15e8b3ad7b0b01a95c5a3ad717176d8901f5941740d9c9c52680a70e293074a9; policy identity=b12ae97b10691cc7ef8b77c3ec03b620304b8be4be8414f2af40f0d3ae6da6be; authorization identity=e7ccd42d064ed95c0bf79b9412fca371632c6174ca9f44f6693cc3469baae07d; destinations=2 context=disposable-fixture status=approved-not-executed execution=not-executed assurance=identity-plus-git-remote-ref
Evidence-Logs: stdout size=2566 bytes sha256=a8985fd6e21bd47428f109b6f61895c4108371417e8c191d9c9d67378c0b15f6; stderr empty sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
Supporting-Gates: physical_extraction_canonical_cutover_approval_gate.py -> PASS 172; sounio_package_support_gate.sh -> PASS; Draft 2020-12 policy and receipt instance validation -> PASS; check_docs_registry.sh plus selftest -> PASS; check_docs_consistency.sh -> PASS; semantic_coordination_gate.sh -> PASS; check_offload_policy.sh -> PASS
Harness-Routing: OrangeFS was full, so immutable archive, compiler, workspaces and logs used node-local /tmp; job 6600 stopped before gates because /usr/bin/time was absent, job 6601 passed the full earlier stack then stopped before the focused gate because Git was absent, and job 6602 passed with Git 2.43.0 provisioned in the ephemeral worker against the same archive and compiler
Fallback-Path: no implementation fallback authorized or used; malformed or stale bindings, non-standalone or dirty repositories, changed local or remote refs, incomplete or extra destinations, changed evidence, incorrect confirmations, failed or mutating rehearsal, occupied output, or forged/rehashed receipt refuse
Legacy-Kept: all canonical scientific-package and research source roots remain present; no canonical-production policy or approval receipt was created; no production destination set, operator decision, recovery plan, cutover execution, or source removal is claimed; every earlier R2.5, R2.6 and R3 interface remains unchanged
Conflicting-Lanes: semantic scanner reported zero exact path collisions; the previously missing SOUNIO-SCIENCE-RESEARCH-BOUNDARY path bindings were added; no compiler, IR, stdlib, package, example, self-hosted, or canonical scientific source file was edited
LLM-Offload: xai/grok-4.3 reviewed authorizer, gate, schemas and docs in four bounded inputs; standalone worktree enforcement and explicit receipt recording of matched CLI confirmations were incorporated, and remaining scope/code-reading disagreements were logged; raw=/tmp/llm-offload-VLRc5v/,/tmp/llm-offload-YLJKjj/,/tmp/llm-offload-OqEQb7/,/tmp/llm-offload-tlTAuZ/
Remaining-Blockers: none for the executable approval-only interface; actual canonical execution still requires a separately authored canonical-production policy and evidence set, an explicit human decision, and the r3-physical-extraction-canonical-cutover-execution interface
Next-Semantic-Interface: r3-physical-extraction-canonical-cutover-execution
```

## R3 Canonical Cutover Execution Integration Receipt

```text
Semantic-Outcome: executable policy-bound Git cutover, exact leased publication, rollback, and post-execution verification boundary for one approved disposable R3 repository set
Concept-Status-Before: executable R0-R3 canonical-cutover approval interface with canonical execution explicitly not performed
Concept-Status-After: executable R0-R3 canonical-cutover execution interface proven in disposable fixtures; Sounio canonical production cutover remains neither authorized nor executed
Distinctions-Added: approval receipt versus execution policy; pre-bound Git transition versus live ref update; published commit versus promoted execution receipt; disposable-fixture execution versus canonical-production permission
Distinctions-Preserved: programming-language core versus scientific package versus research artifact; repository location versus scientific authority; deterministic identity versus independent signature or replay; operator label versus human or organizational authority; verified ordinary rollback versus distributed crash atomicity
Distinctions-Erased: none
Evidence-Source-Snapshot: git archive of 002d5f2277da8f9510b37f8e4d0ac8e9e994a06f, gzip size 339676469 bytes, sha256=16a7cfdddd120cbd47a0b471506126fe3724fd6b1e9e27b772ce9ab73245c642
Evidence-Compiler: unchanged current-source Madaros, 98756167 bytes, sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
Evidence-Run: Slurm job 6613 on gpuorangefs-r770-proxmox -> COMPLETED 0:0 in 00:00:53, MaxRSS=1390324K; R0-R2 178 assertions, R2.5 65 assertions, R2.6 82 assertions, inventory 141 assertions, materialization 167 assertions, authorization 527 assertions, local execution 164 assertions, canonical-cutover approval 172 assertions, and canonical-cutover execution 81 assertions all PASS
Evidence-Execution: execution identity=f7ea56e8028f1a21f6f23afd316bfb93e6a27416067442fdf0da63f28e064d21; policy identity=e55de2ce7f6d82e57c7408cb1fc95948deb7fdc2c36da611ce6ebba1455a406c; approval identity=15e8b3ad7b0b01a95c5a3ad717176d8901f5941740d9c9c52680a70e293074a9; expected fixture commit=789611457cc681226baa2885391d7bbbd29a5fa7; context=disposable-fixture status=executed-and-verified source-removal=executed assurance=identity-plus-git-remote-ref-and-published-commit
Evidence-Logs: focused gate sha256=6d1413e16083ea98787dd92cb8a6bfa5ba8612e3d0135a3611e04a5853e29a4e; stdout size=3065 bytes sha256=06e16b40fd59d757c734e289388202afc1b55a3e786505839c302e41f3d25a53; stderr empty sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
Supporting-Gates: physical_extraction_canonical_cutover_execution_gate.py -> PASS 81; sounio_package_support_gate.sh -> PASS; Draft 2020-12 policy and receipt instance validation -> PASS; check_docs_registry.sh plus five negative selftests -> PASS; check_docs_consistency.sh -> PASS; semantic_coordination_gate.sh -> PASS; check_offload_policy.sh -> PASS
Harness-Routing: one immutable archive and current-source Madaros were copied to node-local /tmp because OrangeFS was full; the composed gate forwarded the same compiler through the complete stack and used Git 2.43.0 for standalone fixture repositories and local bare remotes
Fallback-Path: no implementation fallback authorized or used; invalid context, stale or dirty state, changed evidence, wrong confirmations or commit plan, occupied output, mutating gates, receipt race, ref drift, failed exact recovery, changed destination, reintroduced source, or forged/rehashed receipt refuses
Legacy-Kept: every Sounio canonical scientific-package and research source root remains present; no canonical-production approval, execution policy, destination repository set, human execution decision, production ref update, or execution receipt was created; every earlier R2.5, R2.6, and R3 interface remains unchanged
Conflicting-Lanes: semantic scanner reported zero exact path collisions before the execution lane was validated; no compiler, IR, stdlib, package, example, self-hosted, or canonical scientific source file was edited
LLM-Offload: xai/grok-4.3 reviewed executor, gate, receipt schema, and contract; two gate-maintainability improvements were incorporated and remaining scope or code-reading disagreements were logged; DeepSeek was attempted and returned Insufficient Balance; raw=/tmp/llm-offload-nKnrWX/,/tmp/llm-offload-4NnXUv/,/tmp/llm-offload-89FG0U/,/tmp/llm-offload-RL7Rbi/,/tmp/llm-offload-O2rztG/
Remaining-Blockers: none for the executable disposable-fixture interface; any Sounio production cutover still requires a separately authored canonical-production evidence set and execution policy, real destination repositories and recovery plan, and an explicit human decision
Next-Semantic-Interface: r3-physical-extraction-canonical-production-policy-and-human-decision
```

## R3 Canonical Production Gap Assessment Integration Receipt

```text
Semantic-Outcome: executable deterministic non-authorizing prerequisite-gap observation for one exact canonical source snapshot, repository catalog, and optional proposed target mapping
Concept-Status-Before: executable fixture-only canonical cutover execution interface with production policy, destinations, decision, and execution absent
Concept-Status-After: executable canonical-production gap assessment; current Sounio state has five unmapped planned targets, zero observed mapped destinations, eight missing prerequisites, and no execution authority
Distinctions-Added: repository observation versus hosting attestation; mapping proposal versus mapping approval; observed access label versus human or organizational authority; prerequisite satisfaction versus execution permission; generic continuation versus explicit human cutover decision
Distinctions-Preserved: programming-language core versus scientific package versus research artifact; location versus scientific authority; deterministic identity versus independent signature or replay; fixture proof versus production evidence; sequential observation versus atomic snapshot
Distinctions-Erased: none
Evidence-Source-Snapshot: git archive of 4dc8749a7be001ab0f9d80e5723fc292078c1527, gzip size 339700729 bytes, sha256=484fde3c4881905d60fbf601d8d42a7ba5d389cb6392b9657022c2267a791ede
Evidence-Compiler: unchanged current-source Madaros, 98756167 bytes, sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
Evidence-Run: Slurm job 6615 on gpuorangefs-r770-proxmox -> COMPLETED 0:0 in 00:00:55, MaxRSS=1390896K; R0-R2 178 assertions, R2.5 65 assertions, R2.6 82 assertions, inventory 141 assertions, materialization 167 assertions, authorization 527 assertions, local execution 164 assertions, cutover approval 172 assertions, cutover execution 81 assertions, and production-gap assessment 90 assertions all PASS
Evidence-Fixture: assessment identity=ecb2d8af0ad0f23e4843204a841992171ddaba9cb2bbbd079611610d6e154acf; catalog identity=0ff50a4e8950d7e88b5871bfb7dbf3650a3e32b7e460ce30093ff93a442c0b66; proposal identity=a2d18f6d70ed83f2e4f714fb41df103bbfdd2cfc0a96298b0cea49feebfa0af0; targets=2 status=production-evidence-and-human-decision-required authority=none
Evidence-Current-Observation: catalog observed-at=2026-07-17T21:25:23Z identity=6dae5a00fb0cf176bed2b7e1e9420cede8591a1175a3a58b5d3a555a9844460e file-sha256=ea3285fb3f788f547de5cf4de55930a399f034f21e11763e25e7b46a2460b8c7; assessment source-head=0a88da8cf1c165940cc9aa07f6832992b1206a22 main-head=aff3d4010b462af0d4e79ebc141eb6c39c4eaa50 identity=0fe82728ea24520af7792d4b5cf45c6c20e62c47a09138d0c4b81207e998e816 file-sha256=5d61566bff517177d2088b6e327f4d67dbbc14cdc1aaa02d0369ab24762fbcb3 targets=5 mapped=0 observed-destinations=0 missing-prerequisites=8 status=mapping-proposal-required authority=none
Evidence-Logs: stdout size=3580 bytes sha256=94b7da4b7222ed882248dcdf5557348de1a025dc248b623de732ac521a1624ee; stderr empty sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
Supporting-Gates: physical_extraction_canonical_production_gap_gate.py -> PASS 90; sounio_package_support_gate.sh -> PASS; Draft 2020-12 catalog, proposal, absent-assessment, and proposed-assessment schema instances -> PASS; check_docs_registry.sh plus five negative selftests -> PASS; check_docs_consistency.sh -> PASS; semantic_coordination_gate.sh -> PASS; check_offload_policy.sh -> PASS
Harness-Routing: the production-gap Madaros variable forwarded into the complete cutover-execution stack; immutable archive, compiler, extraction, transaction, fixture repositories, bare remotes, and logs used node-local /tmp; no failed Slurm attempt preceded job 6615
Fallback-Path: none authorized or used; invalid, incomplete, stale, unavailable, dirty, occupied, changed, forged, or rehashed inputs refuse or remain explicitly classified gaps
Legacy-Kept: all canonical sources and repositories remain unchanged; no mapping proposal, repository creation, production evidence set, execution policy, approval, human decision, ref update, or source removal was created or performed
Conflicting-Lanes: semantic scanner reported zero exact path collisions and only this isolated lane dirty; no compiler, IR, stdlib, package, example, self-hosted, or canonical scientific source file was edited
LLM-Offload: xai/grok-4.3 review found no requested-severity issue; a follow-up produced three documented code-reading disagreements converted into explicit gate assertions; DeepSeek returned Insufficient Balance and Gemini/OpenRouter returned HTTP 402 insufficient credits; raw=/tmp/llm-offload-4Gzpnd/,/tmp/llm-offload-388Iwt/,/tmp/llm-offload-xFvI69/
Remaining-Blockers: none for the executable gap-assessment interface; production prerequisites remain intentionally absent and are an explicit permission boundary rather than an implementation Blocker-ID
Next-Semantic-Interface: r3-physical-extraction-canonical-production-policy-and-human-decision
```

## R3 Canonical Production Mapping Decision Integration Receipt

```text
Semantic-Outcome: executable deterministic non-authorizing processing of one reviewed human mapping-selection transcription into explicit revision, provisioning, or proposal-review state
Concept-Status-Before: executable canonical-production gap assessment with five unmapped Sounio targets and no human mapping response
Concept-Status-After: executable mapping-selection processor and verifier; issue #1122 still has no response, so no real decision record, receipt, mapping proposal, repository operation, or cutover exists
Distinctions-Added: mapping-selection provenance versus authenticated human authority; request-new classification versus repository-creation permission; complete reuse selection versus proposed-not-approved mapping; mapping selection versus later explicit cutover decision
Distinctions-Preserved: programming-language core versus scientific package versus research artifact; repository location versus scientific authority; deterministic identity versus independent signature or replay; fixture proof versus production evidence; point-in-time catalog observation versus hosting attestation
Distinctions-Erased: none
Evidence-Source-Snapshot: git archive of aa0e50c6af32f55819d16191735344da5bd1c840, gzip size 339734094 bytes, sha256=ad22c9eca1dd6458a97f55f9063e6f346f70b2cf00470e910ac1a0261a925868
Evidence-Compiler: unchanged current-source Madaros, 98756167 bytes, sha256=6ace9848e8333d959819dbce56b33318185000ae25542696d4aac84960b5bb88
Evidence-Run: Slurm srun job 6635 on gpuorangefs-r770-proxmox -> COMPLETED 0:0 in 00:01:00; R0-R2 178 assertions, R2.5 65 assertions, R2.6 82 assertions, inventory 141 assertions, materialization 167 assertions, authorization 527 assertions, local execution 164 assertions, cutover approval 172 assertions, cutover execution 81 assertions, production-gap 90 assertions, and mapping-decision 204 assertions all PASS; MaxRSS unavailable because Slurm accounting was unreachable
Evidence-Fixture: decision identity=84778c0c29dbe9f0f84b488a62b16144140bd6ca2013d15fa89193f019bb5a2d; receipt identity=d633e64f62830efa1f691d5563a50ee4613a765482435ed87cfdf8e551873870; proposal identity=a2d18f6d70ed83f2e4f714fb41df103bbfdd2cfc0a96298b0cea49feebfa0af0; targets=2 status=proposal-input-complete proposal=proposed-not-approved authority=none
Evidence-Logs: stdout size=4109 bytes sha256=8eedbc7c041abf4c2087fab4843eb1b637d366dc1aee4bdb36f5e33dc1ab4f73; stderr contains only two srun allocation messages, size=92 bytes sha256=37b49d592edaf7aecf7611b86b0d178381e60d6e6434e071fa06ebc5ebe44e5e; streamed payload size=438497280 bytes sha256=877fff07323908461ea2eb813c68ef4ce73861c1e95c501dea347b2381091fc7
Supporting-Gates: physical_extraction_canonical_production_mapping_decision_gate.py -> PASS 204; composed mapping-decision shell -> PASS complete R0-R3 stack; AJV Draft 2020-12 schema compile plus reuse/request/revise instance validation -> PASS; check_docs_registry.sh plus five negative selftests -> PASS; check_docs_consistency.sh -> PASS; semantic_coordination_gate.sh -> PASS; check_offload_policy.sh -> PASS
Harness-Routing: batch job 6629 failed before gates because its inherited /tmp workdir was absent; jobs 6631 and 6633 were stopped before shell startup by cluster batch environment retrieval failure; synchronous Slurm srun job 6635 streamed the exact archive and compiler through stdin and used node-local extraction, fixtures, remotes, home, and temporary files
Fallback-Path: srun was the explicit harness-routing fallback for unavailable sbatch startup; no implementation fallback, production repository fallback, or authority fallback was authorized or used
Legacy-Kept: every canonical scientific-package and research source remains present; all repositories and refs remain unchanged; the existing catalog, gap assessor, proposal schema, fixture cutover interfaces, and earlier R2.5, R2.6, and R3 paths remain intact
Conflicting-Lanes: semantic coordination gate passed; the scanner reported this isolated lane plus unrelated historical compiler worktrees, with no exact collision on the mapping-decision write set and no runtime alert
LLM-Offload: xai/grok-4.3 found and caused removal of one unused subprocess import/catch; follow-up had no major finding; external review preserved the three-state and no-authority boundary; DeepSeek returned Insufficient Balance and Gemini/OpenRouter returned HTTP 402; raw=/tmp/llm-offload-I2t9BK/,/tmp/llm-offload-RZUIyn/,/tmp/llm-offload-FXc28Z/,/tmp/llm-offload-XL3qBP/
Remaining-Blockers: none for the executable non-authorizing processor; issue #1122 has no human response, which is an intentional permission boundary rather than an implementation Blocker-ID
Next-Semantic-Interface: r3-physical-extraction-canonical-production-policy-and-human-decision
```

## Closed Blockers

```text
Blocker-ID: BLK-20260715-science-boundary-raw-ast-closure
Status: closed
Severity: B1
Class: compiler-semantics
Owner: Codex
Lane: SCIENCE-BOUNDARY-R0-R2-20260715
Worktree: /tmp/sounio-science-boundary-r0r2-20260715
Branch: codex/science-boundary-r0r2-20260715
Files-Owned: self-hosted/compiler/module_frontend.sio; self-hosted/compiler/module_parse.sio; self-hosted/compiler/main.sio; scripts/ci/science_boundary_gate.py
Files-Read-Only: examples/projects/hello_pkg/src/main.sio; examples/projects/hello_pkg/src/greet.sio
Do-Not-Touch: bin/souc and bin/madaros outside this serialized lane
Repro: SOUNIO_SCIENCE_BOUNDARY_MADAROS_BIN=/tmp/madaros-science-boundary-r0r2-v5.elf bash scripts/ci/science_boundary_gate.sh
Observed: raw report is incomplete; main.sio has an edge with an empty dependency and unresolved import greet
Expected: complete report with main.sio and greet.sio nodes plus the main.sio -> greet.sio edge
Acceptance-Gate: SOUNIO_SCIENCE_BOUNDARY_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/science_boundary_gate.sh
Evidence-Level: E4
Evidence: current-source Slurm Madaros v13 sha256=0f7f75f03eebee313513e071c11ec529008e73a38d7d38b0b8d6438f661683ea; SOUNIO_SCIENCE_BOUNDARY_GATE_PASS tests=178
Fallback-Path: host syntax audit is advisory-only and cannot produce OK or E-SRB-001/002
Legacy-Kept: yes; [epistemic] remains read-only compatibility metadata
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: closed; preserve the real transitive witness in the required gate
Closure: v11 reports complete main.sio and greet.sio nodes plus their edge, and the shared typechecker closure returns verdict=0
```

```text
Blocker-ID: BLK-20260715-science-boundary-madaros-rebuild-segv
Status: closed
Severity: B1
Class: bootstrap-runtime
Owner: compiler bootstrap lane
Lane: SCIENCE-BOUNDARY-R0-R2-20260715
Worktree: /tmp/sounio-science-boundary-r0r2-20260715
Branch: codex/science-boundary-r0r2-20260715
Files-Owned: self-hosted/compiler/module_frontend.sio; self-hosted/compiler/main.sio
Files-Read-Only: scripts/ci/build_modular_madaros.sh; scripts/dev/souc-build-lock.sh
Do-Not-Touch: shared workspace compiler artifacts and build locks
Repro: stream the current self-hosted and stdlib sources to Slurm partition all, then invoke the v4 raw seed with --native-compile self-hosted/compiler/main.sio -o madaros-current.elf
Observed: compact modular path falls back to full IR, reports parser errors around line 5205 while loading 119 modules, then segfaults at typecheck; no ELF is emitted
Expected: current-source Madaros ELF emitted with nonzero size
Acceptance-Gate: current-source Slurm build emits an ELF which passes --science-boundary-closure examples/projects/hello_pkg/src/main.sio
Evidence-Level: E4
Evidence: /tmp/madaros-science-boundary-r0r2-v10.build.log; /tmp/madaros-science-boundary-r0r2-v11.build.log; v11 sha256=02cce2ae8a2c517b4119ccc9c3e16ae556b4eb561cc626a7fc9794a8ffe82498
Fallback-Path: none; stale or host-scanned closure cannot authorize strict builds
Legacy-Kept: yes
LLM-Offload: logged:.claude/llm_offload_log.md
Next-Action: closed; use scripts/ci/build_modular_madaros.sh rather than raw Madaros self-compilation for current-source rebuilds
Closure: canonical source-tracking bootstrap emitted v10 and v11 on Slurm; the raw self-compile segfault remains outside the accepted build path
```
