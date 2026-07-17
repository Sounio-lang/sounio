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
R2.6 local registry attestation, and R3 physical extraction inventory are
executable. The named gates
prove pass, refuse, `UNKNOWN`, deterministic receipt identity, source
sensitivity, evidence and receipt tamper refusal, absence of a final ELF after
strict refusal, and a real transitive raw-AST import witness. The current-source
Madaros was built through the canonical source-tracking bootstrap path on Slurm.

R2.5 adds no new scientific claim class. It makes the existing receipt a
promotion prerequisite for one local, opt-in release bundle and preserves the
identity-versus-assurance distinction.

R2.6 binds that bundle to a local catalog policy without publication. R3 binds
the declared roots to an exact-file ownership plan while keeping extraction
status `not-executed`; neither interface promotes scientific authority.

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

`r3-physical-extraction-materialization`

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
