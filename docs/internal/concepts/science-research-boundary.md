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

The R0-R2 host attestor, compiler integration, and R2.5 package release boundary
are executable. The named gates
proves pass, refuse, `UNKNOWN`, deterministic receipt identity, source
sensitivity, evidence and receipt tamper refusal, absence of a final ELF after
strict refusal, and a real transitive raw-AST import witness. The current-source
Madaros was built through the canonical source-tracking bootstrap path on Slurm.

R2.5 adds no new scientific claim class. It makes the existing receipt a
promotion prerequisite for one local, opt-in release bundle and preserves the
identity-versus-assurance distinction.

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

`registry-attestation-spec`

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
Write-Set: tools/science_boundary/package_release.py; schemas/sounio.package-release-bundle.v1.schema.json; bin/{madaros,souc}; scripts/ci/package_boundary_release_gate.{py,sh}; docs/ecosystem/{CURATED_PACKAGES.md,SOUNIO_TOML_SPEC.md,ECOSYSTEM_ROADMAP_2026.md,curated-package-release-inventory.tsv}; docs/{architecture,internal/concepts}/science-research-boundary.md; docs/internal/concepts/registry.tsv
Read-Set: tools/science_boundary/attestor.py; schemas/sounio.{claim-contract,package-boundary-receipt}.v1.schema.json; package manifests; science-rings.tsv
Positive-Witness: strict package build emits a deterministic runnable bundle that passes full round-trip verification
Negative-Witness: UNKNOWN, unauthorized claim, tamper, mutation, missing claim, and occupied destination refuse without promoting or overwriting a final bundle
Acceptance-Gate: SOUNIO_PACKAGE_BOUNDARY_MADAROS_BIN=<rebuilt-current-source-ELF> bash scripts/ci/package_boundary_release_gate.sh
Integration-Target: origin/main after review
Authoritative-Only-If: both the 178-assertion R0-R2 gate and the R2.5 gate pass with the same current-source raw Madaros AST collector
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
