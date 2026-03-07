<!-- docs:meta
topic_id: repo.governance.acceptance-report
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A0
source_of_truth: docs/governance/topic-registry.v1.json#repo.governance.acceptance-report
-->

# Docs Acceptance Report

This is the editor-in-chief acceptance snapshot generated from `docs/governance/topic-registry.v1.json`.

## Verdict

- Status: accepted for the current documentation-governance wave when the listed validation surfaces pass.
- Dual-canon sync contract is active across repo docs, website docs, and localized docs metadata.
- Historical and archived repo docs are labeled and redirected back to the current canonical surface through the authority matrix.

## Scope Summary

- Total governed topics: 181
- Repo-backed topics: 147
- Website-backed topics: 47
- Dual-canon topics: 13
- Authority count `archived`: 7
- Authority count `dual`: 13
- Authority count `historical`: 29
- Authority count `repo_only`: 98
- Authority count `website_only`: 34

## Ownership Summary

- A0: 1 topics
- A1: 1 topics
- A2: 62 topics
- A3: 7 topics
- A4: 21 topics
- A5: 17 topics
- A6: 34 topics
- A7: 38 topics

## Locale Acceptance

- Docs collection topics with full six-locale coverage: 30/30
- All governed website docs topics are present in `en`, `pt`, `el`, `zh`, `ja`, and `es`.
- English-only website collections allowed by policy and marked in the registry: tutorials: 6; showcases: 9; blog: 2

## Evidence-Bearing Topics

- repo.docs.implementation.paper-artifact-packaging-spec: scripts/paper/package_paper_artifacts.sh, scripts/paper/paper_submission_pack.sh
- repo.frontdoor.docs-index: docs/governance/DOCS_AUTHORITY_MATRIX.md
- repo.governance.acceptance-report: docs/governance/topic-registry.v1.json, docs/governance/DOCS_AUTHORITY_MATRIX.md
- repo.governance.authority-matrix: docs/governance/topic-registry.v1.json
- repo.paper.cpt-psp.outline: paper/reproduce.sh, scripts/paper/paper_repro_gate.sh
- repo.paper.epistemic-types.benchmarks.external-baselines-summary: paper/reproduce.sh, scripts/paper/paper_repro_gate.sh
- repo.paper.epistemic-types.benchmarks.l4-gemm-summary: paper/reproduce.sh, scripts/paper/paper_repro_gate.sh
- repo.paper.epistemic-types.benchmarks.nvidia-l4-benchmarks: paper/reproduce.sh, scripts/paper/paper_repro_gate.sh
- repo.paper.epistemic-types.readme: paper/reproduce.sh, scripts/paper/paper_repro_gate.sh
- repo.paper.oopsla2027.outline: paper/reproduce.sh, scripts/paper/paper_repro_gate.sh
- repo.paper.paper: paper/reproduce.sh, scripts/paper/paper_repro_gate.sh
- repo.paper.readme: paper/reproduce.sh, scripts/paper/paper_repro_gate.sh, scripts/paper/paper_submission_pack.sh
- repo.paper.sounio-arxiv-draft: paper/reproduce.sh, scripts/paper/paper_repro_gate.sh
- website.docs.gpu: artifacts/omega/gpu_runtime_attest_gate.v1.json
- website.docs.vancomycin-uncertainty: website/public/docs/assets/vancomycin-ship/check_pass.png

## Validation Surfaces

- `bash paper/reproduce.sh`
- `bash scripts/check_docs_consistency.sh`
- `bash scripts/check_docs_registry.sh`
- `bash scripts/fast_gate.sh`
- `bash scripts/paper/paper_repro_gate.sh`
- `bash scripts/paper/paper_submission_pack.sh`
- `node website/scripts/check-docs-parity.mjs`
- `npm --prefix website run check:quality`
