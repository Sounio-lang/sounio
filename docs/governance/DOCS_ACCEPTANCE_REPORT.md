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

- Total governed topics: 1284
- Repo-backed topics: 1134
- Website-backed topics: 163
- Dual-canon topics: 13
- Authority count `archived`: 25
- Authority count `dual`: 13
- Authority count `historical`: 389
- Authority count `repo_only`: 707
- Authority count `website_only`: 150

## Ownership Summary

- A0: 1 topics
- A1: 1 topics
- A2: 725 topics
- A3: 10 topics
- A4: 31 topics
- A5: 35 topics
- A6: 424 topics
- A7: 57 topics

## Locale Acceptance

- Docs collection topics with full six-locale coverage: 37/37
- All governed website docs topics are present in `en`, `pt`, `el`, `zh`, `ja`, and `es`.
- English-only website collections allowed by policy and marked in the registry: tutorials: 42; showcases: 63; blog: 21

## Evidence-Bearing Topics

- repo.docs.implementation.paper-artifact-packaging-spec: scripts/paper/package_paper_artifacts.sh, scripts/paper/paper_submission_pack.sh
- repo.frontdoor.docs-index: docs/governance/DOCS_AUTHORITY_MATRIX.md
- repo.governance.acceptance-report: docs/governance/topic-registry.v1.json, docs/governance/DOCS_AUTHORITY_MATRIX.md
- repo.governance.authority-matrix: docs/governance/topic-registry.v1.json
- website.docs.gpu: artifacts/omega/gpu_runtime_attest_gate.v1.json
- website.docs.vancomycin-uncertainty: website/public/docs/assets/vancomycin-ship/check_pass.png

## Validation Surfaces

- `bash paper/reproduce.sh`
- `bash scripts/check_docs_consistency.sh`
- `bash scripts/check_docs_registry.sh`
- `bash scripts/fast_gate.sh`
- `node website/scripts/check-docs-parity.mjs`
- `npm --prefix website run check:quality`
