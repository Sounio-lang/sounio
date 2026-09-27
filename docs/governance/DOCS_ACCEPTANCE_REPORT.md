<!-- docs:meta
topic_id: repo.governance.acceptance-report
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A0
source_of_truth: docs/governance/topic-registry.v1.json#repo.governance.acceptance-report
-->

# Docs Acceptance Report

This is the editor-in-chief acceptance snapshot for the documentation-governance wave.

## Verdict

- Status: accepted for the current documentation-governance wave when the listed validation surfaces pass.
- Dual-canon sync contract is active across repo docs, website docs, and localized docs metadata.
- Historical and archived repo docs are labeled and redirected back to the current canonical surface through the authority matrix.

## Scope, ownership, locale and evidence numbers

This file intentionally does not carry whole-corpus counts (total governed topics, per-authority
and per-owner breakdowns, evidence-bearing topics, the validation-surface list). Every one of
those numbers is a pure function of every governed doc present in the tree at scan time, so a
snapshot committed by one PR goes stale the instant any *other* PR adds or removes a governed doc
-- even though neither PR touched the other one's files. Get the live numbers on demand instead
of trusting a committed snapshot that races against concurrent merges:

    node scripts/docs/report_acceptance.mjs

Per-topic governance (metadata headers, locale coverage, evidence artifacts, broken links) stays
gated exactly as before, from `docs/governance/topic-registry.v1.json`; only the aggregate corpus counters
moved out of the committed, gated surface.
