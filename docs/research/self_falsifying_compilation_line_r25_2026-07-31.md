<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r25-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r25-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R25 — research authority is path-default historical

**Date:** 2026-07-31
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `RESEARCH_AUTHORITY_IS_PATH_DEFAULT_HISTORICAL__GATE_REJECTS_CURRENT`
**Parents:** `self_falsifying_compilation_line_r22_2026-07-29.md` (last_validated is a literal), `self_falsifying_compilation_line_r23_2026-07-30.md` (validated_by is path ownership), `self_falsifying_compilation_line_r1_2026-07-26.md` (hermeticity)
**Harness:** `scripts/research/self_falsifying_compilation_line_r25_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r25_gate.sh`

---

## 1. Result

**Third field** in the same `docs:meta` family. R22 is a date constant. R23 is
path ownership under a validation name. This rung is `authority` — named as if
it recorded whether a document is current canon or lineage.

> **For every path under `docs/research/`, `authority` is not measured. It is
> `ACTIVE_RESEARCH_DOCS.has(relPath) ? 'repo_only' : 'historical'`, and
> `ACTIVE_RESEARCH_DOCS` is a Set of three path literals. The CI checker enforces
> the field and requires the auto-inserted lineage status note. A research page
> that claims to be current (`repo_only`) without membership of that three-item
> whitelist is a gate failure. The gate is green when almost every research
> finding declares it is historical lineage.**

Verdict: `SELF_FALSIFYING_R25_VERDICT RESEARCH_AUTHORITY_IS_PATH_DEFAULT_HISTORICAL__GATE_REJECTS_CURRENT`.

This page itself is stamped `authority: historical` and carries the lineage note
while declaring `Status: EXECUTABLE`. The defect is not described at arm's length.

## 2. Where it comes from

```
scripts/docs/governance_registry.mjs:32   const ACTIVE_RESEARCH_DOCS = new Set([
                                            'docs/research/RESEARCH_VALIDATION_SUMMARY.md',
                                            'docs/research/epistemic_algebra_review.md',
                                            'docs/research/vancomycin-uncertainty.md',
                                          ]);
scripts/docs/governance_registry.mjs:392  authority: ACTIVE_RESEARCH_DOCS.has(relPath)
                                            ? 'repo_only' : 'historical',
```

`formatHistoricalStatusNote` then inserts the boilerplate "preserved for lineage"
block. `check_docs_registry.mjs` rejects both authority mismatch and a missing
status note on historical pages.

## 3. Verified, and how

Measured 2026-07-31; the contract re-measures on every run.

| clause | | |
|---|---|---|
| `V1_WHITELIST_IS_THREE` | Set at :32 has exactly three path literals, all under `docs/research/` | currency is a three-name list |
| `V2_DEFAULT_IS_HISTORICAL` | path rule :387–392 is ternary on that Set → `repo_only` / `historical` | default is lineage |
| `V3_CORPUS_IS_LINEAGE_DEFAULT` | 320 research topics: historical 317, repo_only 2, dual 1; 317/317 historical pages carry the lineage note; all 317 non-whitelist paths are historical | almost everything is lineage by default |
| `V4_GATE_REJECTS_CURRENT` | hermetic synced farm; unmodified → rc=0; R24's page given `authority: repo_only` → rc=1 with `expected "historical"` | claiming currency fails |

**Update 2026-08-15**: `ACTIVE_RESEARCH_DOCS` grew from three paths to four
when `rna_cayley_dickson_confirmatory_preregistration_2026-08-09.md` was
whitelisted (cherry-picked from a branch-audit finding, see
`docs/audit/BRANCH_AUDIT_2026-08-15.md`). `V1_WHITELIST_IS_THREE` and
`V3_CORPUS_IS_LINEAGE_DEFAULT` now check against a `WHITELIST_SIZE = 4`
constant in the contract script instead of a bare `3` literal, so the next
legitimate whitelist change updates one named constant instead of re-deriving
which of several `== 3` checks needs to move. The clause ID keeps the name
`V1_WHITELIST_IS_THREE` for rung continuity even though the count is now 4 --
it is a label, not a live assertion of the number three.

## 4. Why this rung belongs to this line

R22 and R23 showed fields that answer the wrong question under measurement names.
`authority` varies (`historical` / `repo_only` / `dual`) and still does not
measure currency: it measures membership of a three-path Set plus a path prefix.
An EXECUTABLE research finding of this line is green in CI only when it agrees
to be historical lineage — the same family of inverted enforcement.

## 5. What this is NOT

- **Not a claim that historical is always wrong.** Many pages are genuinely
  lineage. The finding is that the field cannot tell the difference, and the
  gate enforces the path default.
- **Not a claim about the three whitelist entries.** Whether those three deserve
  `repo_only` is unmeasured here.
- **Not fixed.** Expanding the whitelist, or deriving authority from last
  commit / gate run, is a separate rung; doing it here would destroy the
  evidence.
- **Not a compiler change.**

## 6. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r25_contract.py
bash scripts/ci/self_falsifying_compilation_line_r25_gate.sh
```

Needs `node` and the governance scripts. Leaves the working tree byte-identical.

## 7. AI disclosure

Finding, contract, gate and spec drafted under human direction (2026-07-31)
while FO residuals on `lower.sio` were claimed by another agent. All four
clauses machine-measured. No clinical content. GAIDeT-ICMJE 2025.
