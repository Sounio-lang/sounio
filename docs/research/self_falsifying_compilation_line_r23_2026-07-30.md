<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r23-2026-07-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r23-2026-07-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R23 — validated_by is path ownership

**Date:** 2026-07-30 (closed by inversion 2026-08-16)
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `VALIDATOR_IS_PRESERVED__GATE_ACCEPTS_THE_TRUE_VALIDATOR`
**Parents:** `self_falsifying_compilation_line_r22_2026-07-29.md` (the sibling field in the same docs:meta block; a green gate that certifies a date literal), `self_falsifying_compilation_line_r1_2026-07-26.md` (claims bound to no gate; the hermeticity rule this rung obeys)
**Harness:** `scripts/research/self_falsifying_compilation_line_r23_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r23_gate.sh`

---

## 1. Result

R22 found that `last_validated` is a string literal the docs-registry gate
enforces. This rung is the other field in the same `docs:meta` block that looks
like a measurement and is not.

> **`validated_by` is filled from `topic.owner_agent`, and for every path under
> `docs/research/` the owner is the path-prefix literal `A6`. The field is named
> as if it recorded who validated the document. The CI checker enforces equality
> to the path owner: a document that records a different validator is a gate
> failure. The gate is green exactly when the field answers a directory question
> under a validation name.**

Verdict at discovery (2026-07-30): `VALIDATED_BY_IS_PATH_OWNERSHIP__GATE_REJECTS_TRUE_VALIDATOR`.

## 1a. Closure — Closed by inversion, 2026-08-16

**#1752 closed the defect, sibling and same day as R22's.** The provenance
pair is now preserve-per-document: `preservedProvenance`
(governance_registry.mjs) keeps an existing well-formed record — a real
calendar date and a non-empty validator — and the owner-agent default applies
only where a document carries no record or an empty one. The checker enforces
shape instead of equality, while the structural four (topic_id, authority,
audience, source_of_truth) remain registry-authoritative.

**The instrument was inverted the same day, not retired**, for the same
reason as R22: a receipt says the bug is gone; a guard has arms against it
coming back. The inverted clauses:

| clause (inverted) | | |
|---|---|---|
| `V1_GENERATOR_PRESERVES_VALIDATOR` | the owner_agent default sites still exist (fallback) but are bypassed; `preservedProvenance` and `metadataFieldsForTopic` measured over crafted records — a blank or absent validator falls back, a real name wins | the field has an input now |
| `V2_CORPUS_VALIDATORS_WELL_FORMED` | census over every governed repo doc: every declared validator non-empty; owner-label equality no longer required | shape is the contract, not ownership |
| `V3_STRUCTURE_STAYS_REGISTRY_BOUND` | forged structural fields do not reach the expected fields; a blank validator falls back to the owner default | the inversion loosened only the pair |
| `V4_TRUTHFUL_VALIDATOR_IS_ACCEPTED` | hermetic synced farm, five arms: unmodified → `rc=0`; `validated_by: human` → `rc=0` **accepted**; the real name **survives a re-sync** (the exact regression #1752 fixed); a blanked validator → `rc≥1` `expected a non-empty validator`; a forged `topic_id` → `rc≥1` `metadata mismatch for topic_id` | the truth accepted, and the checker still bites |

Verdict: `SELF_FALSIFYING_R23_VERDICT VALIDATOR_IS_PRESERVED__GATE_ACCEPTS_THE_TRUE_VALIDATOR`.

**Sibling of R22.** Same generator, same checker, same meta block. R22's field is
a universal constant; this field varies by path prefix and is still not a
validation record. Counts move with the corpus and live in §3.

## 2. Where it comes from

```
scripts/docs/governance_registry.mjs:650    `validated_by: ${topic.owner_agent}`,
scripts/docs/governance_registry.mjs:731    validated_by: topic.owner_agent,
scripts/docs/governance_registry.mjs:387    if (relPath.startsWith('docs/research/')) {
                                                 owner_agent: 'A6',
```

`formatRepoMetadataBlock` and `metadataFieldsForTopic` both set `validated_by`
from `owner_agent`. For research paths the owner is not discovered; it is the
literal `'A6'` in `inferRepoTopicDetails`. The checker
(`check_docs_registry.mjs` `metadataMismatch`) compares every document's field
against `metadataFieldsForTopic(topic)` and fails on any difference.

## 3. Verified, and how

*(The clause table below is the receipt of the ORIGINAL instrument as it ran
on 2026-07-30; those clause names belonged to the pre-inversion contract and
are kept verbatim as history. The clauses that run today are the inverted set
in §1a.)*

Corpus figures measured 2026-07-30; the contract re-measures them on every run.

| clause | | |
|---|---|---|
| `V1_FIELD_EQUALS_OWNER_AGENT` | two generator sites; `metadataFieldsForTopic` over 1216 topics → match=1216, mismatch=0 | field is owner_agent |
| `V2_PATH_PREFIX_OWNS_RESEARCH` | path rule at :387; 310 research topics, all owner A6; 310 docs declare `validated_by: A6` | research is one label |
| `V3_CORPUS_IS_PATH_OWNERSHIP` | 1066 governed repo docs: match=1066, mismatch=0 | never a non-owner validator |
| `V4_GATE_REJECTS_TRUE_VALIDATOR` | hermetic synced farm; unmodified → rc=0; one research page given `validated_by: human` → rc=1 with `metadata mismatch for validated_by: expected "A6"` | the check rejects a non-owner name |

**V4 has two arms on purpose**, as in R22. Unmodified farm must stay green;
only then is one subject given a validator string that is not any path-owner
label (`human`). Both arms are pinned by the gate.

**The farm is hermetic.** Same real-copy / symlink construction as R22's V4 so
the farm sync cannot write through. Working-tree mtimes of governance artifacts
and the subject are fingerprinted before and after.

## 4. Why this rung belongs to this line

R22 showed a field that is a constant and a gate that certifies the constant.
This rung shows a field that *varies* — A2, A4, A5, A6, A7 by directory — and is
still not answering the question its name asks. Variation is not evidence. The
value is path ownership under a validation name, and the checker will not let a
document say otherwise.

The observation generalises with R22: **any field a generator fills from a
non-measurement (literal or path rule) and a checker enforces is a certified
fiction when the field's name claims a measurement.**

**And it applies to this page.** Registering this spec stamps it
`validated_by: A6` because it lives under `docs/research/`, by an owner it was
never shown to. The defect is not described at arm's length; the description
carries it.

## 5. What this is NOT

- **Not a claim that A6 did or did not validate** any document. Nothing here
  shows who reviewed what. The finding is that the field cannot answer that
  question, and the check that reads it cannot notice.
- **Sibling of R22, not a duplicate.** R22 is about a universal date literal.
  This is about a path-derived owner label misnamed as validation authorship.
- **Not fixed in this rung — and closed later, on the record.** The 2026-07-30
  rung deliberately changed nothing: doing it inside the rung that discovered
  the defect would have destroyed the evidence. The closure came afterwards
  and separately: #1752 (2026-08-16) made the pair preserve-per-document and
  shape-checked, and the same change inverted this instrument into the guard
  described in §1a. The original indictment above is the receipt for why the
  guard exists; the two verdicts (at discovery, and now) are both on this page
  on purpose.
- **Not a compiler change.**

## 6. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r23_contract.py
bash scripts/ci/self_falsifying_compilation_line_r23_gate.sh
```

Needs `node` and the governance scripts; the gate refuses rather than passing
if any is absent. Leaves the working tree byte-identical.

## 7. AI disclosure

Finding, contract, gate and spec drafted under human direction (2026-07-30)
while a parallel agent held the Madaros FO method-on-Call residual. All four
clauses are machine-measured. No clinical content. GAIDeT-ICMJE 2025.

Inversion (2026-08-16) drafted under human direction as part of #1752,
sibling of R22's inversion: the contract and gate were rewritten to guard the
fixed property, and the closure (§1a) was recorded beside the original
finding. All inverted clauses are machine-measured. No clinical content.
GAIDeT-ICMJE 2025.
