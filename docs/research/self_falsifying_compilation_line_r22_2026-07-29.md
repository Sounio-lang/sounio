<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r22-2026-07-29
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r22-2026-07-29
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R22 — the gate that certifies a literal

**Date:** 2026-07-29 (closed by inversion 2026-08-16)
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `PROVENANCE_IS_PRESERVED__GATE_ACCEPTS_THE_TRUE_DATE`
**Parents:** `self_falsifying_compilation_line_r1_2026-07-26.md` (claims bound to no gate; the hermeticity rule this rung obeys), `self_falsifying_compilation_line_r20_2026-07-28.md` (an oracle never committed), `self_falsifying_compilation_line_r21_2026-07-28.md` (the preceding rung)
**Harness:** `scripts/research/self_falsifying_compilation_line_r22_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r22_gate.sh`

---

## 1. Result

This line has catalogued checks that guard nothing: R1 found claims bound to no
gate, R5 found a gate nobody ran, R20 found an oracle never committed to any
branch. R22 is the inverse, and the worst-behaved member of the family — a
check that **does** run, on every push, is **green**, and certifies a string
literal.

Every governed repository document carries a `docs:meta` block with a field
named `last_validated`. It has the form of a measurement. It is a constant.

> **`last_validated` is a quoted literal at two sites in the generator, the same
> value for every topic, declared by every governed repository document, and
> older than the repository's first commit. The check wired into CI enforces the
> literal: a document that records the date it was really validated is a gate
> failure. The gate is green exactly when the field it guards carries no
> information.**

Verdict at discovery (2026-07-29): `VALIDATION_DATE_IS_A_LITERAL__GATE_REJECTS_THE_TRUE_DATE`.

## 1a. Closure — Closed by inversion, 2026-08-16

**#1752 closed the defect.** The provenance pair is now preserve-per-document:
`preservedProvenance` (governance_registry.mjs) keeps an existing well-formed
record — a real calendar date and, when present, a non-empty validator — and
the generator's defaults apply only where a document carries no record or a
malformed one. The checker stopped enforcing constant equality on the pair and
enforces shape instead, while the structural four (topic_id, authority,
audience, source_of_truth) remain registry-authoritative. The motivating
incident — a real 2026-08-13/`claude` header regressed to the placeholder by
a mandatory sync — is recorded in `.claude/llm_offload_log.md` (2026-08-16,
WAIVED row).

**The instrument was inverted the same day, not retired.** A rung that ends
as a document saying "the bug is gone" has no arms against the bug coming
back; a guard has five. The original four clauses demonstrated the defect;
the inverted four guard the fix:

| clause (inverted) | | |
|---|---|---|
| `V1_GENERATOR_PRESERVES_PROVENANCE` | the default literal still exists at two sites (a headerless doc still gets stamped) but is a FALLBACK; `preservedProvenance` and `metadataFieldsForTopic` measured over crafted records | the field has an input now |
| `V2_CORPUS_PAIR_IS_WELL_FORMED` | census over every governed repo doc: every declared date is a real `YYYY-MM-DD`, every validator non-empty; uniformity no longer required | shape is the contract, not uniformity |
| `V3_STRUCTURE_STAYS_REGISTRY_BOUND` | forged structural fields in a doc's meta do not reach the expected fields; a malformed date falls back to defaults | the inversion loosened only the pair |
| `V4_TRUTHFUL_DATE_IS_ACCEPTED` | hermetic synced farm, five arms: unmodified → `rc=0`; git-true date → `rc=0` **accepted**; the true date **survives a re-sync** (the exact regression #1752 fixed); a malformed date → `rc≥1` `expected a YYYY-MM-DD date`; a forged `topic_id` → `rc≥1` `metadata mismatch for topic_id` | the truth accepted, and the checker still bites |

**The inverted instrument earned its keep on first run.** It found that the
fix's original shape test (`^\d{4}-\d{2}-\d{2}$`) accepted `2026-13-45` —
shaped like a date, and not a day anyone validated anything on. An impossible
date carries no more information than the placeholder literal did: the
original defect, one size smaller. The contract was tightened to
`isRealValidationDate` (calendar-valid, leap years included) in the same
change, shared by generator and checker.

Verdict: `SELF_FALSIFYING_R22_VERDICT PROVENANCE_IS_PRESERVED__GATE_ACCEPTS_THE_TRUE_DATE`.

**No counts in the headline, and R1 is the reason.** R1's headline carried a
measured count and went stale the moment a claim was added to the manifest it
counted. The counts here — 1063 documents, 1213 topics, 85 days — are corpus
figures that move whenever a document is added, so they live in §3 with the
date they were measured, and the claim above is stated in the form that does not
drift. This rung moved its own denominator while being written: registering this
spec took the corpus from 1062 documents to 1063.

## 2. Where it comes from

```
scripts/docs/governance_registry.mjs:649    'last_validated: 2026-03-07',
scripts/docs/governance_registry.mjs:730    last_validated: '2026-03-07',
```

The first is in `formatRepoMetadataBlock`, the second in
`metadataFieldsForTopic`. Both take a `topic` argument. Neither lets it, or the
filesystem, or git, or any gate result reach the field. Topic-independence is
**measured** rather than argued: calling `metadataFieldsForTopic` over every
topic in the registry yields exactly one distinct value.

The enforcement is the other half. `scripts/docs/check_docs_registry.mjs`
compares each document's parsed meta against `metadataFieldsForTopic(topic)`
field by field (`metadataMismatch`, :126–131, called at :159 and :172), and the
check runs on every push — `.github/workflows/ci.yml` →
`scripts/dev/check_docs_registry.sh`. So the arrow points the wrong way. The
corpus is uniform because uniformity is what passes.

## 3. Verified, and how

*(The clause table below is the receipt of the ORIGINAL instrument as it ran
on 2026-07-29; those clause names belonged to the pre-inversion contract and
are kept verbatim as history. The clauses that run today are the inverted set
in §1a.)*

Corpus figures measured 2026-07-29 at `7d376f4b8` plus this rung's own commit;
they move with the corpus, and the contract re-measures them on every run.

| clause | | |
|---|---|---|
| `V1_VALUE_IS_A_LITERAL` | two sites, both quoted literals; one distinct value over 1213 topics | the field has no input |
| `V2_ONE_DATE_FOR_EVERY_DOC` | census of the declared field over 1063 governed repo docs: `'2026-03-07': 1063` | one date, no exceptions |
| `V3_DATE_PRECEDES_THE_REPO` | 0 commits older than the declared date; oldest commit 2026-05-31, i.e. **85 days** later | the corpus claims a validation older than its own history |
| `V4_GATE_REJECTS_THE_TRUE_DATE` | hermetic farm, **synced to consistency first**; then unmodified → `rc=0`, one truthful date → `rc=1` with `metadata mismatch for last_validated: expected "2026-03-07"` | the check rejects the truth |

**V3 replaces a measurement that was tried and discarded.** The first attempt
asked, per document, whether the declared date preceded the document's own
creation. It cannot: this repository's history begins 2026-05-31, so git
addition dates are not creation dates for anything older, and an early draft of
this rung produced per-document figures that its own spot-check contradicted.
The surviving question needs no per-document dating and cannot be got wrong:
**no commit in this repository is older than the date every document claims.**

**V4 has two arms on purpose.** An instrument that fails on everything measures
nothing. The synced farm is checked *unmodified* first and must come back green
(`rc=0`); only then is one document given the date git records for its addition
(2026-07-28 for R21's spec), and the same checker must reject it. Both arms are
pinned by the gate, so this rung cannot decay into a one-armed instrument.

**The farm is synced before it is measured, and that is a correction.** As first
written, V4 measured the farm as-copied — so it inherited the repository's
registry staleness, and went red whenever *anyone added a document without
re-running the sync*. That happened **four times in the twelve hours after this
rung landed**: this spec's own registration, then three from a co-working agent,
the last of them committed (`1f3cdb484`), which turned the docs gate red on the
branch and R22 with it. The clause was not wrong to refuse — the farm genuinely
no longer reproduced green — but staleness is already the docs-registry gate's
job, and a rung that duplicates another gate's alarm reports its neighbour's
news instead of its own.

The farm is now brought to consistency with the canonical sync **before** either
control runs, so the clause asks one question only: *given a consistent corpus,
does the checker still reject a document that tells the truth?* The separation
is verified in the condition that motivated it — V4 passes while the
repository's own docs gate is red.

**Hermeticity had to be strengthened to allow it.** R1's `B5_HERMETIC` excludes
gates that dirty the working tree, and a sync *writes* — `sync_governance_metadata.mjs`
rewrites the three governance artifacts unconditionally (`:84-86`). Hardlinks
would have written straight through into the real files. So every tree the sync
can touch is now a **real** copy (`docs`, `examples`, `paper`, `spec`,
`README.md`, and `website/src/content` — 2.4 MB of the 706 MB tree, the rest
symlinked around a real spine), and the clause **asserts** non-interference
rather than assuming it: the mtime and size of the three governance artifacts
and the subject document are fingerprinted before the farm is built and compared
after, with the gate failing on any breach.

Cost, measured: the real copy is 2.6 s cold, sub-second warm; a whole-tree
hardlink copy of 29 GB was measured at 2 min 09 s and rejected. Total rung cost
is unchanged at **~1.1 s** warm.

## 4. Why this rung belongs to this line

R1's headline went stale because its bound-claim count was a real measurement
and the corpus moved under it. That is the benign failure: a measured number
drifts, and running the gate catches it.

This is the malign one. The number never drifts, because it is not measured.
No amount of running the gate can surface it — running the gate is what
*enforces* it. The failure is invisible to the mechanism this line has spent
21 rungs building, and it was found by reading the generator, not by executing
anything. A repository can be fully wired, fully green, and still carry a field
that answers a question nobody is asking it.

The observation generalises past dates: **any field a generator fills with a
constant and a checker enforces is a certified constant, not a certified fact.**
`last_validated` is the instance that happens to be dated.

**And it applies to this page.** Registering this spec stamped it with the same
block as everything else: `last_validated: 2026-03-07`, `validated_by: A6` — a
document written on 2026-07-29 declaring validation nearly five months before it
existed, by an owner it was never shown to. The defect is not described here at
arm's length; the description carries it.

## 5. Cost, and the receipt for it

Two documents in the tree carried a date other than the literal —
`docs/papers/main/168-theorem-preprint.md` and `docs/papers/oopsla2027/outline.md`.
At `7d376f4b8` they were **gate failures**, among the 10 errors the docs-registry
check reported on this branch:

```
- docs/papers/main/168-theorem-preprint.md metadata mismatch for last_validated: expected "2026-03-07"
- docs/papers/oopsla2027/outline.md metadata mismatch for last_validated: expected "2026-03-07"
```

Whatever those two dates meant, the only way to make CI green was to overwrite
them with the constant, and the governance sync that closed the gate did
exactly that. The receipt is kept at `scripts/research/r22/checker_at_7d376f4b8.txt`
rather than paraphrased.

## 6. What this is NOT

- **Not a claim that the documents are unvalidated.** Nothing here shows that
  they were or were not reviewed. The finding is narrower and harder: the field
  cannot answer that question, and the check that reads it cannot notice.
- **Not the sweep's fault.** The governance metadata sweep in flight when this
  was found is the canonical tool's own output; the constant predates it and
  applies to every governed document. Blaming the sweep would have been the
  comfortable error.
- **Not fixed in this rung — and closed later, on the record.** The 2026-07-29
  rung deliberately changed nothing: deriving the field from evidence rewrites
  the meta block of every governed document, and doing it inside the rung that
  discovered the defect would have destroyed the evidence. The closure came
  afterwards and separately: #1752 (2026-08-16) made the pair
  preserve-per-document and shape-checked, and the same change inverted this
  instrument into the guard described in §1a. The original indictment above is
  the receipt for why the guard exists; the two verdicts (at discovery, and
  now) are both on this page on purpose.
- **Not a compiler change.**

## 7. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r22_contract.py
bash scripts/ci/self_falsifying_compilation_line_r22_gate.sh
```

Needs `node` and the two governance scripts; the gate refuses rather than
passing if any is absent. Leaves the working tree byte-identical.

## 8. AI disclosure

Finding, contract, gate and spec drafted under human direction (2026-07-29).
The literal sites were found by reading `governance_registry.mjs` while triaging
an uncommitted governance sweep; all four clauses are machine-measured. One
earlier measurement was discarded as unsound and the reason is recorded in §3
rather than removed. No clinical content. GAIDeT-ICMJE 2025.

Inversion (2026-08-16) drafted under human direction as part of #1752: the
contract and gate were rewritten to guard the fixed property, the closure
(§1a) and the real-date tightening it surfaced were recorded, and the original
finding was left in place as the receipt. All inverted clauses are
machine-measured. No clinical content. GAIDeT-ICMJE 2025.
