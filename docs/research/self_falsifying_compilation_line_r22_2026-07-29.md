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

**Date:** 2026-07-29
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `VALIDATION_DATE_IS_A_LITERAL__GATE_REJECTS_THE_TRUE_DATE`
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

Verdict: `SELF_FALSIFYING_R22_VERDICT VALIDATION_DATE_IS_A_LITERAL__GATE_REJECTS_THE_TRUE_DATE`.

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

Corpus figures measured 2026-07-29 at `7d376f4b8` plus this rung's own commit;
they move with the corpus, and the contract re-measures them on every run.

| clause | | |
|---|---|---|
| `V1_VALUE_IS_A_LITERAL` | two sites, both quoted literals; one distinct value over 1213 topics | the field has no input |
| `V2_ONE_DATE_FOR_EVERY_DOC` | census of the declared field over 1063 governed repo docs: `'2026-03-07': 1063` | one date, no exceptions |
| `V3_DATE_PRECEDES_THE_REPO` | 0 commits older than the declared date; oldest commit 2026-05-31, i.e. **85 days** later | the corpus claims a validation older than its own history |
| `V4_GATE_REJECTS_THE_TRUE_DATE` | hermetic hardlink farm; unmodified → `rc=0`, one truthful date → `rc=1` with `metadata mismatch for last_validated: expected "2026-03-07"` | the check rejects the truth |

**V3 replaces a measurement that was tried and discarded.** The first attempt
asked, per document, whether the declared date preceded the document's own
creation. It cannot: this repository's history begins 2026-05-31, so git
addition dates are not creation dates for anything older, and an early draft of
this rung produced per-document figures that its own spot-check contradicted.
The surviving question needs no per-document dating and cannot be got wrong:
**no commit in this repository is older than the date every document claims.**

**V4 has two arms on purpose.** An instrument that fails on everything measures
nothing. The farm is checked *unmodified* first and must reproduce the green
result (`rc=0`); only then is one document given the date git records for its
addition (2026-07-28 for R21's spec), and the same checker must reject it. Both
arms are pinned by the gate, so this rung cannot decay into a one-armed
instrument.

**The farm is hermetic and cheap.** R1's `B5_HERMETIC` excludes gates that
dirty the working tree, so the demonstration must not edit a real document. The
four trees the checker *walks* are copied with hardlinks — same bytes, no data
movement — and every other root entry is symlinked, because the checker only
resolves those (link targets, related artifacts). That distinction is the
difference between a 0.4 s farm and a 130 s one; a whole-tree hardlink copy of
29 GB takes 2 min 09 s and was measured before being rejected. Total rung cost:
**~1.1 s**.

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
- **Not fixed.** No change is made to the generator or the checker in this
  rung. Deriving the field from evidence (last gate run, last commit, an
  explicit review record) rewrites the meta block of every governed document
  and changes what the check means; that is a separate rung with its own
  falsifier, and doing it inside the rung that discovered the defect would
  destroy the evidence.
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
