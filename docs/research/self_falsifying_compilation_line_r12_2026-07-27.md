<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r12-2026-07-27
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r12-2026-07-27
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R12 — the N-version search: the measure this line proposed was already tested at 200× the scale, and it does not work

**Date:** 2026-07-27
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `PRIOR_ART_HAS_ARTEFACT_MEASURE__CLAIM_NARROWS_FOURTH`
**Parents:** `self_falsifying_compilation_line_r6_2026-07-26.md` (the measure this rung went to defend), `self_falsifying_compilation_line_r9_2026-07-26.md` (the audit that actually worked, and did not use this measure), `docs/papers/oopsla2027/outline.md` §8.5(ii) (which named this exact threat and predicted this exact outcome)
**Harness:** `scripts/research/self_falsifying_compilation_line_r12_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r12_gate.sh`

---

## 0. This rung stopped at Phase 0, on purpose

R12 was planned in five phases: an outside search, a pre-registration, a
hand-built corpus of 12 implementation pairs, a measurement, and a
retrospective. **Only Phase 0 ran.** It returned a pre-registered terminating
outcome, and the remaining phases were **not** executed.

That is the whole point, so it is stated first rather than buried: R9, R10 and
R11 each returned zero and each should have stopped. The rung format guarantees
a shippable artifact whether or not anything is found, which prices a
well-audited null at the same value as a discovery. This rung declares its stop
condition **before** the expensive work and then honours it.

---

## 1. Result

> **The measure R6 proposed — structural independence of two derivations,
> computed from source artifacts — is not new, and its central assumption was
> empirically refuted three weeks ago at 224 problems × 12 models. R6's measure
> uses strictly less information than the one that failed.**

Verdict: `SELF_FALSIFYING_R12_VERDICT PRIOR_ART_HAS_ARTEFACT_MEASURE__CLAIM_NARROWS_FOURTH`.

Both halves of the pre-registered Phase 0 search fired.

### 1.1 An artifact-level independence measure exists, and it was just tested

**Nogueira, Pattabiraman, Vieira & Campos, "A Systematic Methodology for
Evaluating Failure Independence in LLM-Generated Code", arXiv:2607.02808,
submitted 2 July 2026** (Coimbra CISUC/LASI · UNC Charlotte · UBC).

| | Their study | R6/R12 as planned |
|---|---|---|
| structural measure | **CodeBLEU** — lexical n-gram (0.1), weighted n-gram (0.1), **AST similarity (0.4)**, **dataflow similarity (0.4)** | canonicalised-AST similarity only |
| scale | **224 problems × 12 models × 5 languages × 3 prompting strategies** | 6 constructed pairs + 6 controls |
| ground truth for "wrong" | augmented test suites | hand-labelled misreadings |

Their result, which is exactly the question R12 was going to ask:

- Structural diversity is **moderate** — mean CodeBLEU 0.41 (Python), 0.53
  (Java), 0.56 (C), 0.58 (C++).
- Implementations nonetheless **"fail on the same tests far more often than
  expected under independence."**
- Three- and five-version ensembles realise only **0.43** and **0.44** of the
  reliability gain achievable under independence; **below 0.3** when drawn from
  a single model.
- **"Manual fault analysis shows that even different failure patterns often
  share root causes."**

They cite Knight & Leveson (IEEE TSE 12(1), 1986 — 27 independently developed
versions, one million test cases) for the original negative, attributed to
*shared interpretation of the specification*. AST-based similarity between
solutions is also already in use for N-version ensemble selection (Zheng et al.,
cited there as [29]; EnsLLM, arXiv:2503.15838, pairs CodeBLEU with CrossHair).

### 1.2 "Structurally distant, semantically identical" is a named, mature field

Type-4 (semantic) clones are defined as *"program fragments which are
functionally similar without being textually similar"* — and, in the clone
literature's own words, are **"structurally different enough that a model clone
detector may not find them to be within its structural similarity threshold."**

That is a description of R6's failure mode written by another field. AST-based
detectors are documented as struggling precisely here, because they rely on
syntactic similarity and cannot capture functional equivalence.

---

## 2. Why this terminates the branch rather than narrowing it

The pre-registration said outcome C would narrow the claim a fourth time. It
does more than that, and the difference matters.

**R6's measure consults strictly less information than CodeBLEU does**:
canonicalised syntax, and nothing else — no dataflow (weight 0.4), no lexical
n-grams (weight 0.2), which is **60 % of CodeBLEU by weight**. The richer
measure was tested against behavioural failure independence at 200× the scale
R12 could construct by hand, and it does not predict it. **A measure with less
information cannot do better on the same question.**

*Stated carefully, because the loose version is the slip this line keeps making.*
This is a claim about **what information is consulted**, not that R6's
comparison is a sub-computation of CodeBLEU's. It is not: R6 canonicalises
identifiers and diffs the dumps with `SequenceMatcher`; CodeBLEU's syntactic
component matches AST subtrees. The two syntactic measures are **different**,
and neither contains the other. What is nested is the *input*, not the
*algorithm* — and the argument only needs the input. An earlier draft of this
section said "a proper subset of the information CodeBLEU uses", which reads as
the containment that was never checked; same shape as the ord-3 and R8
four-vs-three catches, caught before shipping this time.

Consequences, taken rather than deferred:

1. **R12 Phases 2–4 are not run.** Twelve hand-built implementation pairs would
   be an underpowered replication (n = 6, no statistical power, as the
   pre-registration already conceded) of a settled negative.
2. **The compiler rule is not built.** The obvious next rung — a `corroborator`
   claim field where the compiler refuses codegen unless a second derivation
   measures as independent — rests on the premise that structural independence
   is worth requiring. That premise is refuted. It was separately pre-registered
   at **0/3** on the historical replay for independent reasons (one sub-token
   error, which a corroborator cannot reach; two shared misinterpretations,
   which a corroborator would agree with; and R3's E6 falsifier was written
   *knowing* the correction, so it is not self-starting).
3. **The transfer direction is against us, not for us.** Their population is
   twelve *different models* across five languages; ours is one corpus written
   by one author over time. If twelve independently-trained models still fail
   together, a single author's two derivations certainly do. The result carries
   *a fortiori* into our setting.

**What is untouched:** R6's *empirical* finding — 343/1081 pairs of this
repository's contracts share a derivation, almost all of it one copy-pasted
`cds` — stands. Copy-paste detection is Type-2 and works. What falls is the
*inference* from "structurally distant" to "independent evidence". The measure
still says reliably where evidence is **shared**; it says nothing reliable when
it reports evidence is **independent**. It is a one-sided test, and the line has
been reading it as two-sided since R6.

---

## 3. What actually worked in this arc, and it was not this

R9 audited two predictive kernels successfully. Its ground truth was
**rank-deficiency of the left-multiplication matrix** (`x` is a zero divisor iff
`L_x` is singular) — recorded at the time as *"a route the corpus's own
predicates never take."*

That is not structural independence. Nothing about `L_x` singularity is far from
`cds` in AST distance by design; it is far in **derivational route** — a
different characterisation of the same proposition, drawn from different
mathematics. R7 did the same thing with an independent recursive CD derivation
adjudicated *first against axioms* rather than against the corpus.

**Route independence is what caught things. Code independence is what got
measured for six rungs.** The literature has no mechanical measure for route
independence either — Nogueira et al. leave "what does predict failure
independence" open, and their best available lever (heterogeneous models) only
reaches 0.43 of the independent-case gain. Neither do we: R9 needed judgement to
pick `L_x`, and R9 itself reported `NO_ADJUDICATOR` on `zd_line` rather than
fake one.

Stated as the honest frontier, not as a claim: **the independence that matters
is of the derivational route, it is what worked here, and nobody — including
this line — can compute it.**

---

## 4. What this is NOT

- **Not a refutation of R6's measurement.** §2. The 343 pairs are still 343
  pairs; `cds` is still copy-pasted across the corpus.
- **Not a claim that the neighbouring work is about compile-time gating.**
  Nogueira et al. study runtime fault tolerance under majority voting. Nobody
  applies this at compile time. But applying a measure that is already known not
  to work, in a new place, and finding it does not work there either, is a
  corollary and not a contribution — which is why the branch stops.
- **Not an exhaustive search.** Two targeted questions, answered. The
  mutation-testing / test-suite-independence thread from outline §8.5(i) and the
  reproducibility-badging thread (iii) remain unchecked.
- **Not a compiler change.** The R6–R12 arc remains Python-only, and this rung
  is the one that says the compiler change should not be made.

---

## 5. Reproduce

```bash
python3 -u scripts/research/self_falsifying_compilation_line_r12_contract.py
# expect: both Phase-0 clauses fire, the pinned external figures verify,
#         SELF_FALSIFYING_R12_VERDICT PRIOR_ART_HAS_ARTEFACT_MEASURE__CLAIM_NARROWS_FOURTH

bash scripts/ci/self_falsifying_compilation_line_r12_gate.sh
# expect: SELF_FALSIFYING_COMPILATION_LINE_R12_GATE_OK
```

The contract pins the **external facts the narrowing rests on** — the arXiv
identifiers, the CodeBLEU component weights, the study's scale, and the 0.43 /
0.44 reliability figures — so that re-widening this spec's claim silently
requires editing a figure the gate checks. The token is verified at **every**
occurrence in this file, not only the `Status:` line: R11 shipped with a stale
headline and a green gate because the guard checked one line.

---

### 5.1 The pin guard failed its own negative test — the fifth sub-token catch

The first version of `C1_PRIOR_ART_PINNED` pinned the bare string `0.43`.
Negative test N2 softened the §1.1 headline — *"realise only **0.43**"* →
*"realise only **roughly half**"* — and **the gate stayed green**, because
`0.43` still occurred in §3 and §5. The pin was matching the document, not the
sentence that carries the claim.

That is this line's sub-token failure, for the fifth time, committed **inside
the guard written to prevent it** — after R11 shipped a "three hazards" headline
for hours while the count underneath was five, gate green throughout, for
exactly the same reason.

The fix is not a longer string, it is a **cardinality**: a figure that carries a
claim must occur **exactly once**, so there is no second occurrence to keep the
check satisfied. `C1` now distinguishes `unique` pins (figures) from `present`
pins (identifiers, which should recur wherever the source is cited — demanding
uniqueness there would penalise citing properly). Eight negative tests now fire:
softening the headline figure, softening its restatement in §3, duplicating a
pinned phrase into ambiguity, softening CodeBLEU's dataflow weight, drifting the
token in prose only, flipping the one-sided concession, deleting the Phase-0
stop, and building the declined corpus.

**The generalisable part:** a guard that asks *"does this string appear?"* is
satisfied by the corpus. A guard that asks *"does it appear exactly once?"* is
satisfied only by the sentence. Every check in this line up to here asked the
first question.

---

## 6. Provenance of the external claims

Every figure in §1.1 was read from the paper itself, not from a search summary.
The first automated fetch of arXiv:2607.02808 returned a summary that flagged
the paper as *"future-dated; appears to be generated content"* — a small model
mistaking a July 2026 submission for fiction. The paper was then verified two
independent ways: the arXiv abstract page, and direct decompression of the PDF's
own content streams, which agree verbatim. Building a branch-terminating
conclusion on an unverified summary would have been the failure this line
studies.

---

## 7. AI disclosure

Search, verification, gate and spec drafted under human direction (2026-07-27).
The external figures are quoted from arXiv:2607.02808 and machine-pinned by the
contract, uniquely (§5.1). The decision to stop at Phase 0 was pre-registered in
the plan before the search ran. §5.1 records a guard that passed its own
positive test and failed its negative one; it is reported because it was hit,
not anticipated. No clinical content. GAIDeT-ICMJE 2025.
