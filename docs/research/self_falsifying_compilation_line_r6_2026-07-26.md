<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r6-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r6-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R6 — evidential independence as a static property, and the discovery that a third of this corpus is not independent

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `INDEPENDENCE_CHECKABLE__CORROBORATION_BINDS`
**Parents:** `self_falsifying_compilation_line_2026-07-26.md` (R0 §3: the impossibility whose *antecedent* this rung attacks), `self_falsifying_compilation_line_r3_2026-07-26.md` (falsifiers reach outside the claim but are not self-starting), `docs/papers/oopsla2027/outline.md` (§7.1: what prior art already binds)
**Harness:** `scripts/research/self_falsifying_compilation_line_r6_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r6_gate.sh`

---

## 0. The move, and its ceiling — stated before the results

R0 §3 proved: shared misinterpretation is undetectable **when the compiler's
only evidence about a proposition is the claim's own check**. R0–R5 treated that
as a wall. It is not a wall, it is an **antecedent** — and antecedents can be
changed. R6 requires a *second, independent derivation* of the proposition and
asks the question nobody in the prior art asks:

> `build.rs` binds a build to a check's **exit status**. Snapshot testing binds
> it to a check's **literal output**. R2 bound it to the **proposition** a check
> reports. None of them ask **where the evidence came from**.

**Ceiling, stated up front rather than conceded later.** Structural
independence is a **checkable lower bound**, nothing more. Two files can share
no code and still encode the same misunderstanding, because the misunderstanding
lives in the author's head. R3 demonstrated exactly this on this corpus: its
falsifier shared no code with the harness it refuted (measured below at
similarity `0.151`) and still fired only because the author already knew the
correction. This rung rules out the cheapest failure — a corroborator that
reuses the harness's own derivation — and rules out **nothing** about shared
authorship.

---

## 1. Result

> **Independence is checkable, the guard discriminates — and 343 of 1081 pairs
> of this repository's research contracts (31.7 %) share a derivation, so they
> are not independent evidence of one another.**

| Clause | Result |
|---|---|
| `I1_IMPORT_CLOSURE` | **VACUOUS on this corpus.** Research contracts import no repo-local modules at all — only stdlib and numpy — so import disjointness passes for every pair and rules out nothing. |
| `I2_DERIVATION_DISJOINT` | **2/2** designed pairs classified as expected: the R3 falsifier vs the E6 harness it refutes → `independent` (max body similarity `0.151`); a corroborator that copy-pastes the harness's `cds`/`o`/`e` → `shared` (`1.000`). |
| `I3_CORPUS_SWEEP` | **343/1081 pairs (31.7 %)** share a derivation at similarity ≥ 0.90, over all 47 research contracts. |

Verdict: `SELF_FALSIFYING_R6_VERDICT INDEPENDENCE_CHECKABLE__CORROBORATION_BINDS`.

### 1.1 What is actually shared

Almost entirely one function: **`cds`, the Cayley–Dickson sign table**,
copy-pasted verbatim across the functor-F and CD-tower contracts
(`chingon_zd`, `routon_zd`, `trigintaduonion_zd`, `zd_qec_prediction`,
`g2_zd_fibers`, `functor_f_g2_covariance`, `cd_tower_nullity_histogram_law`,
`e_series_semantic_germ`, `r2_continuous_law_theorem`, …), plus `cd_sigma`
across the ZD-fiber spectral contracts.

**This is the finding.** Those contracts look like a family of independent
results that corroborate one another. They are not: they share one
multiplication table. If `cds` encodes an error, every result built on it
inherits that error identically, every gate stays green, and cross-checking
between them establishes nothing at all. That is a *structural* version of the
shared misinterpretation R0 measured behaviourally — and unlike R0's, it is
detectable by machine, today, without knowing which claim is wrong.

---

## 2. Two notions of independence, one of which died on contact

**Import-closure disjointness** is the obvious formalisation and it is
**vacuous here**, which is worth recording because it is the version a reader
would propose first. These contracts are self-contained scripts; nothing imports
anything local, so the check passes universally. In a corpus of self-contained
scripts a misunderstanding propagates by **copy-paste**, not by dependency.

**Derivation disjointness** — structural comparison of function bodies after
canonicalising identifiers — is the notion that discriminates. Renaming does not
defeat it; the copy-paste fixture matches at `1.000` after every local was
renamed to a positional placeholder.

### The triviality floor, and why it is post-hoc

The first run flagged the *independent* pair, matching the E6 contract's `conj`
against R3's `cd_conj` at similarity `1.000`. Both are 24-node bodies that
negate every component but the first — there is one natural way to write that,
so structural identity there carries no evidence of reuse. Bodies below **50 AST
nodes** are therefore excluded. The observed distribution has a clean gap
(trivial helpers 9–42 nodes; real derivations 64+: `cds` 223, `cd_mul` 131,
`o` 83) and the floor sits in it.

**That floor was chosen after seeing those two functions**, so re-testing it on
those two functions would be fitting a threshold to its own answer. This is why
`I3` exists: the sweep runs every pair in the corpus, and it is the sweep — not
the two designed cases — that shows the guard discriminates rather than flagging
everything or nothing.

---

## 3. What this buys, precisely

- **A compile-time obligation nobody has**: not "did the check pass" but "was
  the corroborating evidence produced independently of the claim". Mechanically
  decidable; cheap; needs nothing new from the author beyond a second artifact.
- **A corpus diagnostic**: 31.7 % here. Any project that cross-checks its
  results can now measure whether its cross-checks are independent, and the
  answer is not obviously yes.
- **A constructive complement to R0 §3.** The proposition says an independent
  derivation is *required*. R6 does not refute it — it makes the required
  independence a property the machine can verify, so the requirement can be
  enforced instead of hoped for.

---

## 4. What this is NOT

- **Not a solution to shared misinterpretation.** See §0. A lower bound only.
- **Not a claim that the 343 pairs are wrong.** `cds` may well be correct.
  The claim is that they are **not independent evidence**, so agreement among
  them is not corroboration.
- **Not a compiler change.** Deliberately: R2 cost four compiler builds under
  lock contention. The property is proved in Python first; a `corroborator`
  claim field is the next rung, and only worth building because this one came
  out `CORROBORATION_BINDS` rather than `GUARD_VACUOUS`.
- **Not tuned-to-pass.** The floor is post-hoc and §2 says so; the evaluation
  that matters is the 1081-pair sweep, not the two designed cases.

---

## 5. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r6_contract.py
# expect: I1 VACUOUS, I2 2/2 as expected, I3 343/1081 sharing a derivation,
#         SELF_FALSIFYING_R6_VERDICT INDEPENDENCE_CHECKABLE__CORROBORATION_BINDS

bash scripts/ci/self_falsifying_compilation_line_r6_gate.sh
# expect: SELF_FALSIFYING_COMPILATION_LINE_R6_GATE_OK
```

The sweep is O(pairs × functions²) and takes a couple of minutes. Counts move as
contracts are added — re-run rather than quoting §1.

---

## 6. AI disclosure

Harness, fixture, gate and spec drafted under human direction (2026-07-26). All
similarity figures are machine-computed and re-runnable. The triviality floor is
post-hoc and labelled as such. No clinical content. GAIDeT-ICMJE 2025.
