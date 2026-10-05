<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r9-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r9-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R9 — finishing the audit: six kernels corroborated, two with no adjudicator, and the method's boundary measured

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `TRUSTED_BASE_PARTIALLY_AUDITABLE`
**Parents:** `self_falsifying_compilation_line_r8_2026-07-26.md` (mapped 12 kernels, audited one), `self_falsifying_compilation_line_r6_2026-07-26.md` (why a corroborator may not import what it audits)
**Harness:** `scripts/research/self_falsifying_compilation_line_r9_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r9_gate.sh`

---

## 0. The question R8 left open

R8 audited the kernel that dominates the blast radius and flagged the method's
boundary in advance:

> *"A shared kernel encoding a **choice** rather than a **theorem** would have no
> adjudicator."*

R9 finishes the audit and turns that sentence into a measurement. Each remaining
kernel is **classified before being audited**:

| Class | Meaning |
|---|---|
| `PREDICTIVE` | asserts a structural fact checkable against independent ground truth — **these can be wrong** |
| `ALGEBRAIC` | pinned down by laws (group axioms, root conditions) |
| `MECHANICAL` | a regrouping or lookup with one possible behaviour |
| `CONVENTION` | encodes a choice; **no adjudicator exists**, and saying so is the result |

Ground truth is computed by **rank-deficiency of the left-multiplication
matrix** — `x` is a zero divisor iff `L_x` is singular. The corpus's own
predicates never take that route, so it is genuinely independent evidence rather
than the same computation restated.

---

## 1. Result

> **Six corroborated, zero divergences, two with no adjudicator. The two kernels
> that could have been wrong — the predictive ones — hold exactly.**

| Kernel | Class | Verdict | Evidence |
|---|---|---|---|
| `expected_labels` | `PREDICTIVE` | **CORROBORATED** | level 4: 7 labels, level 5: 22 labels — **exact match** against the brute-force census |
| `missing_diagonal` | `PREDICTIVE` | **CORROBORATED** | **all 7 fibers** at level 4 and **all 22** at level 5 match the actually-missing pairs |
| `compute_fibers` | `MECHANICAL` | CORROBORATED | 7 fibers regrouped by an independent implementation |
| `p_add` / `p_sub` | `ALGEBRAIC` | CORROBORATED | 300 random pairs vs a dense-vector implementation, plus `f−f=0`, `(f+g)−g=f`, no retained zero coefficients |
| `cusp_wells` | `ALGEBRAIC` | CORROBORATED | 400 random cusps; max `│x³+ax+b│ = 1.38e-14`, all satisfy `3x²+a>0` |
| `cd_sigma` (variant 2) | `ALGEBRAIC` | CORROBORATED | 5 440 products vs the `cd_sigma` R8 audited, 0 disagreements |
| `zd_line` | `CONVENTION` | **NO ADJUDICATOR** | see §2 |
| `chk` | `CONVENTION` | NOT EXTRACTABLE | nested inside another function; not a module-level kernel |

Verdict: `SELF_FALSIFYING_R9_VERDICT TRUSTED_BASE_PARTIALLY_AUDITABLE`.

**The token says `PARTIALLY` and is not being changed.** Six of eight is a good
outcome; the criteria were fixed before running, two kernels have no
adjudicator, and `FULLY_AUDITED` would be false.

### 1.1 The one that mattered

`missing_diagonal` is not bookkeeping. It asserts a **theorem about the tower**:
that the pairs absent from a fiber are exactly `{(a, a ⊕ label) : a ∈ D∖{0}}`
where `D` is the `F₂`-span of `{r, 2^m, …, 2^{b−1}}` — i.e. that a defect born at
one level propagates upward under doubling in a specific, predicted pattern.

That is a claim the algebra can refute. It does not: every fiber at levels 4 and
5 matches a census computed by singular-value rank, which knows nothing about
birth levels or xor-spans. Several level-5 and level-6 results in this corpus
rest on that prediction, and until now it had never been checked against
anything but itself.

---

## 2. Where the method stops, and why it is principled

`zd_line` is defined in terms of `nullspace(Lmat4(b))` — helpers living in its
own file. To audit it, this harness would have to reconstruct those helpers, and
the cheapest way to do that is to copy them from the file under audit. **That is
exactly the failure R6 measures**: a corroborator that imports the derivation it
is checking inherits whatever that derivation encodes.

So `zd_line` is reported as `NO_ADJUDICATOR` rather than given a fake audit. The
boundary is not a gap in effort; it is the independence requirement biting. An
honest audit of `zd_line` needs someone to re-derive the ZD-line structure from
the algebra without looking at the existing implementation — which is a research
task, not a harness.

`chk` is simpler: it is nested inside another function and is not a module-level
kernel at all. R8's cluster detection picked it up because the fingerprinting
walks all function definitions. That is a small imprecision in R8's map, recorded
rather than hidden.

---

## 3. The trusted base, complete

| | |
|---|---:|
| irreducible kernels mapped (R8) | 12 |
| audited in R7/R8 (the sign table, 3 distinct derivations) | 3 |
| audited here | 6 |
| no adjudicator | 2 |
| **corroborated share of the mapped base** | **9 / 11 extractable** |

The corpus's shared foundation is now checked against independent evidence
everywhere it can be. What remains uncorroborated is one kernel whose audit
would require redoing the research it encodes, and one artefact of the mapping.

---

## 4. What this is NOT

- **Not a proof.** Levels ≥ 6 are untested for the predictive kernels (the
  census is `O(2^b)` pairs × rank of a `2^b × 2^b` matrix); random sampling
  bounds the algebraic ones, it does not exhaust them.
- **Not a claim the corpus is correct.** It says the *shared base* agrees with
  independent computation. Results built on that base can still be wrong in
  their own reasoning — R0 §3 and R4 measured exactly that failure mode.
- **Not a decoupling.** Those contracts still share code and still are not
  independent evidence of each other.
- **Not a compiler change.** The whole R6–R9 arc is Python-only, deliberately.

---

## 5. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r9_contract.py
# expect: 6 corroborated, 0 diverged, 2 without an adjudicator,
#         SELF_FALSIFYING_R9_VERDICT TRUSTED_BASE_PARTIALLY_AUDITABLE

bash scripts/ci/self_falsifying_compilation_line_r9_gate.sh
# expect: SELF_FALSIFYING_COMPILATION_LINE_R9_GATE_OK
```

The zero-divisor census dominates the runtime (rank of a `32×32` matrix for each
of 465 index pairs at level 5). Kernels are extracted by AST and compiled in
isolation; the surrounding modules are never imported.

---

## 6. AI disclosure

Harness, gate and spec drafted under human direction (2026-07-26). All figures
are machine-computed and re-runnable. The ground-truth oracle is re-derived
inside the harness rather than imported, for the reason R6 measures. No clinical
content. GAIDeT-ICMJE 2025.
