<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r7-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r7-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R7 — auditing the shared kernel: the corpus's single point of failure is sound, and now it is earned

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `SHARED_KERNEL_CORROBORATED`
**Parents:** `self_falsifying_compilation_line_r6_2026-07-26.md` (found the shared kernel), `self_falsifying_compilation_line_2026-07-26.md` (R0 §3: why agreement among dependent checks is not evidence)
**Harness:** `scripts/research/self_falsifying_compilation_line_r7_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r7_gate.sh`

---

## 0. Why this rung exists

R6 measured that **343 of 1081** pairs of this repository's research contracts
share a derivation, and that what they share is almost entirely one function:
`cds`, the Cayley–Dickson sign table, copy-pasted verbatim across the functor-F
and CD-tower corpus. Those results look mutually corroborating. They are not —
they rest on a single point of failure, and if `cds` were wrong, every result
built on it would inherit the same error identically while every gate stayed
green.

**That is what independence checking is for: it says where to look.** R7 looks.

---

## 1. Result

> **Zero disagreements in 5 440 basis products across levels 3–6.** The
> copy-pasted kernel agrees exactly with an independent re-derivation, and the
> independent derivation passes the axioms that let it adjudicate.

| Clause | Result |
|---|---|
| `A1_ORACLE_AXIOMS` | **PASS** — `e_i² = −1`, anticommutativity, `e_0` the unit, level-3 alternativity to `7.11e-15`, and zero divisors absent at level 3 / present at level 4. |
| `A2_COPIES_AGREE` | **PASS** — 6 copies of `cds` extracted from 6 different contracts; **0** disagreements between them. They are the same function. |
| `A3_KERNEL_VS_ORACLE` | **PASS** — level 3: 64 products, level 4: 256, level 5: 1024, level 6: 4096. **0 disagreements at every level.** |

Verdict: `SELF_FALSIFYING_R7_VERDICT SHARED_KERNEL_CORROBORATED`.

### What "independent" means here, concretely

`cds` computes the sign of `e_i e_j` by an **iterative descent over bit
positions**, maintaining a running sign and swapping/masking operands. The
oracle computes it by **recursive Cayley–Dickson doubling on split arrays** —
`(a,b)(c,d) = (ac − d̄b, da + bc̄)` — where no sign table appears at all: the
signs fall out of the recursion. R6 measured these two at structural similarity
`0.151`, i.e. they are not variants of one another.

So the agreement is not two copies of one idea agreeing with itself. It is two
routes to the same structure constants meeting at 5 440 points.

---

## 2. Adjudication, decided before the comparison

If the two had disagreed, the comparison alone could not say **which** was
wrong. The oracle is therefore checked first against properties that hold for
Cayley–Dickson algebras regardless of implementation — squares,
anticommutativity, the unit, octonion alternativity, and the level-3/level-4
zero-divisor boundary. An oracle failing those is not evidence against anything,
and the harness reports `SHARED_KERNEL_UNTESTABLE` in that case rather than
blaming `cds`.

Those axioms passed, so the oracle had standing. It then agreed.

---

## 3. What this changes

**Before:** nine-plus contracts agreeing about Cayley–Dickson arithmetic, which
established nothing, because they were the same code.

**After:** the shared kernel has one genuinely independent corroboration at
5 440 points. The dependency structure R6 exposed is unchanged — the contracts
are still not independent evidence of each other — but the thing they all depend
on is no longer unexamined.

This is the first result in the whole line where the machinery **prevented**
something rather than measuring a limitation. Not by blocking a build: by
telling us which of 47 contracts' worth of code was load-bearing, so that
auditing one function retired the risk in all of them.

**Cost asymmetry worth noting.** R6's sweep plus this audit is a few minutes of
compute. Independently re-deriving each of the 343 pairs' results would be the
entire research programme over again. Finding the shared dependency is what made
the audit affordable.

---

## 4. What this is NOT

- **Not a proof that `cds` is correct.** It agrees with one independent
  derivation on levels 3–6 over basis pairs. Levels ≥ 7 and non-basis products
  are untested; two derivations can share a misconception a third would catch.
- **Not a repair of the dependency.** The 343 pairs still share code, so
  agreement among them still is not corroboration. R7 audits the shared point;
  it does not make the corpus independent.
- **Not a general method.** It worked because the shared kernel had an axiomatic
  characterisation to check against. A shared kernel encoding a *choice* rather
  than a *theorem* would have no such adjudicator.
- **Not a compiler change.** Still Python-only, deliberately.

---

## 5. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r7_contract.py
# expect: A1/A2/A3 PASS, 5440 products compared, 0 disagreements,
#         SELF_FALSIFYING_R7_VERDICT SHARED_KERNEL_CORROBORATED

bash scripts/ci/self_falsifying_compilation_line_r7_gate.sh
# expect: SELF_FALSIFYING_COMPILATION_LINE_R7_GATE_OK
```

`cds` is extracted from each contract by AST, compiled in isolation and called
directly — the surrounding module is never imported, so nothing else in those
files runs.

---

## 6. AI disclosure

Harness, gate and spec drafted under human direction (2026-07-26). All figures
are machine-computed and re-runnable. The independent oracle is re-derived in
the harness rather than imported; that independence is the property R6 measures
and this rung relies on. No clinical content. GAIDeT-ICMJE 2025.
