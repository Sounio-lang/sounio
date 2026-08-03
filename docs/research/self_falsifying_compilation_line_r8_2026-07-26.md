<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r8-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r8-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R8 — the trusted base: 47 contracts rest on one function, and it now has three independent derivations

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `TRUSTED_BASE_MAPPED__KERNELS_AGREE`
**Parents:** `self_falsifying_compilation_line_r6_2026-07-26.md` (found the sharing), `self_falsifying_compilation_line_r7_2026-07-26.md` (audited one kernel)
**Harness:** `scripts/research/self_falsifying_compilation_line_r8_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r8_gate.sh`

---

## 0. From two data points to a method

R6 found that a third of this corpus's contract pairs share a derivation. R7
audited the biggest shared function and found it sound. Two data points are not
a method. R8 makes it one, in four steps:

1. enumerate **every** shared derivation, not just the largest;
2. rank by **blast radius** — how many contracts inherit it;
3. **collapse wrappers into the kernels they call**, because a wrapper's blast
   radius is really its kernel's;
4. audit what is left — the irreducible base the corpus actually rests on.

Step 3 is what turns a list into a *base*. `omul`, `mul`, `o` and `sign_matrix`
all multiply by looking up `cds`; auditing them audits `cds` plus a loop.

---

## 1. Result

> **23 shared clusters collapse to 12 irreducible kernels and 11 wrappers. The
> wrappers with the largest reach — `omul` (10), `mul` (6), `g2auto` (4),
> `o` (3) — all reduce to one function. And that function now has **three
> independent derivations** (four implementations) agreeing on 5 440 products.**

| Clause | Result |
|---|---|
| `K1_CLUSTER_MAP` | **23** shared-derivation clusters across **47** contracts, **86** function instances. |
| `K2_TRUSTED_BASE` | **12** irreducible kernels, **11** wrappers. Top kernels: `cds` (radius 17), `cds` variant 2 (9), `compute_fibers` (3), `cd_sigma` (3). Top wrappers: `omul` (10) → `cds`, `mul` (6) → `cds`, `g2auto` (4) → `cds`,`o`, `o` (3) → `cds`. |
| `K3_KERNELS_AGREE` | **3 distinct** derivations of the sign table (4 implementations) compared over **5 440** fully comparable basis products (levels 3–6): **0 disagreements, 0 ungradeable**. |

Verdict: `SELF_FALSIFYING_R8_VERDICT TRUSTED_BASE_MAPPED__KERNELS_AGREE`.

### 1.1 The corpus already contained its own independent check

**Independence is measured, not read off the source** — asserting it from
"recursive vs iterative" is exactly the eyeballing R6 exists to replace. The
matrix, at R6's own 0.90 threshold:

| pair | similarity | |
|---|---:|---|
| `cds` v1 vs `cds` v2 | **0.929** | **same derivation**, two textual variants |
| `cds` v1 vs `cd_sigma` | 0.507 | independent |
| `cds` v2 vs `cd_sigma` | 0.477 | independent |
| `cds` v1 vs oracle | 0.107 | independent |
| `cd_sigma` vs oracle | 0.058 | independent |

**A first draft of this spec claimed four independent derivations. Measuring
says three.** `cds` v1 and v2 sit at 0.929 — above the threshold this line uses
to define shared derivation — so they are one derivation wearing two shirts, and
counting them separately would have inflated the corroboration by a third. The
three that survive:

| Derivation | Where | Shape |
|---|---|---|
| `cds` (v1 + v2) | 26 contract-slots | iterative descent over bit positions, carrying a running sign |
| `cd_sigma` | 3 contracts | **recursive** on the doubling structure |
| oracle | this harness | recursive on split arrays; no sign table exists in it at all |

**`cd_sigma` was already in the repository, in three contracts, structurally
unrelated to `cds` — and the two had never been compared.** The corroboration
this rung reports was sitting unused in the corpus. Independence checking did
not create the evidence; it noticed the evidence was there and that nobody had
put the two side by side.

That is the cheapest form this method takes: before writing a new corroborator,
check whether the corpus already contains one.

---

## 2. What the base actually is

Reduced, the 47 contracts' shared foundation is essentially **one object**: the
Cayley–Dickson sign table. Counting instances, `cds` appears directly in 26
contract-slots (17 + 9 variants) and is reached transitively by a further ~28
through wrappers.

This is the number that matters for risk: an error in that one function would
propagate identically through most of the functor-F and CD-tower results, with
every gate green, and agreement among those results would establish nothing —
because they are one result restated.

It is also the number that makes the audit affordable. **Minutes to check one
function against three others; the whole research programme to re-derive the
results it supports.** Finding the base is what buys the leverage.

---

## 3. What this is NOT

- **Not a proof the base is correct.** Three derivations agreeing on basis
  products at levels 3–6 is strong, not decisive: levels ≥ 7 and non-basis
  products are untested, and three routes can still share a misconception a
  fourth would catch.
- **Not a repair.** The corpus is still structurally dependent; those contracts
  still are not independent evidence of each other. R8 audits what they depend
  on, it does not decouple them.
- **Not a complete audit of the base.** 12 kernels were enumerated; the sign
  table was audited because it dominates the blast radius. `compute_fibers`,
  `zd_line`, `p_sub` and the smaller kernels are mapped and **unaudited**.
- **Not a compiler change.** Still Python-only, deliberately.

---

## 4. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r8_contract.py
# expect: K1 23 clusters / 86 instances, K2 12 kernels + 11 wrappers,
#         K3 3 distinct derivations (4 impls) over 5440 products, 0 disagreements,
#         SELF_FALSIFYING_R8_VERDICT TRUSTED_BASE_MAPPED__KERNELS_AGREE

bash scripts/ci/self_falsifying_compilation_line_r8_gate.sh
# expect: SELF_FALSIFYING_COMPILATION_LINE_R8_GATE_OK
```

Kernels are extracted by AST and compiled in isolation; the surrounding modules
are never imported. Counts move as contracts are added — re-run rather than
quoting §1.

---

## 5. AI disclosure

Harness, gate and spec drafted under human direction (2026-07-26). All figures
are machine-computed and re-runnable, including the independence matrix that
corrected this rung's own headline from four derivations to three. The third
derivation is re-derived inside the harness rather than imported, for the reason
R6 measures. No clinical
content. GAIDeT-ICMJE 2025.
