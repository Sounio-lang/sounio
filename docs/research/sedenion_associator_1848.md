<!-- docs:meta
topic_id: repo.docs.research.sedenion-associator-1848
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-associator-1848
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The associator side of the sedenion tower: 1848 = 11 × 168 (executed exactly)

**One line.** Of the `15·14·13 = 2730` ordered distinct triples of sedenion imaginary units, exactly
**`1848 = 11 × 168 = 11·|PSL(2,7)|` are non-associative**, and the factor 11 splits by output grade as
`11 = 10 + 1`: each of the 14 grades `i⊕j⊕k ≠ 8` carries `120`, and the doubling grade `8` carries
`168` (the octonion associator count). This **confirms the open conjecture** of the zero-divisor
geometry report — that the 11 lives on the *associator* side, not the zero-divisor side.

## The open question this answers

`SEDENION_ZERO_DIVISOR_GEOMETRY_REPORT.md` established that the zero-divisor side factors as `7` xor-
fibers × `42` support-quartets = `168`, with **no** factor 11, and concluded:

> "the 11 in `1848 = 11 × 168` appears to belong to the associator side of the tower geometry … the
> primitive zero-divisor side seems governed instead by 7 xor fibers and 42 support quartets. That is
> not a proof, but a clean computational constraint on any deeper conjecture."

This note discharges that constraint by **executing the associator side exactly**.

## The associator, exactly

For sedenion basis units the associator `[e_i, e_j, e_k] = (e_i e_j)e_k − e_i(e_j e_k)` is a single
component `e_(i⊕j⊕k)` with integer coefficient

```
A(i,j,k) = σ(i,j)·σ(i⊕j,k) − σ(j,k)·σ(i,j⊕k)   ∈ {−2, 0, +2}
```

(`σ` is the Cayley–Dickson sign, reused verbatim). A triple is **non-associative** iff `A ≠ 0`.

## Result

| Quantity | Value |
|---|---|
| ordered distinct triples of `{1..15}` | 2730 |
| **non-associative** | **1848 = 11 × 168** |
| — output grade `i⊕j⊕k = 8` (doubling seam) | **168** (= octonion associator count) |
| — the other 14 grades | 1680 = **10 × 168** (each grade exactly 120) |
| octonion sub-tower `{1..7}` | 168 |

So `1848 = 14·120 + 168 = 10·168 + 1·168 = 11·168`. The **e8 doubling grade** carries exactly one
copy of the octonion `168`; the remaining ten copies are spread uniformly (`120` each) across the
other 14 grades. The e8 seam is again distinguished — the same doubling grade that bounds the
zero-divisor geometry (`sedenion_e8_boundary.md`) is the grade that carries the `168` here.

## The deeper structure: why grade 8 carries 168

Decompose the `455 = C(15,3)` **unordered** triples by how many of their 6 orderings are
non-associative:

| ordering-class | unordered triples | per grade |
|---|---|---|
| 0 (associative) | **35** | — |
| 2 (semi) | **168** | 12 per grade `≠ 8` |
| 6 (fully non-associative) | **252** | 16 per grade `≠ 8`, **28 at grade 8** |

The **doubling grade 8 is distinguished**: *all* 28 of its support-triples are fully non-associative
(`G8_NOTFULL = 0`), giving `28 × 6 = 168` ordered. Every other grade splits `16 × 6 + 12 × 2 = 120`.
That is the exact reason grade 8 carries `168` and the others `120` — and a *second* `168` appears as
the semi-class (`12 × 14 = 168`). So the associator side is threaded by `168` three times over: the
grade-8 full slice, the octonion sub-tower, and the semi-class.

## Honest scope

The number `1848 = 11 × 168` is not new (it is the relation the report was probing, from the
de Marrais / tower-geometry literature). What is delivered here is: an **exact execution** that this
`1848` is precisely the ordered non-associative basis-triple count; the **grade decomposition**
`11 = 10 + 1` locating the extra copy at the `e8` grade; and a **three-leg certification** (souc,
Python oracle, Lean `native_decide`). No group-theoretic interpretation of the 11 is claimed beyond
the exact counts.

## Certification

- **Executed in Sounio:** `tests/run-pass/sedenion_associator_1848.sio` (decidable integer arithmetic,
  self-contained). Verdict `ASSOC OK`.
- **Cross-toolchain:** `scripts/ci/sedenion_associator_1848_gate.sh` diffs souc vs the Python oracle
  (`scripts/research/sedenion_associator_1848_oracle.py`) on `TOTAL/GRADE8/OTHER/OCT` and the 15
  per-grade counts. Registered in CI (Contracts). Confirmed under `bin/souc` and stage2.
- **Lean:** `formal/lean4/SounioSedenionAssociator1848.lean` `native_decide`-proves `total_1848`,
  `grade8_168`, `other_1680`, `oct_168`, and the ordering-class theorems `class0_35`, `class2_168`,
  `class6_252`, `grade8_all_full`. `lake build` green; verified by the Lean Proofs CI job.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_associator_1848.sio
python3 scripts/research/sedenion_associator_1848_oracle.py
bash scripts/ci/sedenion_associator_1848_gate.sh
(cd formal/lean4 && lake build SounioSedenionAssociator1848)
```
