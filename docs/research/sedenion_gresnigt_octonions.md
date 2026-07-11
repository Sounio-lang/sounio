<!-- docs:meta
topic_id: repo.docs.research.sedenion-gresnigt-octonions
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-gresnigt-octonions
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Gresnigt's three generation-octonions + a G₂ (color-side) automorphism — executed against the source

**One line.** Reproduced, in this repo's Cayley–Dickson convention, the three octonion subalgebras
Gresnigt uses to build three fermion generations (arXiv:2306.13098), and exhibited an **explicit order-3
monomial automorphism** of 𝕊 that cyclically permutes them. That automorphism lies in **G₂** (it is a
color-sector map); it is **not** Gresnigt's family `S₃` (the Brown direct factor, non-monomial), and this
is **not** a bridge from the zero-divisor monomial-168 to fermion generations. **Erratum E1 stands.**

## What was reproduced (primary-source cross-check)

Gresnigt (EPJC 2019/2024; arXiv:2306.13098) builds three generations from three octonion subalgebras of
𝕊 sharing a common quaternion, cyclically related by the family `S₃`. The paper's copies (its basis) are
the index sets `𝕆₁={1,4,5,8,9,12,13}`, `𝕆₂={2,4,6,8,10,12,14}`, `𝕆₃={3,4,7,8,11,12,15}` with intersection
`ℍ = ⟨e₄,e₈,e₁₂⟩`. Certified here, in this repo's convention:

- each `𝕆_i` is a **genuine (zero-divisor-free) octonion**, and all three are **ambient-equivalent** — each
  has exactly 6 non-anticommuting internal left-mult pairs in the 16-dim `Cℓ(8)` (`sedenion_clifford8.md`);
  `𝕆₁∩𝕆₂∩𝕆₃ = {4,8,12}`.
- the pure 3-cycle `φ = (e₁e₂e₃)(e₅e₆e₇)(e₉e₁₀e₁₁)(e₁₃e₁₄e₁₅)` (all signs `+1`, fixing `e₀,e₄,e₈,e₁₂`) is a
  **genuine algebra automorphism** of 𝕊 (verified on all 256 products), of **order 3**, cyclically
  permuting `𝕆₁→𝕆₂→𝕆₃` and fixing `ℍ` and `e₈`;
- `φ = g ⊕ g` is **diagonal** under `𝕊 = 𝕆 ⊕ 𝕆·e₈`, with `g = (e₁e₂e₃)(e₅e₆e₇)` an automorphism of the base
  octonion `{e₁..e₇}`. So **`φ ∈ G₂ = Aut(𝕆)`** — proved directly (diagonal + `g ∈ Aut(𝕆)`), not via any
  simplicity argument.

## Honest boundary — this is NOT a physics bridge

An adversarial review corrected an over-correction here; the boundary is sharp:

- **`φ ∈ G₂`, and `G₂ ⊃ SU(3)_color`**, so `φ` is a **color-sector** operation. Permuting the three
  octonions **as sets** is something a G₂ element can do; it does **not** make `φ` the family symmetry.
- Gresnigt's **family `S₃`** is, by Brown's theorem `Aut(𝕊)=G₂×S₃`, the direct factor **disjoint from
  G₂** — used *precisely because* it is outside G₂, so that **family ≠ color**. Identifying `φ` with the
  family `S₃` would conflate the two sectors the construction keeps separate (a category error).
- Therefore this **does not** connect the zero-divisor monomial-168 to fermion generations, and **does not
  amend Erratum E1**, which correctly places the monomial-168 inside G₂ (`docs/papers/sedenion-fano-geometry.md`).
- A genuine bridge would require building the fermion ideals `T_i` and showing Brown's `S₃` acts on those
  **states** as a monomial-168 element. The ideals are not built here; the 3-set permutation match is
  necessary but nowhere near sufficient. Open.

What this brick *is*: a clean, triple-legged reproduction of the paper's octonion triple, plus the
explicit G₂/color realization of their 3-set permutation.

## Certification (3 legs)
- **souc**: `tests/run-pass/sedenion_gresnigt_octonions.sio` → `GRESNIGT OK` (bin/souc AND stage2 agree).
- **Python oracle**: `scripts/research/sedenion_gresnigt_octonions_oracle.py`; gate
  `scripts/ci/sedenion_gresnigt_octonions_gate.sh`.
- **Lean `native_decide`**: `formal/lean4/SounioSedenionGresnigtOctonions.lean` — `phi_automorphism`,
  `phi_order_3`, `phi_fixes_quaternion_e8`, `phi_in_G2`, `phi_cycles_octonions`.

## Reproduce
```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_gresnigt_octonions.sio
python3 scripts/research/sedenion_gresnigt_octonions_oracle.py
bash scripts/ci/sedenion_gresnigt_octonions_gate.sh
(cd formal/lean4 && lake build SounioSedenionGresnigtOctonions)
```
Source: Gresnigt NG, arXiv:2306.13098 (EPJC 2023); octonion triple + shared quaternion from §4 / App. A.
