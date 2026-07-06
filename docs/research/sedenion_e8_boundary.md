<!-- docs:meta
topic_id: repo.docs.research.sedenion-e8-boundary
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-e8-boundary
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The e8-boundary of the sedenion zero-divisor set (executed exactly)

**One line.** Of the 112 mixed-half signed two-support sedenion primitives, **exactly 84 participate in a
zero-divisor pair and 28 participate in none**, and the 28 dead ones are **exactly the `e8` family** —
the imaginary unit that doubles the octonions into the sedenions, plus its xor-grade-8 diagonal. The
zero-divisor geometry lives strictly *away from the doubling seam*.

**Prior work and contribution (read this first).** The `84/28` split and the e8-characterization were
already established **empirically, in Python**, in `SEDENION_ZERO_DIVISOR_GEOMETRY_REPORT.md` (finding #2:
*"the excluded cases are precisely the ones touching `e8` or having xor-label 8"*). This document does
**not** claim that discovery. Its contribution is threefold: (a) **executes** the boundary in the running
Sounio language as decidable ℤ-equality (no float, no tolerance); (b) **cross-verifies** the 28 *specific*
excluded triples element-wise across two souc builds and an independent Python implementation (guarding
against souc's silent-miscompile mode — a bare `PASS` is not proof of execution); and (c) identifies the
boundary as the **exact justification of the 168-census's generation filter** (`hi∈{9..15}, lo^hi≠8`),
which previously assumed it silently. All three legs transcribe the *same* Cayley–Dickson sign law, so the
cross-check certifies implementation-agreement, not spec-independence; the independent-spec leg is Lean's
`native_decide` (see "Lean-friendly next target").

## Setup

A **mixed-half two-support primitive** is `v = e_lo (±) e_hi` with a lower index `lo ∈ {1..7}`
(octonionic) and an upper index `hi ∈ {8..15}` (the doubled copy), with an independent sign. There are
`7 × 8 × 2 = 112` of them. (Every *participating* primitive zero-divisor vertex is mixed-half — see
`SEDENION_ZERO_DIVISOR_GEOMETRY_REPORT.md` — so this is the right candidate universe.)

A primitive `a` **participates** iff there exists a primitive `b` with `a · b = 0` in the 16-dimensional
Cayley–Dickson algebra, tested as decidable integer equality of all 16 product components.

## Result

| Quantity | Value |
|---|---|
| mixed-half candidates | 112 |
| participate | **84** |
| excluded (participate in nothing) | **28** |
| — of which touch `e8` (`hi == 8`) | 14 |
| — of which are diagonal (`lo XOR hi == 8`) | 14 |
| overlap of the two excluded families | 0 |

**The boundary is an exact algebraic invariant:**

> A mixed-half primitive is **excluded** ⟺ `hi == 8` **or** `lo XOR hi == 8`.

Both families are the pure doubling generator `e8 = ℓ` (the Cayley–Dickson unit with
`𝕊 = 𝕆 ⊕ 𝕆·ℓ`):

- `hi == 8`: the vectors `e_lo ± e_8 = e_lo ± ℓ` (an octonionic unit plus the bare doubling generator).
- `lo XOR hi == 8` ⟺ `hi = lo + 8`: the diagonal `e_lo ± e_{lo+8} = e_lo ± e_lo·ℓ = e_lo·(1 ± ℓ)`
  (an octonionic unit twisted by `(1 ± ℓ)`).

So the excluded set is exactly the primitives algebraically tied to the doubling generator `ℓ`. A
mixed-half primitive annihilates iff it avoids `e8` **both** as a support element **and** as an
xor-grade. This is the *pre-geometric boundary* of the zero-divisor locus: the "seam" of the
octonion→sedenion doubling is inert to annihilation.

## Why this is more than a re-count

The executable 168-census (`tests/run-pass/sedenion_zd_census_168.sio`) **generates** its 84
representatives with the filter `hi ∈ {9..15}` and `lo XOR hi ≠ 8` — i.e. it *silently assumes* the e8
boundary. This result proves that filter is not an arbitrary convenience but the **exact** boundary: the
28 primitives it drops annihilate with **nothing at all**. The `84 → 336 → 168 = |PSL(2,7)|` census
therefore rests on a characterized algebraic wall, not a hand-tuned generator.

## Robustness

The `84 / 28` split and the invariant are stable when the partner `b` ranges over **all** two-support
primitives (not just mixed-half) and for **both** multiplication orders `a·b` and `b·a`
(non-commutativity does not move the boundary). Verified in `scripts/research/sedenion_e8_boundary_oracle.py`.

## Certification

- **Executed exactly in Sounio:** `tests/run-pass/sedenion_e8_boundary.sio` (self-contained Lean-bridge
  `prim_prod`, no `[i64;2048]` import → no #637; decidable integer equality, no float). Verdict line
  `INVARIANT HOLDS`, gated by the run-pass output gate.
- **Cross-toolchain verified:** `scripts/ci/sedenion_e8_boundary_gate.sh` diffs the 28 **specific**
  excluded primitives emitted by souc against the independent Python oracle
  (`scripts/research/sedenion_e8_boundary_oracle.py`, transcribed from `ir_cd_sigma`) — element-wise
  identical, both reporting `84 / 28 / 14 / 14 / INVARIANT HOLDS`. Registered in CI (Contracts job).
  Confirmed identical under **two** souc builds (committed `bin/souc` and a fresh stage2) and the Python
  toolchain.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_e8_boundary.sio
python3 scripts/research/sedenion_e8_boundary_oracle.py
bash scripts/ci/sedenion_e8_boundary_gate.sh        # cross-verify: CROSS-VERIFIED 28/28
```

## Lean-friendly next target

Prove, by `native_decide`, that the participation predicate over the 112 mixed-half primitives has
support exactly the complement of `{hi = 8} ∪ {lo ⊕ hi = 8}` — turning the executed boundary into a
formal theorem alongside `SounioZeroDivisorBridge.lean`'s `prim_count_84`.
