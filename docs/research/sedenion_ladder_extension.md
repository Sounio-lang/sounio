<!-- docs:meta
topic_id: repo.docs.research.sedenion-ladder-extension
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-ladder-extension
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The sedenion extension of the Furey ladder: rank 3 → 4 (interpretation OPEN)

**One line.** Frente B, vector 4/3 Part B — the *novelty* end, executed exactly. Under the
octonion → **sedenion** doubling, the octonion Standard-Model generation **persists** (B1) and the
doubling adds **exactly one** more fermionic mode (greedy maximal rank `3 → 4`, B2). This is **not** a
clean second generation (which would need `6`), and the sedenion zero-divisor geometry is a **separate**
state-level structure (B3). Every number is exact integer arithmetic, cross-verified on three
independent legs. The particle-physics **interpretation is explicitly OPEN**.

## Setup (the same ladder, one dimension up)

Part A (`docs/research/furey_octonion.md`, the sibling brick) shows Sounio reproduces Furey's
`ℂ ⊗ 𝕆 → one Standard-Model generation`. Over the octonion **left-multiplication** matrices `L_a`
(`8×8`, `L_a[a^b][b] = cd_sigma(a,b,3)`), a single-pair ladder op is the complex `8×8` matrix
`A = α(a,b) = (−L_a, L_b)` (real, imaginary parts). This is Furey's `A = 2α`, so the fermionic
normalisation carries a factor `4`. The three ops from the disjoint pairs `(1,2), (3,4), (5,6)` satisfy

```
{A_i, A_j} = 0            {A_i, A_j†} = 4·δ_ij·I ,
```

i.e. a fermionic creation/annihilation ladder of **three** modes → one generation (occupation
`n ∈ {0,1,2,3}`, the `C(3,n) = 1,3,3,1` multiplicities are `SU(3)` color).

This note applies the exact same construction to the **sedenion** left-multiplication matrices
(`cd_sigma` at `bits = 4`, `16×16`): `A = α(a,b) = (−L_a, L_b)`, adjoint `= (Reᵀ, −Imᵀ)`, anticommutator
`{X,Y} = XY + YX`, all over ℤ.

## B1 — the octonion generation persists inside the sedenion

The three octonion ladder ops `(1,2), (3,4), (5,6)`, now realised as `16×16` complex matrices, **still**
satisfy the full fermionic algebra:

```
{A_i, A_j} = 0    and    {A_i, A_j†} = 4·δ_ij·I₁₆      for all i,j ∈ {1,2,3}.
```

So the entire Standard-Model generation of Part A **survives intact** inside the sedenion — the doubling
does not spoil it. Executed value: `B1_OK 1`.

## B2 — the maximal fermionic rank is 4, not 6

The natural hope is that doubling `𝕆 → 𝕊` doubles the ladder into a **second** generation (rank
`3 → 6`). It does **not**. We compute the maximal set of mutually-fermionic single-pair ladder ops by a
deterministic **greedy** sweep: iterate pairs `(a,b)` with `1 ≤ a < b ≤ N−1` in order, keep a `chosen`
list, and add `α(a,b)` iff it is **self-fermionic** (`{X,X†} = 4·I`, `{X,X} = 0`) **and** cross-fermionic
with every already-chosen `Y` (`{X,Y} = 0` **and** `{X,Y†} = 0` — distinct, independent modes).

| algebra | index range | greedy maximal fermionic rank |
|---|---|---|
| octonion (`𝕆`, `8`-dim) | `1..7` | **3**  → one generation |
| sedenion (`𝕊`, `16`-dim) | `1..15` | **4** |

Executed values: `OCT_RANK 3`, `SED_RANK 4`. **The doubling adds exactly ONE fermionic mode**, `3 → 4`.
One extra creation/annihilation mode is *not* a second `SU(3)` generation (that would be `6`, two disjoint
triples). The naive "second generation" reading fails; the exact answer is a single additional mode.

## B3 — honest scope: the zero divisors live at the state level, not here

The sedenion is famous for its **zero divisors**, and the sibling ZD-geometry bricks
(`docs/research/sedenion_e8_boundary.md`, `sedenion_zd_fibers.md`, `sedenion_zd_quartets.md`,
`sedenion_automorphism_168.md`) map the `84 / 168 / E8`-boundary combinatorics of that structure. It is
important to state where that structure does — and does **not** — touch the present ladder:

> The basis **units** `e_i` never multiply to zero; a sedenion zero divisor is a **2-support** element
> (e.g. `e_i ± e_j`). The ladder generators here are built from single units, so the zero-divisor
> geometry enters at the **STATE / spinor** level, **not** at the ladder-generator level.

So B1/B2 (this note) and the ZD-geometry bricks describe **two different layers**: the fermionic ladder of
generators, and the zero-divisor geometry of states. They are complementary, not the same object.

## Interpretation: OPEN

The facts above are exact and certified. Their **physical** reading is **not** settled and is presented
here as speculative/open:

- B1 is a clean, positive result: the octonion → SM generation is a genuine sub-structure of the sedenion.
- B2 is a *negative* result against the most naive hope: doubling does **not** give two generations; it
  gives `3 → 4`, one extra mode. Whether that single mode carries any Standard-Model meaning (a sterile
  mode? an artifact of the greedy single-pair restriction? something visible only once state-level ZD
  structure is included, per B3?) is **an open question** — this brick does not claim an answer.

This is the whole point of the brick: Sounio spans, **exactly**, from the *established* (octonion/SM,
reproduced) to the *genuinely unexplored* (the sedenion ladder), and reports the unexplored end
honestly — the exact algebra, and an interpretation flagged open.

## Certification (exact, over ℤ — three independent legs)

- **souc**: `tests/run-pass/sedenion_ladder_extension.sio` → `SEDEXT OK`. Self-contained (`cd_sigma`
  copied verbatim, avoiding the stdlib-import defect #637); `main()` kept tiny (the `16×16` matrices live
  inside helper functions) to sidestep the monolithic-`main` codegen SIGSEGV. Runs correctly and
  **identically** under **both** `bin/souc` and the fresh stage2 compiler.
- **Python oracle**: `scripts/research/sedenion_ladder_extension_oracle.py` — a clean re-implementation of
  the Part B facts; CI gate `scripts/ci/sedenion_ladder_extension_gate.sh` diffs the souc value lines
  against the oracle (`B1_OK`, `OCT_RANK`, `SED_RANK`, `SEDEXT OK`).
- **Lean `native_decide`**: `formal/lean4/SounioSedenionLadderExtension.lean` → `b1_persists`,
  `oct_rank_3`, `sed_rank_4` (Mathlib-free, no `sorry`; the `16×16` complex greedy over indices `1..15`
  checks in ~10 s, so kept a `@[default_target]`). All three facts — including `SED_RANK = 4` — are pinned
  by the Lean kernel, not only by souc + oracle.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_ladder_extension.sio
python3 scripts/research/sedenion_ladder_extension_oracle.py
bash scripts/ci/sedenion_ladder_extension_gate.sh
(cd formal/lean4 && lake build SounioSedenionLadderExtension)
```
