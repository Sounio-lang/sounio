<!-- docs:meta
topic_id: repo.docs.research.zd-completeness-pincer-dag-2026-08-10
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.zd-completeness-pincer-dag-2026-08-10
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The completeness pincer: every edge marked PROVED or MEASURED

Step 0 of the current plan. The claim the paper wants is: **the pair (tr A², tr A³) is a complete
invariant of the ZD-fibre geometry, ∀n.** This file is the dependency DAG, with each edge's real
status, so that nothing downstream is built on a fit.

## Edges

| # | edge | status | where |
|---|---|---|---|
| E1 | `tr(A²)` injective in the fibre invariant `g` | **PROVED ∀n** | 2-adic valuation argument (2026-08-04) |
| E2 | `A_σ = JᵀMJ`, `JJᵀ = 2I`, exact spectral halving, `rank ≤ 2^{n−2}−1` | **PROVED ∀n** | fibre antisymmetry (2026-07-31) |
| E3 | the within-fibre deviation of `tr(A³)` IGNORES `g` | **MEASURED** | §31/(III), 2026-08-04 — never proved |
| E4 | the transfer recursion `s3(m+1) = 8·s3 + 24·cp2 + …`, `cp2(m+1) = 4·cp2 + …` | **MEASURED** | §57.49 — 75 in-sample + 92 out-of-sample transitions, 0 failures |
| E5 | the base case `Δs3(j) = 1728·[j,3]₂`, `Δcp2(j) = 0` | **MEASURED** | §57.49 — exact at `m = j`, `j = 3..7` |
| E6 | obligation (ii): `resB` holds off the six lines, for `g(W) = 0` | **PROVED ∀n** | Tiers 100–108, both reference families |
| E7 | the maximal-seam exception is a mask artifact | **MEASURED** (twice recomputed) | §57.49 |
| E8 | the 168-orbit theorem (PSL(2,7) acting on fibres) | **PROVED ∀n** | 2026-07-11 |

## What that means

Three measured edges sit in the chain: **E3, E4, E5**. Fable's count is confirmed against the tree.

- **E3 is the dangerous one.** If the within-fibre deviation does not ignore `g`, the reference pairs
  do not represent their classes and E4/E5 are about the wrong objects. It must be closed or
  declared. Open question, recorded here rather than assumed: is E3 a consequence of E6 (which
  covers exactly `g(W) = 0`) plus the orbit action of E8, or is it the still-open V1 in disguise?
- **E4 and E5 are the two lemmas that make the headline true**, and Fable's reduction applies to
  both: since the inhomogeneity is label-independent it cancels in every within-fibre difference, so
  what must be proved is only the HOMOGENEOUS pair
  `Δcp2(m+1) = 4·Δcp2(m)` and `Δs3(m+1) = 8·Δs3(m) + 24·Δcp2(m)`,
  plus the base. The 7-dimensional affine system, the closed forms in `H`, and `cp3` are discovery
  scaffolding and are NOT proof targets.

## The `8 = 2³` remark, recorded so it is not lost

A cubic trace under a Cayley–Dickson doubling whose folding map satisfies `JJᵀ = 2I` acquires `2³`.
That is where the proof of the `s3` line should come from, and it routes through E2 — which promotes
E2 from infrastructure to load-bearing.

## Status of the deviation law itself

NOT proved. `D[tri3] = 1728·8^(m−j)·[j,3]₂` rests on E4 + E5, both measured. The Lean development
proves its supporting obligation (E6), not the law.

## Step 1, first result — and a null control I had to run against myself

The homogeneous pair verified 206/206 on within-fibre pairs. **That number is weaker than it looks,
and the check that showed it is the kind this lane exists to force.**

Within a fibre, `Δcp2 = 0` in 95/95 pairs tested — `cp2` is fibre-constant. So:

- the line `Δcp2(m+1) = 4·Δcp2(m)` is **VACUOUS** on within-fibre pairs: it reads `0 = 0`;
- the `24·Δcp2` term **never fires** within a fibre, so the second line is not
  `Δs3(m+1) = 8·Δs3(m) + 24·Δcp2(m)` there but simply `Δs3(m+1) = 8·Δs3(m)`.

So E4 was one edge and is really two, of very different kinds:

| | content | status |
|---|---|---|
| **E4a** | `cp2` is fibre-constant, hence `Δcp2 = 0` | **assembly-reachable now** |
| **E4b** | `Δs3(m+1) = 8·Δs3(m)` within a fibre | **the open lemma** |

**E4a is closer than the DAG said.** `cp2_count` (Tier 95) proves `cp2 = −(H−2)(H−6)` on the `g = 0`
class CONDITIONAL on the four-sign law for its summand — and that law is exactly what Tiers 96–108
established for the reference labels: the interior through `cp2_summand_core` + `P1_mul_P3_mask` +
Bridge 1 + `starP_all_{octonion,pow2}_labels`, and the borders directly in Tiers 97–98. Discharging
`cp2_count`'s hypothesis is assembly, not new mathematics, and it turns `Δcp2 = 0` on the reference
class from measurement into a theorem.

**E4b is where the mathematics is**, and Fable's structural remark is the lead: `8 = 2³` on a cubic
trace under a doubling whose folding map satisfies `JJᵀ = 2I`. That routes through E2.

## E4b is not a new lemma — it is §57.50's obligation (i), on differences

Chasing E4b to its floor: `tri3_level_transfer` is already a THEOREM (Tier 90) —

    tri3 at level m+1  =  tri3 at level m  +  3·T1 + 3·T2 + T3

with `T1, T2, T3` the three ε-weighted orthant sums, themselves theorems. Taking within-fibre
differences kills the label-independent inhomogeneity, so what E4b needs is exactly

    3·ΔT1 + 3·ΔT2 + ΔT3  =  7·Δs3 + 24·Δcp2          [95/95 within-fibre pairs, m = 3,4,5]

which is §57.50's obligation (i) restricted to differences. So the DAG collapses further than the
last revision said: **E4b is not an independent lemma**, it is the evaluation of the ε-sums'
combination, and Tier 90 already supplies everything except that evaluation.

### The DAG, current

    E1, E2, E6, E8                    PROVED
    E4a  (cp2 fibre-constant)          assembly of Tiers 95–108 — not yet written
    E4b  = obligation (i)              THE open lemma: evaluate 3T1 + 3T2 + T3
    E5   (base case at m = j)          open, and where the q-binomial's combinatorial content sits
    E3   (deviation ignores g)         open, and the dangerous one

Three open, two of them (E4b, E5) being the two Fable named, and E3 being the one that decides
whether the other two are about the right objects.

## E4a attempted, not landed — with the diagnosis

`cp2_summand_law` assembles the four-sign law from Tiers 96–108: interior via `cp2_summand_core` +
Bridge 1 + `P3² = 1` (no `resB` needed at all), the rest matched to the border lemmas. The structure
is right — every locus has its lemma and the target values match — but it does not compile: 9 errors,
all mechanical, and of two kinds.

1. **`simp only` mangles the goal shape before the border lemma can be rewritten in.** After
   `simp only [if_neg …]` the goal reads `P3 0 b W m * P3 b (0 ^^^ W) W m` while the lemma is stated
   with `P3 b W W m` — the `0 ^^^ W` was not normalised. Rewriting the `Nat.zero_xor` FIRST, before
   any `simp`, is the fix.
2. **`fun h => hb0 h.symm` fails to infer `h`'s type** in argument position. Needs an explicit
   annotation or a `have`.

The file is restored to green and the attempt is preserved at `scratchpad/tier109_attempt.lean`.
Recorded because the first lift attempt failed the same way — mechanically, with a clean diagnosis —
and landed on the second pass once the diagnosis was in hand.

## E4a — LANDED on the second pass

`cp2_summand_law` compiles. The diagnosis written after the first attempt was the whole fix: stop
driving the `if`s by hand with `simp only` (which let the goal shape drift out from under the border
lemmas), put the case facts in the context and supply only the value equation, and let `grind`
evaluate. Twelve loci, one `grind` each.

With it, `cp2_count`'s hypothesis is discharged for any label whose interior satisfies `starP` — and
Tiers 107–108 prove exactly that for both reference families. So **E4a is a theorem**: `cp2` has the
closed form `−(H−2)(H−6)` on the reference class, hence `Δcp2 = 0`, hence the `24·Δcp2` term of the
transfer vanishes within a fibre.

### DAG after this

    E1, E2, E6, E8, E4a               PROVED
    E4b  = obligation (i)             THE open lemma: evaluate 3T1 + 3T2 + T3
    E5   base case at m = j           open — the q-binomial's combinatorial content
    E3   deviation ignores g          open — decides whether E4b and E5 are about the right objects

## Step 2 — E4b taken apart: three closed forms, and a second maximal-seam artifact

E4b was "evaluate `3T1 + 3T2 + T3`", filed as irreducible because §57.50 measured each piece
outside the probed span. **That was a fit broken by one label per level.** Off the maximal seam
`W = 2^m`, each piece has a closed form (`H = 2^(m+1)`):

    T1 = s3 + 4·cp2 + 16H − 64
    T2 = s3 + 4·cp2 −  8H + 64
    T3 = s3          + 48H − 176

486/486 off-seam label-levels at `m = 3..7`, out of sample at `m = 8`. At `W = 2^m` the three
deviate by exactly `(1, −2, 3)·2(H−4)(H−8)`, and `(3,3,1)·(1,−2,3) = 0` — so the combination never
saw it. Second maximal-seam artifact in the lane, after the `288·[m−1,2]₂` term.

Consequences for this DAG:

- **E4b is no longer one opaque lemma.** Obligation (i) follows by arithmetic from the three forms,
  on and off the seam. Within a fibre, `ΔT1 = ΔT2 = Δs3 + 4Δcp2` and `ΔT3 = Δs3`; with E4a all four
  coincide and `Δs3(m+1) = 8Δs3(m)` is orthant counting.
- **The `8 = 2³` remark is CORRECTED.** The `8` is `1 + 3 + 3 + 1`, the orthant count of a triple sum
  over a doubled index set — every orthant contributes `s3` plus a `span{cp2, H, 1}` correction.
  E2 is not load-bearing for this edge on the reading given above.
- **E4b's status is now MEASURED-with-a-proof-shape**, three closed forms, each a Lean target.

### DAG after Step 2

    E1, E2, E6, E8, E4a               PROVED
    E4b  = the three closed forms     MEASURED, shape known — three lemmas, not one evaluation
    E5   base case at m = j           open — the q-binomial's combinatorial content
    E3   deviation ignores g          open — decides whether E4b and E5 are about the right objects

Scope note, since E4a and E4b differ in it: **E4a holds on the reference class** (`refLabel`, i.e.
`g(W) = 0`) — that is where `cp2`'s closed form and hence `Δcp2 = 0` are proved. **E4b's three forms
hold for every label off the maximal seam**, no reference hypothesis. Rows must not inherit each
other's quantifiers.

### Step 2, addendum — the excluded label is not featureless

The seam deviation `2(H−4)(H−8)` equals `192·[m−1,2]₂` exactly (m = 3..9) — the same q-binomial as
the `288·[m−1,2]₂` term §57.49 deflated as a mask artifact. The artifact was in the *combination*:
`(3,3,1)` annihilates the direction `(1,−2,3)` the seam signal lives in. The seam itself carries a
real `[m−1,2]₂`. It is **not** E5's `1728·[m,3]₂` — different q-binomial — but it is the same
location, so E5's row should be read as "the maximal seam is where the q-binomials enter", twice
sighted. Basis check: `{cp2, H, 1}` is exact, unique and minimal; `H²` gets coefficient 0; `g`
cannot replace `cp2`. See §57.70.

### Step 3 — Tier 110: `T3` is a theorem

`weight3_pinned` (kernel-clean): `T3 = s3 − 6·(M³)₀₀ + 12·(M²)₀₀ − 8`, for every `W ≠ 0` and every
`m`, with no off-seam hypothesis. The general form `tri3_epsZero` holds for arbitrary `f` and is
reusable. So the first of E4b's three closed forms is reduced to two scalar evaluations —
`(M²)₀₀` and `(M³)₀₀` — neither of which is a triple sum. Both are to be stated for ALL labels:
`(M²)₀₀ = −(H−2)` unconditionally, and `(M³)₀₀ = 32 − 10H − 96·[m−1,2]₂·[W = 2^m]` with the
q-binomial IN the statement, since `weight3_pinned` carries no off-seam hypothesis. See §57.72.

    E1, E2, E6, E8, E4a               PROVED
    E4b  T3 leg                       PROVED down to two row-0 scalars (Tier 110)
    E4b  T1, T2 legs                  MEASURED closed forms; Tier 93 reduction in hand
    E5   base case at m = j           open
    E3   deviation ignores g          open

### Step 4 — Tier 111: the double walk falls, unconditionally

`walk2_value` (kernel-clean): `(M²)₀₀ = 2 − 2^(m+1)`, every label, no hypothesis beyond
`W < 2^(m+1)`. It reduces to `antisym` on a transposed pair one level up — and the reason `c = W`
is not special is that the partner `W + 2^(m+1)` lies outside `c`'s range, which is also the
structural reason the double walk carries no q-binomial and the triple walk does.

**The T3 leg of E4b is now ONE statement**: `(M³)₀₀ = 32 − 10H − 96·[m−1,2]₂·[W = 2^m]`. See §57.73.

### Step 5 — Tier 112: the T3 leg is ONE quadratic form

`weight3_quad` (kernel-clean, all `W ≠ 0`, all `m`): **`T3 = s3 + 6·Q − 8`**, with
`Q = Σ_{b,c} P3(0,b)P3(b,c)P3(0,c)` a quadratic form in the row-`0` vector — the object Tier 91 had
only measured. Route: `P3 c 0 = −P3 0 c + 2·[c=0]` (from Tier 111), which turns the triple walk into
`−Q + 2·(M²)₀₀`.

`Q` is measured `8H − 28` off the seam and `(H−2)²` at it, via a pointwise law for `ρ_b` with the
special locus `{0,W} ⊕ {0,2^m}` (236/236 labels). Not proved: `ρ_b = 4` on the bulk is a genuine
cancellation, so it needs a different technique from Tiers 111–112. See §57.74.

### Step 6 — `Q`'s recursion mapped; the base case is the open part

`Q(m+1) = Q(m) + B + C + D` by block decomposition (four `P3_block**_total` lemmas, already
theorems). Off the seam `B = 16`, `C = 16 − 2H`, `D = 10H − 32`, giving `Q(m+1) = Q(m) + 8H`; at
`W = 2^m` all three are quadratic in `H` and the level above still lands on `8H′ − 28`, so the
recursion HEALS across the seam. 236/236 labels, `m = 3..6`, full scan.

Left open: the six closed forms in Lean (same shape as Tiers 90–93), and **the base case, which is
a family** — labels with top bit exactly `m` (`2^m < W < 2^(m+1)`) have no level below to descend
to. The `W ↦ W ⊕ 2^m` conjugation that would have collapsed that family is **REFUTED** (entrywise
agreement exactly `H²/2` — chance). And `Q` is not E5 in disguise: the two q-binomials are not
proportional. See §57.75.

### Step 7 — Tier 113: the T2 leg's three constant walks, PROVED

Parallel to Tiers 110–111 on T3 (row-0 pins), the T2 leg's Tier 94 measured expansion

    T2 = s3 − 4(M³)_WW − 4(M²)_WW + 4·cp2
          − 8(M Π_W M)_WW − 8(Π_W M²)_WW + 2^(m+3) − 8

has three label-independent constant walks. Kernel-clean in Lean (no `sorry`):

| theorem | identity | ∀ labels |
|---|---|---|
| `walk2_at_W` | `(M²)_WW = H − 2` | yes |
| `walk_MPiM_WW` | `(M Π_W M)_WW = −(H − 2)` | yes |
| `walk2_0W` | `(M²)_{0W} = H − 2` (= `(Π_W M²)_WW`) | yes |

Pointwise laws: `P3_seam_row_col`, `P3_row0_colW`, `P3_seam_coset`, via `A4_sub` and
`cdSigma_prod_neg`. Arithmetic packaging: `t2_walk_arith` / `t2_constant_walks` collapse the
Tier 94 remainder to `−4·(M³)_WW`, so once the expansion itself is a theorem,

    T2 = s3 + 4·cp2 − 4·(M³)_WW

and the measured closed form is exactly `(M³)_WW = 2(H−8) + 96·[m−1,2]₂·[W=2^m]`.

Also landed: `sumLtI_sigRow`, the seam-index twin of `sumLtI_epsZero` (infrastructure for a
future weight-2 pin expansion parallel to `tri3_epsZero`).

#### DAG after Tier 113

    E1, E2, E6, E8, E4a               PROVED
    E4b  T3 leg                       PROVED down to Q / (M³)₀₀ (Tiers 110–112)
    E4b  T2 leg                       three constant walks PROVED; open: Tier 94 expansion
                                      as theorem + (M³)_WW evaluation
    E4b  T1 leg                       MEASURED closed form; weight-1 word in hand
    E5   base case at m = j           open
    E3   deviation ignores g          open

### Step 7 — Tiers 114–115: six forms → three → one

The four blocks' weights FACTOR: `(1+ε(c))·(1−ε(b)σ(b)σ(c)τ(b,c))`. First factor kills the column
`c=0`; second is `0` off a union of four lines. That is why `Q` cancels in the recursion —
pointwise. Per block: `A:1`, `B:−εσστ`, `C:ε(c)·[B]`, `D:ε(c)`.

    D = Q + 2H − 4    PROVED (Tier 114), unconditional  — 2 of the 6 measured values
    C = B − 2H        PROVED (Tier 115), unconditional  — couples the other 4
    B = −Q + 8H − 12  the single remaining obligation

The seam-dependence of the six was never in the blocks; it is `Q`'s. `B` now reduces to four SINGLE
sums (two already theorems). See §57.76.

### Step 8 — Tier 116: the split is a theorem

`quadSplit` (kernel-clean): `Q(m+1) = A + B + C + D`, the step §57.76's first ledger draft was
missing. `quad_level_transfer` assembles `Q(m+1) = 16H − 28` from it plus Tiers 114-115, with
`B = −Q + 8H − 12` as an explicit hypothesis — the one open obligation. `Q(m)` cancels, so no base
case enters above the level where the label first exists. See §57.77.

### Step 9 — B is proved; the T3 leg is closed

`blockB_value`: `B = −Q + 8H − 12`, every label. It discharges Tier 116's hypothesis, giving
`quad_level_value` (`Q(m+1) = 16H − 28`, no base case — `Q(m)` cancels) and `weight3_closed`
(`T3 = s3 + 48H − 176`) — §57.69's form **for the labels the transfer reaches** (`W < 2^n` at level
`n`). The top-bit-`n` labels above the seam were measured and are not covered. For the reference
class that costs nothing above the base: `W < 8` is unaffected, and for `W = 2^p` the uncovered
level is `n = p`, which is exactly where E5's base case already sits. See §57.78.

### Step 10 — Tiers 120–122: the T2 leg (kernel-clean, uncommitted)

Numbered after the T3 chain so as not to renumber 117–119.

| Tier | content |
|---|---|
| **120** | seam walks: `(M²)_WW = H−2`, `(MΠM)_WW = −(H−2)`, `(M²)_{0W} = H−2` |
| **121** | `weight2_D_pinned`: T2 loses `sigRow` → four walks in `X=tauW⊙P3` |
| **122** | corner evaluated: `Σ X(a,W)P3(W,W)X(W,a) = 2−H` ⇒ `+4·corner = −4(H−2)` |

`weight2_D_corner` leaves three walks. Open on T2: the two middle pin walks (expect `(M³)_WW`
each), bulk `tr(XMX)`, carrier `X=P3+2Π−4e₀e_Wᵀ`, and `(M³)_WW` itself.

#### DAG now

    E1, E2, E6, E8, E4a               PROVED
    E4b  T3 / Q                       **CLOSED** for W < 2^n (Tiers 110–119)
    E4b  T2                           120–122 PROVED; open: 2 pin walks + tr(XMX) + carrier + (M³)_WW
    E4b  T1                           MEASURED closed form
    E5, E3                            open

### Step 10 — the T1 leg mapped; the shared scalar proved

`t1_weight` (Tier 123) factors the weight-1 orthant's two coset flips. The map is then MEASURED
(236/236, seam included):

    T1 = s3 + 4·cp2 − 6·(M³)₀₀ − 4·(M³)_WW + 64 − 36H

with `R1 + R2 = 4·cp2` — §57.69's `4·cp2` is the two coset flips one each.

`walk3_at_W` (Tier 124) PROVES `(M³)_WW = Q − 6H + 12`, unconditional: row `W` is row `0` re-signed,
so the seam walk is `Q`'s quadratic form weighted. **This is the scalar T1 and T2 share** — the
concurrent lane's T2 tier needs the same object.

    E4b, T3 leg    PROVED for W < 2^n at level n
    E4b, T1 leg    Tiers 123–126: scalars + twelve CORE label-independent terms PROVED;
                   single-pin IDs + τ-carrier expansion still open
    E4b, T2 leg    Tiers 120–122, 127–128: D-pin, corner, S₂=(M³)_WW PROVED;
                   S₁ still open (measured = S₂); bulk tr(XMX) open
    E5, E3         open

See §57.79.

### Step 11 — T1 CORE's twelve label-independent terms are theorems

Tiers 125–126: the eight double-pin sums are each `2 − H` and the four corners cancel, so the
`+64 − 32H` of the CORE identity is proved. Everything reduces through Tier 124's re-signings to
`Σε`, `Σσ`, `Σεσ`.

    E4b, T1 leg   scalars PROVED; CORE's 12 label-independent terms PROVED;
                  remaining: the 5 single-pin cyclicity identifications, and the carrier
                  expansion of the two taus (R1, R2, R3, rank-one)
    E4b, T2 leg   concurrent lane landed Tiers 120–122 at 53b823b726

See §57.80.

### Step 12 — T1's CORE fully accounted; the both-Pi carrier term closes

Tier 129: the five single pins identified (`cyc_pin_last`, `cyc_pin_mid`, both for arbitrary pin
`p`), giving `3(M³)₀₀ + 2(M³)_WW` — so every term of CORE is now a theorem.
Tier 131: the both-`Π_W` term collapses to `Σ_b ε(b⊕W)·P3(b,b) = 4 − H`, hence `R3 = 16 − 4H`.

    E4b, T1 leg   CORE fully proved; carrier: R3 PROVED, R1/R2 and the rank-one corrections open
    E4b, T2 leg   concurrent lane, Tiers 120-122 and 127-128 landed; consumes walk3_at_W

See §57.81.

### Step 13 — every piece of T1's carrier expansion is a theorem

Tiers 133–137: `R1 = 2cp2 + 4H − 8` and `R2 = 2cp2 − 4H + 8` (so `R1 + R2 = 4cp2`), and the
rank-one corrections `= −16` (two equal sums with opposite signs cancel; `−8 − 8` survives).
With CORE (125–126, 129) and `R3` (131), all five pieces are proved.

    E4b, T1 leg   ALL PIECES PROVED; remaining: the nine-way pointwise expansion of
                  X(b,c)·X(c,a) routing the T1 triple sum onto them (the quadSplit analogue)
    E4b, T2 leg   concurrent lane
    E5, E3        open

See §57.82.
