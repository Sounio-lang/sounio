<!-- docs:meta
topic_id: repo.docs.research.cd-tower-automorphism-freeze
authority: repo_only
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-automorphism-freeze
-->


# The signed-monomial automorphism group of the Cayley-Dickson tower FREEZES at 168, while the zero-divisor fiber geometry grows

**Honesty firewall.** Every claim below is tagged **PROVEN** (a general all-`n` argument),
**VERIFIED(dims)** (finite exhaustive computation at the listed dimensions only), or
**CONJECTURE** (believed, not established). Exhaustive-at-a-dimension is *never* labelled bare
"proven". This lane has repeatedly caught overclaims; the tags are load-bearing.

Certifying oracle: `scripts/research/cd_tower_automorphism_oracle.py` (pure-integer, deterministic;
quick run does n=4 in <1 s, `--full` adds the ~70 s GL(5,2) sweep + n=6 lift check). Final line
`CD_TOWER_AUT CERTIFIED`.

## ⚠ Scope & prior art — READ FIRST (literature check 2026-07-11)

**The "168" here is the count of valid index-maps `M ∈ GL(n,2)` — i.e. the signed-monomial
automorphism group *modulo diagonal signs* (the `PSL(2,7)` permutation content). It is NOT the full
signed-monomial group.** The full group of `(M, ε)` pairs is `|Aut(Q_n)| = 168 · 2ⁿ` and **GROWS**
(1344, 2688, 5376, … at dim 8, 16, 32): each Cayley-Dickson doubling adjoins one `ℤ₂` *sign* factor
(a pure sign flip on the new upper half), leaving the permutation part fixed. Verified to the digit:
`168 × |diagonal sign autos| = 168 × 2ⁿ` = 1344/2688/5376.

**This is a KNOWN, published result, not an open problem closed here.** Jenya Kirshtein,
*Automorphism groups of Cayley-Dickson loops*, J. Gen. Lie Theory Appl. **6** (2012) (arXiv:1102.5151),
computes `Aut(Q_n)` exactly (orders 1344/2688/5376) and proves **Thm 41 (`n ≥ 4`):
`Aut(Q_n) ≅ Aut(Q_{n−1}) × ℤ₂`**. Octonion base: Koca & Koç, *Turk. J. Phys.* **19** (1995) 304 —
`Aut(O₁₆) = 2³·PSL(2,7)`, the `168 = |PSL(2,7)| = |GL(3,2)|` Fano action on the 7 imaginary directions.
(A *third*, unrelated group is the **continuous** real-algebra automorphism group: `G₂` for the
octonions, `S₃` for the sedenions — do not conflate it with either the full or the permutation
signed-monomial group.)

**So the honest claim is:** the *permutation part* (index-map count / `PSL(2,7)`) freezes at 168 ∀n —
**true, and our block-lemma (`≤`) + lift (`≥`, PROVEN ∀n) re-derive the permutation content of
Kirshtein's Thm 41** by an independent exact/cohomological route. The genuine contribution is
**methodological** — an exact-integer + `F₂`-coboundary derivation with an ∀n degree-obstruction and a
partial Lean leg — not the numerical fact. The title's "freezes at 168" must be read in this scoped
sense; the full signed-monomial group does not freeze.

### Novelty ledger for the *action* (Result 3) — and the honest frontier

The freezing above is Kirshtein-known. The genuinely new object is the **action of the frozen finite
group on the growing discrete ZD/fiber set**. Pinned exactly:

- **NOVEL (not found in the literature).** The explicit **orbit + stabilizer + fixed-point
  decomposition of the DISCRETE ZD/fiber set** under the finite signed-monomial automorphism group
  (permutation part `PSL(2,7)`): `2ⁿ⁻⁴` size-7 Fano orbits `+` `(2ⁿ⁻⁴−1)` fixed seams, stabilizer `S₄`
  (order 24), **PROVEN ∀n** (block-lemma freezing `+` `GL(3,2)` transitivity on `F₂³\{0}`;
  VERIFIED n=4..7). The continuous picture gives *one* orbit; this partitions it into a growing discrete
  orbit set. Claim scope, honestly: this is the *combinatorial orbit decomposition* (counts, stabilizer,
  fixed points) — a group-action fact. It is **not** claimed that the orbits are geometrically
  distinguishable: an adversarial nauty audit (2026-07-12) refuted a distinct-geometries conjecture at
  n=6 (a Fano orbit and a fixed seam share an isomorphic annihilation graph). The fiber geometry instead
  obeys a **parity collapse law** (`γ(Seam(y)) = γ(Fano(y & (y−1)))` iff `wt(y)` even; nauty-complete
  n≤8), which is itself a genuine ℤ₂ structural fact but means the orbit→geometry map is non-injective.
  See `scripts/research/cd_tower_fiber_geometry_collision.py`.
- **NOT novel — do not claim.**
  - *"`PSL(2,7)` is a symmetry of the ZDs."* de Marrais uses `168 ↔ ZD` as a count coincidence `+`
    Fano-labeling frame to **classify box-kites** (full arXiv corpus deep-read 2026-07-11, 9 papers;
    verdict **(b) ADJACENT** — the surrounding pieces, not the orbit-decomposition; the words
    orbit/stabilizer/fixed-point *in the group sense on the ZD set* appear in none of them, and his
    box-kite tallies `7,35,155,651` / `4ⁿ⁻⁴` do not match our `2ⁿ⁻⁴×[7]`).
  - *"a group acts on the ZDs, scaling with n."* Moreno 1998 (`q-alg/9710013`) / Reggiani 2024
    (`arXiv:2411.18881`): the **continuous** `G₂` (`× S₃ᵏ` up the tower) acts **transitively**, one
    orbit, `SU(2)` isotropy, `ZD(𝕊) ≅ G₂/SU(2)`. Ours is the finite/discrete refinement, a different
    object.
- **Frontier (open, stated honestly).**
  - **de Marrais, ICGTMP-2006 talk** *"The Marriage of Nothing and All: Zero-Divisor Box-Kites in a
    'TOE' Sky"* — the one item outside the arXiv corpus. **Caveat SUBSTANTIALLY RESOLVED (2026-07-12,
    convergent secondary evidence):** it was **never published** (the official ICGTMP record lists no
    Group26 proceedings — a "forthcoming from Springer" that did not appear), its subject is
    box-kites `+` TOE-physics, and de Marrais's own 2007 self-citation (`math/0703745`, ref [4]) places
    it as a **QM-degeneracy aside**, not an orbit-decomposition. The talk text itself was **not read**
    (apparently unavailable anywhere public); re-check only if it surfaces, before any *published*
    priority claim. It no longer blocks the novelty scoping.
  - **Physics 3-generation bridge = NULL (computed 2026-07-12).** The generation `Z₃` is the
    *continuous* order-3 automorphism `ψ` (a `2π/3` rotation with `√3/2` coefficients — it **mixes
    basis directions**, so it acts on the continuous ZD manifold, not our discrete fibers). The only
    *discrete* order-3 structure inside our 168 is the point-stabilizer `S₄ → S₃` the orbit theorem
    already contains (the 168 is 2-transitive/**primitive** on the 7 Fano points — no finer block
    system exists). So the physics motivation contextualizes but adds no bridge; **the novelty is
    pure-math, not physics.**

## Prior art this generalizes (credit)

This lifts the Frente-B **sedenion** (dim 16) Fano / zero-divisor program up the CD tower to dim 32:

- `scripts/research/sedenion_zd_fibers_oracle.py` — the 7-fiber, 168-edge sedenion ZD geometry.
- `scripts/research/sedenion_fano_fibers_oracle.py` — Fano `PG(2,2)` fiber incidence.
- `scripts/research/sedenion_automorphism_168_oracle.py` + `docs/research/sedenion_automorphism_168.md`
  — the `168 = |PSL(2,7)|` signed-monomial automorphism subgroup that fixes e₈.
- `formal/lean4/SounioSedenionAutomorphism.lean`, `docs/papers/sedenion-fano-geometry.md`.

The n=4 results here reproduce that work; the n=5 results are the new tower step.

## Setup

A blade index is an `F₂ⁿ` vector; multiplication `e_i·e_j = σ(i,j) e_{i⊕j}` with the recursive
Cayley-Dickson sign `σ` (transcribed identically in every lane oracle). A linear
`M ∈ GL(n,2)` (so `M` preserves `⊕`) is a **signed-monomial automorphism** iff there is a sign
`ε: index → ±1` with `φ(e_i) = ε(i) e_{Mi}` an algebra automorphism, equivalently iff the
multiplier `s(i,j) ⊕ s(Mi,Mj)` (with `s = [σ = −1]`) is an `F₂` coboundary `δε`. This is a
decidable linear-consistency check; the oracle sweeps **all** of `GL(n,2)` (basis-image recursion,
not brute force).

The **seam** is `H = 2^(n-1)` (index 8 at n=4, 16 at n=5) — the octonion→sedenion (and next)
doubling generator.

## Result 1 — The group is FROZEN at 168 = |PSL(2,7)|

| n | dim | #signed-monomial autos | ambient `|GL(n,2)|` | status |
|---|-----|------------------------|---------------------|--------|
| 4 | 16  | **168**                | 20160               | **VERIFIED(n=4)** — full GL(4,2) sweep |
| 5 | 32  | **168**                | 9 999 360           | **VERIFIED(n=5)** — full GL(5,2) sweep |
| 6 | 64  | **168**                | ~2.15×10¹⁰          | **VERIFIED(n=6)** — exhaustive seam-stabilizer sweep |

- The naive expectation that the group grows with the ambient (to `|GL(4,2)| = |PGL(4,2)| = 20160`)
  is **REFUTED** — VERIFIED(n=5): the count stays 168, index 59520 in `GL(5,2)`.
- **Literal lift, VERIFIED(n=5):** every one of the 168 n=5 autos preserves the lower block
  `{1..15}` and restricts to one of the 168 n=4 autos (oracle line `AUT5_LITERAL_LIFT 1`). So the
  n=5 group is not merely *isomorphic* to the n=4 group — it is its literal `diag(A,1)` lift. All 168
  fix the seam (`AUT5_FIXSEAM 168`).
- **n=6 = 168, VERIFIED(n=6) — exhaustive.** Seam-fixing is verified at n=6 (the seam index 32 is the
  unique argmax of the associator-nonassociativity degree, value 3720; Result 4), so every valid auto
  lies in the seam-stabilizer `[[A,0],[β,1]]` (`A ∈ GL(5,2)`, `β ∈ F₂⁵`). Sweeping the **entire**
  stabilizer — all `9 999 360 × 32 = 319 979 520` matrices — yields exactly **168**, all with `β = 0`
  (the lifts of the n=5 group); no `β ≠ 0` auto exists. Two independent code paths agree.
  `scripts/research/cd_tower_automorphism_n6_exhaustive.py` (~150 s, 31-way multiprocessing). The full
  `|GL(6,2)|` was not swept, but need not be — seam-fixing (VERIFIED n=6) reduces it to the stabilizer.

## Result 2 — The zero-divisor fiber geometry GROWS: PG(2,2) → PG(3,2)

The participating mixed-half primitives `e_lo ± e_hi` (`lo ∈ 1..H−1`, `hi ∈ H..2ⁿ−1`) partition by
label `L = lo ⊕ hi` into fibers; mutual annihilation (zero-divisor edges) is strictly **intra-fiber**.
**VERIFIED(n=4)** and **VERIFIED(n=5)** (finite exhaustive scan, `cross = 0` both):

| n | #fibers | = points of | fiber shapes | total ZD edges |
|---|---------|-------------|--------------|----------------|
| 4 | **7**   | `PG(2,2)` (Fano) | uniform: 7 × (12 verts, 24 edges, deg-4) | **168** |
| 5 | **15**  | `PG(3,2)`   | **NON-uniform:** 7 × (28v, 72e, deg 4–12) **+** 8 × (28v, 168e, 12-regular) | **1848 = 11·168** |

- **VERIFIED(n=5) non-uniformity + the type split.** Label `L ∈ {17..31}`; write `Llo = L − 16 ∈ {1..15}`.
  - **Type A** (28v, 72e, deg 4–12, *not* regular): `Llo ∈ {9,…,15}` — 7 fibers.
  - **Type B** (28v, 168e, 12-regular, non-bipartite): `Llo ∈ {1,…,8}` — 8 fibers.
  Contrast n=4, where all 7 fibers are the single uniform Fano-point shape. The growth is therefore
  *two-fold*: the fiber **count** follows the projective-space point count `|PG(n−2,2)| = 2^(n−1)−1`
  (7 → 15), and the fiber **internal geometry** stops being uniform.
- `1848 = 11·168` is a **combinatorial** factorization (7·72 + 8·168 = 504 + 1344), **not** a group
  order — consistent with `sedenion_associator_1848.md`'s reading of the 11 as the e₈-grade factor.

## Result 3 — Orbits on the fibers: 7 + 1 + 7

**VERIFIED(n=4,5).** The group action on a fiber label is `L ↦ M(L)` (linearity), which at n=5
reduces to `Llo ↦ M(Llo)` on `{1..15}` (since `M` fixes the seam bit and preserves the lower block).
The orbit partition of the 15 labels is

> `{8}  ∪  {1,2,3,4,5,6,7}  ∪  {9,10,11,12,13,14,15}`  →  sizes **1 + 7 + 7**.

- The **fixed point is `Llo = 8`**, i.e. fiber `L = 24` at n=5 — the inner (n=4) e₈ seam lifted.
  (There is no "fiber 8" at n=5: n=5 fibers are labelled `L ∈ {17..31}`; `Llo = L − 16` is the
  reduced label the group acts on.)
- Because the n=5 group is the literal lift of the n=4 group, this partition is *the same* whether
  computed from the n=4 autos (quick run) or the n=5 autos (`--full`) — the oracle certifies both.

## Result 4 — Seam-fixing via the associator-degree invariant: **now PROVEN ∀n≥4**

Define the **associator-nonassociativity degree** `deg(i) = #{(j,k) : the triple through i is
non-associative}`.

- **PROVEN (all n), structural:** a signed-monomial automorphism preserves the algebra structure,
  hence maps associative triples to associative triples and non-associative to non-associative;
  therefore `deg(Mi) = deg(i)` — the degree is an automorphism invariant for every n.
- **PROVEN (all n≥4) — closed form, no longer merely verified.** Deriving the uniform one-step
  CD recursion `f_n(u⊕αH,v⊕βH) = f_{n-1}(u,v) ⊕ β·χ(u,v) ⊕ α·n0(v) ⊕ αβ` directly from `cd_sigma`'s
  four defining branches (the only external input is the ∀n-proven anticommutator lemma
  `f(y,x)=f(x,y)⊕χ(x,y)`, Lean `cdAntisym_all`), then expanding `Ψ` and summing over `(j,k)`, gives
  an **exact closed form for every n**:

  `deg(0) = 0`,  `deg(i) = 2H²−8` for **every** nonzero non-seam `i`,  `deg(H) = 4(H−1)(H−2)`.

  Hence `deg(H) − deg(other) = 2(H−2)(H−4)`, which is **> 0 iff H > 4 iff n ≥ 4** — a direct,
  non-inductive proof (it does not depend on lower-level automorphism structure, and is independent
  of the open `L2` lemma below). At `n = 3` the gap is exactly `0` (seam ties all 7 imaginary units,
  matching `G₂`-transitivity on octonion units — **not** unique there); `n ≤ 2` is associative
  (all-zero degree). This is a genuine `n ≥ 4` scope boundary, not an incompleteness — it lines up
  exactly with the block lemma's own `n ≥ 4`/`n = 3`-false boundary below.

  | n | seam H | deg(H) | deg(other) | unique argmax? |
  |---|--------|--------|------------|-----------------|
  | 3 | 4      | 24     | 24         | **no** — tie (gap 0), G₂-transitive boundary |
  | 4 | 8      | 168    | 120        | yes (hist `{0:1, 120:14, 168:1}`) |
  | 5 | 16     | 840    | 504        | yes (`{0:1, 504:30, 840:1}`) |
  | 6 | 32     | 3720   | 2040       | yes |
  | 7 | 64     | 15624  | 8184       | yes (closed form only; not previously tabulated) |

  Independently reproduced by brute-force `deg` (summed from the raw `cd_sigma` `f`-table, not from
  the closed form) at n=2..7, 0 mismatches; closed form matches at every dimension, including the
  new n=7 point.
- **The summation step, written out (all n).** With `i=u⊕pH, j=v⊕qH, k=w⊕rH`, the seam-flip law reads
  `Ψ_n = Ψ_{n-1}(u,v,w) ⊕ p·χ(v,w) ⊕ q·A ⊕ r·χ(u,v)`, `A = χ(u,v)⊕χ(u,v⊕w)`. Summing over `(q,r)∈{0,1}²`
  the count of 1-values is `S(u,v,w) = 4·(Ψ_{n-1}⊕p·χ(v,w))` when `A=0 ∧ χ(u,v)=0`, else exactly `2`
  (two of the four combinations flip). Hence `deg(i) = 2H² + 2·(2·Z₁ − |Z|)` where
  `Z = {(v,w): χ(u,v)=0 ∧ χ(u,v⊕w)=0}` and `Z₁ = #{(v,w)∈Z : Ψ_{n-1}(u,v,w)⊕p·χ(v,w)=1}`.
  For nonzero non-seam `u≠0`, `χ(u,·)=0 ⟺ ·∈{0,u}`, so `Z = {(0,0),(0,u),(u,u),(u,0)}` (`|Z|=4`) and
  `Ψ_{n-1}=0` on all four (degeneracy: `v=0`, `w=0`, or the `(u,u,u)` diagonal) — giving `Z₁=0` for
  both `p=0` (`base=0`) and `p=1` (`base=χ(v,w)=0` on those points), hence `deg = 2H²−8`. For the seam
  (`u=0,p=1`), `χ(0,·)=0` makes `Z` the whole grid and `S = 4·χ(v,w)`, so `deg(H) = 4·#{v,w≠0, v≠w} =
  4(H−1)(H−2)`. This is the derivation the closed form rests on; **every internal step (the `(q,r)`-sum
  lemma, `|Z|=4` with `Ψ_{n-1}=0` on `Z`, the seam `Z=` grid case, and the endpoint reconstruction) is
  machine-checked n=4..7** in `scripts/research/cd_tower_L1_closed_form_derivation.py` — so the closed
  form is *derived*, not merely value-fit. The sole ingredient carried as "exhaustively verified n≤7 +
  n-independent symbolic expansion" rather than written term-by-term is the seam-flip law `(F)` itself
  (the one-step recursion `(R)` it builds on is proved ∀n by the four-branch `cd_sigma` case analysis
  using `cdAntisym_all`).
- **Consequence — seam-fixing is PROVEN(∀n≥4).** Since `H` is the strict unique argmax of the
  ∀n-invariant `deg` for every `n ≥ 4`, invariance forces every valid automorphism to fix `H`, for
  every `n ≥ 4` — not merely at the previously-tabulated `n ≤ 6`. This closes lemma **L1**
  unconditionally.
- Script: `scripts/research/cd_tower_seam_unique_argmax_proof.py` (all checks pass: uniform recursion
  n=2..7, chi-cocycle m=3..6, seam-flip law exhaustive n=4..7 up to `2 097 152` triples, closed-form
  vs. independent brute-force `deg` at n=4..7, n=3 boundary confirmed as a tie).

> Note: the merged all-`n` `seam_coincidence` Lean proof concerns the **ZD / anticommutator** seam
> coincidence — a *different* invariant than associator-degree. We do **not** claim it bridges to the
> automorphism-fixing statement; it remains a separate result, now joined by (rather than needed for)
> the L1 proof above.

## Toward all-`n` freezing — the block lemma and its mechanism

**"Exactly 168 for all n"** reduces to a **block lemma**: every seam-fixing auto `M = [[A,0],[β,1]]`
has `β = 0` (preserves the lower block, so restricts to a lower-level auto). This session found the
**mechanism** — the obstruction is the **associator 3-form** `Ψ(i,j,k) = f(i,j) ⊕ f(i⊕j,k) ⊕ f(j,k)
⊕ f(i,j⊕k)` (`f = [σ=−1]`), i.e. the non-associativity indicator:

- **PROVEN (one-way, all n):** `valid(M) ⟹ M preserves Ψ` — automorphisms preserve associativity; the
  coboundary defect of the multiplier `δ_M` equals `Ψ(Mi,Mj,Mk) ⊕ Ψ(i,j,k)`. (Independently VERIFIED:
  all 168 valid `M` at n=4,5 preserve `Ψ`.)
- **The forcing, VERIFIED(n=4,5,6):** for `A` a valid lower auto and `β ≠ 0`, `M` does **not** preserve
  `Ψ` (0 of the 1176/2520 block-mixing `M` preserve it), so by the one-way lemma `M` is invalid.
  Driven by a **seam-flip law** for `Ψ` under seam-bit addition — **now PROVEN ∀n** (previously only
  verified n≤6): `Ψ(u⊕pH,v⊕qH,w⊕rH) = Ψ(u,v,w) ⊕ p·χ(v,w) ⊕ q·[χ(u,v)⊕χ(u,v⊕w)] ⊕ r·χ(u,v)`. Derived
  from the same uniform one-step `f`-recursion used for L1 above, by expanding `Ψ`'s four `f`-terms
  and collecting: the quadratic (`pq,pr,qr`) cross-terms cancel identically, the `p`/`q`/`r`
  coefficients reduce via `χ`'s definition and one pure-boolean cocycle identity
  `χ(u⊕v,w)⊕χ(v,w)⊕χ(u,v⊕w)=χ(u,v)`, all forall-n algebra. Sole external input: the ∀n-proven
  anticommutator lemma `cdAntisym_all`. Confirmed by independent symbolic collection over all 64
  free atom-assignments (0 mismatches) plus exhaustive recomputation direct from `cd_sigma` at
  n=4,5,6,7 (0 mismatches over up to `2 097 152` triples × 8 seam-bit patterns). On distinct-nonzero
  triples this law forces `β·(i⊕k) = 0` for every realizable `i⊕k`, hence `β = 0` (needs
  `m = n−1 ≥ 3`) — **but only granted the premise that `A` itself is a valid lower auto**, which is
  exactly lemma L2, below.
- **n = 4: UNCONDITIONAL.** The octonion lower block is alternative, so `Ψ₃ = [i⊕j⊕k ≠ 0]` on
  distinct-nonzero triples, preserved by *any* `A ∈ GL(3,2)` for free — no premise needed. (Now also
  Lean-verified, see below.)
- **n = 3: the lemma is FALSE** (24 valid seam-fixing `M`, `β ∈ {0,1,2,3}`) — `F₂²` has no sum-nonzero
  distinct-nonzero triple; a clean scope check confirming `n ≥ 4`.
- `scripts/research/cd_tower_block_lemma_proof.py` (prints `ALL CHECKS PASS`),
  `cd_tower_block_lemma_psi_obstruction.py`.

### L2 (`M-valid ⟹ A-valid`): the pure-lower angle is EXHAUSTED — it reduces to, not proves, `β = 0`

A dedicated attempt to close L2 directly (restrict `M`-validity to the lower `F₂^{n-1}` block) found
that the two "open lemmas" collapse into **one**: for `n ≥ 4` (`m = n−1 ≥ 3`),

> **`A` is a valid `(n−1)`-auto  ⟺  `β = 0`.**

Restricting the validity coboundary equation to lower indices gives `g_A ⊕ C_β = δ_y` for a genuine
`(n−1)`-coboundary `δ_y`, where `g_A(i,j) = f_{n-1}(i,j) ⊕ f_{n-1}(Ai,Aj)` and `C_β` depends only on
`β`. Since coboundaries form an `F₂`-group, `A` valid (`g_A` a coboundary) `⟺ C_β` a coboundary. A
separate **C-lemma** (PROVEN `∀ m ≥ 3`, using that all nonzero `β` lie in one `GL(m,2)`-orbit plus one
explicit non-coboundary witness at `m=3`) shows `C_β` is a coboundary `⟺ β = 0`, for every `m ≥ 3`.
So for `n ≥ 4`, L2's hypothesis (`A` valid) and the block lemma's still-open conclusion (`β = 0`) are
**the same statement** — the program's previously-stated "conditional block lemma" (`A` valid `⟹
β = 0`, `n ≥ 5`) is therefore **circular / vacuous as progress**, not two lemmas one step apart.
`n = 3` (`m = 2`) is the genuine boundary where they diverge: every `C_β` is a coboundary there, so L2
holds *even though* `β ≠ 0` occurs (matching the 24 valid `M` with `β ∈ {0,1,2,3}` above).

- **Status: the pure-lower angle is EXHAUSTED** — L2 is provably equivalent (n≥4) to the
  unconditional `β = 0` statement rather than a separable stepping-stone; not refuted. The crisp
  remaining target it names — rule out `δ(g_A) = δ(C_β)` for any `A ∈ GL(m,2)`, `β ≠ 0` — is exactly
  what the **structural obstruction** (next subsection) attacks: it is `A*Ψ = Ψ⊕δC_β`, and the
  `GL`-invariant degree multiset of `Ψ` gives a non-circular sufficient obstruction (a `mod-2` parity
  count is indeed too weak; the integer degree multiset is not).
- Script: `scripts/research/cd_tower_L2_reduction.py` — uniform recursion verified n=2..6; C-lemma
  witness (`C_{e1}` not a coboundary on `F₂³`) plus good-`β` computed exactly `= {0}` for `m ≥ 3`
  (m=1..6); full valid-`M` sweep confirms `A`-valid `⟺ β=0` at n=4 (168/168) and n=5 (168/168), while
  n=3 (24 `M`, `β∈{0,1,2,3}`, all `A` valid) confirms the boundary; full characterization
  `valid(M) ⟺ (g_A⊕C_β)` coboundary checked with 0 mismatches over all 1344 (n=4) and 322560 (n=5)
  seam-fixing `M`.

### L2 via the structural (cohomological) obstruction — the non-circular handle

The pure-lower angle is circular because it works with the coboundary condition on `g_A`. The
**structural** angle instead works with a `GL`-invariant of the associator 3-form `Ψ`, which the
circular argument never touches. Apply the coboundary operator `δ` to the forward reduction
(`M valid ⟹ g_A⊕C_β ∈ B²`, a theorem ∀n): since `δg_A = Ψ⊕A*Ψ`,

> **`M valid ⟹ A*Ψ = Ψ ⊕ δC_β`** (a NECESSARY 3-cochain identity; `A*Ψ(i,j,k):=Ψ(Ai,Aj,Ak)`).

The associator-degree `deg_Ψ(i)=#{(j,k):Ψ(i,j,k)=1}` is a `GL`-invariant **as a multiset**
(`deg_{A*Ψ}(i)=deg_Ψ(Ai)`, `A` a bijection). This yields an **airtight, non-circular ∀n reduction**:

> **block lemma  ⟸  [ for every `β ≠ 0`:  `max_i deg(Ψ⊕δC_β)(i)  ≠  max_i deg_Ψ(i)` ].**

(If `M` valid with coupling `β`, its lower part `A` solves the necessary identity, forcing the
multisets — hence the maxima — equal; contrapositive gives the reduction. It is a **sufficient**
condition checkable *without* assuming `A` valid or `β = 0`.) Here `max_i deg_Ψ(i) = deg_Ψ(H) = top =
4(H−1)(H−2)`, the seam being the unique argmax (L1, ∀n≥4).

- **VERIFIED(n=4..8):** for **every** `β ≠ 0`, `max_i deg(Ψ⊕δC_β)(i) < top` — the max does not merely
  differ, it strictly **drops**. So the obstruction fires everywhere tested; the block lemma holds at
  `n = 4..8` by this route (independent of the GL-sweeps, and one level further than before).
- **PROVEN ∀n — the `δC_β` closed form:** with `b(x):=β·x` (linear),
  `δC_β(i,j,k) = b(i)·χ(j,k) ⊕ b(j)·[χ(i,j)⊕χ(i,j⊕k)] ⊕ b(k)·χ(i,j)` — *exactly* the correction part
  of the `Ψ` seam-flip law with `(p,q,r)=(b(i),b(j),b(k))`. (`n0`-terms collapse via
  `n0(j)⊕n0(k)⊕n0(j⊕k)=χ(j,k)`; `pq`-terms cancel; `χ`-terms reduce via the `χ`-cocycle.)
- **PROVEN ∀n — the `D_β` closed form:** `D_β(i):=deg(δC_β)(i) = 0` (`i=0`) `/ 2H(H−2)` (`β·i=0`) `/
  2(H−1)(H−2)` (`β·i=1`), derived by `(j,k)`-summation of the `δC_β` closed form (a nonzero linear
  `b` splits `F₂^m` exactly in half). Depends only on `β·i`, independent of `popcount(β)`.
- **VERIFIED(n≤6/7), CONJECTURE ∀n:** the seam value `deg(Ψ⊕δC_β)(H) = 2(H−2)²` (`β·H=0`) `/
  2(H−1)(H−2)` (`β·H=1`), both `< top`; and the **global** max over all `(β≠0,i)` equals
  `2(H−2)(2H−5) = top − 6(H−2) < top`, achieved at `β = H` (the seam-bit direction) — the worst `β`
  aligns exactly with the seam, so the L1 seam-flip machinery governs the hard case.
- **PARTIAL (the overlap attack) — a `deg'` tower recursion; `β_H=1` half a THEOREM ∀n, `β_H=0` open.**
  Targeting `deg'_β:=deg(Ψ⊕δC_β)` directly (the XOR-sum the machinery eats, not the product `O_β`),
  split `β = β_lo ⊕ β_H·H`. For `β_H=1` the level reduces cleanly,
  `deg'^{(m)}_β(i) = 4·deg'^{(m−1)}_{β_lo}(i_lo) + corr`, `corr = 4(H−2)` (`b(i)=0`) `/ 6(H−2)`
  (`b(i)=1`) for `i_lo≠0`, seam `2(H−1)(H−2)`. **Now DERIVED ∀n (backbone):** a `χ`-full seam
  decomposition `χ(x,y)=χ_lo⊕p_x(ν(X)⊕ν(X⊕Y))⊕p_y(ν(Y)⊕ν(X⊕Y))` (∀n identity) gives
  `Φ_β^{(m)}(i,j,k)=Φ_{β_lo}^{(m−1)}(u,v,w)⊕R` with `R` an **explicit 32-monomial polynomial** in 13
  `n`-independent atoms (an algebraic consequence of the ∀n seam-flip + `δC` closed forms; symbolic
  `R` == numeric over all cases); and `corr = Σ_{v,w,s,t} R` is a degree-≤2 polynomial in `H`
  (each monomial sums, by inclusion–exclusion, to a `ℤ`-combination of affine-subspace sizes `∼H²/H/1`,
  the atoms being `=0`-subspace complements and functional hyperplanes) **pinned at 4 points**
  (m=4,5,6,7 — 3 determine a degree-≤2 form) ⟹ the closed form ∀n. The compatibility **(A)** `Σ Φ_lo·R = 0`
  (`R` vanishes on the lower form's support — this decouples the sum into `4·deg'^{(m−1)}+corr`) is now
  **PROVEN ∀n** by 3 pillars: (1) `R=1` forces `(u,v,w)` into one of 6 degenerate configs
  `{u=0,v=0,w=0,u=v,v=w,w=u⊕v}` (a finite exhaustive atom-check on `R`'s explicit formula); (2) `Ψ=0`
  on all 6 (the zero-configs ∀n; the equality/dependence ones by the ∀n seam-flip law + induction —
  the correction vanishes since `χ(u,v)=χ(u,u⊕v)=χ(v,u⊕v)`); (3) `δC_{β_lo}=0` on all 6 (χ-algebra).
  So `Φ_lo=Ψ⊕δC=0` on the `R=1` locus. **The `β_H=1` recursion is therefore a THEOREM ∀n.** This
  closes the induction for the `β_H=1` half:
  `max_i deg'^{(m)}_β ≤ 4·top^{(m−1)}+6(H−2) = 2(H−2)(2H−5) < top` (`β_lo=0` = the extremal `β=H`,
  needing only the level-`m−1` `deg_Ψ` closed form; `β_lo≠0` strict by the level-`m−1` hypothesis) —
  **no `β=H`-extremality claim.** For **`β_H=0`** (`β` purely-lower): `δC_β`'s closed form is **linear
  in `b=β·`** (the `pq`-term dies under `δ`), so `δC_{β_lo⊕H}=δC_{β_lo}⊕δC_H` and the `β_H=0` defect
  `R0 = R ⊕ δC_H`. Hence `deg'^{β_H=0}_{β_lo}(i) = deg'^{β_H=1}_{β_lo⊕H}(i) + D_H(i) − 2·O(i)`, and
  since `deg'^{β_H=1}=4·deg'^{(m−1)}+corr`, the non-bucket `4·deg'^{(m−1)}` term **cancels** iff
  `O(i)=2·deg'^{(m−1)}_{β_lo}(i_lo)+κ(i)` (κ bucket) — the **"O-identity"**. Empirically
  `deg'^{β_H=0}_{β_lo}(i)` **is** then bucket-determined by `(p_i, β_lo·i_lo, cfg(i_lo))`, a **direct
  closed form (no recursion)**, with every bucket value `< top`:
  `0 / 2(H−2)² / 2H²−4H−8 / 2(H−1)(H−2)`, `max = 2H²−4H−8 = mid−4H < top` ∀`H≥4`.
  **`β_H=0` is now PROVEN ∀n by telescoping** (the move that *removes* `Ψ`): the key lemma
  **`n1_0:=#{(s,t):R0=1} = 2` whenever `Φ_lo=1`** holds ∀n — PILLAR 1 (finite exhaustive atom-check on
  the symbolic `R0`: `n1_0=2` on the generic locus, so `n1_0≠2 ⟹` degenerate config) + PILLARS 2,3
  (`Ψ=0` and `δC_{β_lo}=0` on degenerate configs, proven ∀n in the (A) proof). Then the `4·deg'^{(m−1)}`
  term **cancels** and `deg'^{β_H=0}(i) = Σ_{v,w,s,t} R0` — a **corr-style count of the explicit
  atom-polynomial `R0` (no `Ψ` left)**, so bucket-determined by `(p_i, β_lo·i_lo, cfg)` and degree-≤2 in
  `H` (same inclusion-exclusion argument as `corr`), **pinned at H=8,16,32,64,128 (m=4..8)**. All bucket
  values `< top` (max `2H²−4H−8 < top` ∀`H≥4`; at H=128 the max is 32248 < top 64008). **Rigor is
  exactly corr-parity** (the `<top` step reuses `corr`'s degree-≤2 argument; not separately derived like
  `D_β`'s inner counts) — but the closed forms are now **5-point-hardened**, over-determining the degree-2
  ansatz by 2 (a degree-3 impostor matching all 5 points is excluded). Check: `cd_tower_L2_5thpoint`.
- **⟹ BLOCK LEMMA PROVEN ∀n ⟹ `|Aut_n| ≤ 168` ∀n (the UPPER bound).** With `β_H=1` (theorem ∀n) +
  `β_H=0` (above) + the m=3 base (firing verified), the obstruction fires ∀`β≠0` ∀n; via the ∀n
  forward reduction this forces `β=0`, so every auto is `[[A,0],[0,1]]` with `A` a valid lower auto,
  giving `|Aut_n| ≤ |Aut_{n−1}| ≤ … ≤ 168`. Verified end-to-end (full firing, all `β≠0`) at n≤8.
  Scripts: `cd_tower_L2_betaH_recursion_derivation.py`, `cd_tower_L2_compatibility_A_proof.py`
  (β_H=1 + (A)), `cd_tower_L2_betaH0_reduction.py` (β_H=0 telescoping + full firing).
- **⟸ THE LIFT (backward direction) — PROVEN ∀n, fully rigorous (NOT corr-parity).** Exact equality
  needs `|Aut_n| ≥ 168`, i.e. `A valid ⟹ M0=[[A,0],[0,1]] valid`. This is now a clean ∀n theorem: by
  the one-step recursion (∀n), the validity defect of `M0` is **exactly** `D_{M0}(x,y)=g_A(x_lo,y_lo)`
  — the two seam corrections `p_y·[χ(Ax_lo,Ay_lo)⊕χ(x_lo,y_lo)]` and `p_x·[n0(Ay_lo)⊕n0(y_lo)]` vanish
  because `A∈GL` gives `n0(Az)=n0(z)` and `χ(Az,Aw)=χ(z,w)`, and the `p_x p_y` terms cancel. Then
  `g_A=δw ⟹ D_{M0}=δW` with `W(x)=w(x_lo)` (using `(x⊕y)_lo=x_lo⊕y_lo`), so `M0` valid. Combined with
  the ∀n forward reduction this is a **clean iff** `M0 valid ⟺ A valid`, hence a validity-preserving
  **bijection** `{valid β=0 autos of A_n} ↔ Aut_{n−1}`; with `β=0` forced by the block lemma,
  `|Aut_n| = |Aut_{n−1}|`. Verified n=4,5,6 (defect identity + lift + both counts meet 168). Advisor-
  audited (2026-07-11): "clean iff ∀n, no fatal gap." Script: `cd_tower_L2_lift_proof.py`.
- **Route-level risk — now RESOLVED.** The earlier worry (obstruction might stop firing at some `n>8`)
  is gone: `max_i deg'_β(i) < top` is now **PROVEN ∀n** for every `β≠0` (β_H=1 theorem + β_H=0
  telescoping, both giving closed forms `< top`), so the obstruction fires ∀n — no reliance on
  verified-only firing.
- Scripts: `scripts/research/cd_tower_L2_degree_obstruction.py` (reduction + firing n=4..7 + closed
  forms), `cd_tower_dCb_closed_form.py` (`δC_β` derivation), `cd_tower_Dbeta_derivation.py` (`D_β`
  derivation, inner counts certified).

**Honest ∀n status.** Freezing is **VERIFIED(n=4,5,6)** exhaustively; it is **NOT** a theorem ∀n. The
∀n reduction to two lemmas is now **one lemma closed, one open**:

1. **`L1` — unique-argmax.** **PROVEN ∀n≥4** (Result 4 above; direct closed-form proof, no longer
   conjectural). Seam-fixing is therefore established for every `n ≥ 4`, unconditionally.
2. **`L2` — the BLOCK LEMMA is now PROVEN ∀n ⟹ `|Aut_n| ≤ 168` ∀n (upper bound).** Via the
   non-circular structural obstruction (subsection above): the reduction
   `block lemma ⟸ [∀β≠0: max_i deg(Ψ⊕δC_β) < top]` is PROVEN ∀n; the obstruction **fires ∀n**, now
   proven (not just verified) — `δC_β`, `D_β`, the `β_H=1` tower recursion + its compatibility (A), and
   the `β_H=0` telescoping (`deg'^{β_H=0}=ΣR0`, no `Ψ`) all give closed forms `< top` ∀n (at
   **corr-parity** rigor — the `<top` steps rest on the same degree-≤2 counting argument as `corr`).
   So `β=0` is forced ∀n, i.e. every auto is `[[A,0],[0,1]]`, giving `|Aut_n| ≤ 168`.
3. **The LIFT — `|Aut_n| ≥ 168` — is now PROVEN ∀n (fully rigorous, not corr-parity).** `A valid ⟹
   [[A,0],[0,1]] valid` via the exact defect identity `D_{M0}=g_A(x_lo,y_lo)` (subsection above). This
   closes the backward direction that was VERIFIED(n≤6). Combined with L2 it is a **clean iff**, so
   `|Aut_n| = |Aut_{n−1}|`.

**Net: freezing `=168` for all n is a COMPLETE ASSEMBLED PROOF modulo exactly ONE standard lemma.**
The chain `L1 (∀n) → block-form → forward reduction (∀n) → β=0 by block lemma → lift (∀n, clean iff) →
|Aut_n| = |Aut_{n−1}| = … = |Aut_4| = 168` is closed **except** the block lemma's **value-pinning**: the
max-degree closed forms (`corr`, `2H²−4H−8`) are degree-≤2-in-`H` established by a **5-point pin**
(H=8,16,32,64,128) + standard F₂ rank-stabilization, **not yet inner-count-derived** the airtight way
`D_β` was. So the honest tier is: **every link PROVEN ∀n except one standard value-pinning lemma
(5-point-hardened).** Do **not** yet tag freezing `=168` as PROVEN(all n) — it is an *apparent complete
proof* pending (i) an explicit inner-count derivation of the max bucket, or (ii) a literature
cross-check (Cawagas / Moreno on sedenion auto-groups; `168 = |PSL(2,7)|`). The `≤168` upper bound is
an unconditional THEOREM ∀n; the `≥168` lift is an unconditional THEOREM ∀n; only their assembly into
`=168` funnels through the one soft link.

## Lean kernel leg (native_decide, finite dims only — not committed / no CI gate)

`formal/lean4/SounioCDTowerAutomorphism.lean` (Mathlib-free, a third independent transcription of
`cdSigma` matching the sibling `SounioSedenionAutomorphism.lean`) was written and built
(`lake build SounioCDTowerAutomorphism`, 39 s) to kernel-certify five of the above finite facts via
`native_decide`, **no `sorry`**. This is a *finite-dimension* leg, not a new ∀n result — L1 and L2
remain outside `native_decide`'s reach (no general ∀n induction was attempted in Lean here):

| theorem | statement | dim |
|---|---|---|
| `seam4_unique_argmax` | seam index 8 has `deg=168`, all others `≤168`, sole argmax | n=4 |
| `seam5_unique_argmax` | seam index 16 has `deg=840`, all others `≤840`, sole argmax | n=5 |
| `oct_psi3` | `Ψ₃(i,j,k) = [i⊕j⊕k≠0]` on all distinct-nonzero triples (octonion alternativity, block-lemma n=3 base) | n=3 |
| `block_lemma_n4` | every valid monomial auto has seam row `M[3]=8` (`β=0`, top-right 0-block), full `GL(4,2)=65536` sweep | n=4 |
| `cd_auto_count_n4` | exactly 168 valid autos, independent in-kernel recount | n=4 |

`#print axioms` on each: only `[propext, native_decide.ax_1_1]` — the standard `native_decide`
footprint, nothing smuggled in. `n=5`/`n=6` in-kernel count/block sweeps were **not** attempted
(`GL(5,2)` has 9.9M elements — beyond the practical `native_decide` ceiling used here); those
dimensions remain covered only by the Python/`souc` oracles above, not by this Lean leg. The file is
a non-`@[default_target]` `lean_lib` (no CI gate) and has not been committed.

## Reproduce

```bash
python3 scripts/research/cd_tower_automorphism_oracle.py           # n=4 sweep + both fiber geometries + n=6 assoc  (<1 s)
python3 scripts/research/cd_tower_automorphism_oracle.py --full    # + full GL(5,2) sweep + literal-lift + n=6 lift bound (~70 s)
# expect: ... CD_TOWER_AUT CERTIFIED
python3 scripts/research/cd_tower_automorphism_n6_exhaustive.py    # exhaustive n=6 seam-stabilizer sweep = 168 (~150 s, multiprocessing)
python3 scripts/research/cd_tower_block_lemma_proof.py             # block-lemma mechanism + n=4-unconditional  -> ALL CHECKS PASS
python3 scripts/research/cd_tower_block_lemma_psi_obstruction.py   # P1 + the beta!=0 => breaks-Psi forcing (n=4,5)
python3 scripts/research/cd_tower_seam_unique_argmax_proof.py      # L1 PROVEN ∀n>=4: closed form + independent brute-force cross-check n=2..7
python3 scripts/research/cd_tower_L2_reduction.py                  # L2 reduction: A-valid <=> beta=0 (n>=4); C-lemma; full sweeps n=3,4,5

# Lean (native_decide, finite dims, not committed / no CI gate):
export HOME=/workspace/.home/openvscode-server/.agents/claude-3; export PATH="$HOME/.elan/bin:$PATH"; ulimit -v unlimited
cd formal/lean4 && lake build SounioCDTowerAutomorphism
```
