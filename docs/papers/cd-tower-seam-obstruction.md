<!-- docs:meta
topic_id: repo.docs.papers.cd-tower-seam-obstruction
authority: repo_only
audience: researchers
last_validated: 2026-07-10
validated_by: claude-opus-4-8
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.cd-tower-seam-obstruction
-->

# Paper 3: The Anticommutator Obstruction to Zero-Division Across the Cayley–Dickson Tower — an Executable, Machine-Checked Certification in Exact Sign-Cocycle Form

## Claim

In every Cayley–Dickson (CD) algebra `A_n` (dimension `2^n`), for the "lower × upper" basis-index
pairs `(l,u)` (`1 ≤ l < 2^{n-1} ≤ u < 2^n`), zero-division of `e_l + e_u` is governed by a single
**sign-cocycle predicate** on the multiplication cocycle `σ`. This paper does **not** claim the
underlying *mathematical fact* as new: that these basis pairs are zero divisors, *with an explicit
2-term annihilator*, is a proved general-`n` theorem (Moreno 1998; Biss–Dugger–Isaksen 2008). What is
new here is (i) an **independent, machine-checked, Mathlib-free ∀n proof of the entire seam
coincidence** `off-seam ⟺ 2-term zero divisor ⟺ XOR-winner ⟺ {L_l,L_u} ≠ 0` **in exact integer
σ-form** (`seam_coincidence`: `isZD = offSeam ∧ hasXorAnnih = offSeam ∧ anti0 = ¬offSeam`, ∀ `bits≥4`
on loHi) — both the converse (`off-seam ⟹ ZD`, an ordinary induction over the tower with an octonion
base) and its forward obstruction (`on-seam ⟹ not a ZD`), plus the operator bridge, all with kernel
axioms `[propext, Quot.sound]` plus a single `native_decide` octonion-base anchor; (ii) an `O(N)`
decision reduction that pushes the cross-check to dimension `1024`; and (iii) a primary-source
cross-check against Moreno's original annihilator. The σ-recursion machinery gives a *combinatorial*
form of the Moreno/BDI statement — its own, self-contained proof in the Sounio convention, not a
transcription.

> **Core contract:** Exactness is a property of the computation, not of the number. Every claim below
> is tagged as exactly one of: **proved** (Lean, no `sorry`, no Mathlib), **decided** (Lean
> `native_decide` at a fixed dimension), **computed** (Python probe cross-checked against brute
> force), **cited** (external published theorem), or **conjectured** (empirically verified over a
> stated range; not yet proved in the Sounio encoding). No tag is silently upgraded.

## Setup

`A_n = A_{n-1} ⊕ A_{n-1}·e_{top}`, `top = 2^{n-1}`. Basis units `e_0..e_{2^n-1}` (`e_0` real), with a
sign cocycle `σ(a,b) ∈ {±1}` such that `e_a·e_b = σ(a,b)·e_{a⊕b}` (indices as bit-strings; the
product index is the bitwise XOR `a⊕b`). Left multiplication `L_i(x) = e_i·x`. The **top seam** of the
lower×upper pairs is `{(l,u) : u = top or l⊕u = top}`; its complement is the **off-seam** set,
`off-seam(l,u) := ¬(u = top ∨ l⊕u = top)`. The base identity `L_i² = −I` is `σ(i,j)·σ(i, i⊕j) = −1`.

## 1. The predicates and the status-tagged coincidence

Over the lower×upper locus we relate four predicates on `(l,u)` (`SounioCDTowerSeam.lean`,
`SounioCDConverse.lean`):

- `offSeam(l,u)` — combinatorial (off the top seam).
- `hasXorAnnih(l,u)` — **defined as** the existence of a **2-term** annihilator `e_a + s·e_b` with
  `a,b ≥ 1`; §2 shows this forces `b = a⊕(l⊕u)` and reduces to a four-sign condition.
- `isZD(l,u)` — the brute search for a 2-term annihilator (`SounioCDTowerSeam.isZD`).
- `anti0(l,u)` — the operator anticommutator vanishing, `{L_l, L_u} = 0` (`SounioCDTowerSeam.anti0`).

The content is the coincidence `offSeam ⟺ hasXorAnnih ⟺ isZD ⟺ ¬anti0`. **Every link is now a proved
∀n theorem** on the loHi locus (`bits ≥ 4`), bundled as `seam_coincidence`:

| Link | Direction | Status |
|---|---|---|
| `offSeam ⟹ hasXorAnnih` | the converse proper | **proved ∀n** (`converse_holds`) — ordinary induction (`Q_all`; octonion base by `native_decide`; six exhaustive seam cases) + the BDI doubling recursion `converse_recursion'`; discharges `ConverseConjecture` (`converse_conjecture_proved`). Empirically verified n≤10 (§2); matches Moreno/BDI (§4) |
| `hasXorAnnih ⟹ isZD` | reduction (sufficiency) | **proved ∀n** (`hasXorAnnih_sound`, on `l,u < 2^bits`, `l ≠ u`) |
| `isZD ⟹ hasXorAnnih` | reduction (necessity) | **proved ∀n** (`hasXorAnnih_complete`) — the reverse reduction `annih_forces` (any 2-term annihilator forces `b=a⊕(l⊕u)` and the four-sign winner, axioms `[propext, Quot.sound]`), with the `a=0`/`{0,d}`-orbit corner (`P0_neg_of_onSeam`) routed through `converse_holds`. Upgrades the old n=4 anchor; `xorAnnih_eq_isZD_all` gives the full `hasXorAnnih == isZD` on loHi ∀n |
| `¬offSeam ⟹ ¬isZD` | forward obstruction | **proved ∀n** (`isZD_eq_offSeam`) — on-seam pairs have no XOR-winner (`hasXorAnnih_false_of_onSeam`, every orbit loses via the edge lemmas) composed with `hasXorAnnih_complete`. A purely combinatorial route (no operator/`anti0` detour needed) |
| `anti0 ⟺ ¬offSeam` | operator bridge | **proved ∀n** (`seam_eq_anti0`) — via the identity `Q(c)=P(c)` (the anticommutator four-sign product equals the converse winner product, two `cdSigma_cocycle` rewrites), so the operator condition reduces to the same `P`-analysis. Generalizes `coincidence_16/32` to all n (`anti0_eq_offSeam_all`) |

All links carry axioms `[propext, Quot.sound]` plus the single inherited k=3 octonion `native_decide`
base anchor. `seam_coincidence` states the four-way identity as one theorem: `isZD = offSeam ∧
hasXorAnnih = offSeam ∧ anti0 = ¬offSeam`, ∀ bits≥4 on loHi. The one overclaim we still avoid: **no**
claim about zero divisors with annihilators of length `> 2` — every predicate here is about 2-term
factors by definition (Grok-verified exhaustive for 2-term factors, §2).

## 2. The 2-term reduction (the crux, in exact σ-form)

A 2-term right factor `e_a + s·e_b` (`a ≠ b`, `s ∈ {±1}`) satisfies `(e_l+e_u)(e_a + s·e_b) = 0`
**only if** the four output indices `{l⊕a, l⊕b, u⊕a, u⊕b}` coincide in cross-pairs, forcing

```
b = a ⊕ (l⊕u).
```

Given that `b`, annihilation holds iff both cancellation equations are solvable in `s`:
`σ(l,a) + s·σ(u,b) = 0` and `s·σ(l,b) + σ(u,a) = 0`, i.e. iff the **four-sign product**

```
P(a) := σ(l,a)·σ(u,a)·σ(l, a⊕l⊕u)·σ(u, a⊕l⊕u) = +1,    with then  s = −σ(l,a)·σ(u,b).
```

An independent math-review (grok-4.1) confirmed this reduction is correct and **exhaustive for
2-term factors**. It collapses the `O(N²)` 2-term search to `O(N)` per pair (`hasXorAnnih`), a fast
path *validated equal to* the brute `isZD` at n=4,5,6 and then run to n=7,8,9,10 (dim 128..1024) with
**zero counterexamples** (`cd_tower_converse_probe.py`).

**Structural fact (now proved ∀n).** The winner set `{a : P(a)=+1}` is never empty on the off-seam
loHi locus — this is exactly what `Q_all` + `converse_recursion'` establish (a non-exceptional
disagreeing orbit downstairs maps to an upstairs winner). Its *cardinality* remains an empirical
observation: `≥ 8`, taking tower-shaped values `8·(2^k−1)` at the tested levels
(`Σ_a P(a) = 4·#agreeing-orbits − N`), not needed for the non-emptiness proof.

## 3. Ingredients proved / decided in the Sounio encoding

- **`e_i² = −1` (diagonal), for all n — proved** (Mathlib-free structural induction on bit-lists,
  `SounioCDCocycle.diag`, PR #718).
- **`L_i² = −I` (full cocycle lemma) + basis-unit anticommutation — proved ∀n on the bit-list sign**
  (`SounioCDCocycle.Lsq` and `SounioCDCocycle.antisym`, Mathlib-free, kernel-verified axioms
  `[propext, Quot.sound]`, no `sorry`/`native_decide`). Proved via the bundled four-property
  conjunction `cocycle_bundle` (diag ∧ antisym ∧ L ∧ R) by a length-indexed ∀-pairs induction — the
  simultaneous induction the Door-1 target always required. This closes Door 1 as framed, and the
  standing anticommutation piece open since #718.
- **Representation bridge `sgn = cdSigma`, for all n — proved** (`sgn_eq_cdSigma` for `1 ≤ n`;
  `xorL_bitsOf`, `isZ_bitsOf`; all axioms `[propext, Quot.sound]`), generalizing the n=4,5
  `native_decide` anchors `agree4/agree5`. Hence **`cdSigma_cocycle` — `L_i²=−I` on the canonical Nat
  sign `cdSigma`, for all n** (`∀ n i j, i<2^n → j<2^n → i≠0 → cdSigma i j n · cdSigma i (i⊕j) n = −1`).
  The **representation gap is closed**: the cocycle lemma `L_i²=−I` is unconditional on `cdSigma` for
  all n, and (see below) the `¬offSeam ⟺ anti0` operator bridge is now **also proved ∀n**
  (`seam_eq_anti0`), so the forward obstruction `¬offSeam ⟹ ¬isZD` is a ∀n theorem (`isZD_eq_offSeam`).
- **Reduction sufficiency `hasXorAnnih ⟹ isZD` — proved ∀n** (`hasXorAnnih_sound`, Mathlib-free, no
  `sorry`/`native_decide`; axioms `[propext, Quot.sound]`), on `l,u < 2^bits`, `l ≠ u`. From a
  `hasXorAnnih` witness `a` it constructs the explicit `isZD` certificate — the XOR-linked pair
  `{a, a⊕(l⊕u)}` with sign `s = −σ(l,a)σ(u,b)`, ordered — and discharges `annih` at every output
  index. (The three hypotheses are load-bearing: the theorem is false for `l=u` or unbounded indices;
  all hold on loHi.)
- **The converse proper `offSeam ⟹ hasXorAnnih` — proved ∀n** (`converse_holds`, `∀ bits≥4` on loHi;
  and `converse_conjecture_proved : ConverseConjecture` composing it with `hasXorAnnih_sound`).
  Mathlib-free, axioms `[propext, Quot.sound]` + the single k=3 octonion-base `native_decide` anchor.
  The engine is `Q_all`: *every distinct-nonzero pair in `A_k` (`k≥3`) has a non-exceptional
  disagreeing orbit* (`P=−1`), by **ordinary** induction on the level. Base = octonions `A_3` (a
  division algebra; `native_decide`). Step = six exhaustive cases on the seam position `H=2^k`: three
  seam-element **edges** collapse to the cocycle/antisym/diagonal identities (`edge_m_eq_H`,
  `edge_m_eq_H_plus_l`, `edge_l_eq_H`); **both-low** and **both-high** pairs *inherit* their witness
  from the level below via sign-stability (`P_stable_low`, `fVal_high_stable`); **mixed** pairs use the
  explicit witness `a=m_lo` (`mixed_witness_disagree`). The doubling step `P_(l,u)(a)=−P_(l,u_lo)(a)`
  (`converse_recursion'`, unconditional via the proved `cdAntisym_all`) turns a downstairs disagreeing
  orbit into an upstairs `hasXorAnnih` winner.
- **Reduction necessity `isZD ⟹ hasXorAnnih` — proved ∀n** (`hasXorAnnih_complete`, loHi, `bits≥4`).
  The core is `annih_forces` (axioms `[propext, Quot.sound]`, no native anchor): *any* 2-term
  annihilator `(a,b,s)` forces `b = a⊕(l⊕u)` and the four-sign winner product `= +1` — the reverse of
  `hasXorAnnih_sound`. Subtlety handled: the brute `isZD` admits an `a=0` factor `1+s·e_d` on the
  `{0,d}` orbit that `hasXorAnnih` excludes; on-seam it cannot annihilate (`P0_neg_of_onSeam`), off-seam
  `converse_holds` supplies the real winner. `xorAnnih_eq_isZD_all` then gives the full
  `hasXorAnnih == isZD` on loHi for all n (generalizing the n=4 anchor `xorAnnih_eq_isZD_16`).
- **Forward obstruction `¬offSeam ⟹ ¬isZD` — proved ∀n** (`isZD_eq_offSeam`, in fact `isZD = offSeam`
  on loHi ∀ `bits≥4`). On-seam pairs have no XOR-winner (`hasXorAnnih_false_of_onSeam`: every orbit
  loses, via the edge lemmas + `P0_neg_of_onSeam`), composed with `hasXorAnnih_complete`. A purely
  combinatorial route — no operator/`anti0` detour.
- **Operator bridge `anti0 ⟺ ¬offSeam` — proved ∀n** (`seam_eq_anti0`; `anti0_eq_offSeam_all` over
  `loHi`). Key identity `anti0_QP`: the anticommutator four-sign product `Q(c)` equals the converse
  winner product `P(c)` (two `cdSigma_cocycle` rewrites), so `{L_l,L_u}=0 ⟺ ∀c P(c)=−1 ⟺ ¬offSeam`
  reuses the converse `P`-analysis wholesale. Generalizes `coincidence_16/32` to all n.
- **The full coincidence — `seam_coincidence`:** `isZD = offSeam ∧ hasXorAnnih = offSeam ∧
  anti0 = ¬offSeam`, one theorem, ∀ `bits≥4` on loHi, axioms `[propext, Quot.sound]` + the single k=3 base.
- **Converse anchors — decided:** `converse_16` (brute), `converse_sharp_16/32/64` (sharp σ-form),
  `xorAnnih_eq_isZD_16`, `coincidence_16/32` (now all superseded ∀n by the theorems above but retained
  as fast regressions), `native_decide`, no `sorry`.
- **Primary-source regression — decided:** `moreno_e1_e10` — Moreno's own example, `e₁+e₁₀`
  annihilated by `e₁₅−e₄`, discharged via `annih` (`l=1,u=10,a=15,b=4,s=−1`; note `4 = 15⊕11`,
  i.e. `b=a⊕d` — the XOR link appears literally in the 1998 source).

## 4. Relation to the published literature (honest three-tier)

- **KNOWN ∀n (high confidence).** These basis pairs are zero divisors, with an explicit 2-term
  XOR-linked annihilator. **Moreno, Thm 2.9** (`q-alg/9710013`): for alternative trace-zero norm-1
  `a,b`, `(a,b) ∈ A_{n+1}` is a ZD iff `−2 ∈ spec(L²_{a+b})`; his opening worked example is exactly
  `e₁+e₁₀ ⟂ e₁₅−e₄`. **Biss–Dugger–Isaksen, Prop 11.1** (`math/0511691`): `Ann(a,b) = {(x,(ax)b)}`,
  yielding the same 2-term annihilator for every distinct imaginary basis pair. Basis units are
  alternative, so `e_l+e_u` is in scope. Our `moreno_e1_e10` and the `hasXorAnnih` reduction are a
  faithful, executable instance of this machinery.
- **NOT confirmed (medium confidence).** That the criterion fires *exactly on the off-seam set*, in
  closed combinatorial form, for **all** n. This is most plausibly the content of Zhilina's
  "doubly alternative zero divisors / hexagons" (IJAC 31:4 (2021); J. Math. Sci. 272:4 (2023)), which
  we could not access (paywalled, not on arXiv). **Action item, not a citation of proof.**
- **UNVERIFIED — a correctness caveat, not a result.** Moreno/BDI's anticommutator lives on
  `{L_{e_l}, L_{e_{u'}}}` with *both* indices in the lower algebra `A_{n-1}` (`u' = u − top`), whereas
  our seam theorem is stated on `{L_l, L_u}` at the actual `A_n` indices. These are related through
  the doubling generator but are **not literally the same operator pair** (the sign flavor differs
  too). We treat the **Sounio convention as primary** — it is self-consistently `native_decide`-verified
  and stands on its own — and flag the Moreno correspondence as an unchecked bridge, not an identity.

## 5. The honest ledger

- **Proved (∀n):** `e_i²=−1`; **`L_i²=−I` and basis-unit anticommutation** — on the bit-list sign
  (`Lsq`, `antisym`, `cocycle_bundle`) **and on the canonical Nat sign** (`cdSigma_cocycle`, via the
  proved `sgn = cdSigma` bridge). And — on the loHi locus for all `bits ≥ 4` — **the entire seam
  coincidence** (`seam_coincidence`: `isZD = offSeam ∧ hasXorAnnih = offSeam ∧ anti0 = ¬offSeam`), i.e.
  all four links: the converse `offSeam ⟹ hasXorAnnih` (`converse_holds`, discharging
  `ConverseConjecture`), sufficiency `hasXorAnnih ⟹ isZD` (`hasXorAnnih_sound`), necessity
  `isZD ⟹ hasXorAnnih` (`hasXorAnnih_complete`, via `annih_forces`), the forward obstruction
  `¬offSeam ⟹ ¬isZD` (`isZD_eq_offSeam`), and the operator bridge `anti0 ⟺ ¬offSeam` (`seam_eq_anti0`,
  via `Q(c)=P(c)`). Axioms `[propext, Quot.sound]` throughout, plus the single k=3 octonion
  `native_decide` base anchor (inherited from the converse's octonion base).
- **Decided (fixed n), retained as fast regressions (all superseded ∀n above):** `xorAnnih_eq_isZD_16`
  (necessity n=4); `coincidence_16/32` (the seam coincidence n=4,5,6); Moreno's example.
- **Computed:** brute==fast reduction n=4,5,6; the converse frontier n=7..10 (dim ≤1024) — now a
  cross-check of proved theorems, not the evidence base.
- **Cited (external, ∀n):** ZD-status + explicit 2-term XOR annihilator for these basis pairs
  (Moreno Thm 2.9; BDI Prop 11.1) — our `seam_coincidence` now supplies an *independent, machine-checked*
  proof of the same statement (and its converse) in the Sounio convention.
- **Still open in this encoding:** the *closed-form* off-seam⟺criterion identity in the published
  literature's own framing (possibly Zhilina); and the unchecked Moreno seam-index correspondence (§4)
  — a caveat about matching *their* operator pair, not a gap in our proofs. `ConverseConjecture` — the
  conjecture this file was built around — is a theorem (`converse_conjecture_proved`), and the seam
  coincidence it sits inside is now fully closed ∀n on loHi.

## 6. Reproduce, and the next formal step

```bash
export PATH="$HOME/.elan/bin:$PATH"
(cd formal/lean4 && lake build SounioCDConverse SounioCDTowerSeam SounioCDCocycle)   # green, no sorry
python3 scripts/research/cd_tower_converse_probe.py     # validate O(N) vs brute n≤6, then n=7..10
python3 scripts/research/cd_tower_seam_oracle.py        # forward / seam tower
```

**Done — the seam coincidence is fully closed ∀n on loHi** (`seam_coincidence`): `Door 1 — L_i²=−I +
antisym` proved ∀n (`cocycle_bundle`) on both `sgn` and canonical `cdSigma`; the converse
`offSeam ⟹ hasXorAnnih` (`converse_holds`, discharging `ConverseConjecture`); sufficiency
`hasXorAnnih ⟹ isZD` (`hasXorAnnih_sound`); necessity `isZD ⟹ hasXorAnnih` (`hasXorAnnih_complete`,
via the reverse reduction `annih_forces`); the forward obstruction `¬offSeam ⟹ ¬isZD`
(`isZD_eq_offSeam`); and the operator bridge `anti0 ⟺ ¬offSeam` (`seam_eq_anti0`, via `Q(c)=P(c)`).
Every "single remaining open theorem" of the earlier drafts is now closed.

The engine is the counting statement `Q` on the sign cocycle
(`scripts/research/cd_tower_converse_counting.py`), now a **theorem** (`Q_all`): *every
distinct-nonzero pair `(l,m)` in `A_k` (`k≥3`) has a non-exceptional disagreeing orbit* (`P=−1`),
where an orbit is `{a, a⊕d}`, `d=l⊕m`, `P` is constant on orbits, and the two exceptional orbits are
`{0,d}` and `{l,m}`. Proved by **ordinary induction on the level** (`native_decide` octonion base,
six exhaustive seam cases), then lifted through the doubling recursion `P_(l,u)(a) = −P_(l,u_lo)(a)`.

The proof levers, all now formal:
- **Base:** octonions `A_3` are a division algebra — *every* orbit disagrees (`oct_all_disagree` /
  `Q_base_bool`, `native_decide`; the sole native anchor).
- **Doubling / stability:** `P_stable_low` (both-low, low block) and `fVal_high_stable` (both-high, low
  block) — the sign is level-invariant on the low block, so these pairs **inherit** a disagreeing
  witness from the level below.
- **Explicit witness (a σ-identity):** a mixed pair `(l, 2^{k−1}+m_lo)` with `m_lo ∉ {0,l}` disagrees
  at `a = m_lo` (`mixed_witness_disagree`).
- **Seam-element edges:** the three pairs touching `H=2^{k−1}` collapse to `cdSigma_cocycle`,
  `cdAntisym_all`, `cdSigma_diag` (`edge_m_eq_H`, `edge_m_eq_H_plus_l`, `edge_l_eq_H`).

**On the earlier "both-low residual seam":** it is **resolved by inheritance**, and was never an
independent obstruction. Both-low pairs do *not* need the recursion to "peel" them — they **inherit**
their disagreeing witness from the level below via sign-stability (`P_stable_low`), exactly like
both-high pairs (`fVal_high_stable`). The seam was genuinely non-trivial until this reduction was
found; once it is seen, the induction is ordinary and every case closes, with no hard both-low core
remaining. (This supersedes the provisional "genuinely hard case, cf. Zhilina" note in earlier drafts.)

## References

- Moreno G, "The zero divisors of the Cayley–Dickson algebras over the reals," Bol. Soc. Mat.
  Mexicana (3) 4 (1998) 13–28. arXiv `q-alg/9710013`. (Thm 2.9; opening `e₁+e₁₀` example.)
- Biss D, Dugger D, Isaksen D, "Large annihilators in Cayley–Dickson algebras," Comm. Algebra 36
  (2008) 632–664. arXiv `math/0511691`. (Prop 11.1; doubling recursion Lemmas 8.5–8.10.)
- Biss, Christensen, Dugger, Isaksen, "…II," arXiv `math/0702075`; "Eigentheory…," arXiv `0905.2987`.
- Zhilina S, "Orthogonality graphs of real Cayley–Dickson algebras I," IJAC 31:4 (2021) 663–689; "On
  doubly alternative zero divisors…," J. Math. Sci. 272:4 (2023) 496–518. *(Not accessed; flagged.)*
- de Marrais R, "The 42 Assessors and the Box-Kites…," arXiv `math/0011260` (heuristic; nomenclature).
- Cawagas RE, Discuss. Math. 24 (2004) 251; Kivunge & Smith, Comment. Math. Univ. Carolin. (2004).
- In-repo lineage: `sedenion_seam_bridge.md` (#713), `cd_tower_seam.md` (#717), `SounioCDCocycle.lean`
  (#718); Paper 1 `exact-168-executable.md`, Paper 2 `sedenion-fano-geometry.md`.
