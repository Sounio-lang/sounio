<!-- docs:meta
topic_id: repo.docs.papers.cd-tower-seam-obstruction
authority: repo_only
audience: researchers
last_validated: 2026-07-08
validated_by: claude-opus-4-8
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.cd-tower-seam-obstruction
-->

# Paper 3: The Anticommutator Obstruction to Zero-Division Across the Cayley–Dickson Tower — an Executable, Machine-Checked Certification in Exact Sign-Cocycle Form

## Claim

In every Cayley–Dickson (CD) algebra `A_n` (dimension `2^n`), for the "lower × upper" basis-index
pairs `(l,u)` (`1 ≤ l < 2^{n-1} ≤ u < 2^n`), zero-division of `e_l + e_u` is governed by a single
**sign-cocycle predicate** on the multiplication cocycle `σ`. This paper does **not** claim the
underlying mathematics as new: that these basis pairs are zero divisors, *with an explicit 2-term
annihilator*, is a proved general-`n` theorem (Moreno 1998; Biss–Dugger–Isaksen 2008). What is new
here is an **executable, Mathlib-free, `native_decide`-checked certification** of the phenomenon in
exact integer σ-form, an `O(N)` decision reduction that pushes empirical verification to dimension
`1024`, and a primary-source cross-check against Moreno's original annihilator.

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

## 1. Three predicates and the status-tagged chain

Over the lower×upper locus we relate three predicates on `(l,u)` (`SounioCDTowerSeam.lean`,
`SounioCDConverse.lean`):

- `offSeam(l,u)` — combinatorial (off the top seam).
- `hasXorAnnih(l,u)` — **defined as** the existence of a **2-term** annihilator `e_a + s·e_b` with
  `a,b ≥ 1`; §2 shows this forces `b = a⊕(l⊕u)` and reduces to a four-sign condition.
- `isZD(l,u)` — the brute search for a 2-term annihilator (`SounioCDTowerSeam.isZD`).

The content is the chain `offSeam ⟺ hasXorAnnih ⟺ isZD`, each **link separately tagged**:

| Link | Direction | Status |
|---|---|---|
| `¬offSeam ⟹ ¬isZD` | forward obstruction | **proved** given `L_i²=−I`, which is now itself **proved ∀n on the bit-list sign** (`Lsq`, §3); transfer to the Nat `cdSigma` used here is decided at n=4,5 (`agree4/5`), general bridge pending (§3) |
| `hasXorAnnih ⟹ isZD` | reduction (sufficiency) | **proved ∀n** (`hasXorAnnih_sound`, Mathlib-free, axioms `[propext, Quot.sound]`), on the domain `l,u < 2^bits`, `l ≠ u` — both hold on loHi |
| `isZD ⟹ hasXorAnnih` | reduction (necessity) | **decided** n=4 (`xorAnnih_eq_isZD_16`); argument **Grok-verified** exhaustive *for 2-term factors* (§2) |
| `offSeam ⟹ hasXorAnnih` | the converse proper | **decided** n=4,5,6; **conjectured** (verified n≤10, §2); **cited** as following ∀n from Moreno/BDI modulo the index bridge (§4) |

Note the two blocking overclaims we explicitly avoid: (a) we make **no** claim about zero divisors
with annihilators of length `> 2` — every predicate above is about 2-term factors by definition; and
(b) `offSeam ⟺ hasXorAnnih` is stated as a *chain of tagged links*, never as a single proved
biconditional.

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

**Structural observation (empirical, n≤10 — not proved):** the winner set `{a : P(a)=+1}` is never
empty on the off-seam locus; its cardinality is `≥ 8` and takes tower-shaped values `8·(2^k−1)` at
the tested levels, consistent with a counting/recursion proof (`Σ_a P(a) = 4·#agreeing-orbits − N`).

## 3. Ingredients proved / decided in the Sounio encoding

- **`e_i² = −1` (diagonal), for all n — proved** (Mathlib-free structural induction on bit-lists,
  `SounioCDCocycle.diag`, PR #718).
- **`L_i² = −I` (full cocycle lemma) + basis-unit anticommutation — proved ∀n on the bit-list sign**
  (`SounioCDCocycle.Lsq` and `SounioCDCocycle.antisym`, Mathlib-free, kernel-verified axioms
  `[propext, Quot.sound]`, no `sorry`/`native_decide`). Proved via the bundled four-property
  conjunction `cocycle_bundle` (diag ∧ antisym ∧ L ∧ R) by a length-indexed ∀-pairs induction — the
  simultaneous induction the Door-1 target always required. This closes Door 1 as framed, and the
  standing anticommutation piece open since #718. **Residual (representation bridge):** the proof is
  on the bit-list `sgn`; its equality to the Nat `cdSigma` used by the forward/converse predicates is
  `native_decide`-checked at n=4,5 (`agree4/agree5`), so the *fully general* transfer to `cdSigma`
  awaits a separate (mechanical) `sgn = cdSigma` ∀n induction. At the `sgn` level the forward
  obstruction of §1 is now **unconditional**; at the `cdSigma` level it is unconditional for n=4,5 and
  conditional-on-the-bridge beyond.
- **Reduction sufficiency `hasXorAnnih ⟹ isZD` — proved ∀n** (`hasXorAnnih_sound`, Mathlib-free, no
  `sorry`/`native_decide`; axioms `[propext, Quot.sound]`), on `l,u < 2^bits`, `l ≠ u`. From a
  `hasXorAnnih` witness `a` it constructs the explicit `isZD` certificate — the XOR-linked pair
  `{a, a⊕(l⊕u)}` with sign `s = −σ(l,a)σ(u,b)`, ordered — and discharges `annih` at every output
  index. (The three hypotheses are load-bearing: the theorem is false for `l=u` or unbounded indices;
  all hold on loHi.)
- **Converse anchors — decided:** `converse_16` (brute), `converse_sharp_16/32/64` (sharp σ-form),
  `xorAnnih_eq_isZD_16` (reduction == brute at n=4), all `native_decide`, no `sorry`.
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

- **Proved (∀n, bit-list `sgn`):** `e_i²=−1`; **`L_i²=−I` (`Lsq`) and basis-unit anticommutation
  (`antisym`)** via `cocycle_bundle`; and the reduction sufficiency `hasXorAnnih ⟹ isZD`
  (`hasXorAnnih_sound`, on `l,u < 2^bits`, `l ≠ u`).
- **Proved (∀n on `sgn`; on `cdSigma` for n=4,5 + conditional-on-bridge beyond):** the forward
  obstruction `¬offSeam ⟹ ¬isZD` (its `L_i²=−I` hypothesis is now discharged at the `sgn` level).
- **Decided (fixed n):** `sgn=cdSigma` representation bridge at n=4,5 (`agree4/5`); seam coincidence
  n=4,5,6; reduction `hasXorAnnih==isZD` n=4; sharp converse n=4,5,6; Moreno's example.
- **Computed:** brute==fast reduction n=4,5,6; the converse frontier n=7..10 (dim ≤1024).
- **Cited (external, ∀n):** ZD-status + explicit 2-term XOR annihilator for these basis pairs
  (Moreno Thm 2.9; BDI Prop 11.1).
- **Conjectured / open in this encoding:** `offSeam ⟹ hasXorAnnih` as a *formal* ∀n theorem, and the
  closed-form off-seam⟺criterion identity (possibly Zhilina). `ConverseConjecture : Prop` is stated,
  **unasserted** — its honest status is "follows ∀n from Moreno/BDI modulo a seam-index correspondence
  we have not formally checked; empirically verified n≤10; not yet proved in the Sounio encoding."

## 6. Reproduce, and the next formal step

```bash
export PATH="$HOME/.elan/bin:$PATH"
(cd formal/lean4 && lake build SounioCDConverse SounioCDTowerSeam SounioCDCocycle)   # green, no sorry
python3 scripts/research/cd_tower_converse_probe.py     # validate O(N) vs brute n≤6, then n=7..10
python3 scripts/research/cd_tower_seam_oracle.py        # forward / seam tower
```

**Done:** `hasXorAnnih ⟹ isZD` proved ∀n (`hasXorAnnih_sound`); and **Door 1 — `L_i²=−I` + `antisym`
proved ∀n** on the bit-list sign (`cocycle_bundle`), discharging the forward obstruction's hypothesis.
**Remaining open theorems:** (i) the `sgn = cdSigma` ∀n representation bridge (mechanical induction;
currently `native_decide` at n=4,5) — closing it makes the forward obstruction unconditional on
`cdSigma` too; (ii) the converse proper `offSeam ⟹ hasXorAnnih` ∀n, which needs an existence argument
(a winning `a`), route via the BDI doubling recursion (base case `A_3`).

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
