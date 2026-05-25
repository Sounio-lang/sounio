<!-- docs:meta
topic_id: repo.docs.papers.main.168-dual-pathway-correction
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.main.168-dual-pathway-correction
-->

# Correction & deepening of Theorem 2 — "A Dual Pathway to 168"

**Status:** machine-checked correction note for the v2 revision of `168-dual-pathway.typ`
(Authorea DOI 10.22541/au.177507195.52530214/v1). Do **not** treat the original Theorem 2
as proved; this note states the corrected theorem and its proof, with the exact obstruction.

*Written in English to match the preprint and v2 revision; the mathematical content was
developed in Portuguese in the verification arena (session transcript). Basis vectors are
written in plain `e_1`, `e_10` form (not Unicode subscripts) so the witness pastes cleanly
into the Typst source.*

All claims are (i) hand-derivable below and (ii) verified by exact integer arithmetic in two
independent codebases:
- Python oracle: `scripts/research/generate_sedenion_phi_map_injectivity.py` (reuses the
  compiler-faithful Cayley–Dickson sign `cd_sigma`); artifact
  `artifacts/research/sedenion_phi_map_injectivity.v1.json` (27/27 checks pass).
- Sounio (self-hosted compiler): `tests/stdlib/nn/test_fractal_sedenion_e2e.sio`, tests
  **T9/T10** (10/10 passed). Gate: `scripts/ci/sedenion_phi_injectivity_gate.sh`.

**Scope of the two codebases.** Sounio T9/T10 verify the *algebraic core* — I1–I4 and the
126/42 counts — in integer arithmetic. The GL(3,2) sign cocycle of §5 (group theory: building
the 168-element group, solving the sign-lift, the 28224 cocycle check) is verified in the
**Python oracle only**; the Sounio side mirrors the algebraic identities, not the group action.

---

## 0. The error in Theorem 2

The published Theorem 2 asserts a **GL(3,2)-equivariant bijection** `Φ̄` between the 168 ordered
non-collinear Fano triples and the unordered primitive zero-divisor pairs, with the unordered
descent justified by: *"For triples with different (p,q), exhaustive enumeration confirms that
Φ̄ remains injective"* (line 237).

**This is false.** A single exhibitable witness (plain-subscript form, paste-safe):

> `T = (1,2,5)` (s = 1 xor 2 xor 5 = 6) and `iota(T) = (5,6,1)` are distinct admissible triples
> with **different** `(p,q)` — `(1,2)` vs `(5,6)` — yet
> `Phi_bar(T) = Phi_bar(iota T) = { (e_1 + e_10)/sqrt(2),  (e_5 + e_14)/sqrt(2) }`.

There are exactly **42** such collisions; the unordered image has cardinality **126**, not 168.
The map `Φ̄` is **not** injective.

---

## 1. Definitions (unchanged from the preprint)

Triple `T=(p,q,r)`, distinct indices in `{1,…,7}`, `s := p⊕q⊕r`. **Admissible** = `s≠0` and
`{p,q,r,s}` distinct; there are exactly **168** (the parity condition `p⊕q=r⊕s` of the
preprint is *vacuous* — it is equivalent to `s=p⊕q⊕r`). With `δ_a=+1`,

- `δ_b = −σ(p,r)·σ(s,q)`  (Eq.12). **Ground truth:** this is the *unique* `δ_b∈{±1}` making
  `a·b=0` (verified 168/168 against the sedenion product), so Eq.12 is correct.
- `Φ(T) = (a,b)`, `a=(e_p + e_{q+8})/√2`, `b=(e_r + δ_b e_{s+8})/√2`.
- `σ` is **antisymmetric** on distinct nonzero indices: `σ(j,i) = −σ(i,j)` (imaginary
  octonions anticommute).

## 2. The order-swap involution ι

Reversing `(a,b)→(b,a)` and re-canonicalising to `δ_a=+1` form sends `b=(e_r+δ_b e_{s+8})`
to the `a`-slot, forcing `p'=r`, `q'=s`; the trailing `a` becomes the `b`-slot, `r'=p`; and
`s' = p'⊕q'⊕r' = r⊕s⊕p = q`. Hence

> **ι(p,q,r) = (r, s, p)**,  with `s(ιT)=q`.

`ι² = id` (apply twice: `ι(r,s,p)=(p,q,r)`), it is **fixed-point-free** (`ιT=T` ⇒ `r=p`), and
it maps the 168 onto itself: the domain is **84 ι-orbits of size 2**. *(Verified.)*

## 3. The four identities

- **I1 (δ_b ∘ ι = δ_b).** `δ_b(ιT) = −σ(r,p)σ(q,s) = −(−σ(p,r))(−σ(s,q)) = −σ(p,r)σ(s,q) = δ_b(T)`,
  using antisymmetry. *(168/168.)*  This already refutes the original conjecture
  `δ_b(T)·δ_a(ιT) = −1`: with `δ_a≡+1` and `δ_b∘ι=δ_b`, the product is `δ_b(T)=±1`, never `−1`.
- **I3 (Sign Cancellation Lemma).** `σ(p,r)σ(s,q)σ(s,p)σ(q,r) = −1`. *(168/168; unchanged.)*
- **I2 (collision criterion).** As unordered sets,
  `Φ̄(T)={(e_p+e_{q+8}),(e_r+δ_b e_{s+8})}`,
  `Φ̄(ιT)={(e_r+e_{s+8}),(e_p+δ_b e_{q+8})}` (using `r'=p,s'=q` and I1). The low indices `p≠r`
  identify the vectors; matching gives `Φ̄(T)=Φ̄(ιT) ⇔ δ_b(T)=+1`. *(168/168.)*
  **Uniqueness (no enumeration):** if `Φ̄(T)=Φ̄(U)` with `T≠U`, then since `Φ` is *ordered*-
  injective (preprint line 233), `Φ(U)` must be the reverse `(b,a)`; the `a`-slot of `U` would
  be `b`, whose high coefficient is `δ_b(T)` — but `a`-slots have high coefficient `+1`, forcing
  `δ_b(T)=+1` and `U=ιT`. So every collision is an ι-orbit, and only for `δ_b=+1`.
- **I4 (involution closes).** ι is a fixed-point-free involution on the 168. *(Verified.)*

## 4. Corrected Theorem

> **Theorem 2′.** For the 168 ordered non-collinear Fano triples with `Φ` as above:
> 1. **Ordered injectivity** holds and is analytic (no enumeration): `Φ` is injective into the
>    336 ordered primitive zero-divisor pairs (168 distinct images).
> 2. **The unordered map `Φ̄` is exactly 2-to-1**, with image **126**: it is injective on the
>    84 triples with `δ_b=−1` and 2-to-1 (collapsing each ι-orbit) on the 84 with `δ_b=+1`.
>    `126 = 84 + 42`, with **42 unordered collisions**, each an ι-orbit `{T, ιT}`.
> 3. **Incompatibility (bijectivity ⊥ GL(3,2)-equivariance).** A bijection `168 ↔ 168` unordered
>    ZD pairs exists, but only via a **non-equivariant** gauge (one orientation bit per ι-orbit);
>    **no GL(3,2)-equivariant bijection exists.**

**Proof of (3) — the obstruction.** Releasing `δ_a` to a gauge `α(T)∈{±1}` gives
`a=(e_p+α e_{q+8})`, `b=(e_r+α δ_b e_{s+8})`; this stays a zero divisor (preprint Theorem 1,
"for any `δ_a`") and stays in `𝒫₊` (`α` multiplies the *high* index; the leading coefficient on
`e_p` is `+1` for any `α`, and `𝒫₊` contains both high signs). The non-collision condition is
`α(ιT) = −δ_b(T)·α(T)`. `Φ_α` is GL(3,2)-equivariant **iff** `α` is GL(3,2)-invariant. But
GL(3,2) acts **transitively** on the 168 (single regular orbit; preprint line 92, verified), so
an invariant `α` is constant, whence non-collision forces `δ_b≡−1` — false (84 are `+1`).
Hence no equivariant gauge. A per-orbit (non-equivariant) `α` yields a genuine `168↔168`
bijection (verified: 0 collisions). ∎

## 5. The sign cocycle (the obstruction, in closed form)

**Headline (verified):** `δ_b(g·T)/δ_b(T) = ∏_{x∈Q(T)} ε_g(x)` in **28224/28224** cases, and this
quantity is trivial **exactly** on the order-21 Frobenius subgroup `F₂₁ = 7:3` of GL(3,2).

Each `g∈GL(3,2) ⊂ G₂=Aut(𝕆)` acts as `g·e_i = ε_g(i)·e_{g i}` with signs `ε_g:{1..7}→{±1}`
chosen so `g` preserves the octonion product; the automorphism condition is
`σ(gi,gj) = σ(i,j)·ε_g(i)ε_g(j)ε_g(i⊕j)`. Substituting into `δ_b` and using
`p⊕r = s⊕q` (since `(p⊕r)⊕(q⊕s)=(p⊕q)⊕(r⊕s)=0`) so that `ε_g(p⊕r)ε_g(s⊕q)=1`:

> **δ_b(g·T) = ω(g,T)·δ_b(T),  with  ω(g,T) = ∏_{x∈{p,q,r,s}} ε_g(x) = ⟨ε_g, Q(T)⟩.**

`ω` is well-defined independent of the coboundary freedom in `ε_g` (because `Q=p⊕q⊕r⊕s=0`),
and is a **genuine 2-cocycle** (T-dependent: 147 of 168 group elements give a non-constant
`ω(g,·)`), **not** a group character. It is trivial **exactly** on `F₂₁` — the sign-free octonion
automorphisms (the only `g` with `ε_g≡+1`). The non-triviality of `[ε_g]` off `F₂₁` is precisely
the obstruction of §4(3), and shows that the preprint's naive equivariance `g·e_p=e_{gp}`
(line 239) holds only on this index-8 subgroup.

## 6. Fano structure of the defect

The map `orbit ↦ {(p,q),(r,s)}` is a bijection on the 84 ι-orbits. Each orbit determines a
**pencil-pair**: two Fano lines `{p,q,v}`, `{r,s,v}` through the common point `v=p⊕q=r⊕s`
(equivalently a flag, via `(v, excluded line)`), plus a relative orientation. There are **21**
line-pairs (= flags), each carrying **4** orientations, and **every** line-pair splits exactly
**2/2** by `δ_b`. Thus the 42 collisions `↔` 21 flags × 2 anti-coherent orientations, and
`δ_b` is precisely the **orientation-coherence sign**. So `126 = 84 + 42` is Fano-indexed.
*(All counts verified.)* This Fano-orientation sign is **consistent with** the GL(3,2) cocycle of
§5 — both are realised by `δ_b` — but the identification of the two as the *same* class is not
proved here and is left as a conjecture for the v2 (it would say the orientation bit is the
restriction of `ω` to a transversal of `F₂₁`).

## 7. Recommended v2 edits

- Replace Theorem 2 by **Theorem 2′** (§4). Keep ordered injectivity (line 233) verbatim — it
  is correct and analytic.
- Correct line 237: the unordered descent is **not** injective; state the 2-to-1 structure and
  cite I1/I2/I4 (no enumeration).
- Correct line 239: the naive index action preserves the product only on `F₂₁`; the general
  automorphism carries the sign cocycle `ε_g`, and `δ_b` transforms by `ω(g,T)=∏_Q ε_g` (§5).
- Add §5–§6 as the deepening: the incompatibility theorem and the Fano-indexed sign cocycle are
  a stronger, fully-proved result than the original (false) equivariant bijection.
- Errata note *(suggested wording; author to refine — including how/whether to credit the arena)*:
  "v1 Theorem 2 claimed a GL(3,2)-equivariant unordered bijection; the unordered map is in fact
  2-to-1 (image 126), the obstruction being a GL(3,2) sign cocycle (closed form, §5). The
  discrepancy was found during machine verification and is checked in two independent codebases."

## 8. Reproduce

```
python3 scripts/research/generate_sedenion_phi_map_injectivity.py     # 27/27 checks, ALL PASS
bin/souc run tests/stdlib/nn/test_fractal_sedenion_e2e.sio            # 10 / 10 passed (T9,T10)
bash scripts/ci/sedenion_phi_injectivity_gate.sh                      # PASS (both codebases)
```
