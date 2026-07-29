<!-- docs:meta
topic_id: repo.docs.research.functor-f-ord3-module-decomp-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-ord3-module-decomp-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — what the ord-3 secondary module actually is (honest: it fills the class coordinate space; `2·V₃` is CD-doubling)

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `ORD3_IMAGES_FILL_CLASS_COORD_SPACE`
**Parent:** `functor_f_ord3_symmetry_fill_spec_2026-07-25.md` (`NO_INVARIANT_FILL`)
**Harness:** `scripts/research/functor_f_ord3_module_decomp_contract.py`

---

## 0. The result — and an overclaim, corrected

An earlier draft of this rung announced `M = 2·V₃` as *"the exact representation-theoretic
fingerprint of the ord-3 secondary operation"* — a genuinely-uncomputed discovery. **On scrutiny
(advisor review + full module anatomy) that framing was an overclaim, and is corrected here.**
It is the same label-drift failure mode already self-caught twice in this thread (associator-vs-`φ`
at `E₆`; `S₄`-vs-order-192 group id): the computation was right, the *label on the object* drifted.

The honest statement:

> Fix a Fano-line support-class `L` (6 zero-divisors `b = e_i+e_j`, `i,j` in `{L, L+8}`). The
> ord-3 secondary images `{(x·y)·b : x,y ∈ F(b)=ker L_b}` are **non-degenerate**: each single `b`
> already spans a **4-dimensional** image, and the six images together fill **exactly the 6-dim
> coordinate space of the class's indices**, `M = span{e_i, e_{i+8} : i ∈ L}` (containing all six
> `b`, and reaching the 6th dimension beyond the 5-dim span of the `b` themselves).
>
> As a module for the order-192 group `G = 2³:S₄` that 6-space **is** `2·V₃` — but the `2` is
> merely the **Cayley-Dickson lower/upper doubling** (the lift is `diag(g,g)`, so `G` acts
> *identically* on `span{e₁,e₂,e₃}` and `span{e₉,e₁₀,e₁₁}`), and `V₃` is the octonion-automorphism
> action on a Fano line's 3 coordinates (absolutely irreducible). So **`2·V₃` fingerprints the class
> *coordinate* structure — doubling × Fano-line action — not the ternary operation's content.**

The only genuinely **operation-dependent** fact is the **non-degeneracy** (clause `M2`): the
operation's images *fill* this coordinate space. `2·V₃` itself is structural bookkeeping forced by
the coordinate support and the CD doubling — modest, not a discovery.

---

## 1. Results

| Clause | Result | Reading |
|---|---|---|
| `M1_MODULE` | `|G|=192`; `dim M = 6`; `G`-stable (dev `1.8e-15`) | `M` is a genuine 6-dim `G`-module. |
| `M2_NONDEGENERACY` | per-`b` image dims `[4,4,4,4,4,4]`; the six fill **exactly** `span{e_i,e_{i+8}:i∈L}`; contains all 6 ZD `b` | **the one operation-dependent fact:** the ord-3 images are non-degenerate — they surject onto the class's full 6-dim coordinate space. |
| `M3_2xV3` | `⟨χ,χ⟩=4`, `⟨χ,1⟩=0`, `dim End_G(M)=4` (computed in-harness, `=M₂(ℝ)`), non-abelian, `{3,3}` split over **4 seeds** | `M` is `2·V₃`, `V₃` absolutely irreducible at multiplicity 2 (the `{3,3}` split rules out the quaternionic `{6}` alternative). |
| `M4_DEFLATION_CD_DOUBLING` | `V₃ = G|span{e₁,e₂,e₃}` abs. irreducible; upper half `span{e₉,e₁₀,e₁₁}` **identical** (dev `0.0`) | the `2` is the **Cayley-Dickson doubling**; `2·V₃` is the class **coordinate** structure, not a fine operation-invariant. |

Verdict: `FUNCTOR_F_ORD3MOD_VERDICT ORD3_IMAGES_FILL_CLASS_COORD_SPACE`.

---

## 2. Why the `2·V₃` deflates (the anatomy that corrected the headline)

- `support(M) = {1,2,3,9,10,11}` **exactly**, and `dim M = 6` ⟹ `M` is *precisely* the coordinate
  subspace `span{e₁,e₂,e₃,e₉,e₁₀,e₁₁}` — the class's lower Fano line `L={1,2,3}` and its upper copy
  `L+8`. Nothing finer.
- The sedenion lift is `diag(g,g)`: `G` acts the **same** on the lower and upper triples. Hence
  `M = V₃ ⊕ V₃` with `V₃ = G|span{e₁,e₂,e₃}` — the multiplicity `2` is the CD doubling, full stop.
- `V₃` is the (standard) octonion-automorphism action on a Fano line's 3 points, absolutely
  irreducible (`⟨χ_{V₃},χ_{V₃}⟩ = 1`).
- So `2·V₃` follows from `support + doubling` alone; it carries **no** information about the ternary
  operation beyond `M2` (that the images reach all six coordinates).

The `2·V₃` decomposition inference itself is sound (advisor-confirmed, independently re-enumerated):
`dim End_G(M)=4` non-abelian `⟹ M₂(ℝ)` or `ℍ`; the `{3,3}` split (over multiple seeds) rules out
`ℍ` (which would give `{6}`), forcing `V₃` absolutely irreducible at multiplicity 2. What was wrong
was only the *label* — calling a coordinate-space fact a fingerprint of the operation.

---

## 3. What this is / is NOT

- **Is:** an honest anatomy — the ord-3 images are non-degenerate (fill the class coordinate
  6-space), which as a `G`-module is `2·V₃ =` CD-doubling of the Fano-line octonion action.
- **Not** a genuinely-new invariant of the ternary operation — the earlier "fingerprint of the
  ord-3 operation" headline was an overclaim, retracted here.
- **Not** a claim about the group/PSL(2,7)/its irreps being new (all standard); **not** symbolic
  (numerical certificate, machine precision); **not** the Petitot conjecture (`D3`-quarantined);
  **not** clinical.

---

## 4. Reproduce

```bash
python3 scripts/research/functor_f_ord3_module_decomp_contract.py
# expect: M1..M4 PASS, FUNCTOR_F_ORD3MOD_VERDICT ORD3_IMAGES_FILL_CLASS_COORD_SPACE
```

Builds the order-192 group, forms the ord-3 module `M`, and verifies: `M` = the class coordinate
6-space (per-`b` dim 4, non-degenerate fill); `dim End_G(M)=4` in-harness; the `{3,3}` split over
4 seeds; and the CD-doubling deflation (`upper ≡ lower`, `V₃` absolutely irreducible).

---

## 5. AI disclosure

Probe, contract, and note produced under human direction (2026-07-26), continuing the "find
something genuinely uncomputed" push. **An initial framing (`2·V₃` = the fingerprint of the ord-3
operation, committed `62fd3ebc1`) was an overclaim; advisor review + full anatomy showed `M` is just
the class's 6-dim coordinate space and the `2` is Cayley-Dickson doubling — corrected in this rung
(verdict changed to `ORD3_IMAGES_FILL_CLASS_COORD_SPACE`).** The sound, modest content is: the ord-3
images are non-degenerate (fill the class coordinate space), and that space is `2·V₃` = doubling ×
Fano-line octonion action. Certificate numerical (machine precision), not symbolic; the two harness
gaps the review flagged (in-harness `dim End`, multi-seed genericity) are now closed. §10 math-review
(Grok `[OK]`) had validated the `2·V₃` *inference*; the deflation is a framing correction, not a math
error, so no re-offload. No new group, no semantic claim, no clinical content. GAIDeT-ICMJE 2025.
