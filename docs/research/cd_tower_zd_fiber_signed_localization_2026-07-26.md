<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-signed-localization-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-signed-localization-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — the geometry localizes to the signed resonance graph (an ∀n narrowing)

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `ZD_FIBER_GEOMETRY_LOCALIZES_TO_SIGNED_GRAPH`
**Parent:** `cd_tower_zd_fiber_spectral_classifier_2026-07-26.md` (spectrum = complete geometry invariant, `n≤8`)
**Harness:** `scripts/research/cd_tower_zd_fiber_signed_localization_contract.py`

---

## 0. The result

The prior rung showed the adjacency spectrum classifies the fiber geometries (`n≤8`) but left the
**∀n** question open. This rung **narrows** it — halving the object and localizing all the geometry
to a signed cocycle where the ∀n machinery already exists:

> Each fiber's annihilation graph is the **Z₂ signed double cover** of a "lo-graph" on the
> `2^{n-1}-1` lo-labels (closed-form adjacency, verified `n≤8`). Hence, from the block form
> `[[A₊,A₋],[A₋,A₊]]`:
>
> **`spec(G_n(L)) = spec(A_R) ∪ spec(A_σ)`** — the fiber spectrum splits into the **unsigned**
> resonance graph `A_R` and the **signed** resonance graph `A_σ` (`ε`-weighted), each on **half** the
> vertices. (`L1`; verified directly via the algebra product for **all** fibers `n=6,7`; a structural
> identity ∀n given the closed-form rule.)
>
> **The geometry lives in the sign.** `A_σ` **alone** is a complete invariant — `#distinct A_σ
> spectra = 3·2^{n-5}`, the full classification, for `n=6,7,8` — while the unsigned `A_R` is
> strictly **coarser** (`4, 8, 16`). (`L2`.)

So the open **∀n** classification reduces to the **signed resonance graph `A_σ`** (half the
vertices), whose signs `ε = −P₁` are a product of `cd_sigma` values — and that cocycle's ∀n law (the
**seam-flip law**) is already **proven ∀n in Lean** (`SounioSeamFlip.lean`). The natural ∀n attack
surface is now explicit.

---

## 1. Results

| Clause | Result | Reading |
|---|---|---|
| `L1_DOUBLE_COVER` | `spec(G_n(L)) = spec(A_R) ∪ spec(A_σ)`, **all** fibers `n=6,7` | halves the vertex count; ∀n via the `[[A₊,A₋],[A₋,A₊]]` block form (standard: `spec(B+C)∪spec(B−C)`). |
| `L2_GEOMETRY_IN_SIGN` | `#A_σ spectra = 6,12,24 = 3·2^{n-5}` (complete); `#A_R spectra = 4,8,16` (coarser), `n=6,7,8` | the signed half `A_σ` carries the entire classification; the unsigned half does not. |
| `L3_LOCALIZATION` | ⟹ ∀n problem reduces to `A_σ` (signed cocycle, seam-flip proven ∀n) | a genuine narrowing — **not** a proof of ∀n. |

Verdict: `CD_TOWER_ZDLOC_VERDICT ZD_FIBER_GEOMETRY_LOCALIZES_TO_SIGNED_GRAPH`. §10 Grok `[OK]` (block-
form spectrum "textbook"; counts accepted; "genuine localization, ∀n disclaimer correctly maintained").

---

## 2. Honest bounds (what this is NOT)

- **Not** a proof of ∀n completeness/injectivity — a **narrowing**. `A_σ` still grows (`2^{n-1}-1`
  vertices, half of `G_n`); a spectral doubling recursion for `A_σ` remains the ∀n target.
- **Not** "the unsigned half is universal": `A_R` **varies** by class (some fibers are cocktail-party
  `K_{(2^{n-2}-1)×2}`, others are not). The precise statement is `A_σ` complete, `A_R` strictly coarser.
- The double-cover spectral identity itself is **textbook** spectral graph theory; novelty is only its
  **application** to localize this CD-fiber classification to the sign cocycle.
- **Not** symbolic beyond a numerical eigenvalue computation; **not** `D3`; **not** clinical.

---

## 3. Reproduce

```bash
python3 scripts/research/cd_tower_zd_fiber_signed_localization_contract.py
# expect: L1 (n=6,7) OK, L2 (n=6,7,8) OK, VERDICT ZD_FIBER_GEOMETRY_LOCALIZES_TO_SIGNED_GRAPH
```

---

## 4. AI disclosure

Probe, contract, and note produced under human direction (2026-07-26), pushing the ∀n frontier of the
`PSL(2,7)` orbit-theorem thread at the user's request ("attack the ∀n recursion"). **New:** the
double-cover reduction `spec(G_n)=spec(A_R)∪spec(A_σ)` (all fibers `n=6,7`) and the localization of the
full geometry to the **signed** resonance graph `A_σ` (complete alone; `A_R` coarser) — narrowing the
open ∀n problem to a signed-cocycle spectral question on half the vertices, where the seam-flip law is
Lean-proven ∀n. §10 Grok `[OK]` on all claims; the block-form spectral identity is textbook (cited),
novelty scoped to the CD-fiber application. Numerical certificate; ∀n OPEN. No semantic claim, no
clinical content. GAIDeT-ICMJE 2025.
