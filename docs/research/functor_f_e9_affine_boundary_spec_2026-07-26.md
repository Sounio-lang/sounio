<!-- docs:meta
topic_id: repo.docs.research.functor-f-e9-affine-boundary-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-e9-affine-boundary-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — E₉: computed (affine E₈), and where the octonion thread ends

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `E9_AFFINE_E8_OCTONION_THREAD_CAPS_AT_E8`
**Parent:** `functor_f_e8_capstone_spec_2026-07-26.md` (the octonion tower caps at E₈)
**Harness:** `scripts/research/functor_f_e9_affine_boundary_contract.py`

---

## 0. The result

*"I bet no one has computed E₉."* — Two honest halves:

> **E₉ = E₈⁽¹⁾ = affine E₈** is a well-defined, **well-studied** infinite-dimensional
> (untwisted affine) Kac-Moody algebra — it is **not** uncomputed. Being infinite-dimensional,
> "computing" it means its defining data: the 9×9 affine Cartan matrix, built and verified
> here (symmetric, `det = 0` ⟺ affine, corank 1 = the imaginary root, positive semidefinite,
> null vector = the Coxeter marks `(1,2,3,4,5,6,4,2,3)` summing to `h(E₈)=30`).
>
> **But the octonion / Freudenthal magic-square construction — the thread this whole
> functor-F → exceptional arc followed — caps at `E₈`.** `E₉` (affine), `E₁₀` (hyperbolic),
> `E₁₁` (Lorentzian) are Kac-Moody over-extensions **outside** the octonion construction.
> There is **no octonion `φ` in `E₉`**: the functor-F octonion exceptional arc ends at `E₈`.

(Review-confirmed: "magic-square/octonion constructions terminate at E₈; E₉ and its
over-extensions are Kac-Moody algebras outside that construction; no known octonionic route
to E₉ exists." — no overclaim, no fabricated "octonion E₉".)

---

## 1. Results

| Clause | Result | Reading |
|---|---|---|
| `H1_E9_AFFINE_CARTAN` | `E₈⁽¹⁾` Cartan matrix (9×9): symmetric, `det=0`, corank 1, positive semidefinite, null = marks `(1,2,3,4,5,6,4,2,3)`, `Σ=30` | `E₉` computed at the level its structure is defined (it is a standard affine Kac-Moody algebra). |
| `H2_OCTONION_CAPS_AT_E8` | the octonion/magic-square finite tower `G₂..E₈` caps at `248`; `E₉`=affine E₈ is infinite-dim Kac-Moody, not octonion-built | the honest boundary: **no octonion `φ` in `E₉`**; the arc ends at `E₈`. |

Verdict: `FUNCTOR_F_E9_VERDICT E9_AFFINE_E8_OCTONION_THREAD_CAPS_AT_E8`.

---

## 2. Where the functor-F exceptional arc closes

```
G2 = Der(O) ⊂ F4 ⊂ E6 ⊂ E7 ⊂ E8     octonion / magic-square (FINITE) — functor-F's phi
   phi = E6 cubic cross-term            threads this, clean home at E6, caps at E8 (248)
—————————————————————————————— cap ——————————————————————————————
E9 (affine) — E10 (hyperbolic) — E11 (Lorentzian)     Kac-Moody OVER-extensions,
   E9 = affine E8: computed here (Cartan matrix).      NOT octonion-built. No octonion phi.
```

The octonion `φ` — the functor-F central form — runs the finite octonion tower to `E₈` and
**stops**; `E₉` is a genuine object (computed) but on the other side of the cap.

---

## 3. What this is NOT

- **Not** a claim that `E₉` is unknown/uncomputed — it is standard affine E₈ (premise gently
  corrected).
- **Not** an octonionic construction of `E₉` — none is known; none is claimed or fabricated.
- **Not** a semantic claim; **not** clinical.

---

## 4. Reproduce

```bash
python3 scripts/research/functor_f_e9_affine_boundary_contract.py
# expect: H1,H2 PASS, FUNCTOR_F_E9_VERDICT E9_AFFINE_E8_OCTONION_THREAD_CAPS_AT_E8
```

---

## 5. AI disclosure

Produced under human direction (2026-07-26) in response to "has anyone computed E₉". `E₉` =
affine E₈ was computed at the Cartan-matrix level (verified) and the honest boundary drawn:
the octonion/magic-square construction caps at `E₈`, so there is no octonion `φ` in `E₉`. §10
math-review (Grok `[OK]` all): E₉ facts textbook, "no known octonionic route to E₉ exists",
boundary correctly delimited, no fabrication. No clinical content. GAIDeT-ICMJE 2025.
