<!-- docs:meta
topic_id: repo.docs.research.functor-f-ord3-symmetry-fill-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-ord3-symmetry-fill-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — ord-3 fill, settled by symmetry: the secondary op is an invariant-free S₄-module

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `NO_INVARIANT_FILL` (settles `NO_CANONICAL_FILL` at every level)
**Parents:** `functor_f_ord3_quotient_fill_spec_2026-07-25.md` (`NO_CANONICAL_FILL`), `functor_f_fano_psl27_thread_spec_2026-07-25.md` (`PSL27_THREADS_THE_TOWER`)
**Harness:** `scripts/research/functor_f_ord3_symmetry_fill_contract.py`

---

## 0. The question

`NO_CANONICAL_FILL` showed the *bare* sedenion algebra cannot canonically fill the 2-dim
ord-3 quotient; the `PSL(2,7)` thread then supplied a real symmetry. Does that symmetry
select the canonical secondary value the bracketing could not?

---

> **Group-identification correction (2026-07-26).** A later computation (while hunting for genuinely-uncomputed structure) found the acting symmetry is **not** the abstract `S₄` (order 24) as first stated: the 24 line-fixing collineation-representatives are a transversal that **generates a group of order 192** = `(ℤ₂)³ ⋊ S₄`, itself inside the full signed-automorphism group of 𝕆 of order **1344 = 8·168 = (ℤ₂)³ : PSL(2,7)** (sign-kernel `(ℤ₂)³`, verified). The invariant-free **result is unaffected** — the invariant test uses the generating-set common-kernel, which computes the *generated group's* invariants regardless of the exact order. Only the group label was wrong; it is corrected throughout.

---

## 1. Result

| Clause | Result | Reading |
|---|---|---|
| `Z1_SINGLE_FIBRE_STAB_TRIVIAL` | the stabiliser of a single ZD `b` (its fibre **as a subspace**) inside the lifted `PSL(2,7)` is **trivial** (order 1) | there is no group to average a single quotient `Q` over — the naive symmetrisation is empty. |
| `Z2_CLASS_S4` | the acting group is the **signed-octonion-automorphism line-stabiliser, order 192** = `(ℤ₂)³ ⋊ S₄` (CORRECTION 2026-07-26: not the abstract `S₄` of order 24 — the 24 collineation-reps are a transversal generating the order-192 group; it sits in the full signed-auto group of 𝕆 of order `1344 = 8·168 = (ℤ₂)³:PSL(2,7)`); it acts on the 8-dim support-class and permutes its **6 fibres** (its collineation quotient is `S₄`) | the symmetry lives at the class level, not the single fibre. |
| `Z3_NO_S4_INVARIANT_IN_SPAN` | `dim(S₄-invariants ∩ class-level secondary span) = 0` for **all 7 classes** | the secondary content carries **no** `S₄`-invariant vector. The two ambient `S₄`-invariants (`e₀`, `e₈=ℓ`) lie in the *complement* of the secondary span, not in it. |
| `Z4_NO_GENUINE_FILL` | the ord-3 secondary operation is an **invariant-free `S₄`-module** | no canonical value can exist at bare-algebra, single-fibre, *or* class-symmetry level. |

Verdict: `FUNCTOR_F_ORD3SYM_VERDICT NO_INVARIANT_FILL`.

---

## 2. The honest resolution — and the two bugs caught on the way

`NO_CANONICAL_FILL` is now **explained**, not merely observed: the ord-3 secondary
operation is a canonical module (for the order-192 group) with **no invariant line**, so there is nothing for a
symmetry-canonical *value* to be. The canonical object is the **module** (canonical as a
representation), not a scalar. This closes the fill question the parent rung left open:
imposing structure could still define one, but the algebra + its full `PSL(2,7)`/`S₄`
symmetry do **not**.

> **Two computational near-misses recorded (not suppressed).** An early character-based
> count gave a non-integer trivial multiplicity (a broken restriction of the group action
> to an SVD basis), and a throwaway intersection script used the SVD *null-space* instead
> of the *row-space* — which made `e₀, e₈` look like invariants *inside* the secondary span
> and briefly suggested a positive fill. Scrutiny fixed both: the correct
> `dim(K ∩ row-space) = 0`, and `e₀, e₈` are in the complement. The clean negative is the
> real result; the flip-flops are logged here because a positive on this exact question is
> the failure mode this whole line was built to avoid.

---

## 3. What this is NOT

- **Not** a claim that *no* imposed `A∞`/differential structure could ever fill the
  quotient — only that the bare algebra and its `PSL(2,7)`/`S₄` symmetry do not.
- **Not** an obstruction to the `PSL27_THREADS_THE_TOWER` result — that thread is about
  *indexing/symmetry* of the fibres; this is about a *canonical value* in the secondary
  quotient, a different object.
- **Not** D3, not an identity, not clinical.

---

## 4. Place in the ladder

```
NO_CANONICAL_FILL   bare algebra: 2-dim quotient reachable, bracketing-swept, no value
PSL27_THREADS_THE_TOWER   the fibres carry a real PSL(2,7)/S4 symmetry
NO_INVARIANT_FILL   that symmetry has no invariant in the secondary span either =>
                    the secondary op is an invariant-free S4-module; the fill question is
                    settled negatively at every level the algebra+symmetry provide.
```

---

## 5. Reproduce

```bash
python3 scripts/research/functor_f_ord3_symmetry_fill_contract.py
# expect: Z0..Z4 PASS, FUNCTOR_F_ORD3SYM_VERDICT NO_INVARIANT_FILL
```

Pure Python (numpy); builds the full 168-element `PSL(2,7)`; embeds a core axiom-audit.

---

## 6. AI disclosure

Probe, contract, and note produced under human direction (2026-07-25) after the operator
pushed to keep looking. The result is a clean negative reached after two computational
near-misses (a broken character restriction; an SVD null-space/row-space confusion), both
caught by scrutiny and recorded in §2 rather than shipped as a positive. Claims bounded by
the five named clauses. Commit gated on the §10 math-review offload. No clinical content.
GAIDeT-ICMJE 2025.
