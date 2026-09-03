<!-- docs:meta
topic_id: repo.docs.research.functor-f-field-functoriality-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-field-functoriality-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — field functoriality: additive up to a G₂-covariant ord-1 correction

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `K_CHARACTERISED` (4/4)
**Parents:** `functor_f_phi_fp_equivariant_spec_2026-07-25.md` (`E_GREEN`), `rupture-r4-fano-field_2026-07-25.md` (`R4_GREEN`)
**Harness:** `scripts/research/functor_f_field_functoriality_contract.py`

---

## 0. The question

The single-line rungs settled how F behaves on *objects* (configurations) under
`G₂`. A functor also acts on *morphisms*. In the R4 field of seven squares the
morphisms are the **cross-line couplings**: perturbing one factor of the trilinear
associator `[x,y,z]` by an off-line direction. Does F respect the **composition** of
two couplings — is it additive — or is there an obstruction?

This is the one composition question whose answer is not fixed in advance (composing
the induced `(a,b)`-maps is vacuously true; additivity of the *jet* is not), and it
is on-theme: the programme is about non-associativity as rupture, and here
non-associativity is exactly what can break the functor.

---

## 1. Set-up

Two couplings on **different slots** of the associator:

```
α(δ₁, δ₂) = [ x + δ₁·e_{u₁} , y + δ₂·e_{u₂} , z ]
          = [x,y,z] + δ₁[e_{u₁},y,z] + δ₂[x,e_{u₂},z] + δ₁δ₂·[e_{u₁},e_{u₂},z].
```

Additivity of the deformation — `F(both) − F(base) = (F(A) − F(base)) + (F(B) −
F(base))` — holds **iff the cross term `δ₁δ₂·[e_{u₁},e_{u₂},z]` vanishes.** (Two
couplings on the *same* slot are additive trivially, because the associator is linear
in each slot; that case carries no content and is excluded.)

Sweep: all 7 Fano base lines `(i,j,k)`, all `C(4,2)=6` unordered pairs of off-line
units → **42 configurations** (`δ₁,δ₂ = 0.7,0.3`, fixed).

---

## 2. Result (measured, all 42 configurations)

| Clause | Measured | Reading |
|---|---|---|
| `K1_RESIDUAL_IS_CROSS_ASSOCIATOR` | residual `= δ₁δ₂[e_{u₁},e_{u₂},z]` **exactly**, worst `‖res−cross‖ = 0.0` | the obstruction *is* the cross associator |
| `K2_ADDITIVE_IFF_ASSOCIATIVE` | `14` associative couplings → residual `0`; `28` cross-line → residual `≠0`; `0` violations | F is a strict functor exactly on associative couplings |
| `K3_CORRECTION_IS_ORD1` | all `28` corrections have `‖[e_{u₁},e_{u₂},z]‖ = 2` | the correction is an **ord-1** object (non-Fano associator magnitude) |
| `K4_CORRECTION_G₂_COVARIANT` | over `28×200` `g∈G₂`: `‖R‖` dev `6.7e-16`, pairing-`b` dev `1.3e-15`, argmax-`b` breaks `1.12` | the correction is **natural** (`b_cov`-covariant) |

**Verdict `K_CHARACTERISED`:** *F is a functor on the R4 field up to a G₂-covariant
ord-1 correction; it is strictly additive exactly when the coupling stays inside an
associative (Fano) subalgebra, and the failure otherwise is precisely the associator
of the two off-line directions — an ord-1 object of fixed magnitude 2, itself
`G₂`-covariant.*

---

## 3. Why this matters

`rupture-programme-synthesis §5` draws an architecture diagram in which the ord-1
associator column and the field-level R4 box are connected, but the connection is
never computed — it is asserted by adjacency. This rung **computes that edge**: the
obstruction to field-level functoriality of F is not a new object, it is the ord-1
associator re-appearing one level up, and it is covariant, so it is a *natural
correction*, not noise. The 14/28 split is the field made precise: F composes cleanly
within a quaternionic (associative) sub-field and picks up exactly one graded,
covariant defect when a coupling crosses between sub-fields.

This upgrades open-edge #1 of the synthesis ("Functor F — a formal functor, not only
path classes") from "path classes only" to "a functor on configurations and on field
morphisms, with a computed, covariant, ord-1 coherence defect."

---

## 4. Contract clauses

| Clause | Statement | PASS = |
|---|---|---|
| `K1_RESIDUAL_IS_CROSS_ASSOCIATOR` | additivity residual `= δ₁δ₂[e_{u₁},e_{u₂},z]` to `1e-12`, all configs | obstruction identified exactly |
| `K2_ADDITIVE_IFF_ASSOCIATIVE` | residual `=0 ⟺ same_line(u₁,u₂,z)`; no violation | clean characterisation |
| `K3_CORRECTION_IS_ORD1` | every nonzero correction has `‖assoc‖ = 2` | correction lives in ord-1 |
| `K4_CORRECTION_G₂_COVARIANT` | `‖R‖` & pairing-`b` invariant, argmax-`b` breaks, over `28×200` `g` | correction is natural |

---

## 5. What this is NOT

- **Not a claim that F fails.** F *is* a functor; the correction is a controlled,
  covariant ord-1 term, and it is exactly zero on associative couplings.
- **Not a construction of the full 2-categorical coherence data.** We compute the
  first coherence defect, not a full associator-of-associators tower.
- **Not D3, not clinical.**

---

## 6. Suggested next edges

1. **Coherence / pentagon.** Is the ord-1 correction a *coherent* associator — does it
   satisfy a pentagon-type identity across three couplings? (ord-1 → 2-cocycle.)
2. **Order dependence.** `δ₁,δ₂` is fixed here; test whether swapping coupling order
   changes only the sign of the correction (expected: `[u₁,u₂,z] = −[u₂,u₁,z]`).
3. **Lift the correction into Φ_fp** at the field level: does the covariant `b_cov` of
   the residual select a well-defined path class for the *cross*-field configuration?

---

## 7. Reproduce

```bash
python3 scripts/research/functor_f_field_functoriality_contract.py
# expect: K1..K4 PASS, FUNCTOR_F_FIELD_VERDICT K_CHARACTERISED
```

Pure Python (numpy); CD sign law self-contained and audit-visible.

---

## 8. AI disclosure

Probe, contract, and note produced under human direction (2026-07-25). The composition
question was posed and the residual-vs-cross-associator framing was set with an advisor
review that rejected the two vacuous readings (same-slot additivity; composing induced
maps) and fixed the discriminating computation. Claims bounded by the four named
clauses and the measured table in §2. Commit gated on the §10 math-review offload
(`bin/llm-offload -t math-review -p xai`). No clinical content. GAIDeT-ICMJE 2025.
