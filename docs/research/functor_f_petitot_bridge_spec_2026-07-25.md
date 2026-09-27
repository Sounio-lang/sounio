<!-- docs:meta
topic_id: repo.docs.research.functor-f-petitot-bridge-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-petitot-bridge-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — the algebra → Petitot bridge: cusp-canonical, butterfly-obstructed

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `B_OBSTRUCTED` (characterised negative, not an identity)
**Parents:** `functor_f_g2_tower_closure_spec_2026-07-25.md` (`Q_GREEN`), `petitot-semantic-potential.md` (§3 divergence)
**Harness:** `scripts/research/functor_f_petitot_bridge_contract.py`

---

## 0. What this is, and the verdict type fixed in advance

This is the cross-column edge: does the algebraic column (associator / G₂ forms) reach
the morphodynamic column (Petitot's catastrophe stratification)? The programme forbids
the identity reading (D3: "Petitot bifurcation set ≡ ZD/associator locus"). So the
verdict type was **named before computing**: an *operational* bridge with a stated
divergence, in the `H/K` honest mould — never a green identity.

The honest outcome is a **characterised obstruction**: the bridge closes canonically at
the **cusp** (binary contrariety) and is **obstructed at the butterfly** (the mediating
"complex/neutral term"). The much-warned dimension coincidence is exactly what happens.

---

## 1. Method discipline (why this is falsifiable)

Per the advisor guard: **derive the controls, count the invariants, and state the count
BEFORE looking at any well-count.** A butterfly has 3 controls `(t,v,w)`; any surjection
onto a 3-parameter family hits the 3-well pocket somewhere, so "3 wells found" proves
nothing. The test is whether the pocket **boundary** is an algebraic locus, and whether
the algebra even supplies a *canonical* control of the right type.

---

## 2. Findings

| Clause | Result | Reading |
|---|---|---|
| `B0_CORE_AUDIT` | inherited octonion core passes its axioms | foundation verified before use |
| `B1_DIVERGENCE_SURVIVES` | isolated Fano square associator `= 0` → Booleanizable (cusp), not butterfly | the §3 divergence is preserved — a bridge that broke it would break the algebra |
| `B2_CUSP_CANONICAL` | 2 coupled lines → **exactly 2** canonical continuous G₂-invariants (depth + tilt) | the cusp (contrariety) closes canonically — this re-derives `R3_GREEN` |
| `B3_DIM_MATCHES` | 3 coupled lines → **exactly 3** independent continuous G₂-invariants | dimension matches the butterfly's `(t,v,w)` — **necessary only** |
| `B4_BUTTERFLY_FACTOR_VANISHES` | canonical G₂ 3-form `φ(α₁,α₂,α₃)` **vanishes over all 840** 3-line configs (worst `0.0`; *measured*, structural proof flagged open in §3); the 3 independent invariants are three same-type single-axis **depths** | **no canonical `x⁴` "butterfly factor"** — reaching the 3-well pocket needs a `t` chosen by hand (fabrication) |

The functional-rank chart (`B2`,`B3`) is the Jacobian of the invariant vector with
respect to the coupling strengths `δ` (the field configuration's only continuous DOF);
the rank is bounded by `#δ`, so it is `≤2` (2 lines) / `≤3` (3 lines) **regardless of
how many invariants exist** — the count does not rest on a completeness theorem for the
invariant ring.

Verdict: `FUNCTOR_F_PETITOT_VERDICT B_OBSTRUCTED`.

---

## 3. The obstruction, precisely

The cusp control pair is canonical, exactly as `Φ_fp` already showed: `a = A₀ + ‖α‖²/4`
(depth, even) and `b = τ + ⟨α, e_m⟩/2` (tilt, odd) are **read off** the jet. For the
butterfly one needs a *third, distinct* unfolding direction — the `x⁴` "butterfly
factor" `t` that opens the three-well pocket. The canonical **antisymmetric** cubic
G₂-invariant of three vectors — the 3-form `φ(α₁,α₂,α₃)` (note `⟨c, a×b⟩ = φ(a,b,c)`, so
the cross-product pairing is not an extra invariant) — is the natural source for such a
control, and it **vanishes on the coupling jets** (measured over 840 configs). The
remaining continuous invariants are **symmetric** depths (Gram norms), three of the
*same* type (one per coupling) — three copies of the cusp datum, not the butterfly's
three *distinct* controls. Hence no canonical control of the `x⁴` type is supplied; this
conclusion needs only the measured rank (`= 3`, domain-bounded) and `φ(jets) = 0`, not a
completeness theorem for the invariant ring.

So the `3 ↔ 3` count is a **coincidence of dimension**, precisely as warned. The
algebra canonically forces the cusp stratum and **does not** canonically force the
butterfly (mediating) stratum. This sharpens Petitot §3: the octonion model reproduces
contrariety but leaves the *complex/neutral term* algebraically un-forced — consistent
with "the isolated square is Booleanizable", against Petitot's need for extra topology.

*(Open sub-question, flagged: `φ(α₁,α₂,α₃) ≡ 0` looks like a structural identity for
associator jets, not a numerical accident over 840 samples. A proof — likely from the
jets lying in a coassociative arrangement — would upgrade `B4` from measured to
theorem. Not attempted here.)*

---

## 4. What this is NOT

- **Not** "Petitot is wrong" — the cusp half matches; only the butterfly (3-well) half
  is algebraically un-forced.
- **Not** an identity claim in either direction (D3 respected).
- **Not** a proof that no bridge exists — only that the *canonical* algebraic controls
  do not reach the butterfly; a different, non-algebraic construction is not excluded.
- **Not** D3, not clinical.

---

## 5. Place in the ladder

```
G_GREEN         uniformity across 7 lines
H_CHARACTERISED argmax-b obstruction, b_cov fix
E_GREEN         continuous-orbit equivariance
K_CHARACTERISED field functoriality up to ord-1 correction
P_GREEN         correction is the coherent G2 3-form
Q_GREEN         G2 form-tower closes; algebraic invariants terminate at ord-2
B_OBSTRUCTED    algebra->Petitot: cusp canonical, butterfly obstructed (cross-column)
```

The algebraic column is saturated (`Q`) and the first cross-column probe returns a
clean, located obstruction (`B`) rather than a forced match — the programme's preferred
kind of result.

---

## 6. Suggested next edges

1. **Prove `φ(α₁,α₂,α₃) ≡ 0`** (the coassociative-arrangement identity) — upgrades `B4`.
2. **ord-M instead of ord-P**: the Ollivier–Ricci side, now with the lesson that a
   canonical control-type match (not just a dimension count) is the bar.
3. **External write-up**: `G→Q` (closed algebraic story) + `B` (honest cross-column
   obstruction) is a complete, self-contained arc for the paper skeleton.

---

## 7. Reproduce

```bash
python3 scripts/research/functor_f_petitot_bridge_contract.py
# expect: B0..B4 PASS, FUNCTOR_F_PETITOT_VERDICT B_OBSTRUCTED
```

Pure Python (numpy); CD sign law self-contained; embeds the `B0` core axiom-audit.

---

## 8. AI disclosure

Probe, contract, and note produced under human direction (2026-07-25). The verdict type
(operational, never identity) and the count-before-well-count discipline were fixed by
an advisor review that predicted the dimension-coincidence failure mode; the invariant
count was computed and surfaced to the operator as a checkpoint before any butterfly
evaluation. `φ(jets) = 0` measured over 840 configs (exact); the wording was tightened
after a math-review flagged "identically" as an overreach (proof open) and flagged an
unnecessary completeness claim — the obstruction argument was re-grounded on the
domain-bounded rank and the measured vanishing alone. Claims bounded by the five named
clauses. Commit gated on the §10 math-review offload. No clinical content.
GAIDeT-ICMJE 2025.
