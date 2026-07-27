<!-- docs:meta
topic_id: repo.docs.research.functor-f-g2-tower-closure-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-g2-tower-closure-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — G₂ form-tower closure: the rupture invariants terminate at ord-2

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `Q_GREEN` (5/5)
**Parent:** `functor_f_g2_coherence_spec_2026-07-25.md` (`P_GREEN`)
**Harness:** `scripts/research/functor_f_g2_tower_closure_contract.py`

---

## 0. The question

The coherence rung proved `φ·φ = δδ − δδ − ψ`: contracting the ord-1 3-form produces
the ord-2 4-form `ψ`. Natural next question (coherence §7.2): does contracting `ψ`
generate an **ord-3** invariant, or does the graded tower of rupture forms
`{δ, φ (ord-1), ψ (ord-2)}` **close**?

---

## 1. The closed tower (measured, exact integer coefficients)

On `Im(𝕆)` (indices `1..7`), every pairwise contraction of `{φ, ψ}` returns a member
of `{δ, φ, ψ}`:

| Identity | Value | Rung tie |
|---|---|---|
| `φ_{abe} φ_{cde}` | `δ_{ac}δ_{bd} − δ_{ad}δ_{bc} − ψ_{abcd}` | ord-1·ord-1 → ord-2 (`P_GREEN`) |
| `ψ_{aefg} ψ_{befg}` | `24·δ_{ab}` | ord-2·ord-2 (3-index) → scalar/`δ` |
| `ψ_{abef} ψ_{cdef}` | `4(δ_{ac}δ_{bd} − δ_{ad}δ_{bc}) − 2·ψ_{abcd}` | ord-2·ord-2 (2-index) → `δ` + ord-2 |
| `φ_{aef} ψ_{bcef}` | `−4·φ_{abc}` | ord-2·ord-1 → **ord-1** |

All hold with worst deviation `0.0`. `24·7 = 168 = ‖ψ‖²` recovers the standard G₂
4-form norm — an independent consistency check on the octonion core.

**No ord-3 object is generated.** The tower closes on `{δ, φ, ψ}`; contraction never
escapes the span. The graded rupture invariants of the programme **terminate at
ord-2**.

---

## 2. What is and isn't the claim

> **Framing (not an overclaim).** Identities (1)–(4) are the **standard G₂ contraction
> identities** (Bryant / Karigiannis normalisation). This rung does not discover them;
> it verifies them exactly on *this repo's* octonion core and draws the consequence for
> the rupture programme: the ord-1 associator 3-form and the ord-2 co-associator
> 4-form it generates are **exactly** the closed G₂ contraction algebra, so the
> "order-of-singularity" tower of §3 of the synthesis does not produce an unbounded
> ladder of new algebraic invariants — the algebraic column saturates at ord-2 = G₂'s
> representation data.

---

## 3. Consequence for the programme

`rupture-programme-synthesis §3` tabulates orders of singularity (ord-1 associator,
ord-2 `det L_x` / annihilation, ...) and warns "do not collapse sensors". This rung
sharpens the **algebraic** column specifically: within it, ord-1 (`φ`) and ord-2
(`ψ`) are not two independent sensors that could spawn a third — they are the two
G₂-invariant forms, closed under the algebra's own contraction. Any higher algebraic
rupture invariant is a polynomial in `{δ, φ, ψ}`, not a new tensor. (This says nothing
about the *non-algebraic* orders — ord-M curvature, ord-P bifurcation — which remain
separate by construction.)

---

## 4. Contract clauses

| Clause | Statement | PASS = |
|---|---|---|
| `Q0_CORE_AUDIT` | inherited octonion core passes its axioms | foundation verified before use |
| `Q1_PSI_NORM` | `ψ_{aefg} ψ_{befg} = 24 δ_{ab}` | 4-form norm / trace closes |
| `Q2_PSI_SELF` | `ψ_{abef} ψ_{cdef} = 4(δδ) − 2ψ` | ord-2 self-contraction closes on `δ`+ord-2 |
| `Q3_PHI_PSI_MIXED` | `φ_{aef} ψ_{bcef} = −4 φ` | ord-2·ord-1 closes back on ord-1 |
| `Q4_TOWER_CLOSES` | with `φ·φ = δδ − ψ`, all contractions stay in `{δ,φ,ψ}` | no ord-3 invariant generated |

Verdict: `FUNCTOR_F_TOWER_VERDICT Q_GREEN`.

---

## 5. What this is NOT

- **Not a new theorem in G₂ geometry** — the identities are standard; the deliverable
  is their exact verification on the repo core plus the programme consequence.
- **Not a closure claim about non-algebraic orders** (M, P) — only the algebraic column.
- **Not D3, not clinical.**

---

## 6. The Functor F ladder (complete algebraic column)

```
G_GREEN         uniformity across 7 lines
H_CHARACTERISED argmax-b obstruction, b_cov fix
E_GREEN         continuous-orbit equivariance, pole-flip
K_CHARACTERISED field functoriality up to ord-1 correction
P_GREEN         correction is the coherent G2 3-form; phi.phi = dd - psi
Q_GREEN         tower {delta,phi,psi} closes; rupture invariants terminate at ord-2
```

---

## 7. Suggested next edges

1. **Cross-column, not more algebra.** The algebraic column is saturated; the open
   frontier is ord-1/2 ↔ ord-M (Ollivier–Ricci) or ord-P (bifurcation), which need a
   functor, not a contraction — the synthesis marks these `⇏ without a functor`.
2. **`ψ` as a field-level Φ_fp dial** (carried over from coherence §7.3): does the
   invariant 4-form select a well-defined cross-field path class?
3. **External write-up.** The five-rung algebraic-column result (`G→Q`) is now a
   self-contained, gated story suitable for the paper skeleton.

---

## 8. Reproduce

```bash
python3 scripts/research/functor_f_g2_tower_closure_contract.py
# expect: Q0..Q4 PASS, FUNCTOR_F_TOWER_VERDICT Q_GREEN
```

Pure Python (numpy); CD sign law self-contained; embeds the `Q0` core axiom-audit.

---

## 9. AI disclosure

Probe, contract, and note produced under human direction (2026-07-25). Closed forms
were measured empirically (integer coefficients `24, 4, −2, −4`) then verified exactly;
the standard-identity status is stated to avoid a novelty overclaim. The inherited
octonion core was independently axiom-audited per standing instruction to verify other
agents' work. Claims bounded by the five named clauses. Commit gated on the §10
math-review offload (`bin/llm-offload -t math-review -p xai`). No clinical content.
GAIDeT-ICMJE 2025.
