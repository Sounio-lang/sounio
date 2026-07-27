<!-- docs:meta
topic_id: repo.docs.research.functor-f-g2-coherence-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-g2-coherence-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — coherence of the ord-1 correction: the G₂ contraction identity

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `P_GREEN` (5/5)
**Parent:** `functor_f_field_functoriality_spec_2026-07-25.md` (`K_CHARACTERISED`)
**Harness:** `scripts/research/functor_f_g2_coherence_contract.py`

---

## 0. The question

Field functoriality (`K_CHARACTERISED`) showed the coupling-composition defect is the
cross associator `[e_{u₁},e_{u₂},e_z]`, an ord-1 object of magnitude 2. Is that
correction **coherent** — controlled by an identity — or is it an isolated term? The
suggested edge asked for a "pentagon". The honest, computable form of that question is
the **G₂ contraction identity**: does the associator 3-form close under
self-contraction onto its dual 4-form?

> **Framing (not an overclaim).** This is the *algebraic* coherence identity of G₂
> (a contraction / 2-cocycle-type relation among the structure forms), **not** the
> literal Mac Lane pentagon — the octonions carry no monoidal 4-fold structure for
> which a pentagon would be stated. What is proved is the exact structure identity that
> governs how ord-1 associators compose and that produces the graded ord-2 datum.

---

## 1. The two forms

For imaginary basis units (`a,b,c ∈ 1..7`):

```
phi_{abc} = <e_a e_b, e_c>                       -- the G2 3-form (structure constants)
[e_a, e_b, e_c] = -2 * sum_d psi_{abcd} e_d       -- the associator defines the 4-form psi
```

Both are computed directly from the octonion product (not fitted). Measured:

- `phi` is **totally antisymmetric**, values in `{-1,0,1}` — the G₂ 3-form.
- `psi` is **totally antisymmetric**, values in `{-1,0,1}`, and **G₂-invariant**
  (`max dev 3.2e-15` over 50 automorphisms) — the co-associator 4-form.

---

## 2. The identity (exact)

```
sum_e phi_{abe} phi_{cde} = delta_ac delta_bd - delta_ad delta_bc - psi_{abcd}
```

holds over all `7⁴` index tuples with **worst deviation `0.0`** (`P3`). This is the
defining G₂ contraction identity: the self-contraction of the 3-form is a sum of the
metric-square term and the dual 4-form. It is the coherence law of the associator —
the composition of two ord-1 corrections is not free, it closes onto `δδ − δδ` plus
exactly one graded ord-2 object `psi`.

*(A sign remark, since it is easy to get wrong: with the associator convention
`[e_a,e_b,e_c] = -2 psi e_d` above, the identity carries `-psi`. The probe first tried
`+psi` and missed by exactly `2` on every `psi = ±1` entry — the tell of a pure sign
convention, not a real failure; the contract fixes the sign and the residual is `0`.)*

---

## 3. Tie back to the field correction (`P4`)

For every one of the 42 field configurations,

```
[e_{u₁}, e_{u₂}, e_z] = -2 * psi_{u₁ u₂ z ·}    (exact)
```

so the ord-1 coupling-composition correction of `K_CHARACTERISED` **is** the invariant
4-form `psi` with one index left open. The magnitude-2 seen there is `2·|psi| = 2`.
Thus the field defect is a graded, G₂-invariant, coherent object — the ord-2 rung of
the same structure, not a new phenomenon.

---

## 4. Contract clauses

| Clause | Statement | PASS = |
|---|---|---|
| `P0_CORE_AUDIT` | inherited octonion core satisfies its axioms (identity, `e_i²=-1`, anticommutativity, **alternativity**) | foundation independently verified before use |
| `P1_PHI_3FORM` | `phi` totally antisymmetric, values `{-1,0,1}` | it is the G₂ 3-form |
| `P2_PSI_4FORM` | `psi` totally antisymmetric, `{-1,0,1}`, G₂-invariant | it is the invariant co-associator |
| `P3_CONTRACTION_IDENTITY` | `phi·phi = δδ − δδ − psi` exactly, all `7⁴` | the coherence law holds |
| `P4_CORRECTION_IS_PSI` | field correction `= -2·psi[u₁,u₂,z,·]`, all 42 configs | ord-1 defect is the graded 4-form |

Verdict: `FUNCTOR_F_COHERENCE_VERDICT P_GREEN`.

---

## 5. What this establishes for the programme

The `rupture-programme-synthesis` names the associator as "the G₂ 3-form" and draws
G₂ as the algebraic spine binding ord-1 and ord-2. This rung **proves that spine is
the actual G₂ structure**: the ord-1 3-form `phi` and its self-contraction generate
the invariant 4-form `psi`, via G₂'s own defining identity, exactly. The Functor-F
ladder now reads:

```
G_GREEN         uniformity across 7 lines
H_CHARACTERISED argmax-b obstruction, b_cov fix
E_GREEN         continuous-orbit equivariance, pole-flip
K_CHARACTERISED field functoriality up to ord-1 correction
P_GREEN         that correction is the coherent G2 3-form; contraction closes onto psi
```

---

## 6. What this is NOT

- **Not the Mac Lane pentagon** (see §0 framing) — it is the G₂ contraction identity.
- **Not a new proof of G₂ structure constants** — it verifies the standard identity on
  this repo's octonion core and links it to the field correction.
- **Not D3, not clinical.**

---

## 7. Suggested next edges

1. **Order-dependence / antisymmetry of the coupling defect** (was field §6.2):
   `[u₁,u₂,z] = −[u₂,u₁,z]` follows from `psi` antisymmetry — a one-line corollary now.
2. **`psi`-contraction chain**: `sum_ef psi_{abef} psi_{cdef}` — the next identity in the
   G₂ tower; does it close onto `phi` and the metric? (ord-2 → ord-3 bookkeeping.)
3. **Lift `psi` into Φ_fp** as a field-level dial: does the invariant `psi`-correction
   select a well-defined *cross-field* path class?

---

## 8. Reproduce

```bash
python3 scripts/research/functor_f_g2_coherence_contract.py
# expect: P0..P4 PASS, FUNCTOR_F_COHERENCE_VERDICT P_GREEN
```

Pure Python (numpy); CD sign law self-contained; the contract independently
re-audits the inherited octonion core (`P0`) before using it.

---

## 9. AI disclosure

Probe, contract, and note produced under human direction (2026-07-25). The pentagon
edge was reframed to the G₂ contraction identity to avoid a monoidal-pentagon
overclaim; a probe caught a `psi`-sign convention (missed by exactly 2) before the
spec was written. The inherited octonion core was independently axiom-audited per
standing instruction to verify other agents' work. Claims bounded by the five named
clauses. Commit gated on the §10 math-review offload
(`bin/llm-offload -t math-review -p xai`). No clinical content. GAIDeT-ICMJE 2025.
