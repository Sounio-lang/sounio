<!-- docs:meta

topic_id: repo.docs.research.functor-f-fano-psl27-thread-2026-07-25
authority: repo_only
audience: researchers
last_validated: 2026-07-25
validated_by: claude
source_of_truth: repo.docs.research.functor-f-fano-psl27-thread-2026-07-25
-->

# Functor F — one Fano/PSL(2,7) threads the whole order tower

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `PSL27_THREADS_THE_TOWER` (a **positive** cross-order unification)
**Parents:** `functor_f_g2_coherence_spec_2026-07-25.md` (`P_GREEN`, the φ 3-form), `functor_f_ord3_secondary_ternary_spec_2026-07-25.md` (`SECONDARY_TERNARY_LOCATED`, the ord-3 fibre)
**Prior work threaded:** the sedenion 7-fibre / `168=|PSL(2,7)|` ZD structure (frente-B PR #660; Moreno; Kirshtein; repo memory `cd-tower-168-acts-on-zd-fibers`)
**Harness:** `scripts/research/functor_f_fano_psl27_thread_contract.py`

---

## 0. The result, and why it is the frontier

Every prior cross-column probe (Petitot ord-P, Ollivier–Ricci ord-M) came back a
located **obstruction**, and the ord-3 edge came back **located but unfilled**. Pushed to
stop retreating to obstructions and look at the **symmetry** rather than the bare
products, one finds a **positive** result the earlier passes walked past:

> **The same Fano plane and `PSL(2,7)` action index and move all three layers of the
> functor-F order tower.** The octonion G₂ 3-form `φ` (ord-1) has 7 Fano lines as its
> structure constants. The sedenion zero-divisor geometry (ord-2), and the secondary
> ternary operation that lives on it (ord-3), are organised by the **same** 7 Fano lines,
> and a `PSL(2,7)` collineation acts on all three by the **same** permutation.

This is an **operational unification** of the two columns the programme had kept
separate — the **algebraic** column (`φ`, `ψ`, the closed contraction tower) and the
**ZD-geometric** column (the 168-orbit fibres): they share one indexing (the Fano plane)
and one symmetry (`PSL(2,7)`). It is a shared structure across layers, **not** a claim
that the three layers are a single object, and **not** an identity between them (D3).

---

## 1. Results

| Clause | Result | Reading |
|---|---|---|
| `W1_ZD_7FIBRES` | the 42 sedenion ZD `e_i+e_j` fall into **7 fibres of 6** | the ord-2 ZD set has the Fano cardinality (`7 lines × 6 ordered pairs`). |
| `W2_FIBRE_EQ_PHI_LINE` | each `e_i + e_{8+k}` has fibre-line = the Fano line through `(i,k)` (its octonion-support is that line's complement); the **7 fibre-lines are exactly the 7 `φ` structure-constant lines** | **the new identity**: the ZD fibre index *is* the functor-F 3-form. |
| `W3_PSL27_EQUIVARIANCE` | an explicit `PSL(2,7)` Fano collineation (a signed-permutation octonion automorphism, order 3) lifted diagonally to `Aut(𝕊)` permutes the ZD fibres by the **same `π`** it induces on the `φ` lines — verified on all 42 ZD | the symmetry is shared, not coincidental. |
| `W4_ORD3_ON_FIBRE` | the ord-3 secondary operation sits on these fibres (`ker L_b = ker R_b = ` 4-dim, uniform) | ord-1 `φ`, ord-2 ZD, ord-3 secondary all thread one `PSL(2,7)`. |

Verdict: `FUNCTOR_F_FANO_VERDICT PSL27_THREADS_THE_TOWER`.

---

## 2. What is prior and what is new (honest split)

- **Prior.** That the sedenion ZD organise into 7 fibres carrying a `168=|PSL(2,7)|`
  action is established (frente-B PR #660 "L=lo⊕hi, 7×12, 168"; Moreno's fibre
  construction; Kirshtein's signed-group analysis; the repo's own orbit-theorem memo).
  The octonion `φ` 3-form / Fano plane is textbook.
- **New (this rung).** The **explicit identity** `fibre-line = φ-line` (the complement
  map, exact on all 42 ZD), and the **equivariant thread** that makes ord-1 (`φ`), ord-2
  (the ZD fibre), and ord-3 (the secondary ternary operation located there) three faces
  of the *same* `PSL(2,7)` — i.e. the functor-F algebraic tower and the ZD-orbit geometry
  share one Fano/PSL(2,7) structure rather than sitting as two adjacent columns. This is the connection the `φ`-tower work
  (`G_GREEN…Q_GREEN`) makes available and that the cross-column passes had not drawn.

---

## 3. Why this is not D3 and not trivial doubling

- **Not D3.** No identity "the ZD locus *is* the associator/Petitot bifurcation set" is
  claimed. What is shared is an **indexing and a symmetry** (the Fano plane and its
  automorphism group), an operational unification, not an equation between the objects.
- **Not merely CD bookkeeping.** That sedenion multiplication is built from octonions by
  doubling is what makes the connection *possible*; it does not make the specific
  statements automatic — `W2`'s exact `fibre-line = φ-line` correspondence and `W3`'s
  equivariance over all 42 ZD are concrete, checkable facts, and they are what tie the
  new `φ`-tower to the prior 168-orbit. The contribution is the identification, executed.

---

## 4. Contract clauses

(as in §1; `W0_CORE_AUDIT` verifies both the octonion `bits=3` and sedenion `bits=4`
cores before use.)

---

## 5. What this is NOT

- **Not** a new proof of the 168-orbit theorem (prior work) — it threads it to `φ`.
- **Not** a claim that `PSL(2,7)` is the full `Aut(𝕊)` — it is the Fano-collineation
  subgroup that this rung exhibits acting equivariantly.
- **Not** a `∀`-`PSL(2,7)` structural proof — `W3` is computational evidence over the 42
  ZD for one explicit order-3 generator (itself re-verified in `Aut(𝕊)`); building all 168
  and proving transitivity is the §7.1 next edge.
- **Not** D3, not an identity, not clinical.

---

## 6. Place in the ladder — the first positive cross-order result

```
G_GREEN..Q_GREEN            algebraic column closed (φ, ψ, tower terminates ord-2)
B_OBSTRUCTED, M_CHARACTERISED   cross-column ord-P, ord-M: located obstructions
SECONDARY_TERNARY_LOCATED / NO_CANONICAL_FILL   ord-3: located on the ZD fibre, unfilled
PSL27_THREADS_THE_TOWER     ord-1 φ, ord-2 ZD, ord-3 secondary = one Fano/PSL(2,7)
```

The frontier is genuinely positive here: rather than another obstruction, the algebra's
own symmetry **unifies the orders it had been probing separately**.

---

## 7. Suggested next edges

1. **Full `PSL(2,7)` orbit on `φ`/`ψ`/ZD jointly** — build all 168 collineations and
   confirm the joint action is a single transitive `PSL(2,7)` on the 7-fibre × φ-line
   incidence (this rung exhibits one order-3 generator).
2. **`ψ` (4-form) under the same thread** — does the co-associator's coassociative
   4-planes align with the ZD fibres under `PSL(2,7)` too?
3. **The ord-3 quotient as a `PSL(2,7)`-module** — decompose the 2-dim quotient `Q`
   (`NO_CANONICAL_FILL`) under the fibre stabiliser; a distinguished sub-rep would be the
   symmetry-canonical secondary value the bare-algebra bracketing could not supply.

---

## 8. Reproduce

```bash
python3 scripts/research/functor_f_fano_psl27_thread_contract.py
# expect: W0..W4 PASS, FUNCTOR_F_FANO_VERDICT PSL27_THREADS_THE_TOWER
```

Pure Python (numpy); Cayley-Dickson `bits=3,4`; embeds the `W0` core axiom-audit;
constructs an explicit `PSL(2,7)` collineation (bit-rotation + solved signs).

---

## 9. AI disclosure

Probe, contract, and note produced under human direction (2026-07-25) after the operator
pushed to stop settling for obstruction-characterisations and look at the symmetry. The
`fibre-line = φ-line` identity and the `PSL(2,7)` equivariance are measured exactly over
all 42 ZD. The 7-fibre/168 structure is credited as prior work; the contribution is the
`φ`-tower thread. Claims bounded by the five named clauses. Commit gated on the §10
math-review offload. No clinical content. GAIDeT-ICMJE 2025.
