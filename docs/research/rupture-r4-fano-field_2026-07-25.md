<!-- docs:meta
topic_id: repo.docs.research.rupture-r4-fano-field-2026-07-25
authority: repo_only
audience: researchers
last_validated: 2026-07-25
validated_by: grok
source_of_truth: docs/research/rupture-r4-fano-field_2026-07-25.md
-->

# R4 — multi-line Fano field obstruction (system-level)

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** `R4_PARTIAL` (field census + mixing jet); multi-line Φ path classes open  
**Harness:** `scripts/research/rupture_r4_fano_field_contract.py`  
**Parents:** `rupture-r3-fano-phi_2026-07-25.md` (R3_GREEN), `petitot-semantic-potential.md` §2–3,
`rupture-abcd-claims_2026-07-24.md`

---

## 1. Why R4 after R3_GREEN

R3 closes the **neighbourhood of one Fano line** \(L_\star\):

- jet \([e_i+\varepsilon e_u,e_j,e_k]=\varepsilon[e_u,e_j,e_k]\);
- Φ_fp even/odd → Path C (contrariety) / Path D (contradiction).

Petitot’s square in isolation is still the wrong level for the octonion model: the
algebraic non-Booleanisability of 𝕆 is a property of the **field of squares**
(seven quaternion subalgebras), not of one line. R4 certifies that field.

---

## 2. Field facts (executed)

| ID | Statement | Result |
|---|---|---|
| **F1** | Each of the 7 Fano lines is internally associative | PASS |
| **F2** | Every pair of lines meets in **exactly one** unit (Fano incidence) | PASS (21 pairs) |
| **F3** | Cross-line non-Fano triples on a worked pair have \(\|\mathrm{assoc}\|=2\) | PASS |
| **F4** | Census on all \(\binom{7}{3}=35\) Im triples: 7 Fano (assoc 0) + 28 non-Fano (assoc 2) | PASS |
| **F5** | Two-line mixing \(x=e_{a_1}+\delta e_{a_2}\) yields a **linear cross jet**; residual \(\|\alpha\|=2\) at \(\delta=1\); pure L1 perturbation of the same form gives \(\|\alpha\|=0\) | PASS |
| **F6** | Worked pair \(L_1=(1,2,3)\), \(L_2=(1,4,5)\) share \(e_1\) | PASS |

### The system residual (load-bearing contrast)

```text
F5_BASE_ON_L1          [e2,e1,e3] ||α|| = 0     (all on L1)
F5_L1_INTERNAL_PERTURB e2+e3 on L1   ||α|| = 0
F5_RESIDUAL_CROSS_δ1   e2+e4 (L1⊕L2) ||α|| = 2
```

Coupling a second square through the shared term produces an obstruction that
**does not appear** under internal perturbation of a single square. That is the
field-level rupture object.

---

## 3. Relation to R3 Φ_fp

| Object | Algebraic support | Cancel with single-line \(\tau\)? |
|---|---|---|
| R3 neighbourhood jet | one line + off-line unit | even/odd split; Path C cancels *odd control*, not \(\alpha\) itself |
| R4 cross jet | two lines through shared unit | residual \(\|\alpha\|\) stays \(O(1)\); not an L1-only phenomenon |

R3’s \(\tau\) dials the **cusp control plane** for one square’s ambient shadow.
R4’s cross associator is present **before** any control map: it is the reason a
system of squares cannot be Booleanised by treating each square in isolation.

---

## 4. Verdict ladder

| Verdict | Meaning |
|---|---|
| `R4_PARTIAL` | F1–F6 field census + mixing jet — **current** |
| `R4_GREEN` | multi-line Φ into control space with path-class separation (open) |

**D3** (identity of Petitot potential with algebraic locus) remains **forbidden**.

---

## 5. Reproduce

```bash
python3 scripts/research/rupture_r4_fano_field_contract.py
# expect: R4_VERDICT R4_PARTIAL, R4_CONTRACT_OK

bash scripts/ci/rupture_abcd_contracts_gate.sh
# expect: RUPTURE_ABCD_CONTRACTS_OK (includes R3_GREEN + R4_PARTIAL)
```

---

## 6. AI disclosure

Field contract under human direction (2026-07-25). No clinical claims.
GAIDeT-ICMJE 2025.
