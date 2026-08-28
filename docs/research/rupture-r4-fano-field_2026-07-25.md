<!-- docs:meta
topic_id: repo.docs.research.rupture-r4-fano-field-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rupture-r4-fano-field-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# R4 — multi-line Fano field obstruction + multi-line Φ_fp

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** `R4_GREEN` (field census + multi-line path classes); D3 still forbidden  
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
(seven quaternion subalgebras), not of one line. R4 certifies that field **and**
lifts Φ_fp so its path classes are sourced from the **cross-line** jet.

---

## 2. Field facts F1–F6 (executed)

| ID | Statement | Result |
|---|---|---|
| **F1** | Each of the 7 Fano lines is internally associative | PASS |
| **F2** | Every pair of lines meets in **exactly one** unit | PASS (21 pairs) |
| **F3** | Cross-line non-Fano triples have \(\|\mathrm{assoc}\|=2\) | PASS |
| **F4** | Census: 7 Fano (assoc 0) + 28 non-Fano (assoc 2) | PASS |
| **F5** | Two-line mixing jet linear; L1-internal \(\|\alpha\|=0\), cross \(\|\alpha\|=2\) | PASS |
| **F6** | Worked pair \(L_1=(1,2,3)\), \(L_2=(1,4,5)\) share \(e_1\) | PASS |

### System residual (load-bearing contrast)

```text
F5_BASE_ON_L1          [e2,e1,e3] ||α|| = 0     (all on L1)
F5_L1_INTERNAL_PERTURB e2+e3 on L1   ||α|| = 0
F5_RESIDUAL_CROSS_δ1   e2+e4 (L1⊕L2) ||α|| = 2
```

---

## 3. F7 — multi-line Φ_fp path classes (R4_GREEN)

**Source of α (not R3):**

\[
x(\delta)=e_{a_1}+\delta e_{a_2},\qquad
\alpha(\delta)=[x(\delta),e_s,e_z]
\]

with \(a_1\in L_1\setminus\{s\}\), \(a_2\in L_2\setminus\{s\}\), \(s=L_1\cap L_2\),
\(z\) on \(L_1\). When the base triple lies on \(L_1\), \(\alpha(\delta)=\delta\cdot
[e_{a_2},e_s,e_z]\) — pure field term.

**Same Φ_fp as R3** (unit \(A_0=-1\)):

\[
a=A_0+\frac{\|\alpha\|^2}{4},\qquad
b=\tau+\frac{\alpha_m}{2}.
\]

| Path | Dial | Outcome under increasing \(\lvert\delta\rvert\) |
|---|---|---|
| **C (field contrariety)** | \(\tau=-\alpha_m/2\) \(\Rightarrow b\equiv 0\) | monostable **neutral** \(x=0\) |
| **D (field contradiction)** | \(\tau=0\) \(\Rightarrow b=\alpha_m/2\) | monostable **polar**; \(\mathrm{sign}(\delta)\) selects pole |

**Source check:** \(\|\alpha(1)\|=\|\mathrm{pure\ cross}\|\); L1-internal
perturbation of the same form still gives \(\|\alpha\|=0\).

So the path classes are **not** a re-run of single-line off-unit coupling: they
are driven by coupling a second square through the shared term.

---

## 4. Verdict

| Verdict | Meaning |
|---|---|
| `R4_PARTIAL` | F1–F6 only |
| `R4_GREEN` | F1–F7 — **current** |

**D3** remains **forbidden** (no identity of Petitot potential with algebraic locus).

---

## 5. Reproduce

```bash
python3 scripts/research/rupture_r4_fano_field_contract.py
# expect: R4_VERDICT R4_GREEN, F7_MULTI_LINE_PHI_PATHS PASS

bash scripts/ci/rupture_abcd_contracts_gate.sh
# expect: RUPTURE_ABCD_CONTRACTS_OK
```

---

## 6. AI disclosure

Field + multi-line Φ under human direction (2026-07-25). No clinical claims.
GAIDeT-ICMJE 2025.
