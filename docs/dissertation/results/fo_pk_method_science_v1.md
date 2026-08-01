<!-- docs:meta
topic_id: repo.docs.dissertation.results.fo-pk-method-science-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.fo-pk-method-science-v1
-->

---
docs:meta: true
topic: dissertation-results
kind: quantitative-output
drug: oral-css-exemplar
model: FO-GUM-oral-Css
status: implementation-complete
version: fo-pk-method-science-v1
date: 2026-07-31
---

# FO PK method science — quantitative results v1

**Full receipt index (tables + re-run commands):**  
[`docs/research/fo_pk_method_science_receipts_2026-07-31.md`](../../research/fo_pk_method_science_receipts_2026-07-31.md)

**Chapter prose handoff (ready-to-paste EN-UK + claim map):**  
[`docs/dissertation/handoff/fo_pk_method_science_package.md`](../handoff/fo_pk_method_science_package.md)

**Residual §5.4 oral Css closeout (machine-checked surface independence):**  
[`docs/research/fo_pk_residual4_oral_css_closeout_2026-07-31.md`](../../research/fo_pk_residual4_oral_css_closeout_2026-07-31.md)

**Compiler stack:** Madaros FO trust ≥42/42  
**Scope:** Oral steady-state Css under first-order GUM / FO. **Not clinical guidance.**  
**Package status (2026-07-31):** R1–R4 science gates + `fo_residual4_stack_gate.sh` → `ORAL_CSS_RESIDUAL4_CLOSED`.

This results file is the dissertation annex pointer for the four green science receipts R1–R4. Numbers are frozen by CI gates; re-derive before any external claim. Paste-ready methods/results prose lives in the handoff package (§4).

---

## 1. Scientific claim (measured)

Under Madaros, first-order uncertainty propagation for oral average steady-state concentration

\[
C_{\mathrm{ss}} = \frac{F\cdot\mathrm{Dose}/\tau}{\mathrm{CL}_0\,e^{\eta}}
\]

is **surface-independent**: multi-mod stdlib helpers (`epistemic::fo`), dissertation-shaped `Pk` methods, call-result receivers (`make_pk(...).css`), and call-site composition produce the same Var / \(E_2\) / CL / rate freezes. Correlated latents, dosing-interval scaling, and τ-uncertainty are measured as separate green companions.

---

## 2. Core freezes (τ = 12 h)

| Quantity | Value | Receipt |
|----------|------:|---------|
| \(C_{\mathrm{ss}}\) point | 6.666666 | R1, R4 |
| \(\mathrm{Var}(C_{\mathrm{ss}})\) | 0.795833 | R1–R4 |
| \(E_2[C_{\mathrm{ss}}]\) | 6.724 | R1, R3, R4 |
| bias \(E_2 - C_{\mathrm{ss}}\) | 0.057333 | R1 |
| \(\mathrm{Var}(\mathrm{CL})\) | 0.340000 | R1, R4 |
| \(\mathrm{Var}(\mathrm{rate})\) | 4.784722 | R4 |

Seeds: \(F=0.8\pm0.05\), Dose\(=500\pm10\), \(\mathrm{CL}_0=5\pm0.3\), \(\eta=0\pm0.1\).

---

## 3. Correlated latents and τ uncertainty (R2)

Exposure \(E=\mathrm{CL}\cdot V\) with FO on \(\mathrm{CL}_0,V_0,\eta_{\mathrm{cl}},\eta_v\):

\[
\mathrm{Var}(E) = 1575 + 1250\cdot\rho
\]

| \(\rho\) | \(\mathrm{Var}(E)\) |
|---------:|--------------------:|
| 0 | 1575 |
| 0.5 | 2200 |
| 1 | 2825 (= shared peel) |

Css with \(\sigma_\tau=0.5\): \(\mathrm{Var}=0.872993\) (vs 0.795833 at \(\sigma_\tau=0\)); \(E_2=6.735574\).

---

## 4. Dosing-interval series (R3)

| \(\tau\) (h) | \(C_{\mathrm{ss}}\) | \(\mathrm{Var}\) | Scale vs \(\tau=12\) |
|-------------:|--------------------:|-----------------:|---------------------:|
| 8 | 10.000000 | 1.790625 | 2.25 |
| 12 | 6.666666 | 0.795833 | 1 |
| 24 | 3.333333 | 0.198958 | 0.25 |

Law: \(C_{\mathrm{ss}}\propto 1/\tau\), \(\mathrm{Var}\propto 1/\tau^2\).  
Elimination rate \(\mathrm{kel}=\mathrm{CL}/V\) with shared \(\eta\): \(\mathrm{kel}=0.1\), \(\mathrm{Var}=5.2\times10^{-5}\) (η cancels).

---

## 5. Import ↔ method parity (R4)

| Surface | \(\mathrm{Var}(C_{\mathrm{ss}})\) | \(E_2\) | \(\mathrm{Var}(\mathrm{CL})\) |
|---------|----------------------------------:|--------:|------------------------------:|
| `fo_css` (import) | 0.795833 | 6.724 | 0.340000 |
| `pk.css` (method) | 0.795833 | 6.724 | 0.340000 |
| `make_pk(...).css` | 0.795833 | 6.724 | 0.340000 |
| call-site composition | 0.795833 | — | — |

---

## 5b. Oral AUC + half-life (R5, 2026-08-01)

Same seeds as R1 plus \(V_0=50\pm 2\). Shared-\(\eta\) cancel on kel / \(t_{1/2}\).

| Quantity | Value |
|----------|------:|
| AUC point | 80 |
| \(\mathrm{Var}(\mathrm{AUC})\) | 114.6 |
| \(E_2[\mathrm{AUC}]\) | 80.688 |
| kel | 0.1 |
| \(\mathrm{Var}(\mathrm{kel})\) | \(5.2\times 10^{-5}\) |
| \(t_{1/2}\) | 6.931471 |
| \(\mathrm{Var}(t_{1/2})\) | 0.249835 |

Surface parity: method = call-result = free-fn = site/peel on Var freezes.

**R5b:** multi-mod `epistemic::fo::{fo_auc, fo_kel, fo_thalf}` bit-agrees with methods
(`FO_PK_IMPORT_AUC_THALF_GATE_OK`).

---

## 5c. Accumulation ratio + residual fraction (R6, 2026-08-01)

Same \(\mathrm{CL}_0,V_0,\eta\) seeds; fixed \(\tau=12\).  
\(f_{\mathrm{rem}}=\exp(-\mathrm{kel}\cdot\tau)\), \(\mathrm{Rac}=1/(1-f_{\mathrm{rem}})\).

| Quantity | Value |
|----------|------:|
| \(f_{\mathrm{rem}}\) point | 0.301195 |
| \(\mathrm{Var}(f_{\mathrm{rem}})\) | 0.000679 |
| Rac point | 1.431014 |
| \(\mathrm{Var}(\mathrm{Rac})\) | 0.002848 |
| \(E_2[\mathrm{Rac}]\) | 1.434130 |

Method = peel on both Vars; free-fn Rac matches method under FO site budget.
**R6b:** multi-mod `fo_rac`/`fo_frac_rem` agree on freezes (`FO_PK_IMPORT_RAC_GATE_OK`).

---

## 6. How to re-run

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
export MADAROS_RAW_BIN=${MADAROS_RAW_BIN:-artifacts/self-hosted/madaros}

bash scripts/ci/fo_pk_struct_method_driver_gate.sh
bash scripts/ci/fo_pk_struct_rho_tau_driver_gate.sh
bash scripts/ci/fo_pk_struct_multidose_driver_gate.sh
bash scripts/ci/fo_pk_import_method_driver_gate.sh
bash scripts/ci/fo_pk_struct_auc_thalf_driver_gate.sh   # R5
bash scripts/ci/fo_pk_import_auc_thalf_driver_gate.sh   # R5b
bash scripts/ci/fo_pk_struct_rac_driver_gate.sh         # R6
bash scripts/ci/fo_pk_import_rac_driver_gate.sh         # R6b
```

Expected: eight `*_GATE_OK` lines. R1–R4 re-validated 2026-07-31
(residual-4: `ORAL_CSS_RESIDUAL4_CLOSED`); R5–R6b re-validated 2026-08-01.

Full package re-run:

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
export MADAROS_RAW_BIN=${MADAROS_RAW_BIN:-artifacts/self-hosted/madaros}
bash scripts/ci/fo_pk_struct_method_driver_gate.sh
bash scripts/ci/fo_pk_struct_rho_tau_driver_gate.sh
bash scripts/ci/fo_pk_struct_multidose_driver_gate.sh
bash scripts/ci/fo_pk_import_method_driver_gate.sh
bash scripts/ci/fo_pk_struct_auc_thalf_driver_gate.sh
bash scripts/ci/fo_residual4_stack_gate.sh
```

---

## 7. Drivers (source of truth)

| ID | Path |
|----|------|
| R1 | `examples/epistemic_fo_second_order/fo_pk_struct_method_driver.sio` |
| R2 | `examples/epistemic_fo_second_order/fo_pk_struct_rho_tau_driver.sio` |
| R3 | `examples/epistemic_fo_second_order/fo_pk_struct_multidose_driver.sio` |
| R4 | `examples/epistemic_fo_second_order/fo_pk_import_method_driver.sio` |
| R5 | `examples/epistemic_fo_second_order/fo_pk_struct_auc_thalf_driver.sio` |
| R5b | `examples/epistemic_fo_second_order/fo_pk_import_auc_thalf_driver.sio` |
| R6 | `examples/epistemic_fo_second_order/fo_pk_struct_rac_driver.sio` |
| R6b | `examples/epistemic_fo_second_order/fo_pk_import_rac_driver.sio` |

Compiler prerequisite: FO trust 42/42 — `scripts/ci/madaros_gum_fo_trust_gate.sh`.  
Stack map: `docs/audit/MADAROS_FO_GUM_STACK_2026-07-27.md`.

---

## 8. Honest residuals

1. In-driver boolean acceptance after heavy FO can SEGV; gates grep printed tables.  
2. ΣH under multi-site FO load may print ~7.20 vs solo-path 7.292592; Var/\(E_2\) freezes are the primary claims.  
3. This annex is oral Css FO infrastructure, not a full PBPK28 clinical claim.  
4. **Import↔method — residual §5.4 oral Css CLOSED (2026-07-31):**
   Fragment stack through multipass register, method peel, and multi-mod
   registry model **CLOSED**; live R4 green. Full engine for arbitrary
   programs remains open (out of scope). Closeout:
   `docs/research/fo_pk_residual4_oral_css_closeout_2026-07-31.md`.
   Stack: `fo_residual4_stack_gate.sh`.
5. **R6 multi-site FO budget:** Rac-class FO is heavy; drivers keep ≤5–6 FO
   sites or silent-exit (rc=0). Nested Rac methods must inline kel.

---

## 9. Dissertation placement

Suggested subsection (numbering belongs to the prose session):

**§4.x First-order GUM for oral steady-state Css (compiler surfaces)** — warm-up
to GUM-through-ODE / PBPK28 epistemic budgets. Frame: JCGM 100:2008 first-order
propagation is surface-independent across multi-mod stdlib helpers, struct
methods, call-result receivers, and call-site composition.

Ready-to-paste EN-UK paragraphs (opening, R1–R4 freezes, bridge to PBPK, short
citation blurb, mandatory residuals):  
[`docs/dissertation/handoff/fo_pk_method_science_package.md`](../handoff/fo_pk_method_science_package.md) §4–§5.

Cross-links: `VISAO_GERAL.md` Contribution 1; `chapter_04.md` (PBPK28 clinical);
`section_4_10_sobol_hdmr_package.md` (global SA); `m5_gum_4th_order_v1.md` (FO can understate MC).

---

## 10. LLM-offload review

| Provider | Task | Outcome |
|----------|------|---------|
| xAI (Grok) | math-review (annex) | OK on Css identity, τ-scaling, Var(E)=1575+1250ρ, kel cancellation; TIGHTENABLE on symbolic commutativity (addressed in residual 4) |
| xAI + Z.AI | math-review (handoff package) | PASS — dual independent re-derivation of all freezes; zero [WRONG] |

---

*Annex version fo-pk-method-science-v1. Prefer the research receipts file for full tables; prefer the handoff package for chapter prose.*
