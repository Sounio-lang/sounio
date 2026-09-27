<!-- docs:meta
topic_id: repo.docs.research.fo-pk-method-science-receipts-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.fo-pk-method-science-receipts-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# FO PK method science receipts — measured oral Css suite (2026-07-31)

**Status:** `EXECUTABLE` — four green CI gates under Madaros FO trust ≥42/42  
**Compiler:** Madaros (`artifacts/self-hosted/madaros`)  
**Stack audit:** [`docs/audit/MADAROS_FO_GUM_STACK_2026-07-27.md`](../audit/MADAROS_FO_GUM_STACK_2026-07-27.md)  
**Scope:** Oral steady-state pharmacokinetics with first-order GUM / FO uncertainty. **Not clinical guidance.**

This note indexes the dissertation-facing **science receipts** that exercise the Madaros FO stack after method FO, free-fn fields, call-result receivers, correlate, and multi-mod import closed. Every number below is re-derivable by re-running the named gate.

---

## 1. Model

Oral average steady-state concentration:

\[
C_{\mathrm{ss}} = \frac{F \cdot \mathrm{Dose}/\tau}{\mathrm{CL}_0 \cdot e^{\eta}}, \qquad
\mathrm{CL} = \mathrm{CL}_0 e^{\eta}, \qquad
V = V_0 e^{\eta_v}
\]

Exposure product:

\[
E = \mathrm{CL} \cdot V = \mathrm{CL}_0 V_0 \exp(\eta_{\mathrm{cl}}+\eta_v)
\]

Default seeds (unless a driver overrides τ or ρ):

| Parameter | Mean | σ |
|-----------|-----:|--:|
| \(F\) | 0.8 | 0.05 |
| Dose | 500 | 10 |
| \(\tau\) | 12 h | 0 (or 0.5 in ρ-τ driver) |
| \(\mathrm{CL}_0\) | 5 | 0.3 |
| \(V_0\) | 50 | 2 |
| \(\eta\) | 0 | 0.1 |

---

## 2. Receipt family (run all four)

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
export MADAROS_RAW_BIN=${MADAROS_RAW_BIN:-artifacts/self-hosted/madaros}

bash scripts/ci/fo_pk_struct_method_driver_gate.sh      # R1 methods + ρ=1
bash scripts/ci/fo_pk_struct_rho_tau_driver_gate.sh     # R2 ρ-sweep + σ_τ
bash scripts/ci/fo_pk_struct_multidose_driver_gate.sh   # R3 τ-series + kel
bash scripts/ci/fo_pk_import_method_driver_gate.sh      # R4 import ↔ method
```

Measured 2026-07-31 on this workspace: **all four `*_GATE_OK`**.

| ID | Driver | Gate token |
|----|--------|------------|
| R1 | `examples/epistemic_fo_second_order/fo_pk_struct_method_driver.sio` | `FO_PK_STRUCT_METHOD_DRIVER_PASS` |
| R2 | `examples/epistemic_fo_second_order/fo_pk_struct_rho_tau_driver.sio` | `FO_PK_STRUCT_RHO_TAU_DRIVER_PASS` |
| R3 | `examples/epistemic_fo_second_order/fo_pk_struct_multidose_driver.sio` | `FO_PK_STRUCT_MULTIDOSE_DRIVER_PASS` |
| R4 | `examples/epistemic_fo_second_order/fo_pk_import_method_driver.sio` | `FO_PK_IMPORT_METHOD_DRIVER_PASS` |
| R5 | `examples/epistemic_fo_second_order/fo_pk_struct_auc_thalf_driver.sio` | `FO_PK_STRUCT_AUC_THALF_DRIVER_PASS` |
| R5b | `examples/epistemic_fo_second_order/fo_pk_import_auc_thalf_driver.sio` | `FO_PK_IMPORT_AUC_THALF_DRIVER_PASS` |
| R6 | `examples/epistemic_fo_second_order/fo_pk_struct_rac_driver.sio` | `FO_PK_STRUCT_RAC_DRIVER_PASS` |
| R6b | `examples/epistemic_fo_second_order/fo_pk_import_rac_driver.sio` | `FO_PK_IMPORT_RAC_DRIVER_PASS` |
| R7 | `examples/epistemic_fo_second_order/fo_pk_struct_cmax_driver.sio` | `FO_PK_STRUCT_CMAX_DRIVER_PASS` |
| R7b | `examples/epistemic_fo_second_order/fo_pk_import_cmax_driver.sio` | `FO_PK_IMPORT_CMAX_DRIVER_PASS` |
| R8 | `examples/epistemic_fo_second_order/fo_pk_struct_fss_driver.sio` | `FO_PK_STRUCT_FSS_DRIVER_PASS` |
| R8b | `examples/epistemic_fo_second_order/fo_pk_import_fss_driver.sio` | `FO_PK_IMPORT_FSS_DRIVER_PASS` |
| R9 | `examples/epistemic_fo_second_order/fo_pk_struct_ptr_driver.sio` | `FO_PK_STRUCT_PTR_DRIVER_PASS` |
| R9b | `examples/epistemic_fo_second_order/fo_pk_import_ptr_driver.sio` | `FO_PK_IMPORT_PTR_DRIVER_PASS` |
| R10 | `examples/epistemic_fo_second_order/fo_pk_struct_mrt_driver.sio` | `FO_PK_STRUCT_MRT_DRIVER_PASS` |
| R10b | `examples/epistemic_fo_second_order/fo_pk_import_mrt_driver.sio` | `FO_PK_IMPORT_MRT_DRIVER_PASS` |
| R11 | `examples/epistemic_fo_second_order/fo_pk_struct_ld_driver.sio` | `FO_PK_STRUCT_LD_DRIVER_PASS` |
| R11b | `examples/epistemic_fo_second_order/fo_pk_import_ld_driver.sio` | `FO_PK_IMPORT_LD_DRIVER_PASS` |
| R12 | `examples/epistemic_fo_second_order/fo_pk_struct_auct_driver.sio` | `FO_PK_STRUCT_AUCT_DRIVER_PASS` |
| R12b | `examples/epistemic_fo_second_order/fo_pk_import_auct_driver.sio` | `FO_PK_IMPORT_AUCT_DRIVER_PASS` |

```bash
bash scripts/ci/fo_pk_struct_auc_thalf_driver_gate.sh   # R5 AUC + t½ methods
bash scripts/ci/fo_pk_import_auc_thalf_driver_gate.sh   # R5b import ↔ method
bash scripts/ci/fo_pk_struct_rac_driver_gate.sh         # R6 Rac + f_rem methods
bash scripts/ci/fo_pk_import_rac_driver_gate.sh         # R6b import ↔ method
bash scripts/ci/fo_pk_struct_cmax_driver_gate.sh        # R7 Cmax + PTF methods
bash scripts/ci/fo_pk_import_cmax_driver_gate.sh        # R7b Cmin + multi-mod
bash scripts/ci/fo_pk_struct_fss_driver_gate.sh         # R8 f_ss + n90 methods
bash scripts/ci/fo_pk_import_fss_driver_gate.sh         # R8b import ↔ method
bash scripts/ci/fo_pk_struct_ptr_driver_gate.sh         # R9 PTR + DOF methods
bash scripts/ci/fo_pk_import_ptr_driver_gate.sh         # R9b import ↔ method
bash scripts/ci/fo_pk_struct_mrt_driver_gate.sh         # R10 MRT + t90 methods
bash scripts/ci/fo_pk_import_mrt_driver_gate.sh         # R10b import ↔ method
bash scripts/ci/fo_pk_struct_ld_driver_gate.sh          # R11 LD + fe methods
bash scripts/ci/fo_pk_import_ld_driver_gate.sh          # R11b import ↔ method
bash scripts/ci/fo_pk_struct_auct_driver_gate.sh        # R12 AUC_τ SS methods
bash scripts/ci/fo_pk_import_auct_driver_gate.sh        # R12b import ↔ method
```

---

## 3. R1 — Method FO stack (dissertation core)

**Surfaces:** struct-lit methods · `make_pk(...).css` · let-alias · free-fn parity · nested rate/CL · shared-η exposure · distinct η + `correlate(ρ=1)`.

| Quantity | Value |
|----------|------:|
| \(C_{\mathrm{ss}}\) point | 6.666666 |
| \(\mathrm{Var}(C_{\mathrm{ss}})\) | 0.795833 |
| \(E_2[C_{\mathrm{ss}}]\) | 6.724 |
| \(\sum H_{kk}\) | 7.292592 |
| bias \(E_2 - C_{\mathrm{ss}}\) | 0.057333 |
| \(\mathrm{Var}(\mathrm{CL})\) | 0.340000 |
| exposure shared η | 2825 |
| exposure indep η | 1575 |
| exposure ρ=1 | 2825 |

**Derivation (exposure with FO on \(\mathrm{CL}_0,V_0,\eta\)):**  
\(f=\mathrm{CL}_0 V_0 e^{2\eta}\) at means has \(s_{\mathrm{cl}}=50\), \(s_v=5\), \(s_\eta=500\)  
→ \(\mathrm{Var}=225+100+2500=2825\). Independent \(\eta_1,\eta_2\): \(s_{\eta1}=s_{\eta2}=250\) → \(1575\).

---

## 4. R2 — Correlated latents + τ uncertainty

**Exposure** \(\mathrm{Var}=1575 + 1250\cdot\rho\) (cross-term \(2\cdot 250\cdot 250\cdot\rho\cdot 0.01\)):

| \(\rho\) | \(\mathrm{Var}(E)\) |
|---------:|--------------------:|
| 0 | 1575 |
| 0.5 | 2200 |
| 1 | 2825 (= shared peel) |

**Css with \(\sigma_\tau=0.5\)** (vs fixed \(\tau\)):

| Setup | \(\mathrm{Var}(C_{\mathrm{ss}})\) | \(E_2\) |
|-------|----------------------------------:|--------:|
| \(\sigma_\tau=0\) | 0.795833 | 6.724 |
| \(\sigma_\tau=0.5\) | 0.872993 | 6.735574 |

Call-result methods match struct-lit on every row.

---

## 5. R3 — Multi-dose interval series

\(C_{\mathrm{ss}}\propto 1/\tau\), \(\mathrm{Var}(C_{\mathrm{ss}})\propto 1/\tau^2\) relative to \(\tau=12\):

| \(\tau\) (h) | \(C_{\mathrm{ss}}\) | \(\mathrm{Var}\) | Scale vs \(\tau=12\) |
|-------------:|--------------------:|-----------------:|---------------------:|
| 8 | 10.000000 | 1.790625 | 2.25 \(=(12/8)^2\) |
| 12 | 6.666666 | 0.795833 | 1 |
| 24 | 3.333333 | 0.198958 | 0.25 \(=(12/24)^2\) |

**kel = CL/V** with shared \(\eta\): latent cancels → \(\mathrm{kel}=0.1\), \(\mathrm{Var}=0.000052\) (FO from \(\mathrm{CL}_0,V_0\) only).  
Parity: call-result and free-fn match at \(\tau=12\); \(E_2=6.724\).

---

## 6. R4 — Import ↔ method FO parity

Multi-mod `use epistemic::fo::{fo_css, fo_clearance, fo_infusion_rate}` **bit-agrees** with Pk methods, call-result, and call-site composition:

| Surface | \(\mathrm{Var}(C_{\mathrm{ss}})\) | \(E_2\) | \(\mathrm{Var}(\mathrm{CL})\) | \(\mathrm{Var}(\mathrm{rate})\) |
|---------|----------------------------------:|--------:|------------------------------:|--------------------------------:|
| import `fo_css` | 0.795833 | 6.724 | 0.340000 | 4.784722 |
| `pk.css` | 0.795833 | 6.724 | 0.340000 | 4.784722 |
| `make_pk(...).css` | 0.795833 | 6.724 | 0.340000 | — |
| site composition | 0.795833 | — | — | — |

This is the multi-mod + method FO science closeout: stdlib helpers and dissertation structs are interchangeable FO surfaces for this model.

---

## 6b. R5 — Oral AUC + elimination half-life (2026-08-01)

Endpoints under the same seeds as R1 (plus \(V_0=50\pm 2\)):

\[
\mathrm{AUC}=\frac{F\cdot\mathrm{Dose}}{\mathrm{CL}_0 e^{\eta}},\qquad
t_{1/2}=\frac{\ln 2}{\mathrm{kel}},\quad
\mathrm{kel}=\frac{\mathrm{CL}}{V}
\]

With **shared** \(\eta\), kel cancels the latent; \(t_{1/2}=\ln 2\cdot V_0/\mathrm{CL}_0\).

| Quantity | Value | Surfaces |
|----------|------:|----------|
| AUC point | 80 | method / call-result |
| \(\mathrm{Var}(\mathrm{AUC})\) | 114.6 | method = call = free = site |
| \(E_2[\mathrm{AUC}]\) | 80.688 | method = call-result |
| kel point | 0.1 | method |
| \(\mathrm{Var}(\mathrm{kel})\) | \(5.2\times 10^{-5}\) | method |
| \(t_{1/2}\) point | 6.931471 | method |
| \(\mathrm{Var}(t_{1/2})\) | 0.249835 | method = call = free = peel |

Gate: `scripts/ci/fo_pk_struct_auc_thalf_driver_gate.sh` → `FO_PK_STRUCT_AUC_THALF_GATE_OK`.

**R5b import parity:** multi-mod `fo_auc` / `fo_kel` / `fo_thalf` / `fo_clearance`
(`stdlib/epistemic/fo.sio`) bit-agree with Pk methods, call-result, site, and peel
on all R5 freezes. Gate: `fo_pk_import_auc_thalf_driver_gate.sh`.

---

## 6c. R6 — Multi-dose accumulation ratio + residual fraction (2026-08-01)

Under the same \(\mathrm{CL}_0,V_0,\eta\) seeds as R3 kel, fixed \(\tau=12\):

\[
f_{\mathrm{rem}}=\exp(-\mathrm{kel}\cdot\tau),\qquad
\mathrm{Rac}=\frac{1}{1-f_{\mathrm{rem}}}=\frac{1}{1-\exp(-\mathrm{kel}\cdot\tau)}.
\]

Shared \(\eta\) cancels in kel, so FO is from \(\mathrm{CL}_0,V_0\) only (same peel class as \(t_{1/2}\)).

| Quantity | Value | Surfaces |
|----------|------:|----------|
| \(f_{\mathrm{rem}}\) point | 0.301195 | method |
| \(\mathrm{Var}(f_{\mathrm{rem}})\) | 0.000679 | method = peel |
| Rac point | 1.431014 | method |
| \(\mathrm{Var}(\mathrm{Rac})\) | 0.002848 | method = peel (= free, FO budget) |
| \(E_2[\mathrm{Rac}]\) | 1.434130 | method |

Gate: `scripts/ci/fo_pk_struct_rac_driver_gate.sh` → `FO_PK_STRUCT_RAC_GATE_OK`.

**R6b import parity:** multi-mod `fo_rac` / `fo_frac_rem` bit-agree with method
frac and peel on Rac freezes; method Rac Var/\(E_2\) printed from import FO under
multi-site FO residual budget (≤6 heavy FO sites — more sites silent-exit rc=0).
Gate: `fo_pk_import_rac_driver_gate.sh` → `FO_PK_IMPORT_RAC_GATE_OK`.

**Compiler residual (honest):** Rac FO expressions are heavy
\((1/(1-\exp(\cdot)))\). Drivers keep ≤5–6 `variance_of` / `second_order_mean`
sites; nested method FO for Rac requires **inlined kel** (separate kel method
call → SEGV). Documented in driver headers; gates still grep science tables.

---

## 6d. R7 — Multi-dose Cmax / Cmin / peak–trough fluctuation (2026-08-01)

1-compartment multi-dose with effective dose \(F\cdot\mathrm{Dose}\), shared \(\eta\):

\[
C_{\max}=\frac{F\cdot\mathrm{Dose}}{V}\mathrm{Rac},\quad
C_{\min}=\frac{F\cdot\mathrm{Dose}}{V}\frac{f_{\mathrm{rem}}}{1-f_{\mathrm{rem}}},\quad
\mathrm{PTF}=\frac{C_{\max}-C_{\min}}{C_{\mathrm{ss,avg}}}=\mathrm{kel}\cdot\tau.
\]

| Quantity | Value | Where measured |
|----------|------:|----------------|
| \(C_{\max}\) point | 11.448115 | R7 method |
| \(\mathrm{Var}(C_{\max})\) | 2.050059 | R7 method = import |
| \(E_2[C_{\max}]\) | 11.539124 | R7 method |
| \(C_{\min}\) point | 3.448115 | R7 method |
| \(\mathrm{Var}(C_{\min})\) | 0.306096 | R7b import (FO budget) |
| PTF point | 1.200000 | R7 method (= kel·τ) |
| \(\mathrm{Var}(\mathrm{PTF})\) | 0.007488 | R7 method peel = import |

Gates: `fo_pk_struct_cmax_driver_gate.sh`, `fo_pk_import_cmax_driver_gate.sh`.
stdlib: `fo_cmax`, `fo_cmin`, `fo_ptf`.

**FO budget:** ≤3 FO sites when Cmax is Rac-class heavy (4th silent-exits).

---

## 6e. R8 — Fraction of steady state + doses to 90% SS (2026-08-01)

\[
f_{\mathrm{ss}}(n)=1-\exp(-n\cdot\mathrm{kel}\cdot\tau),\qquad
n_{90}=\frac{\ln 10}{\mathrm{kel}\cdot\tau}.
\]

Shared \(\eta\) cancels; FO from \(\mathrm{CL}_0,V_0\) only (peel class).

| Quantity | Value | Surfaces |
|----------|------:|----------|
| \(f_{\mathrm{ss}}(3)\) point | 0.972676 | method |
| \(\mathrm{Var}(f_{\mathrm{ss}}(3))\) | 0.000050 | method = peel = import |
| \(E_2[f_{\mathrm{ss}}(3)]\) | 0.971912 | method |
| \(n_{90}\) point | 1.918820 | method |
| \(\mathrm{Var}(n_{90})\) | 0.019145 | method = peel = import |

Gates: `fo_pk_struct_fss_driver_gate.sh`, `fo_pk_import_fss_driver_gate.sh`.
stdlib: `fo_fss`, `fo_n90`.

---

## 6f. R9 — Peak–trough ratio + degree of fluctuation (2026-08-01)

\[
\mathrm{PTR}=\frac{C_{\max}}{C_{\min}}=\exp(\mathrm{kel}\cdot\tau)=\frac{1}{f_{\mathrm{rem}}},\qquad
\mathrm{DOF}=\frac{C_{\max}-C_{\min}}{C_{\min}}=\mathrm{PTR}-1.
\]

| Quantity | Value | Surfaces |
|----------|------:|----------|
| PTR point | 3.320113 | method |
| \(\mathrm{Var}(\mathrm{PTR})\) | 0.082541 | method = peel = import |
| \(E_2[\mathrm{PTR}]\) | 3.338918 | method |
| DOF point | 2.320113 | method |
| \(\mathrm{Var}(\mathrm{DOF})\) | 0.082541 | method = peel (= PTR var) |

Gates: `fo_pk_struct_ptr_driver_gate.sh`, `fo_pk_import_ptr_driver_gate.sh`.
stdlib: `fo_ptr`, `fo_dof`.

---

## 6g. R10 — MRT + time to 90% SS (hours) (2026-08-01)

\[
\mathrm{MRT}=\frac{1}{\mathrm{kel}}=\frac{V}{\mathrm{CL}},\qquad
t_{90}=\frac{\ln 10}{\mathrm{kel}}=n_{90}\cdot\tau.
\]

Links R5 \(t_{1/2}=\ln 2\cdot\mathrm{MRT}\) and R8 \(n_{90}=t_{90}/\tau\).

| Quantity | Value | Surfaces |
|----------|------:|----------|
| MRT point | 10.000000 | method |
| \(\mathrm{Var}(\mathrm{MRT})\) | 0.519999 (method/import) / 0.520000 (peel) | ULP on FO path |
| \(E_2[\mathrm{MRT}]\) | 10.035999 | method |
| \(t_{90}\) point | 23.025850 | method |
| \(\mathrm{Var}(t_{90})\) | 2.756987 | method = peel = import |
| \(E_2[t_{90}]\) | 23.108743 | method |

Gates: `fo_pk_struct_mrt_driver_gate.sh`, `fo_pk_import_mrt_driver_gate.sh`.
stdlib: `fo_mrt`, `fo_t90`.

---

## 6h. R11 — Loading dose + fraction eliminated (2026-08-01)

\[
\mathrm{LD}=\mathrm{Dose}\cdot\mathrm{Rac},\qquad
f_e=1-f_{\mathrm{rem}}=1-\exp(-\mathrm{kel}\cdot\tau).
\]

\(f_e=f_{\mathrm{ss}}(1)\); \(\mathrm{Var}(f_e)=\mathrm{Var}(f_{\mathrm{rem}})\) (R6).

| Quantity | Value | Surfaces |
|----------|------:|----------|
| LD point | 715.507200 | method |
| \(\mathrm{Var}(\mathrm{LD})\) | 916.939959 | method = peel = import |
| \(E_2[\mathrm{LD}]\) | 717.065032 | method |
| \(f_e\) point | 0.698804 | method |
| \(\mathrm{Var}(f_e)\) | 0.000679 | method = peel (= R6 f_rem) |

Gates: `fo_pk_struct_ld_driver_gate.sh`, `fo_pk_import_ld_driver_gate.sh`.
stdlib: `fo_ld`, `fo_fe`.

---

## 6i. R12 — Steady-state AUC over dosing interval (2026-08-01)

\[
\mathrm{AUC}_{\tau,\mathrm{ss}}=\frac{F\cdot\mathrm{Dose}}{\mathrm{CL}}=C_{\mathrm{ss}}\cdot\tau.
\]

Same freezes as R5 oral AUC; Css·τ surface bit-agrees (multi-dose exposure identity).

| Quantity | Value | Surfaces |
|----------|------:|----------|
| \(\mathrm{AUC}_\tau\) point | 80 | method |
| \(\mathrm{Var}(\mathrm{AUC}_\tau)\) | 114.6 | method = Css·τ = fo_auc |
| \(E_2[\mathrm{AUC}_\tau]\) | 80.688 | method = Css·τ |
| \(C_{\mathrm{ss}}\) point | 6.666666 | method |
| \(\mathrm{Var}(C_{\mathrm{ss}})\) | 0.795833 | method (R1) |

Gates: `fo_pk_struct_auct_driver_gate.sh`, `fo_pk_import_auct_driver_gate.sh`.
stdlib: `fo_auc_tau`, `fo_css_tau` (aliases of `fo_auc` / `fo_css·τ`).

---

## 7. FO compiler surfaces exercised

| Surface | R1–R4 | R5/b | R6–R11/b | R12/b |
|---------|:-----:|:----:|:--------:|:-----:|
| Struct-lit methods | ✓ | ✓ | ✓ | ✓ |
| Multi-mod import | R4 | R5b | R6b–R11b | R12b |
| Shared FO peel | kel | t½ | f_rem…fe | Css·τ |
| \(E_2\) | Css | AUC | Rac…LD | AUC_τ |

Compiler prerequisite: Madaros FO trust gate **42/42** (`scripts/ci/madaros_gum_fo_trust_gate.sh`).

---

## 8. Honest residuals (do not paper over)

1. **In-driver bool acceptance after heavy FO** can SEGV under Madaros; gates therefore **grep printed science tables**, not an in-process `if ok` chain.
2. **ΣH under multi-site FO load** can print ~7.20 vs solo-path 7.292592 (multidose driver); Var/E₂ freezes remain the primary science claims.
3. **Runtime (non-const) mutual FO depth** remains residual; not used by these receipts.
4. **Import↔method residual §5.4 oral Css CLOSED (2026-07-31):** full fragment
   stack + R4; L2 full engine OPEN (out of scope). Closeout:
   `docs/research/fo_pk_residual4_oral_css_closeout_2026-07-31.md`.
   Stack: `fo_residual4_stack_gate.sh`.
5. **Multi-site heavy FO budget (R6–R7):** >~6 nested Rac-class FO sites (R6) or
   >~3 when stacking Cmax var+E₂ (R7) can silent-exit (rc=0). Drivers budget
   FO sites; Cmin Var lives on R7b import under that budget.

---

## 9. Dissertation citation sketch

When citing in dissertation prose (EN-UK):

> First-order GUM uncertainty for oral steady-state Css was executed under Madaros FO (trust ≥42). Struct methods, call-result projections, multi-mod `epistemic::fo` helpers, correlated latent η, dosing-interval scaling, and τ-uncertainty were measured as green CI receipts R1–R4 (2026-07-31); tables re-run via `scripts/ci/fo_pk_*_gate.sh`.

**Ready-to-paste chapter package (methods/results paragraphs + claim map):**  
[`docs/dissertation/handoff/fo_pk_method_science_package.md`](../dissertation/handoff/fo_pk_method_science_package.md)

**Quantitative annex:**  
[`docs/dissertation/results/fo_pk_method_science_v1.md`](../dissertation/results/fo_pk_method_science_v1.md)

**Residual §5.4 oral Css closeout:**  
[`docs/research/fo_pk_residual4_oral_css_closeout_2026-07-31.md`](fo_pk_residual4_oral_css_closeout_2026-07-31.md)  
(`bash scripts/ci/fo_residual4_stack_gate.sh` → `ORAL_CSS_RESIDUAL4_CLOSED`)

Point to this file + the four drivers for the numerical freezes.

---

*Receipts re-validated 2026-07-31 (R1–R4 + residual-4 stack). Re-run the package
commands in the annex before claiming any number in external prose.*
