<!-- docs:meta
topic_id: repo.docs.dissertation.handoff.fo-pk-method-science-package
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.handoff.fo-pk-method-science-package
-->

# FO PK method science — writing package (oral Css, R1–R4)

**For:** Claude Desktop / prose session drafting a methods–results subsection on
first-order GUM under Madaros FO (oral steady-state Css exemplar).  
**From:** Grok Build session 2026-07-31, branch `research/zd-fiber-antisymmetry-lemma-20260731`.  
**Not:** PBPK28 clinical chapter material — that remains `chapter_04.md` and
`section_4_10_sobol_hdmr_package.md`. This package is the **compiler-backed oral
Css FO science closeout** that supports Contribution 1 (GUM-through-model) at the
algebraic / steady-state layer before full ODE/PBPK budgets.  
**Governing numerical annex:** [`docs/dissertation/results/fo_pk_method_science_v1.md`](../results/fo_pk_method_science_v1.md)  
**Full receipt index:** [`docs/research/fo_pk_method_science_receipts_2026-07-31.md`](../../research/fo_pk_method_science_receipts_2026-07-31.md)  
**Residual §5.4 closeout:** [`docs/research/fo_pk_residual4_oral_css_closeout_2026-07-31.md`](../../research/fo_pk_residual4_oral_css_closeout_2026-07-31.md)  
**Compiler stack:** Madaros FO trust ≥42/42 — `scripts/ci/madaros_gum_fo_trust_gate.sh`  
**Package re-validation:** R1–R4 + `fo_residual4_stack_gate.sh` → `ORAL_CSS_RESIDUAL4_CLOSED` (2026-07-31)  
**Scope disclaimer:** Oral Css FO infrastructure and surface parity. **Not clinical guidance.**

Drafting rules: quote freezes from §3 tables only; cite gate path + pass token;
EN-UK orthography; residual §5.4 is **oral Css closed** — do not upgrade
L2 full-engine (arbitrary programs) into a theorem.

---

## 0. Where this sits in the thesis

| Layer | Role | Primary artefact |
|-------|------|------------------|
| Compiler FO stack | Method FO, free-fn fields, call-result, correlate, multi-mod | `docs/audit/MADAROS_FO_GUM_STACK_2026-07-27.md` |
| Science receipts R1–R4 | Measured oral Css freezes under FO | This package + results annex |
| PBPK28 GUM-through-ODE | Full clinical chapter (rapa/sema) | `chapter_04.md`, `pbpk28_epistemic_v1.md` |
| Higher-order / Sobol | Hessian residual, global SA | `m5_gum_4th_order_v1.md`, §4.10 package |

**Suggested placement (prose session decides):** a short subsection under methods
or results, e.g. **§4.x First-order GUM for oral steady-state Css (compiler
surfaces)** — immediately *before* or *as a warm-up to* GUM-through-ODE / PBPK28
epistemic budgets. Frame as: the language executes JCGM 100:2008 first-order
propagation on dissertation-shaped `Pk` APIs with the same freezes as multi-mod
stdlib helpers.

---

## 1. Model (quoteable)

Oral average steady-state concentration under a lognormal clearance latent:

\[
C_{\mathrm{ss}}
  = \frac{F\cdot \mathrm{Dose}/\tau}{\mathrm{CL}_0\, e^{\eta}},
\qquad
\mathrm{CL} = \mathrm{CL}_0 e^{\eta},
\qquad
V = V_0 e^{\eta_v}.
\]

Exposure product (correlated latents):

\[
E = \mathrm{CL}\cdot V = \mathrm{CL}_0 V_0 \exp(\eta_{\mathrm{cl}}+\eta_v).
\]

Default seeds (unless a driver overrides \(\tau\) or \(\rho\)):

| Parameter | Mean | \(\sigma\) |
|-----------|-----:|-----------:|
| \(F\) | 0.8 | 0.05 |
| Dose | 500 | 10 |
| \(\tau\) | 12 h | 0 (0.5 in R2 τ-uncertainty row) |
| \(\mathrm{CL}_0\) | 5 | 0.3 |
| \(V_0\) | 50 | 2 |
| \(\eta\) | 0 | 0.1 |

First-order GUM (JCGM 100:2008): for \(y=f(\mathbf{x})\),

\[
u_c^2(y) \approx \sum_i \sum_j
  \frac{\partial f}{\partial x_i}
  \frac{\partial f}{\partial x_j}
  u(x_i)u(x_j)r(x_i,x_j).
\]

Second-order mean correction (when reported):  
\(E_2[y] \approx f(\boldsymbol{\mu}) + \tfrac12 \sum_k H_{kk}\sigma_k^2\).

---

## 2. Claims → drivers → gates

| ID | Claim (one line) | Driver | Gate | Pass token |
|----|------------------|--------|------|------------|
| R1 | Method FO stack: struct-lit, `make_pk(...).css`, free-fn, shared-η exposure, ρ=1 | `examples/epistemic_fo_second_order/fo_pk_struct_method_driver.sio` | `scripts/ci/fo_pk_struct_method_driver_gate.sh` | `FO_PK_STRUCT_METHOD_DRIVER_PASS` / `*_GATE_OK` |
| R2 | \(\mathrm{Var}(E)=1575+1250\rho\); Css \(\sigma_\tau=0.5\) raises Var | `fo_pk_struct_rho_tau_driver.sio` | `fo_pk_struct_rho_tau_driver_gate.sh` | `FO_PK_STRUCT_RHO_TAU_DRIVER_PASS` |
| R3 | \(C_{\mathrm{ss}}\propto 1/\tau\), \(\mathrm{Var}\propto 1/\tau^2\); kel shared-η cancels | `fo_pk_struct_multidose_driver.sio` | `fo_pk_struct_multidose_driver_gate.sh` | `FO_PK_STRUCT_MULTIDOSE_DRIVER_PASS` |
| R4 | Import `epistemic::fo` bit-agrees with method / call-result / site | `fo_pk_import_method_driver.sio` | `fo_pk_import_method_driver_gate.sh` | `FO_PK_IMPORT_METHOD_DRIVER_PASS` |
| R5 | Oral AUC + \(t_{1/2}\) FO; shared-η kel cancel; surface parity | `fo_pk_struct_auc_thalf_driver.sio` | `fo_pk_struct_auc_thalf_driver_gate.sh` | `FO_PK_STRUCT_AUC_THALF_DRIVER_PASS` |
| R5b | Import `fo_auc`/`fo_thalf` bit-agrees with methods | `fo_pk_import_auc_thalf_driver.sio` | `fo_pk_import_auc_thalf_driver_gate.sh` | `FO_PK_IMPORT_AUC_THALF_DRIVER_PASS` |
| R6 | \(f_{\mathrm{rem}}\) + Rac FO; shared-η peel; method = peel | `fo_pk_struct_rac_driver.sio` | `fo_pk_struct_rac_driver_gate.sh` | `FO_PK_STRUCT_RAC_DRIVER_PASS` |
| R6b | Import `fo_rac`/`fo_frac_rem` bit-agrees on freezes | `fo_pk_import_rac_driver.sio` | `fo_pk_import_rac_driver_gate.sh` | `FO_PK_IMPORT_RAC_DRIVER_PASS` |
| R7 | \(C_{\max}\)/PTF FO; \(C_{\min}\) point; kel·τ identity | `fo_pk_struct_cmax_driver.sio` | `fo_pk_struct_cmax_driver_gate.sh` | `FO_PK_STRUCT_CMAX_DRIVER_PASS` |
| R7b | Import `fo_cmax`/`fo_cmin`/`fo_ptf` freezes (Cmin Var) | `fo_pk_import_cmax_driver.sio` | `fo_pk_import_cmax_driver_gate.sh` | `FO_PK_IMPORT_CMAX_DRIVER_PASS` |
| R8 | \(f_{\mathrm{ss}}(n)\) + \(n_{90}\) FO; shared-η peel | `fo_pk_struct_fss_driver.sio` | `fo_pk_struct_fss_driver_gate.sh` | `FO_PK_STRUCT_FSS_DRIVER_PASS` |
| R8b | Import `fo_fss`/`fo_n90` bit-agrees on freezes | `fo_pk_import_fss_driver.sio` | `fo_pk_import_fss_driver_gate.sh` | `FO_PK_IMPORT_FSS_DRIVER_PASS` |
| R9 | PTR + DOF FO; PTR = exp(kel·τ) | `fo_pk_struct_ptr_driver.sio` | `fo_pk_struct_ptr_driver_gate.sh` | `FO_PK_STRUCT_PTR_DRIVER_PASS` |
| R9b | Import `fo_ptr`/`fo_dof` bit-agrees on freezes | `fo_pk_import_ptr_driver.sio` | `fo_pk_import_ptr_driver_gate.sh` | `FO_PK_IMPORT_PTR_DRIVER_PASS` |
| R10 | MRT + \(t_{90}\) FO; links t½ and n90 | `fo_pk_struct_mrt_driver.sio` | `fo_pk_struct_mrt_driver_gate.sh` | `FO_PK_STRUCT_MRT_DRIVER_PASS` |
| R10b | Import `fo_mrt`/`fo_t90` bit-agrees on freezes | `fo_pk_import_mrt_driver.sio` | `fo_pk_import_mrt_driver_gate.sh` | `FO_PK_IMPORT_MRT_DRIVER_PASS` |
| R11 | LD = Dose·Rac + \(f_e\) FO | `fo_pk_struct_ld_driver.sio` | `fo_pk_struct_ld_driver_gate.sh` | `FO_PK_STRUCT_LD_DRIVER_PASS` |
| R11b | Import `fo_ld`/`fo_fe` bit-agrees on freezes | `fo_pk_import_ld_driver.sio` | `fo_pk_import_ld_driver_gate.sh` | `FO_PK_IMPORT_LD_DRIVER_PASS` |
| R12 | \(\mathrm{AUC}_\tau=C_{\mathrm{ss}}\cdot\tau\) FO identity | `fo_pk_struct_auct_driver.sio` | `fo_pk_struct_auct_driver_gate.sh` | `FO_PK_STRUCT_AUCT_DRIVER_PASS` |
| R12b | Import `fo_auc_tau`/`fo_css_tau` bit-agrees | `fo_pk_import_auct_driver.sio` | `fo_pk_import_auct_driver_gate.sh` | `FO_PK_IMPORT_AUCT_DRIVER_PASS` |

Re-run all twenty:

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
bash scripts/ci/fo_pk_struct_cmax_driver_gate.sh        # R7
bash scripts/ci/fo_pk_import_cmax_driver_gate.sh        # R7b
bash scripts/ci/fo_pk_struct_fss_driver_gate.sh         # R8
bash scripts/ci/fo_pk_import_fss_driver_gate.sh         # R8b
bash scripts/ci/fo_pk_struct_ptr_driver_gate.sh         # R9
bash scripts/ci/fo_pk_import_ptr_driver_gate.sh         # R9b
bash scripts/ci/fo_pk_struct_mrt_driver_gate.sh         # R10
bash scripts/ci/fo_pk_import_mrt_driver_gate.sh         # R10b
bash scripts/ci/fo_pk_struct_ld_driver_gate.sh          # R11
bash scripts/ci/fo_pk_import_ld_driver_gate.sh          # R11b
bash scripts/ci/fo_pk_struct_auct_driver_gate.sh        # R12
bash scripts/ci/fo_pk_import_auct_driver_gate.sh        # R12b
```

Expected: twenty `*_GATE_OK` lines. R1–R4 re-validated 2026-07-31; R5–R12b on 2026-08-01.

Compiler prerequisite (do not claim science freezes without it):

```bash
bash scripts/ci/madaros_gum_fo_trust_gate.sh   # ≥42/42
```

---

## 3. Numerical results to quote (verbatim freezes)

### 3.1 Core Css freezes at \(\tau=12\) h (R1, R4)

| Quantity | Value | Receipts |
|----------|------:|----------|
| \(C_{\mathrm{ss}}\) point | 6.666666 | R1, R4 |
| \(\mathrm{Var}(C_{\mathrm{ss}})\) | 0.795833 | R1–R4 |
| \(E_2[C_{\mathrm{ss}}]\) | 6.724 | R1, R3, R4 |
| bias \(E_2 - C_{\mathrm{ss}}\) | 0.057333 | R1 |
| \(\mathrm{Var}(\mathrm{CL})\) | 0.340000 | R1, R4 |
| \(\mathrm{Var}(\mathrm{rate})\) | 4.784722 | R4 |
| \(\sum H_{kk}\) (solo path) | 7.292592 | R1 |

### 3.2 Correlated exposure and \(\tau\) uncertainty (R2)

Exposure \(E=\mathrm{CL}\cdot V\) under FO on \(\mathrm{CL}_0,V_0,\eta_{\mathrm{cl}},\eta_v\):

\[
\mathrm{Var}(E) = 1575 + 1250\cdot\rho.
\]

| \(\rho\) | \(\mathrm{Var}(E)\) |
|---------:|--------------------:|
| 0 | 1575 |
| 0.5 | 2200 |
| 1 | 2825 (= shared-η peel) |

Css with dosing-interval uncertainty:

| Setup | \(\mathrm{Var}(C_{\mathrm{ss}})\) | \(E_2\) |
|-------|----------------------------------:|--------:|
| \(\sigma_\tau=0\) | 0.795833 | 6.724 |
| \(\sigma_\tau=0.5\) | 0.872993 | 6.735574 |

### 3.3 Dosing-interval series and kel (R3)

Law: \(C_{\mathrm{ss}}\propto 1/\tau\), \(\mathrm{Var}(C_{\mathrm{ss}})\propto 1/\tau^2\).

| \(\tau\) (h) | \(C_{\mathrm{ss}}\) | \(\mathrm{Var}\) | Scale vs \(\tau=12\) |
|-------------:|--------------------:|-----------------:|---------------------:|
| 8 | 10.000000 | 1.790625 | 2.25 \(=(12/8)^2\) |
| 12 | 6.666666 | 0.795833 | 1 |
| 24 | 3.333333 | 0.198958 | 0.25 \(=(12/24)^2\) |

Elimination rate \(\mathrm{kel}=\mathrm{CL}/V\) with **shared** \(\eta\): latent cancels →  
\(\mathrm{kel}=0.1\), \(\mathrm{Var}(\mathrm{kel})=5.2\times 10^{-5}\) (FO from \(\mathrm{CL}_0,V_0\) only).

### 3.4 Surface parity (R4)

| Surface | \(\mathrm{Var}(C_{\mathrm{ss}})\) | \(E_2\) | \(\mathrm{Var}(\mathrm{CL})\) |
|---------|----------------------------------:|--------:|------------------------------:|
| import `fo_css` | 0.795833 | 6.724 | 0.340000 |
| `pk.css` method | 0.795833 | 6.724 | 0.340000 |
| `make_pk(...).css` | 0.795833 | 6.724 | 0.340000 |
| call-site composition | 0.795833 | — | — |

### 3.5 Oral AUC and half-life (R5)

| Quantity | Value |
|----------|------:|
| AUC | 80 |
| \(\mathrm{Var}(\mathrm{AUC})\) | 114.6 |
| \(E_2[\mathrm{AUC}]\) | 80.688 |
| kel | 0.1 |
| \(\mathrm{Var}(\mathrm{kel})\) | \(5.2\times 10^{-5}\) |
| \(t_{1/2}\) | 6.931471 |
| \(\mathrm{Var}(t_{1/2})\) | 0.249835 |

Method = call-result = free-fn = site/peel on all Var freezes. Shared \(\eta\) cancels in kel / \(t_{1/2}\).

### 3.6 Accumulation ratio and residual fraction (R6)

| Quantity | Value |
|----------|------:|
| \(f_{\mathrm{rem}}\) | 0.301195 |
| \(\mathrm{Var}(f_{\mathrm{rem}})\) | 0.000679 |
| Rac | 1.431014 |
| \(\mathrm{Var}(\mathrm{Rac})\) | 0.002848 |
| \(E_2[\mathrm{Rac}]\) | 1.434130 |

Method = peel on both Vars; free-fn matches method. Shared \(\eta\) cancels in kel.
Multi-mod helpers: `fo_frac_rem`, `fo_rac` (R6b).

### 3.7 Multi-dose Cmax / Cmin / PTF (R7)

| Quantity | Value |
|----------|------:|
| \(C_{\max}\) | 11.448115 |
| \(\mathrm{Var}(C_{\max})\) | 2.050059 |
| \(E_2[C_{\max}]\) | 11.539124 |
| \(C_{\min}\) | 3.448115 |
| \(\mathrm{Var}(C_{\min})\) | 0.306096 |
| PTF | 1.200000 |
| \(\mathrm{Var}(\mathrm{PTF})\) | 0.007488 |

PTF \(=\mathrm{kel}\cdot\tau\) after algebra. Multi-mod: `fo_cmax`, `fo_cmin`, `fo_ptf` (R7b).

### 3.8 Fraction of SS and \(n_{90}\) (R8)

| Quantity | Value |
|----------|------:|
| \(f_{\mathrm{ss}}(3)\) | 0.972676 |
| \(\mathrm{Var}(f_{\mathrm{ss}}(3))\) | 0.000050 |
| \(E_2[f_{\mathrm{ss}}(3)]\) | 0.971912 |
| \(n_{90}\) | 1.918820 |
| \(\mathrm{Var}(n_{90})\) | 0.019145 |

Multi-mod: `fo_fss`, `fo_n90` (R8b).

### 3.9 Peak–trough ratio and DOF (R9)

| Quantity | Value |
|----------|------:|
| PTR | 3.320113 |
| \(\mathrm{Var}(\mathrm{PTR})\) | 0.082541 |
| \(E_2[\mathrm{PTR}]\) | 3.338918 |
| DOF | 2.320113 |
| \(\mathrm{Var}(\mathrm{DOF})\) | 0.082541 |

Multi-mod: `fo_ptr`, `fo_dof` (R9b).

### 3.10 MRT and \(t_{90}\) (R10)

| Quantity | Value |
|----------|------:|
| MRT | 10.000000 |
| \(\mathrm{Var}(\mathrm{MRT})\) | 0.519999 |
| \(E_2[\mathrm{MRT}]\) | 10.035999 |
| \(t_{90}\) (h) | 23.025850 |
| \(\mathrm{Var}(t_{90})\) | 2.756987 |
| \(E_2[t_{90}]\) | 23.108743 |

Multi-mod: `fo_mrt`, `fo_t90` (R10b). Peel MRT Var prints 0.520000 (ULP vs method).

### 3.11 Loading dose and \(f_e\) (R11)

| Quantity | Value |
|----------|------:|
| LD | 715.507200 |
| \(\mathrm{Var}(\mathrm{LD})\) | 916.939959 |
| \(E_2[\mathrm{LD}]\) | 717.065032 |
| \(f_e\) | 0.698804 |
| \(\mathrm{Var}(f_e)\) | 0.000679 |

Multi-mod: `fo_ld`, `fo_fe` (R11b).

### 3.12 Steady-state \(\mathrm{AUC}_\tau\) (R12)

| Quantity | Value |
|----------|------:|
| \(\mathrm{AUC}_\tau\) | 80 |
| \(\mathrm{Var}(\mathrm{AUC}_\tau)\) | 114.6 |
| \(E_2[\mathrm{AUC}_\tau]\) | 80.688 |
| \(\mathrm{Var}(C_{\mathrm{ss}}\cdot\tau)\) | 114.6 |

Multi-mod: `fo_auc_tau`, `fo_css_tau` (R12b). Identity with R5 AUC freezes.

---

## 4. Ready-to-paste EN-UK prose

Paste into the thesis chapter; adjust section numbers. Keep the residual
paragraph. Numbers must match §3 freezes.

### 4.1 Opening (methods or results)

> First-order uncertainty propagation for oral average steady-state
> concentration was executed under the Madaros compiler’s FO GUM stack
> (JCGM 100:2008). The algebraic model
>
> \[
> C_{\mathrm{ss}} = \frac{F\cdot\mathrm{Dose}/\tau}{\mathrm{CL}_0\,e^{\eta}}
> \]
>
> uses independent Gaussian seeds on bioavailability \(F\), dose, baseline
> clearance \(\mathrm{CL}_0\), and a zero-mean lognormal latent \(\eta\) (default
> \(\tau=12\,\mathrm{h}\)). The endpoint is deliberately simpler than the
> full PBPK28 GUM-through-ODE budgets of later sections: it isolates whether
> the language’s first-order shadow propagation is **surface-independent**
> across multi-module stdlib helpers, dissertation-shaped struct methods,
> call-result receivers, and call-site composition, before those operators
> are embedded in adaptive integrators.

### 4.2 Core freezes (R1)

> Under the default seeds \(F=0.8\pm 0.05\), \(\mathrm{Dose}=500\pm 10\),
> \(\mathrm{CL}_0=5\pm 0.3\), \(\eta=0\pm 0.1\), and fixed \(\tau=12\,\mathrm{h}\),
> Madaros reports \(C_{\mathrm{ss}}=6.666666\) with combined variance
> \(\mathrm{Var}(C_{\mathrm{ss}})=0.795833\). The second-order mean correction is
> \(E_2[C_{\mathrm{ss}}]=6.724\), a positive bias of \(0.057333\) relative to the
> point evaluation, consistent with convexity of the map in the uncertain
> inputs. Clearance alone freezes at \(\mathrm{Var}(\mathrm{CL})=0.340000\).
> These freezes are reproduced on struct-literal methods, on
> `make_pk(...).css` call-result projections, and on free-function parity
> paths (receipt R1; gate
> `scripts/ci/fo_pk_struct_method_driver_gate.sh`).

### 4.3 Correlated latents (R2)

> Exposure \(E=\mathrm{CL}\cdot V\) with independent baseline means and
> correlated latents \(\eta_{\mathrm{cl}},\eta_v\) yields the linear law
> \(\mathrm{Var}(E)=1575+1250\cdot\rho\). At \(\rho=0\), \(0.5\), and \(1\),
> measured variances are \(1575\), \(2200\), and \(2825\) respectively; the
> unit-correlation case bit-agrees with a shared-latent peel of the same
> product. Separately, admitting dosing-interval uncertainty
> \(\sigma_\tau=0.5\,\mathrm{h}\) raises \(\mathrm{Var}(C_{\mathrm{ss}})\) from
> \(0.795833\) to \(0.872993\) and \(E_2\) from \(6.724\) to \(6.735574\)
> (receipt R2).

### 4.4 Multi-dose interval series (R3)

> Holding all other seeds fixed, the steady-state mean scales as
> \(C_{\mathrm{ss}}\propto 1/\tau\) and the FO variance as
> \(\mathrm{Var}(C_{\mathrm{ss}})\propto 1/\tau^2\). Relative to the \(\tau=12\)
> reference, the \(\tau=8\) and \(\tau=24\) rows freeze at scale factors
> \(2.25=(12/8)^2\) and \(0.25=(12/24)^2\), with absolute variances
> \(1.790625\) and \(0.198958\). The elimination rate
> \(\mathrm{kel}=\mathrm{CL}/V\) with a **shared** latent \(\eta\) cancels that
> channel: \(\mathrm{kel}=0.1\) with residual
> \(\mathrm{Var}(\mathrm{kel})=5.2\times 10^{-5}\) attributable only to
> \(\mathrm{CL}_0\) and \(V_0\) (receipt R3). This cancellation is a structural
> check that FO peels shared channels rather than double-counting them.

### 4.5 Import ↔ method parity (R4)

> Multi-module helpers imported from `epistemic::fo`
> (`fo_css`, `fo_clearance`, `fo_infusion_rate`) bit-agree with the
> dissertation `Pk` methods, with call-result receivers, and with explicit
> call-site composition on every frozen column of
> \(\mathrm{Var}(C_{\mathrm{ss}})\), \(E_2\), \(\mathrm{Var}(\mathrm{CL})\), and
> \(\mathrm{Var}(\mathrm{rate})=4.784722\). Algebraically, the four pure
> surfaces are definitionally equal and the default-seed FO freezes are exact
> rationals (`SounioFoCssSurfaceParity.lean` / `fo_css_surface_parity_gate.sh`).
> Compiler IR surface commutativity for arbitrary programs remains
> **executable evidence** under green CI gates (receipt R4); residual compiler
> limits are listed in §5 of this package.

### 4.6 Bridge to PBPK / higher-order work (one paragraph)

> The oral Css suite does not replace GUM-through-ODE on PBPK14/PBPK28, nor
> the Sobol/Cut-HDMR global budgets of §4.10. Its role is infrastructural: it
> shows that first-order combined standard uncertainty, second-order mean
> bias, correlated latents, and shared-channel cancellation are available on
> the same API shapes a pharmacologist writes (`pk.css`, multi-dose \(\tau\)
> sweeps) and that those shapes agree with the multi-mod stdlib. Full organ
> trajectories, Hessian heatmaps, and Monte Carlo cross-checks remain the
> clinical-chapter claims; first-order GUM alone can understate uncertainty
> for strongly non-linear endpoints (see M5 fourth-order residual and MC
> cross-validation annexes).

### 4.7 Short citation blurb (abstract / related work)

> First-order GUM uncertainty for oral steady-state Css was executed under
> Madaros FO (trust ≥42). Struct methods, call-result projections, multi-mod
> `epistemic::fo` helpers, correlated latent \(\eta\), dosing-interval scaling,
> and \(\tau\)-uncertainty were measured as green CI receipts R1–R4
> (2026-07-31); tables re-run via `scripts/ci/fo_pk_*_gate.sh`.

---

## 5. Honest residuals (must appear if freezes are quoted)

1. **In-driver boolean acceptance after heavy FO** can SEGV under Madaros;
   gates **grep printed science tables**, not an in-process `if ok` chain.
2. **\(\sum H_{kk}\) under multi-site FO load** may print ~7.20 vs solo-path
   7.292592 (multidose driver). Primary science claims are Var / \(E_2\) freezes.
3. **Runtime (non-const) mutual FO depth** remains residual; unused by R1–R4.
4. **Import↔method agreement — residual §5.4 oral Css CLOSED (2026-07-31):**
   Full fragment stack closed (algebra → FoExpr → FO bytecode → pure compile →
   FO_XFER expand → multipass register → method peel → multi-mod registry
   model). Live R4 freezes green. **L2 full engine** (arbitrary programs)
   remains open and is out of dissertation scope for oral Css.
   Closeout: `docs/research/fo_pk_residual4_oral_css_closeout_2026-07-31.md`.
   Stack: `scripts/ci/fo_residual4_stack_gate.sh`.
5. **Not a full PBPK28 clinical claim** — oral Css exemplar only.
6. **Not clinical guidance** — dissertation methodology / compiler science.

---

## 6. FO compiler surfaces exercised

| Surface | R1 | R2 | R3 | R4 |
|---------|:--:|:--:|:--:|:--:|
| Struct-lit methods | ✓ | ✓ | ✓ | ✓ |
| Call-result methods | ✓ | ✓ | ✓ | ✓ |
| Free-fn / site parity | ✓ | | ✓ | ✓ |
| `correlate` | ✓ | ✓ | | |
| Multi-mod import | | | | ✓ |
| Shared FO channel (peel) | ✓ | ✓ | ✓ (kel) | |
| \(E_2\) / Hessian | ✓ | ✓ | ✓ | ✓ |

---

## 7. Cross-links

| Document | Role |
|----------|------|
| `docs/dissertation/results/fo_pk_method_science_v1.md` | Quantitative annex (tables + re-run) |
| `docs/research/fo_pk_method_science_receipts_2026-07-31.md` | Research receipt index |
| `docs/audit/MADAROS_FO_GUM_STACK_2026-07-27.md` | Compiler FO stack map |
| `formal/lean4/SounioFoCssSurfaceParity.lean` | Algebraic surface-parity residual closeout |
| `scripts/ci/fo_css_surface_parity_gate.sh` | Executable ℚ freeze certificate (17/17) |
| `docs/dissertation/handoff/chapter_04.md` | PBPK28 clinical chapter handoff |
| `docs/dissertation/handoff/section_4_10_sobol_hdmr_package.md` | Global SA writing package |
| `docs/dissertation/results/m5_gum_4th_order_v1.md` | Why FO/Hessian can understate MC |
| `docs/dissertation/VISAO_GERAL.md` | Contribution map (GUM-through-ODE) |

---

## 8. LLM-offload / audit trail

| When | Provider | Task | Outcome |
|------|----------|------|---------|
| 2026-07-31 (annex) | xAI (Grok) | math-review | OK on Css identity, τ-scaling, Var(E)=1575+1250ρ, kel cancel; TIGHTENABLE symbolic commutativity → residual §5.4 |
| 2026-07-31 (this package) | xAI / grok-4.3 | math-review | OK on all eight freeze families; no leaps |
| 2026-07-31 (this package) | Z.AI / GLM-5.2 | math-review | Independent re-derivation of Var(Css)=191/240, Var(E)=1575+1250ρ, ΣH_kk=1969/270 (incl. τ diag), kel Var=5.2e-5, σ_τ row, Var(rate)=689/144; truncated at token cap mid wrap-up, **zero [WRONG]** |
| 2026-07-31 (residual-4 Lean) | xAI + Z.AI | math-review | PASS on SounioFoCssSurfaceParity freezes + surface rfl; cert 17/17 |

Re-run `bin/llm-offload -t math-review -p xai` (and `-p zai`) if freezes change.
Every offload appends `.claude/llm_offload_log.md`.

---

*Package version fo-pk-method-science-handoff-v1 (2026-07-31). Re-run the four
gates before quoting any number outside this repository.*
