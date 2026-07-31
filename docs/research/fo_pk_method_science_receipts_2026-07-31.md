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

## 7. FO compiler surfaces exercised

| Surface | R1 | R2 | R3 | R4 |
|---------|:--:|:--:|:--:|:--:|
| Struct-lit methods | ✓ | ✓ | ✓ | ✓ |
| Call-result methods | ✓ | ✓ | ✓ | ✓ |
| Free-fn / site parity | ✓ | | ✓ | ✓ |
| `correlate` | ✓ | ✓ | | |
| Multi-mod import | | | | ✓ |
| Shared FO channel (peel) | ✓ | ✓ | ✓ (kel) | |
| \(E_2\) / Hessian | ✓ | ✓ | ✓ | ✓ |

Compiler prerequisite: Madaros FO trust gate **42/42** (`scripts/ci/madaros_gum_fo_trust_gate.sh`).

---

## 8. Honest residuals (do not paper over)

1. **In-driver bool acceptance after heavy FO** can SEGV under Madaros; gates therefore **grep printed science tables**, not an in-process `if ok` chain.
2. **ΣH under multi-site FO load** can print ~7.20 vs solo-path 7.292592 (multidose driver); Var/E₂ freezes remain the primary science claims.
3. **Runtime (non-const) mutual FO depth** remains residual; not used by these receipts.
4. **Import↔method residual split (six layers, 2026-07-31):** L0–L2
   registration fragment CLOSED; L2 engine install OPEN (R4). Stack:
   `fo_residual4_stack_gate.sh`. Spec:
   `docs/research/fo_css_compiler_residual_half_spec_2026-07-31.md`.

---

## 9. Dissertation citation sketch

When citing in dissertation prose (EN-UK):

> First-order GUM uncertainty for oral steady-state Css was executed under Madaros FO (trust ≥42). Struct methods, call-result projections, multi-mod `epistemic::fo` helpers, correlated latent η, dosing-interval scaling, and τ-uncertainty were measured as green CI receipts R1–R4 (2026-07-31); tables re-run via `scripts/ci/fo_pk_*_gate.sh`.

**Ready-to-paste chapter package (methods/results paragraphs + claim map):**  
[`docs/dissertation/handoff/fo_pk_method_science_package.md`](../dissertation/handoff/fo_pk_method_science_package.md)

**Quantitative annex:**  
[`docs/dissertation/results/fo_pk_method_science_v1.md`](../dissertation/results/fo_pk_method_science_v1.md)

Point to this file + the four drivers for the numerical freezes.

---

*Receipts re-validated 2026-07-31. Re-run the four gates before claiming any number in external prose.*
