<!-- docs:meta
topic_id: repo.docs.audit.dissertation-epistemic-fabrication-sweep-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dissertation-epistemic-fabrication-sweep-2026-08-18
-->

# Dissertation epistemic fabrication sweep — post-KCONF (#1882)

**Date:** 2026-08-18  
**Lane:** grok-cli2 / `fab-sweep-20260818`  
**Main tip:** `ee5541421e` (includes #1882 KCONF layout fix)  
**Instrument:** Madaros **rebuilt from this tip** (`artifacts/self-hosted/madaros`, ELF `\x7fELF`, 100 084 302 B) **and** `SOUNIO_SOUC_ENGINE=lean_single`. No E230 aggregate patch.  
**Method:** dual-engine `souc run` on dissertation / GUM / Budget64 surfaces; classify **bit-pattern class** (~1e18 / 8+ digit floats) vs **suspicious zero** (`var(...)=0` / `std(Knowledge)=0` where lean is non-zero).

Parent: `#1792` (MERGED) installed fail-closed detectors. `#1882` closed the F2 confidence sitofp instance. This note answers: **what else still fabricates on the thesis surface?**

---

## 0. Why dual-engine

Default `bin/souc` is Madaros. A number that is only healthy under lean_single is not thesis-defensible. Every row below was measured on **both**.

---

## 1. Closed in this wave (do not rediscover)

| ID | Symptom | Surface | Madaros (post-#1882) | lean_single |
|---|---|---|---|---|
| **F2 / KCONF** | IEEE bits of ~0.67 printed as ~4.6e18 | `epistemic_pbpk28` TEST 6 | **`0.671038`**, ALL 9 PASS | **`0.671038`**, ALL 9 PASS |
| **F2 witness** | same | `f64_bitcast_boundary_knowledge_conf` | `KCONF_ALL_OK`, R25=0.66 | `KCONF_ALL_OK` |

Mechanism: `Knowledge.confidence` was `is_float:3` with f64 store → sitofp. Fixed in `ir_register_knowledge_layout` (#1882).

---

## 2. Still fabricating — partition (two diseases)

### Family A — **GUM / Knowledge variance collapse** (Madaros-only zeros)

Lean shows real variance; Madaros prints exact `0.000000` and/or `std(Knowledge)=0`.

| # | Surface | Madaros | lean_single | Notes |
|---:|---|---|---|---|
| A1 | `rapamycin_epistemic_adaptive` | `var(blood/brain/periph)=0.000000`; rc≠0; `FABRICATED_ZERO` | `var(blood)=0.000009` …; **PASS** | Canonical F1 from `#1792` / fabrication gate |
| A2 | `rapamycin_rk4_budget` | `var(*)=0`; `std(Knowledge)=0` for all 3 comps; rc=0 (**silent** on vars) | `var(blood)=0.452e-3`; `std(Knowledge)>0` | Same disease as A1; Budget64 std still healthy on both |

**Count:** **2** measured dissertation surfaces (same cause class: Knowledge/GUM variance pipeline under Madaros).  
**Not closed by #1882.** Detectors catch A1; A2 still prints zeros without always failing the suite.

### Family B — **Type-B infinite DOF sentinel print** (Madaros-only `2^63`)

Budget64 stores Type-B / Welch fallback DOF as `1.0e30`. lean_single **prints** `1.000000e30`. Madaros **prints** `9223372036854775808.000000` (= `2^63` as f64).

| # | Surface | Madaros DOF print | lean DOF print | Expanded `k` / `U` |
|---:|---|---|---|---|
| B1 | `rapamycin_iso_budget` | `2^63` (×2 blood/brain) | `1e30` | `k=1.960`, `U` matches lean |
| B2 | `rapamycin_rk4_budget` | `2^63` (×3 comps) | `1e30` | `k=1.960`, Budget64 `std` matches |

**Count:** **2** surfaces (one shared printer: `stdlib/epistemic/budget64.sio` `println(b.effective_dof)`).  
**Hypothesis (not yet root-fixed):** large f64 sentinel `1e30` hits an integer trunc/print path on Madaros (x86 `cvttsd2si` indefinite → `0x8000…` → printed as `2^63`), not the KCONF field-kind split (that would sitofp the IEEE bits of `1e30` → ~5.06e18, which we do **not** see).

Coverage factor and expanded uncertainty remain numerically aligned with lean on these runs — the **lie is the DOF line**, not (yet) the `k×u_c` product. Still thesis-hostile if a panel quotes Effective DOF.

### Family C — **false positives from naive `0.000000` grep** (not fabrication)

| Surface | Why dismissed |
|---|---|
| `epistemic_pbpk28` | Legitimate zero sensitivity / mass lines; confidence healthy `0.671038` |
| `epistemic_pbpk28_hessian` | Index/CSV zeros + `rho_literal=0` on both engines |
| `tacrolimus_trough_gum` | Tiny variance fractions print as `0.000000` on Madaros; lean shows `2e-11` — **display**, and both PASS; not GUM collapse |

---

## 3. Surfaces measured green (no A/B fabrication)

Among others, dual-engine clean on fabrication symptoms:  
`rapamycin_epistemic_pbpk`, `rapamycin_gum_vs_mc`, `epistemic_confidence_print_probe`, `glp1_gipr_gum`, `vancomycin_auc_gum`, `dissertation_{tirzepatide,vancomycin,pbpk_rapamycin,demo,oral_pd}`, `halo_pgx_gate_pass`, `biomaterial_release`, `des_sirolimus`, `cross_drug_iso_budget`, `olanzapine_d2_mtor`, `pop_pbpk_pd`, `madaros_knowledge_value_mul`.

**Resource ceiling (rc=182), not fabrication:** `d2_gum`, `gum_vs_mc` (validation), `rapamycin_clinical`, `pop_sim` — Madaros dies on handle table; lean PASS. Orthogonal to F1/F2/B.

**Other Madaros rc≠0 without fabrication match:** `madaros_gum_fo_knowledge_ops` rc=139; `dissertation_steady_state_demo` rc=1 — triage separately.

---

## 4. Headline numbers

| Question | Answer |
|---|---|
| How many **real** fabricating surfaces remain (post-#1882)? | **3 unique files**: adaptive (A1), rk4_budget (A1+B2), iso_budget (B1) |
| How many **causes**? | **Two**: (A) Knowledge/GUM variance collapse; (B) Budget64 infinite-DOF print/`1e30`→`2^63` |
| Does one fix close several? | **Yes within each family:** A likely one Madaros GUM/Knowledge lowering fix → adaptive + rk4 Knowledge zeros; B one Budget64 DOF print/cast fix → all Effective DOF lines |
| Is F2 (4.6e18 confidence) still open? | **No** on measured surfaces under source-built Madaros |

---

## 5. Next owners (do not conflate)

1. **Family A (thesis-critical):** Madaros Knowledge variance / `variance_of` through ODE — start from A1 (`rapamycin_epistemic_adaptive`) with lean non-zero as control. May need `lower.sio` / GUM lowering; claim before edit.  
2. **Family B:** `budget64` Effective DOF print path for `1e30` sentinel — prefer printing as `inf`/`1e30` without integer trunc; witness `budget64_test` T2 + iso/rk4 DOF lines.  
3. **Do not** reopen KCONF / weaken E170 / treat rc=182 as fabrication.

---

## 6. Reproduce

```bash
export MADAROS_STACK_KB=524288 SOUNIO_STDLIB_PATH=$(pwd)/stdlib
# Madaros must be source-built at a tip containing #1882
./bin/souc run tests/run-pass/rapamycin_epistemic_adaptive.sio | rg 'var\(|FABRIC'
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/rapamycin_epistemic_adaptive.sio | rg 'var\(|PASS'
./bin/souc run tests/run-pass/rapamycin_iso_budget.sio | rg 'Effective DOF|922337|1e\+30'
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/rapamycin_iso_budget.sio | rg 'Effective DOF|1e\+30'
```

Logs: `/tmp/fab-sweep/{madaros,lean_single}/`.

---

## 7. AI disclosure

Sweep and partition by AI agent (grok-cli2) under human direction after #1882. GAIDeT-ICMJE 2025. Numbers re-derived on this worktree; prebuilt ELF without #1882 must not be used to refute F2 closure.
