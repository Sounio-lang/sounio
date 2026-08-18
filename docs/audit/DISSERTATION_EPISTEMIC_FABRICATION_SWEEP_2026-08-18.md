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

Lean shows real variance; Madaros printed exact `0.000000` / `std(Knowledge)=0` on multi-RHS surfaces.

| # | Surface | Before (Madaros) | lean_single | After contour (Madaros) |
|---:|---|---|---|---|
| A1 | `rapamycin_epistemic_adaptive` | `var=0` + `FABRICATED_ZERO` | non-zero; PASS | **non-zero + PASS** (RHS inlined) |
| A2 | `rapamycin_rk4_budget` | `var=0`; silent | non-zero; PASS | **non-zero + PASS** (main-loop RHS inlined) |

#### Root (discriminated 2026-08-18) — **not** the April ζ slot OOB alone

| Probe | Madaros `variance_of` | lean |
|---|---|---|
| Inline Euler-like, 3 channels (iso mini) | **non-zero** | non-zero |
| Same math via **user `fn rhs(...)`** (CALL3) | **`0`** | non-zero |
| `rapamycin_iso_budget` (RHS **inlined** in `main`) | non-zero historically | non-zero |
| adaptive / rk4 (RHS behind **calls**) | `0` before contour | non-zero |

**Cause:** Madaros first-order GUM / FO variance **does not survive user function call boundaries**. lean_single does. The `#1706` 1024 `variance_base_reg` silent-drop → `0.0` remains a related honesty hazard but is **not** required to explain A1/A2: a one-call, 20-step probe already zeros.

**Contour (thesis surfaces):** inline `rhs_*` at call sites (same shape as iso). Compiler follow-up: interprocedural FO transfer. Regression: `tests/run-pass/gum_fo_across_call.sio` (to land with this wave).

**Not closed by #1882.** Contour closes A1/A2 science surfaces; compiler FO-across-call remains open.

### Family B — **large-f64 print saturation → `2^63`** (Madaros-only)

Budget64 Type-B / Welch fallback DOF is stored as `1.0e30`. lean_single prints `1.000000e30`. Madaros prints `9223372036854775808.000000` (`2^63` as f64) on iso/rk4 Effective DOF lines.

| # | Surface | Madaros DOF print | lean DOF print | Expanded `k` / `U` |
|---:|---|---|---|---|
| B1 | `rapamycin_iso_budget` | `2^63` (×2) | `1e30` | `k=1.960`, `U` matches lean |
| B2 | `rapamycin_rk4_budget` | `2^63` (×3) | `1e30` | `k=1.960`, Budget64 `std` OK |

#### `2^63` is ambiguous — two known diseases, same glyph

Project memory already names a **`variance_of` → `2^63`** failure (2026-04-12, `docs/research/zeta_variance_fix_plan.md` / `zeta_variance_deep_investigation.md`): GUM variance slot buffer 1024 entries; deep while-loops (Bogacki adaptive) allocate past the bound; OOB / uninitialized BSS read appears as `0x8000…` = `2^63`. That disease is **depth-dependent** (1-stage ODE safe; adaptive ~10k slots overflows). Contours historically: in-place mutation, lookbehind.

Family B as first written looked like “Budget64 DOF printer only.” A **single-stage discriminant** (no `variance_of`, no loop chain) decides otherwise:

| Probe | Stages / depth | Madaros | lean_single |
|---|---|---|---|
| `print_f64(1.0e30)` / `println(1.0e30)` | **1**, no GUM | **`2^63`** | `1e30` |
| local `effective_dof = 1e30` then `println` | **1** | **`2^63`** | `1e30` |
| Magnitude scan | **1** | `1e15`/`1e18`/`9e18` OK; **`≥1e19` → `2^63`** | all correct |
| KCONF sitofp of IEEE(`1e30`) | n/a | would be ~**5.06e18** | — |

**Verdict: Family B ≠ April `variance_of` overflow.** Same observable (`2^63` in a log), **two causes**:

1. **April / ζ** — variance slot OOB on deep chains (depth-dependent).  
2. **B / print** — Madaros `print_f64`/`println` of magnitudes **above ~i64 max (~9.22e18)** saturates (consistent with `cvttsd2si` indefinite → print as `2^63`). **Deterministic, single-stage.** Budget64 DOF `1e30` is one consumer, not the root.

**Operational rule:** a `2^63` in a log does **not** tell you which disease fired. Fixing the DOF/print path will not clear a deep-chain `variance_of` `2^63`, and fixing slot reset will not clear `print_f64(1e30)`. Anyone who sees a residual `2^63` after one fix must re-run the discriminant table above before concluding the fix failed.

**Note on Family A vs April:** today’s A1/A2 show **collapse to `0.000000`**, not `2^63`. The fabrication-detect note already separates “`variance_of` overflow `2^63` on deep chains” from “F1 collapse to 0 under Madaros adaptive.” Collapse-to-zero may be a sibling of the slot-bound silent return (`slot >= 1024` → no write), but it is **not** the same observable as April’s garbage `2^63`. Do not merge A into the April note without a depth-controlled variance probe.

Coverage `k` / `U` on iso/rk4 still match lean — the B lie is the **DOF print** (and any other ≥1e19 f64 print), not yet the expanded-uncertainty product.

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
| How many **causes**? | **Three named** if counting April ζ separately: (A) Knowledge/GUM variance **collapse to 0**; (B) Madaros **large-f64 print** ≥~1e19 → `2^63`; (ζ) historical deep-chain `variance_of` OOB → `2^63`. B ≠ ζ (discriminant above). |
| Does one fix close several? | **Yes within each family:** A → adaptive + rk4 Knowledge zeros; B → all `print_f64` of ≥1e19 including Budget64 DOF `1e30` |
| Is F2 (4.6e18 confidence) still open? | **No** on measured surfaces under source-built Madaros |
| Is every log `2^63` the same bug? | **No — ambiguous.** Re-run P1 `print_f64(1e30)` vs a deep `variance_of` chain before attributing. |

---

## 5. Next owners (do not conflate)

1. **Family A (thesis-critical, #1792 name):** Madaros Knowledge variance / `variance_of` through ODE — start from A1 (`rapamycin_epistemic_adaptive`) with lean non-zero as control. May need `lower.sio` / GUM lowering; claim before edit.  
2. **Family B (cheaper, less grave):** Madaros `print_f64`/`println` for `|x| ≳ 1e19` — fix once; Budget64 DOF is a client. Witness: one-stage `print_f64(1.0e30)`.  
3. **April ζ** — if deep-chain `variance_of` still yields `2^63` under current Madaros, track separately from B; do not close ζ by fixing print.  
4. **Do not** reopen KCONF / weaken E170 / treat rc=182 as fabrication.

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
