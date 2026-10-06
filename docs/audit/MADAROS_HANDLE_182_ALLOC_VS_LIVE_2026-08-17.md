<!-- docs:meta
topic_id: repo.docs.audit.madaros-handle-182-alloc-vs-live-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-handle-182-alloc-vs-live-2026-08-17
-->

# Handle profile of the five rc=182 sites — allocated vs live

**Date:** 2026-08-17  
**Lane:** grok-cli2 / `handle-182-profile-five`  
**Question:** for each of the five post-#1799 wall-hitters, how many handles are **allocated**, how many are **live at once**, and what is the **ratio**? Does that ratio (or the shape of consumption) differ enough that `rc=182` is a convenience bucket hiding distinct causes?

**Files touched:** this audit only (+ OrangeFS probe copies under `$STAGE/probes/profile/`). **Not** `lower.sio`, **not** `codegen_x86_linux.sio`, **not** `runtime_context.sio` (those claims are held by other lanes).

**Compiler:** source-built Madaros `d0c798e4ed` (`$STAGE/build/madaros`).  
**Stage:** `/orangefs/training/pbpk-suite-remeasure-d0c798e4edcd-20260817T221419Z/probes/profile/`.

---

## 0. Two meanings of “live” (must not conflate)

| Sense | Meaning under native-v2 today |
|---|---|
| **Table occupancy** | `RuntimeContext.handle_count` — monotonic bump; **nothing is freed**, so occupancy = lifetime allocations |
| **Logical live** | Managed objects still needed for the continuing computation (what a sound collector could keep) |

Because reclamation is unimplemented, **table occupancy never falls**. The ratio that decides whether Fix B/A pays off is:

\[
R = \frac{\text{allocated (handle\_count)}}{\text{peak logical live}}
\]

If \(R \gg 1\), the wall is lifetime garbage. If \(R \approx 1\), the program truly needs ~capacity live and reclaiming dead frames will not save it.

---

## 1. Instrument (validated before use)

Sampler: Python fork + `/proc/<pid>/mem` read of the 2 GiB anon mmap base +32 (`handle_count`) / +40 (`capacity`). Context layout from `runtime_context.sio` (read-only). No compiler edits.

### Positive / negative controls (refutation suite)

| Probe | Expectation | **Measured LAST hc** | Verdict |
|---|---|---:|---|
| Managed `S{f64×3}`, N=10 000 | hc = 10 000 | **10 000** | instrument fires |
| Managed `S{f64×3}`, N=50 000 | hc = 50 000 | **50 000** | |
| Keep 2 logical live (`prev`+`s`), N=20 000 | hc ≈ 20 001 | **20 001** | cumulative still tracks allocs, not live |
| Unboxed `P{f64×2}`, N=50 000 | hc = 0 | **0** | no false positives |

A census that returned 0 on the managed controls would have been discarded. It did not.

---

## 2. Hard numbers — the five sites

Capacity on this binary: **4 194 304** (=2²²).

### 2.1 At death (full suite parameters)

| Site | rc | **max/LAST handle_count** | Notes |
|---|---:|---:|---|
| `rapamycin_clinical` | 182 | **4 194 138** | dies in PART B GUM |
| `gum_vs_mc` (full) | 182 | **4 194 113** | dies in MC after GUM |
| `rapamycin_pop_sim` (n=20) | 182 | **4 194 196** | dies in patient loop |
| `d2_gum` | 182 | **4 194 141** | dies in `d2_gum_build` |
| `d2_voi` | 182 | **4 194 216** | dies in `d2_gum_build` (VoI never starts) |

All five asymptote to the **same ceiling** (within ~200 of capacity). The previously cited “>3 000 001” is consistent with a mid-run sample, not a different wall.

### 2.2 Scaling — consumption rate (where the profiles differ)

| Configuration | LAST hc | rc | Implied rate |
|---|---:|---:|---|
| `pop_sim` **n=1** | **559 963** | 2 (assert fails, not 182) | ≈ **5.60×10⁵ handles / patient** |
| `pop_sim` **n=3** | **1 671 921** | 2 | 3 × 5.57×10⁵ (linear) |
| `pop_sim` **n=7** | **3 910 075** | 2 | 7 × 5.59×10⁵ (linear) |
| `pop_sim` n=20 | 4 194 196 | **182** | crosses capacity between patient 7 and 8 |
| `gum_vs_mc` **n_mc=0** (GUM-only, 2×4 sims) | **2 558 201** | 4 | ≈ **3.20×10⁵ handles / GUM sim** |
| `gum_vs_mc` full | 4 194 113 | **182** | GUM + MC tips over |

**Rates are not identical** (patient-length Tsit5 ≈ 560 k; GUM sim ≈ 320 k). That is expected: different step counts / call graphs into the same `tsit5_step_pbpk` / BBB integrate family. It is **not** evidence of a second exit mechanism.

### 2.3 Where the handles come from (shared micro-sink)

`pop_simulate` / `cv_simulate` / `gmc_simulate` all drive adaptive Tsit5 over `PBPKState14` (14×f64 = 112 B → **managed**).

`tsit5_step_pbpk` builds stages with nested `pbpk_state_add` / `pbpk_state_scale` / `pbpk_ode`, each returning a **fresh** `PBPKState14`. Per accepted step this is **O(10¹–10²) managed allocations**, of which only the final `Tsit5StepResult14` (2 states) is retained by the caller; the stage temporaries are logically dead immediately.

Order-of-magnitude check: 560 k handles / patient ÷ ~2 000–5 000 steps ⇒ ~100–280 handles/step — compatible with the nested stage construction.

`d2_gum` / `d2_voi` multiply a chronic oral-BBB integrate (~19 trough runs in `d2_gum_build`) instead of 14-comp Tsit5, but still feed the **same unreclaimed bump**.

---

## 3. Peak logical live (bound)

Precise live-set counting needs a root bitmap (the missing piece). Bounds from source + controls:

| Phase | Peak logical managed (bound) | Why |
|---|---:|---|
| Inside `tsit5_step_pbpk` | **≲ 40** | live stage bindings + `st_in` + `prm`; not thousands |
| Across patients / MC samples | **≲ 50** | prior patient state abandoned; only scalar AUC arrays retained |
| Control A (N managed, live=1) | **1** | measured pattern: hc=N, live=1 |
| Control C (explicit live=2) | **2** | hc=N+1 |

So for one pop patient:

\[
R_{\text{patient}} \approx \frac{560\,000}{\lesssim 40} \gtrsim 10^{4}
\]

For pop n=7 completing under the wall:

\[
R \approx \frac{3.91\times 10^{6}}{\lesssim 50} \gtrsim 10^{4}\text{–}10^{5}
\]

For every wall-hitter at death:

\[
R \approx \frac{4.19\times 10^{6}}{\lesssim 50} \gtrsim 10^{5}
\]

**The ratio is enormous for all five.** Almost all allocated handles are logically dead. That is the reclamation case.

---

## 4. Is `rc=182` a bucket or a family?

Apply the `lower_array` lesson explicitly.

| Hypothesis | Evidence | Keep? |
|---|---|---|
| Five independent runtime bugs | Same exit, same capacity asymptote, linear scaling with outer multiplicity | **No** |
| One call stack for all five | Rates differ (560 k/patient vs 320 k/GUM-sim); `d2_*` vs Tsit5 entrypoints differ | **No** |
| One mechanism (unreclaimed bump) stressed by different multipliers | All data above | **Yes** |
| Distinct **logical-live** profiles that would need different fixes | All have \(R \gg 1\); Fix B/A helps the same way | **No meaningful split for reclaim** |
| Distinct **caller** profiles for Fix C (source discipline) | Yes — Tsit5 `&!` / unbox states vs fewer FD/MC iterations | **Yes, for workarounds only** |

**Verdict:** `rc=182` is a **real family at the runtime layer** (one wall). It is a **convenience bucket only if** you pretend the five call sites are one stack or five unrelated bugs. The **alloc/live ratio does not split them into different causes** — it unifies them: all are lifetime-garbage amplifiers of `tsit5_step_pbpk`-class (or BBB-integrate-class) managed traffic.

`d2_voi` remains the same primary sink as `d2_gum` (`d2_gum_build`); its “VoI” label is still a triage fiction at the failure point.

---

## 5. What this says to the reclamation lanes

1. **Reclamation is the right global fix** — \(R \sim 10^{4}\)–\(10^{5}\) means almost everything in the table is reclaimable in principle.  
2. **Frame/loop watermark (Fix B)** matches the measured shape: per-step / per-patient garbage with tiny retained set.  
3. **Ceiling raises will not change \(R\)** — they only delay the same ratio.  
4. **Do not special-case `d2_voi`** as a separate reclaim problem.  
5. Caller-side Fix C remains valid **per topology** (in-place `PBPKState14` / fewer nested by-value adds in Tsit5) and would drop the **rate**, not the mechanism.

---

## 6. Receipts

```text
$STAGE/probes/profile/
  sample_hc.py
  A_n10k_hc.log B_n50k_hc.log C_live2_hc.log D_unboxed_hc.log   # instrument validation
  pop_n1_hc.log pop_n3_hc.log pop_n7_hc.log pop_n20_hc.log
  gum_k0_hc.log gum_full_hc.log
  clinical_hc.log d2_gum_hc.log d2_voi_hc.log
```

Madaros: `$STAGE/build/madaros` from `build_modular_madaros.sh` on `d0c798e4ed`.

---

## 7. Non-claims

- Did not implement reclamation.  
- Did not edit shared native/compiler files.  
- Logical-live bounds are from source structure + controls, not a tracing GC census (impossible without the root bitmap the bug lacks).  
- Did not re-litigate wrap vs ceiling (codex-2: no mask; fail-closed 182).

---

## 8. Document control

| Date | Change |
|---|---|
| 2026-08-17 | Alloc vs live profiles for five sites; instrument validation; R≫1 unified reclaim case. |
