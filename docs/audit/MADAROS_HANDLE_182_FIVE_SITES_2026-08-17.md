<!-- docs:meta
topic_id: repo.docs.audit.madaros-handle-182-five-sites-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-handle-182-five-sites-2026-08-17
-->

# Angle 4 — the five rc=182 sites: one wall, three topologies

**Date:** 2026-08-17  
**Lane:** grok-cli2 / `handle-182-five-sites`  
**Question:** after #1799, five suite members compile (`Written to a.out`) and die with `madaros: handles full` / exit **182**. Is that the **same** handle consumption everywhere, or is `rc=182` a convenience bucket hiding distinct sinks — the way `lower_array` hid println-SEGV vs handle-wall?

**Answer in one line:** **one mechanism / one wall**; **three call-site topologies** that all feed the same unreclaimed bump allocator. Not five independent bugs. Not one identical call stack either. `d2_gum` and `d2_voi` share the **identical** primary sink.

**Instrument:** Madaros built from source at `d0c798e4ed` (`$STAGE/build/madaros`, md5 `fe91a596…`). Never the prebuilt `bin/souc` alone.  
**Stage:** `/orangefs/training/pbpk-suite-remeasure-d0c798e4edcd-20260817T221419Z` (+ `$STAGE/probes/`).

Prior art (do not re-derive): #555 capacity raise, #651 wrap at 2²⁰, `3da75acd31` (2²⁰→2²², “moves the wall”), unreclaimed bump + unwired `native_v2_emit_gc_empty_frame_reset`, fable-1 dispatch `MADAROS_HANDLE_TABLE_182_LIFETIME_DISPATCH_2026-08-17.md` (branch, mechanism confirmed).

---

## 1. The wall (shared — confirmed again today)

From `self-hosted/native/gc.sio` on current main:

- `native_v2_handle_table_capacity_default() = 4194304` (**2²²**)
- Alloc path: monotonic bump of `RuntimeContext.handle_count` in `nc_core_emit_alloc_into` (`codegen_x86_linux.sio`)
- Overflow → diagnostic `madaros: handles full` → `emit_exit(182)`
- Reset emitter exists and is **deliberately unwired** (stack maps are counts, not per-slot root bitmaps)

### Exact-capacity witnesses (source-built Madaros, this stage)

| Program | N | rc | Note |
|---|---:|---:|---|
| `struct S {a,b,c: f64}` (24 B, managed) | **4 194 303** (=2²²−1) | **0** | `acc=4194303` |
| same | **4 194 304** (=2²²) | **182** | `handles full` — exact slot |
| `struct P {a,b: f64}` (16 B, unboxed) | 5 000 000 | **0** | no handle slots |

Live set in the managed loop is always **1**. The wall is **cumulative lifetime allocations**, not working-set size. That is the whole runtime bug; the five dissertation sites are callers that cross it.

---

## 2. The five sites — death phase (remeasure + re-run)

| # | Test | Last honest progress | Dies in |
|---:|---|---|---|
| 1 | `rapamycin_clinical` | PART A clinical PASSes | PART B GUM FD budget (before table print) |
| 2 | `gum_vs_mc` | Scenario-1 GUM prints `SD_GUM` | first MC loop (default `n_mc=10`) |
| 3 | `rapamycin_pop_sim` | banner “Running N virtual patients…” | patient loop |
| 4 | `d2_gum` | banner only | **`d2_gum_build`** (before any TEST) |
| 5 | `d2_voi` | banner only | **`d2_gum_build`** — VoI never starts |

**`d2_voi` is not a VoI bug.** Its `main` is:

```text
let budget = d2_gum_build(priors)   // ← dies here
let voi    = d2_voi_from_budget(budget)
```

Same primary sink as `d2_gum`. Counting them as two independent 182 families would repeat the `lower_array` labelling error.

---

## 3. Scaling — is it the same consumption?

### 3.1 Population (`rapamycin_pop_sim`) — outer multiplicity of one integrate

Same `pop_simulate` / 14-comp adaptive Tsit5 / `t_end=168h`, vary `n_patients` only.

| `n_patients` | handles full? | Completes integrate loop? |
|---:|---|---|
| 1–7 | **no** | yes (may fail 2 assertion tests; not 182) |
| **8–20** | **yes** | dies during patient loop |

≈ **7 full patient integrates** fit under 2²²; the 8th crosses the wall. Order-of-magnitude: ~0.5–0.6×10⁶ managed allocations per patient-length integrate. Same simulate body as the clinical/MC paths; only the outer count changes.

### 3.2 GUM vs MC (`gum_vs_mc`) — GUM fixed + MC multiplicity

Patch both `mc_auc_sd(..., n_mc, seed)` call sites (`n_mc` is the literal `10` before the seed).

| `n_mc` | Result |
|---:|---|
| 0, 1 | **no** 182; both scenarios finish (tests may fail for sampling) |
| 2 | **no** 182; finishes |
| **3+** | **182** — after Scenario-1 (GUM+MC) and Scenario-2 GUM, dies entering Scenario-2 MC (for `n_mc=3`); for `n_mc=10` dies already in Scenario-1 MC |

So MC samples of `gmc_simulate` are the same *class* of long integrate as population patients. GUM’s 4 sims alone do **not** exhaust the table; adding a few MC samples after GUM does. One wall; outer MC count is the lever.

### 3.3 Clinical GUM (`rapamycin_clinical`) — FD wrapper

PART A: one `cv_simulate` (PASS).  
PART B: `n_params = 7`, one-sided FD → up to **7 more** `cv_simulate` via `cv_perturb` → fresh `PBPKParams14` (≈44×f64 ≈ 352 B → **managed**) each time. Dies at the start of PART B — baseline + early FD already near capacity (same integrate family as pop/MC).

### 3.4 D2 GUM / VoI — chronic oral BBB FD budget

`d2_gum_build`: 1 reference + 2×9 FD trough runs = **19×** `d2g_run_trough_72h` → `oral_halo_bbb_run_params` (72 h chronic). Dies before printing the budget. Heavier per-call than a single 168 h Tsit5 patient in practice (or more managed traffic per step); either way it is still **managed allocs in an unreclaimed bump**, not a second exit code.

---

## 4. Verdict: family vs convenience bucket

| Claim | Verdict |
|---|---|
| Five different runtime bugs | **False** |
| One identical call stack | **False** (`d2_*` ≠ `pop_sim` ≠ clinical PART B) |
| One mechanism (unreclaimed handle bump → 182 at 2²²) | **True** |
| `rc=182` as a suite “family” | **True as a wall family**; **false if read as one call site** |
| Closest analogy to `lower_array` | Label was over-broad once; here the label **is** the mechanism — but it still **aggregates three topologies** |

### Topology map (the useful split)

| Topology | Members | Multiplier pattern |
|---|---|---|
| **T1 — FD GUM around long integrate** | `rapamycin_clinical` PART B; `d2_gum` / `d2_voi` via `d2_gum_build` | ~(1+k·N) integrates; large param structs returned by value |
| **T2 — independent full integrates in an outer loop** | `rapamycin_pop_sim`; `gum_vs_mc` MC arm | K × `*_simulate` |
| **Shared primitive** | all five | `nc_core_emit_alloc_into` managed path; `PBPKParams14` / BBB param bundles / step metrics >16 B |

**One fix closes the wall for all three topologies** (reclamation / precise roots — fix B/A in the lifetime dispatch).  
**Caller discipline (fix C)** can be topology-specific (`&!` step_mut, ≤16 B hot structs, lower K) and may green individual tests without closing the compiler debt.

Raising the ceiling again is **not** a fix (already moved 2²⁰→2²²; still open).

---

## 5. What this means for the suite narrative

From the pbpk_suite re-measure (`DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-17.md`):

- Cluster L (10× SIGSEGV 139) → **5 PASS + 5×182**
- Resource-ceiling “family” grew **2 → 7** (5 ex-L + 2 MC N=2000)

Angle 4 adds: those five (and by extension the two MC N=2000 siblings) are **not** five root causes. They are **one lifetime wall** stressed by **T1/T2 multipliers**. Treating “rc=182” as a single repair ticket is correct at the runtime layer; treating it as “fix d2_voi” or “fix pop_sim” without touching the bump allocator is the convenience-bucket mistake.

---

## 6. Receipts

```text
STAGE=/orangefs/training/pbpk-suite-remeasure-d0c798e4edcd-20260817T221419Z
$STAGE/probes/h_exact_cap.log      # N=2^22 → 182
$STAGE/probes/h_under_cap.log      # N=2^22-1 → 0
$STAGE/probes/h_unboxed.log        # 5e6 × 16B → 0
$STAGE/probes/pop2_n{6,7,8,9}.log  # threshold 7→8 patients
$STAGE/probes/gmc2_k{0,1,2,3,5,10}.log
$STAGE/probes/d2_{gum,voi}_rerun.log
$STAGE/probes/clinical_rerun.log
```

Compiler measured: `$STAGE/build/madaros` from `scripts/ci/build_modular_madaros.sh` on commit `d0c798e4ed`.

---

## 7. Non-claims

- Does not implement reclamation.
- Does not assert exact handles-per-step for every ODE stage (order-of-magnitude from patient scaling only).
- Does not merge fable-1’s dispatch doc; cites it as prior mechanism work.
- Does not reopen #555/#651 history beyond the binding fact: capacity sits at a pure power of two with **no** reclaim.

---

## 8. Document control

| Date | Change |
|---|---|
| 2026-08-17 | Angle 4: five-site topology map; exact 2²² witness; pop 7↔8 and gum n_mc scaling; d2_voi≡d2_gum_build. |
