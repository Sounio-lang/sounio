<!-- docs:meta
topic_id: repo.docs.audit.rc182-d2-headroom-contradiction-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.rc182-d2-headroom-contradiction-2026-08-18
-->

# rc=182 on d2_gum / d2_voi “with 2.36M headroom” — resolved

**Date:** 2026-08-18  
**Lane:** grok-cli2 / `handle-182-d2-contradiction`  
**Trigger:** minimax-cli3 [`RC182_DIAGNOSTIC_GAP_AND_INSTRUMENT_2026-08-17.md`](../../../../.wt/minimax-cli3/docs/audit/RC182_DIAGNOSTIC_GAP_AND_INSTRUMENT_2026-08-17.md) reported `d2_gum` / `d2_voi` exiting **182** at `peak_handle_count = 1 835 008` (≈43.8% of 2²²) with **`pin_count = 0`**, while the other three wall-hitters showed `peak_handle ≈ peak_pin ≈ capacity`. That would contradict “182 = capacity reached.”

**Prior own work:** alloc/live profiles ([`MADAROS_HANDLE_182_ALLOC_VS_LIVE_2026-08-17.md`](MADAROS_HANDLE_182_ALLOC_VS_LIVE_2026-08-17.md)) had mmap-base reads of d2_* climbing to ~4 194 k. This note decides which instrument is telling the truth.

**Discipline:** no edits to `lower.sio` / `codegen_x86_linux.sio` / `gc.sio` / `runtime_context.sio`. Read-only source + OrangeFS probes. Madaros **from source** (`$STAGE/build/madaros`, commit `d0c798e4ed`).

---

## 1. The three hypotheses (as posed)

| ID | Hypothesis |
|---|---|
| **H1** | The counter that trips 182 is **not** the `handle_count` the probe reads |
| **H2** | There is a **second, tighter limit** that nobody named |
| **H3** | Some other condition exits with code **182** (reused code) |

---

## 2. What the codegen actually does (source)

Active native path: `self-hosted/native/codegen_x86_linux.sio`.

Handle slow path (both MIR and core-IR emitters):

1. Compare `handle_count + 1` against `handle_capacity` (`setae` → jump).
2. On overflow: `native_v2_emit_gc_request_metadata(..., handle_table_full)` then **`emit_exit(182)`** (or `nc_core_emit_alloc_fail_into(..., 182)` which prints `madaros: handles full` then exits 182).

There is **no other `emit_exit(..., 182)`** on this backend outside that handle-full path.

Sibling: `codegen.sio` (non-x86_linux) attempts pin/live probe + empty-frame reset before failing 182. That path is **not** what these ELFs use. On x86_linux, `runtime_context.pin_count` is **initialised to 0 and never incremented** (only per-entry `pin_count` fields are zeroed on alloc; the context-level counter stays 0). So a true mmap-base read of `pin_count` **must** be 0 for every program on this backend.

---

## 3. Dual-instrument re-measure (same ELFs, same source Madaros)

Two readers in one process:

- **MMAP-base:** 2 GiB anon mapping start = runtime context (entry trampoline). Read `hc@+32`, `cap@+40`, `pin@+72`.
- **MAGIC-scan (minimax-style):** search readable mappings **≤256 MiB** (explicitly skips the 2 GiB mmap) for the u64 `4194304`, treat `addr-40` as a “context”, apply optional `hc < cap` filter.

### Controls

| Probe | MMAP last hc / pin | MAGIC `under_hc` (hc&lt;cap) | Note |
|---|---|---|---|
| Managed N=10 000 | **10 000 / 0** | **1 835 008** (and garbage pins) | Magic “under” number appears **even when true hc=10k** |

**`1 835 008` is a phantom.** It is not d2’s peak. It is a recurring false candidate from scanning for the capacity immediate outside the real context.

### The five wall-hitters

| Test | MMAP first→last hc | MMAP pin | MAGIC last `under_hc` | rc |
|---|---|---|---:|---:|
| `d2_gum` | 49 → **4 193 876** | **0** | **1 835 008** | **182** |
| `d2_voi` | 59 → **4 193 811** | **0** | **1 835 008** | **182** |
| `rapamycin_clinical` | 66 → **4 193 752** | **0** | 4 186 767 | **182** |
| `rapamycin_pop_sim` | 21 → **4 193 889** | **0** | 4 192 256 | **182** |
| `gum_vs_mc` | 66 → **4 193 723** | **0** | 4 192 256 | **182** |

Capacity = 4 194 304. All five MMAP traces are monotonic climbs from tens to within a few hundred of capacity, then 182. **d2_* are not special.**

Receipts: `$STAGE/probes/contradiction/{d2_gum,d2_voi,clinical,pop20,gum,ctrl10k}_both.log`.

---

## 4. Why minimax’s GAP table said otherwise

From their own [`CURVE_REPORT_5_TESTS.md`](/tmp/handle-instrument/5tests/CURVE_REPORT_5_TESTS.md) (already on disk):

> For d2_gum and d2_voi, the probe sees only the 1,835,008 stack candidate because the other stack addresses are already at or above capacity … the `hc < hcap` filter rejects them.

And:

> The fact that they exit rc=182 means the actual handle_count DID exceed capacity on those candidates before the probe could record it.

The **GAP** doc then treated `1 835 008` as a real peak and inferred a “different failure mode.” That inference does not survive the control (phantom 1 835 008 on a 10k program) nor the mmap-base climb on d2_*.

### On `pin_count ≈ handle_count` for the other three

On the live x86_linux backend, context `pin_count` is never bumped → true pin is **0** for all five (measured). Minimax’s `peak_pin ≈ 4.2M` on the Tsit5 trio is therefore also a **wrong-candidate / race** read (their honest limits section already warns about stack copies and half-written u64s). It is **not** evidence that live set equals capacity, and it must not be used to argue “reclamation cannot help.”

Logical-live bounds from the alloc/live audit (source structure of `tsit5_step_pbpk`, \(R \gtrsim 10^{4}\)) still stand; they never depended on `pin_count`.

---

## 5. Verdict on H1 / H2 / H3

| Hypothesis | Verdict | Evidence |
|---|---|---|
| **H1** (wrong counter) | **Rejected for the runtime; accepted as probe bug** | MMAP `handle_count` reaches capacity on all five; 182 sites compare that same field. The “2.36M headroom” number is magic-scan `under_hc=1835008`, a phantom also seen on the 10k control. |
| **H2** (second tighter limit) | **Rejected** | No second ceiling in the 182 path; d2_* asymptote to the same 2²² wall as the other three. |
| **H3** (182 reused for another condition) | **Rejected on x86_linux** | Only handle-full slow paths emit 182; d2_* show the same monotonic growth-to-wall shape. |

**Bottom line:** d2_gum / d2_voi **do** hit the handle-table ceiling. The apparent headroom was an instrumentation artefact (capacity-magic scan + `hc < cap` filter + stack candidates), amplified by treating `pin_count` as a live proxy on a backend that never updates it.

The wall-family conclusion is **not** overturned. What is overturned is the GAP doc’s claim of a second d2 failure mode based on those two metrics.

---

## 6. Implications for the fleet

1. **Do not design a second bug around d2_*.** Same reclaim problem.  
2. **Prefer mmap-base (or `.data` slot → context pointer) over capacity-magic scans** for external diagnostics; if magic-scanning, **must include `hc >= cap` candidates** and must not treat pin_count as live on x86_linux until something actually increments it.  
3. **In-binary print** of `count/capacity/reason` at the 182 site (minimax’s fix spec) remains the right durable diagnostic — it would have prevented this contradiction.  
4. Prior alloc/live ratios for the five remain the reclaim case; ignore pin≈handle from the fragile scan.

---

## 7. Non-claims

- Does not implement in-binary diagnostics or reclamation.  
- Does not assert minimax’s instrument is useless — the GAP (numbers exist but are not printed) is real; only the d2 headroom **interpretation** is wrong.  
- Does not re-open wrap-vs-ceiling (still a real fail-closed compare).

---

## 8. Document control

| Date | Change |
|---|---|
| 2026-08-18 | Dual-instrument resolve of d2_* “headroom” contradiction; phantom 1835008 shown on 10k control; pin=0 on true context for all five. |
