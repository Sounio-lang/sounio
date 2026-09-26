<!-- docs:meta
topic_id: repo.docs.audit.madaros-handle-182-bbb-per-construction-dispatch-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-handle-182-bbb-per-construction-dispatch-2026-09-26
-->

# Madaros exit 182 on the BBB `darwin_pbpk` family — a per-construction leak, not a capacity limit (dispatch)

**Date:** 2026-09-26
**Status:** mechanism CONFIRMED on the committed ELF and on a Madaros built
from the tree under test (§2). Root cause is the one already dispatched in
[`MADAROS_HANDLE_TABLE_182_LIFETIME_DISPATCH_2026-08-17.md`](MADAROS_HANDLE_TABLE_182_LIFETIME_DISPATCH_2026-08-17.md)
and is still OPEN. This document is a dispatch: no `self-hosted/` file is changed.
**What is new here:** a measured per-operation cost model (§3), a
quantified per-test demand (§4), and a second unreclaimed resource that sits
behind the first (§5). Together they settle "leak or capacity" (§6) and rule
out a ceiling raise as a fix.
**Minimal repro:** [`docs/handoff/repros/handle_table_182_per_construction_madaros.sio`](../handoff/repros/handle_table_182_per_construction_madaros.sio) (26 lines, no imports).

## 1. Symptom

`./bin/souc run` on these five tests compiles successfully. The program starts,
then exits **182** after printing `madaros: handles full` on stderr. The
`*_OK` sentinel never appears. Under `SOUNIO_SOUC_ENGINE=lean_single` all
five print their sentinel and exit 0.

| Test (`tests/stdlib/darwin_pbpk/bbb/`) | Madaros | lean_single |
|---|---|---|
| `test_bbb_gum_budget.sio` | rc=182, `handles full` | rc=0, `BBB_GUM_BUDGET_OK` |
| `test_bbb_hdmr_7d.sio` | rc=182, `handles full` | rc=0, `BBB_HDMR_7D_OK` |
| `test_bbb_pce2d_sobol.sio` | rc=182, `handles full` | rc=0, `BBB_PCE2D_SOBOL_OK` |
| `test_bbb_pce_vs_gum.sio` | rc=182, `handles full` | rc=0, `BBB_PCE_VS_GUM_OK` |
| `test_des_bbb_coupled.sio` | rc=182, `handles full` | rc=0, `DES_BBB_COUPLED_OK` |

## 2. Where the message comes from

The emitted program's allocator hands every **managed** aggregate a slot in
a fixed handle table. That means any struct over 16 B; structs of 16 B or
less are unboxed, see `native_v2_is_small_value_struct_tag` in
`self-hosted/native/gc.sio`.

- Capacity is `native_v2_handle_table_capacity_default() = 4194304` (2²²),
  at 48 B/slot, carved from the single 2 GiB mmap (`gc.sio`).
- Slots come from a monotonic bump of `RuntimeContext.handle_count` and are
  **never returned**. The slow path fails closed with `emit_exit(182)`,
  reason `native_v2_gc_reason_handle_table_full()`
  (`self-hosted/native/codegen_x86_linux.sio`, alloc slow path).
- Reclamation is deliberately absent, not forgotten. Stack maps carry a
  root-kind mask and slot counts, not a per-slot root bitmap. So no point
  exists at which live handles are known.

`self-hosted/native/gc.sio` and `stack_maps.sio` have not changed since the
2026-08-17 dispatch (`git log --since=2026-08-17 -- self-hosted/native/gc.sio self-hosted/native/stack_maps.sio` is empty).

### Instrument

- Tree: `main` at `2e8b76d312`, worktree `/workspace/worktrees/claude-bbb-handles`.
- **Committed ELF:** `bin/madaros-linux-x86_64`, md5 `57c015c1`. `bin/souc --version` itself says it is not built from the tree.
- **Source-built:** bare `make build-madaros` on the same tree produced `artifacts/self-hosted/madaros`, md5 `5764851f`, which `bin/souc --version` reports as a local build of tree `2e8b76d312`. The build resolved through a content-cache hit (key `4784102d…`, a hash of the source) stored by a concurrent build of identical source. It is invoked only through `./bin/souc`.
- **Every Madaros number in §3–§5 and §7 was taken on both ELFs and is identical on both**, to the same print window: repro, control, p1, p3, q4, q5, q6, t1, t2, t3, z1, and all five tests.
- Every emitted ELF run with `ulimit -s 524288`.
- Probes print a progress counter every *P* iterations. The last counter
  before exit 182 bounds the death point to one *P*-window. So one run gives
  handles-per-iteration *k* = 2²² / *N*_die, with no bisection.

## 3. The cost model: constructions cost a slot, nothing else does

Every probe loops 20,000,000 times unless it dies first. `P` is a 24 B struct
(managed) and `Q` a 16 B struct (unboxed).

| Probe | Operation per iteration | Result | Handles / iter |
|---|---|---|---:|
| p0 | read `p.a` of a loop-invariant `P` | 20M, rc=0 | 0 |
| p1 | pass loop-invariant `P` **by value** to a leaf fn | 20M, rc=0 | 0 |
| p3 | pass `P` by value through **three** nested fns | 20M, rc=0 | 0 |
| p5 | pass `&P` to a leaf fn | 20M, rc=0 | 0 |
| q1 | `s = ode(s, c, prm)`: 16 B return, 88 B `prm` by value | 20M, rc=0 | 0 |
| p2 / p4 / q2 / q3 | 16 B `Q` passed, returned, nested as an argument | 20M, rc=0 | 0 |
| q4 | `let p = mk(1.0)` returns a fresh `P` | dies after i=4,186,112 | **1** |
| q5 | `p = step(p, 1.0)`, `var P` updated from a fn | dies after i=4,186,112 | **1** |
| q6 | `R { st: P, err: P }` returned, `p = r.st` | dies after i=2,088,960 | **2** |

Rule, to the slot: **constructing** a managed aggregate costs one handle.
By-value passing, nesting, field reads and `var` reassignment cost none. Each
dead handle is kept for the rest of the process.

Applied to the real library code:

| Probe (real stdlib) | Result | Handles |
|---|---|---:|
| t1: loop of `tsit5_step_pbpk` (`tsit5_pbpk14.sio`) | dies after step 66,560 (window 512) | **63 per Tsit5 attempt** (2²²/63 = 66,576 ∈ (66,560, 67,072]) |
| t3: loop of `bbb_coupled_run`, 168 h rapamycin, `bbb_dt=0.01` | 7 runs complete, dies inside run 8 | **524,288 – 599,186 per run** |

Static count of `tsit5_step_pbpk`: 7 `pbpk_ode`, 42 `pbpk_state_scale`/`pbpk_state_add`
in stages 2–7, 13 in the error estimate, plus the result. That is 63–64
`PBPKState14` (14 × f64 = 112 B, managed) constructions per attempt, and
t1 measures 63. The BBB sub-model's RK4 step is entirely 16 B `BBBState` and
takes **zero** handles.

Independent cross-check: a lean_single copy of the systemic loop in
`bbb_coupled_run` counts **8,669 Tsit5 attempts** per 168 h run (8,213
accepted + 456 rejected). 8,669 × 63 = **546,147 handles/run**. That falls
inside the t3 bracket, and 2²² / 546,147 = 7.68 predicts death inside run 8.

## 4. Per-test demand

`bbb_gum_budget` = 1 reference + 2 × 7 central-difference runs. HDMR and PCE
add their own node evaluations on top.

| Test | `bbb_coupled_run` calls | Handles needed (× 546,147) | × table |
|---|---:|---:|---:|
| `test_bbb_gum_budget` | 15 | 8.19 M | **1.95** |
| `test_bbb_pce_vs_gum` | 8 (PCE) + 15 (GUM) = 23 | 12.6 M | **3.0** |
| `test_bbb_hdmr_7d` | 1 + 7 × 8 (HDMR) + 15 = 72 | 39.3 M | **9.4** |
| `test_bbb_pce2d_sobol` | 8 × 8 (PCE-2D) + 15 = 79 | 43.1 M | **10.3** |
| `test_des_bbb_coupled` | 1 run, 720 h, continuous stent release | > 4.19 M | > 1 |

`des_bbb_run` has its own loop (`stdlib/darwin_pbpk/scenarios/des_sirolimus_bbb.sio`).
Its single 720 h run already exceeds the table, which implies over 66,576
Tsit5 attempts. Its exact demand was not measured.

The step count is not the lever. 8,213 accepted steps over 168 h is a mean
step of ~0.02 h against `dt_max = 0.5`. Explicit Tsit5 is stability-bound on
the stiff 14-compartment model at `tight_ode_config()` tolerances. That is a
property of the model and solver, not of Madaros.

## 5. A second unreclaimed resource: the heap bump (exit 181)

Unboxed ≤16 B values take no handle, but they are still bump-allocated on
the heap, which is never reclaimed either.

| Probe | Result |
|---|---|
| z1: `let q = mk(1.0)` with 16 B `Q`, one allocation per iteration | rc=**181** `madaros: arena full` after i=40,501,248 |
| t2: loop of the real `bbb_rk4_step` (8 × 16 B values per step) | rc=**181** `arena full` after step 5,062,656 |

5,062,656 × 8 = **40,501,248**: both die at the same allocation count.
Heap = 2³¹ − 2²² × 48 − ctx ≈ 1,946 MB, and 1,946 MB / 40.5 M = **48 B per
unboxed 16 B value** (32 B header + 16 B).

For the BBB tests this wall is behind the handle wall. Each run spends
546k × (32 + 112) B ≈ 79 MB on `PBPKState14`, plus 16,800 × 384 B ≈ 6 MB on
BBB RK4 state, so ~85 MB/run. The heap alone would therefore stop
`pce2d_sobol` (79 runs, ~6.7 GB) and `hdmr_7d` even with an infinite handle
table. A fix that only recycles handle *slots* would turn these 182s into
181s. The `gc.sio` comment's claim that "the heap is not starved" holds for
the handle-table sizing question it answers, not for this workload.

## 6. Verdict: leak, not capacity

- **Live set is O(1); consumption is O(steps).** In t1 the live managed set
  is the loop state plus parameters, a handful of objects. The process still
  dies after 66,576 steps because each step's 62 dead intermediates keep
  their slots. The repro dies with at most ~5 values live.
- **No capacity that fits in the address space covers the family.**
  `pce2d_sobol` needs ~43 M slots, about 2^25.4. `gc.sio` records that the
  handle and heap walls coincide near 2²⁴. Past that the 48 B/slot table
  eats the heap that §5 shows is itself exhausted by these runs.
- The lean_single seed passes all five. Its runtime reclaims, or allocates
  differently; this doc did not investigate which. The defect is confined
  to the native-v2 managed/unboxed allocation path in Madaros.

So the 2²² capacity is where the leak becomes visible, not the defect.

## 7. Minimal repro

[`docs/handoff/repros/handle_table_182_per_construction_madaros.sio`](../handoff/repros/handle_table_182_per_construction_madaros.sio)
is an RK2 step on a 24 B struct. It makes four constructions per step, so the
table fills at 2²² / 4 = 1,048,576 steps.

```bash
cd /workspace/worktrees/<yours>   # cut from main
export SOUNIO_STDLIB_PATH=$PWD/stdlib
./bin/souc compile docs/handoff/repros/handle_table_182_per_construction_madaros.sio -o /tmp/r.elf
(ulimit -s 524288; /tmp/r.elf | tail -1; echo rc=$?)
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run docs/handoff/repros/handle_table_182_per_construction_madaros.sio | tail -1
```

| Build | Engine | Program | Last stdout | stderr | rc |
|---|---|---|---|---|---:|
| committed `57c015c1` | Madaros | repro (24 B) | `i=1044480` | `madaros: handles full` | 182 |
| committed `57c015c1` | Madaros | control (16 B, field `c` dropped) | `REPRO_OK` | — | 0 |
| — | lean_single | repro | `REPRO_OK` | — | 0 |
| — | lean_single | control | `REPRO_OK` | — | 0 |
| source-built `5764851f` | Madaros | repro (24 B) | `i=1044480` | `madaros: handles full` | 182 |
| source-built `5764851f` | Madaros | control (16 B) | `REPRO_OK` | — | 0 |

1,044,480 is the last multiple of the 4096-iteration print window below
1,048,576, so the measured cost is exactly 4 handles/step.

Source-built Madaros `5764851f` on the five tests: all compile (rc=0), all exit **182** with `madaros: handles full`, and no sentinel appears. This is identical to the committed ELF.

## 8. Fix directions

The ranking of the 2026-08-17 dispatch (§4 there) stands. This workload
sharpens two points.

**B (escape-scoped frame reclamation) must reset both bumps and copy the
result out.** `tsit5_step_pbpk` constructs 63 values and returns one
`Tsit5StepResult14` referencing two of them. It takes only by-value
arguments and writes no `&!`, global or capture. So the returned value is
the only escape path, and it is exactly the case a frame watermark handles:

- at entry, record `(handle_count, heap_ptr)`;
- at return, relocate the handles ≥ watermark reachable from the return
  value down to the watermark, then reset both bumps.

Resetting `handle_count` without the heap pointer converts 182 into 181
(§5). A reset without relocation is the #555/#651 wipe-under-live-handle
class. Handles below the watermark (for example an argument returned
unchanged) must not move.

Soundness rests on the escape analyzer (`self-hosted/analysis/escape.sio`).
The default must be "escapes, do not reset". The witness suite needs:

- a returned-argument positive control;
- a nested-managed-field return (q6 shape);
- a `&!`-param store negative control.

**C (source-side avoidance) is real and measurable, but it is a workaround.**
The following are estimates, not measurements.

- Fusing each Tsit5 stage into one `PBPKState14` literal, rather than nested
  `pbpk_state_scale`/`pbpk_state_add` chains, drops a step from 63 to about
  16 handles: 7 ODE + 6 stages + 1 error + 2 for the result. That fits
  `gum_budget` (≈ 2.1 M) and `pce_vs_gum` (≈ 3.2 M), but not `hdmr_7d` or
  `pce2d_sobol` (≈ 10–11 M).
- An in-place `&!` step with preallocated stage buffers would approach zero
  handles per step. It must avoid the `let a0 = s.a; s = f(s)` field-alias
  hazard recorded on 2026-09-26.
- Either change lives in `stdlib/darwin_pbpk/tsit5_pbpk14.sio`, a shared
  numerical kernel. Its result must be re-verified bit-for-bit on lean_single
  before it can stand in any dissertation receipt.

**Not a fix:** raising `native_v2_handle_table_capacity_default` (§6).

## 9. Non-goals and scope

- No `self-hosted/` or `stdlib/` file is changed by this dispatch.
- The reclamation design lane
  ([`HANDLE_TABLE_RECLAMATION_DESIGN_2026-08-17.md`](HANDLE_TABLE_RECLAMATION_DESIGN_2026-08-17.md))
  claimed `self-hosted/native/gc.sio`. A patch goes through that lane.
- The exit-181 finding (§5) is recorded because it constrains the 182 fix.
  It is not separately root-caused here.

## AI disclosure

Measurement, analysis and drafting by an AI agent (Claude) under human direction, 2026-09-26. GAIDeT-ICMJE 2025.
