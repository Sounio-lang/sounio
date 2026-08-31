<!-- docs:meta
topic_id: repo.docs.internal.umbrella-gate-map
authority: repo_only
audience: users
last_validated: 2026-08-02
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.umbrella-gate-map
-->

# Umbrella gate map — what each gate is actually blocked by

> **This is a dated snapshot, not a live board.** The table below was measured
> once, on 2026-08-02, against a Madaros built from
> `research/self-falsifying-compilation-line-20260726` — a research branch, not
> `main`. It has not been re-run since, and `main` has moved a great deal in the
> interval. Read the *method* here as current and every *number* as historical;
> to know where the umbrella stands today, re-run the gate. Salvaged from #1603,
> whose other half (`scripts/dev/agent-bus.sh` and the MCP bridge) is a standing
> direction call about a second coordination channel and is deliberately not
> included.


**Measured:** 2026-08-02, `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh`,
run with `env -u SOUC_BIN -u SOUNIO_SOUC_BIN` against a Madaros built from
`research/self-falsifying-compilation-line-20260726`.

**Why this document exists.** The umbrella reports one number, 4/15, and that
number was read four times in a day as "nothing moved" without anyone asking
what the eleven failures were failing *at*. They are not eleven problems. Five
of them are one problem seen five times.

## Read this before running it

- **Pass `env -u SOUC_BIN -u SOUNIO_SOUC_BIN`.** `SOUC_BIN` is set in this pod's
  environment to `/workspace/sounio/bin/souc` — the integration checkout, parked
  on another branch, carrying an older compiler. Without unsetting it the gates
  measure that instead of your worktree and still print a number.
- **A 2–4 line gate stdout is the design, not a crash.** Each gate writes its
  real work into `$OUT_DIR/logs`, and several nest one level further
  (`driver_self → serious_track → *.check.log`). Follow the `out=` path down.
- **The nested temp dirs are deleted on gate exit.** To keep them, run the gate
  directly with its own `SOUNIO_*_DIR` set.

## The map

| gate | status | blocked by |
|---|---|---|
| `e2e_codegen_suite` | **PASS** | |
| `lean_single_fixed_point` | **PASS** | |
| `iso_budget` | **PASS** | |
| `phase_y_gum_pbpk` | **PASS** | |
| `driver_self_compile` | FAIL | **`self-hosted/compiler/native_compile_driver.sio`: 29 errors** — E008 10, E137 9, E012 8, E005 2 |
| `science_spine` | FAIL | ← runs `driver_self_compile` first |
| `gum_primitives` | FAIL | ← runs `driver_self_compile` first |
| `f64_ladder` | FAIL | ← `science_spine` ← `driver_self_compile` |
| `semantic_hardening` | FAIL | ← `science_spine` ← `driver_self_compile` |
| `struct_orchestrator` | FAIL | **not established** — no dependency on the chain above, cause not yet measured |
| `imported_closure_boundary` | FAIL | **`self-hosted/compiler/lean.sio`: 51 errors** — E137 17, E008 12, E009 8, E012 7, E011 2, E005 2 |
| `imported_captured_closure_boundary` | FAIL | same `lean.sio` set |
| `dissertation_pbpk_suite` | FAIL | **2 of 53 tests** (was 30 of 53 under the pre-2026-08-02 compiler) |
| `phase_j_conf_gate` | FAIL | 3 DROP / 1 FAIL / 1 PASS; `conf_reject_demo` exits 1 |
| `kretikos_kaxi_meta` | FAIL | self-check step |

## Where the leverage is

**Five of the eleven failures share one root**: `native_compile_driver.sio` and
its 29 errors. `science_spine` and `gum_primitives` invoke
`driver_self_compile` as their first step; `f64_ladder` and `semantic_hardening`
invoke `science_spine`, which invokes it. None of the four gets to its own
manifest, which is why four gates produce no test tally at all — they never
reach their tests.

So the highest-value target in the umbrella is 29 errors in one file, not a
corpus. The work done on 2026-08-02 took `lean.sio` from 2 649 errors to 51 and
moved exactly two gates' blocker (the closure-boundary pair) without flipping
them, because a gate is binary and 51 is not 0.

**Movement is real and was invisible.** `dissertation_pbpk_suite` went from
30/53 tests failing to 2/53 across the same day. The gate stayed red both times,
so the umbrella's headline number showed nothing. A gate map is how that stops
being invisible.

## What is not established

`struct_orchestrator` shares no dependency with the driver chain and its cause
was not measured — an earlier attempt attributed the driver's 29 errors to it,
but that reading came from a mapper that leaked one shared log directory into
every gate's census and is withdrawn. It needs its own run with a persistent
`SOUNIO_NATIVE_V2_STRUCT_ORCHESTRATOR_DIR`.
