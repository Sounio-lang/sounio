# E230 diagnostic implementation — handoff to a build lane

**Date:** 2026-08-17
**Lane:** drain-minimax-cli1 / `handle-table-ceiling-refusal`
**Code:** `error[E230]`
**Status:** **HALT** — design complete (v1 = commit `73f6599d7e`, v2 =
commit `16db5e9560`); source implementation deferred. This handoff is the
boundary report, not a fix.

## What this lane has shipped

1. **`docs/audit/HANDLE_TABLE_CEILING_REFUSAL_2026-08-17.md`** (v1 design)
2. **`docs/audit/HANDLE_TABLE_CEILING_REFUSAL_REFINEMENT_2026-08-17.md`**
   (v2 with measured evidence: codex-2's no-wrap refutation,
   minimax-cli3's stdlib-layer refusal coordination, d2_gum hard number
   `4193876 of 4194304`)
3. **`scripts/ci/handle_table_ceiling_gate.sh`** (W1 compile-time refusal,
   W3 hot-loop drift refusal, W4 negative control; W2 90% warning
   positive control deferred — see below)
4. **Coordination**: cross-references minimax-cli3's
   `MC28_MC_N_CEILING = 1000` / `MS28_MC_N_CEILING = 500` constants and
   `MC_CEILING_DETECTED` / `MS_CEILING_DETECTED` markers; cross-references
   fable-1's `MADAROS_HANDLE_TABLE_182_LIFETIME_DISPATCH_2026-08-17.md`
   fix directions A/B/C (reclamation, not mine)

## What this lane cannot do

**Source implementation of the E230 diagnostic at the 182 slow path.**

Per FLEET_CONSTRAINTS:
- "**`./bin/souc` is PREBUILT.** Editing compiler source does not change
  it. Build from source before claiming any compiler behaviour."
- "A full self-compile, `make build`, `lake build`, or the test suite is
  not [allowed on this pod]."
- "**Slurm launch is currently broken**" — so I cannot route the build
  through Slurm either.

And per fable-1's honest dispatch (`docs/audit/MADAROS_HANDLE_TABLE_182_LIFETIME_DISPATCH_2026-08-17.md`):
- "**The backend has no accessible compile-path observability.** Madaros
  has ≥4 parallel function-compile implementations ... Env-gated `print`
  diagnostics added to three of these produced **no output** for a test
  program, and the **pre-existing** `native_streaming` diagnostic also
  produced none — so the actual path a given program takes could not be
  traced, and the emitted ELF is not `objdump`-parseable. Debugging the
  emit therefore requires **building backend observability first**."

The E230 diagnostic modifies `self-hosted/native/codegen_x86_linux.sio`,
specifically `nc_core_emit_alloc_failure_diagnostic_into` and the
allocation slow path. Without a build lane to verify, any bug I introduce
silently breaks every native-v2 ELF the lane produces. This is the kind
of "untested instrument" failure the FLEET_CONSTRAINTS explicitly warns
against: an entire false investigation (see issue #1689 in the same
file).

The user's brief was explicit ("implementar") but the constraint is real.
Halt with a clear handoff is the honest move.

## What a build lane needs to do

### Step 1 — apply the patch

`./.scratch/e230_diagnostic.patch` is a unified diff against `origin/main`
that implements the minimum diagnostic. It modifies:

| File | Change |
|------|--------|
| `self-hosted/native/codegen_x86_linux.sio` | `nc_core_emit_alloc_failure_diagnostic_into` now prints count + capacity + reason at both 181 and 182 slow paths. `nc_core_emit_alloc_into` inserts the 90% warning check before the 100% check. |
| `self-hosted/native/runtime_context.sio` | Adds `runtime_context_field_e230_90_warning_fired()` at offset 248 (one i64). `runtime_context_size()` 248 → 256. All 9 consumers of `runtime_context_size()` pick up the new size automatically (per fable-1's note: "no hardcoded 248 anywhere, all 9 consumers call the function"). |
| `self-hosted/native/runtime_init.sio` (or wherever the entry trampoline lives) | Verify the entry trampoline re-reads `runtime_context_size()` instead of using a baked constant. If baked, update. |
| `scripts/ci/handle_table_ceiling_gate.sh` | W2 90% warning positive control — assertion already in place, will PASS once the patch lands. |

### Step 2 — build from source

```bash
bash scripts/ci/build_modular_madaros.sh /tmp/madaros_e230.elf
MADAROS_RAW_BIN=/tmp/madaros_e230.elf SOUNIO_STDLIB_PATH=$(pwd)/stdlib \
  bash scripts/ci/handle_table_ceiling_gate.sh
```

The gate must produce:
- W1 PASS — program with > 4194304 alloc sites → `error[E230]` with count/capacity/reason, nonzero rc
- W2 PASS — program with dynamic count crossing 90% → `warning[E230]` once, then again at 100% with full count/capacity/reason
- W3 PASS — program with loop crossing 100% → `error[E230]` with full count/capacity/reason, nonzero rc
- W4 PASS — trivial program → rc=0, no E230

If W1 / W2 / W3 FAIL with the expected message missing, the patch is wrong.

### Step 3 — measure on the d2_gum class

```bash
MADAROS_RAW_BIN=/tmp/madaros_e230.elf SOUNIO_STDLIB_PATH=$(pwd)/stdlib \
  ulimit -s 524288 && ./bin/souc run stdlib/darwin_pbpk/pd/d2_gum.sio 2>&1 | tail -20
```

Expected new output (one line, replacing the silent `madaros: handles full`):

```
madaros: handles full: count=4194305 of 4194304 (2^22)
```

If the line says anything other than `count=4194305 of 4194304 (2^22)`,
the patch's count extraction is wrong (likely wrong runtime context
field offset, or wrong register convention).

### Step 4 — measure on rapamycin_pop_sim

The 20-patient run that died mid-experiment must now die at the wall
WITH the named count. Without the count, the partial output looks like
data; with the count, the user sees `count=N of 4194304` and knows the
workload exceeded the budget at handle N.

If the gate tests PASS and the d2_gum / rapamycin_pop_sim measurements
show the named count, the patch is correct. Commit it.

## What the patch does NOT do

- **Reclamation.** Owned by fable-1's fix B (function-scoped watermark,
  blocked on observability per his dispatch). My lane is disjoint — I do
  not touch `lower.sio` or any reclamation path. Per the fleet protocol
  ("nao toques em ficheiros que nao sejam os teus; hoje ja tive de
  desfazer tres lanes no mesmo lower.sio"), reclamation is not my work.
- **stdlib-layer refusal.** Owned by minimax-cli3
  (`MC28_MC_N_CEILING = 1000` etc.). My E230 is the compiler-layer
  complement, not the duplicate.
- **Heap wall (181).** The patch touches the 181 path for completeness
  (count + capacity + reason for arena full too) but the user's brief
  was specifically about 182. The 181 path is included because the same
  diagnostic function emits both.
- **Layer 1 / Layer 2 (compile-time static count refusal, runtime
  pre-flight).** Deferred — see v2 §8 for the file list.

## Honest summary

| What | Where | Status |
|------|-------|--------|
| Design (v1) | commit `73f6599d7e` | shipped |
| Refinement (v2) | commit `16db5e9560` | shipped |
| Gate W1/W3/W4 | `scripts/ci/handle_table_ceiling_gate.sh` | shipped |
| Gate W2 (90% positive control) | gate | assertion in place, will pass once patch lands |
| Source patch | `.scratch/e230_diagnostic.patch` | drafted, not committed to source tree |
| Coordination (minimax-cli3, fable-1) | audit doc §4 | cross-referenced |
| Source implementation in compiler | | **DEFERRED — no build lane on this pod** |

The user's brief said "implementar". The premise of "implement" is
"build and verify". This pod has no build lane. So the honest move is to
halt with a clear handoff and let a build lane apply the patch.

If the user wants this lane to commit the source change anyway, the
risk is: any bug in the emitted x86_64 (wrong register, wrong frame
setup, wrong field offset) silently breaks every native-v2 ELF the lane
produces, and the backend has no observability to debug it. The build
lane would have to bisect to find and fix it.

A build lane is a lane with `bin/souc` rebuilt from current source,
access to a Madaros source build, and a CI lane that can run the gate.
None of those exist on this pod.

**Recommendation:** assign the patch application to a build lane. After
W1/W2/W3 PASS on that lane, fast-forward this lane's commit with the
verified change.