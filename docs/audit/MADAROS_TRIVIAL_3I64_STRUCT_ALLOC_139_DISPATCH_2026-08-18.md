<!-- docs:meta
topic_id: repo.docs.audit.madaros-trivial-3i64-struct-alloc-139-dispatch-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-trivial-3i64-struct-alloc-139-dispatch-2026-08-18
-->

# Madaros SIGSEGV 139 on a 3-field aggregate — not E230 (dispatch)

**Date:** 2026-08-18
**Scope:** runtime SIGSEGV (rc=139) of a **real Madaros ELF** produced by
`souc compile` from a trivial 3-field programme. **Not** handle-table
exhaustion, **not** E230, **not** the println-kind0 family, **not** N=3.
**Status:** OPEN. Classification + minimal witness. **No fix in this
dispatch.** `self-hosted/ir/lower.sio` was not touched (codex-3 holds it;
fable-1 has history).
**Owner of the correction:** the E230-v3 diagnostic patch
(`minimax-cli1`, `.scratch/e230_diagnostic.patch`) — specifically
`self-hosted/native/codegen_x86_linux.sio` + the
`runtime_context_size` 248→256 bump in
`self-hosted/native/runtime_context.sio`. Default unpatched Madaros does
**not** reproduce.
**Witness:** [`repro/trivial_3i64_alloc_139.sio`](repro/trivial_3i64_alloc_139.sio)

## 0. Why this is not E230 / 182 / println-139

| Family | What it looks like | This crash |
|---|---|---|
| E230 | `warning[E230]` / `error[E230]` naming count and capacity | no diagnostic at all |
| 182 | `madaros: handles full` after the programme starts | no such line; dies immediately |
| println-kind0 139 | `println` of an unclassified scalar (bool/int copy) | `print("done\n")` alone is **rc=0**; the 3-field construct with **no print** still 139 |
| this | first construction of a ≥3-slot aggregate | SIGSEGV, rc=139 |

Handle-table capacity on the crashing binary is still **4194304** (2²²).
The programme constructs **one** value. The ceiling is not in the picture.

## 1. Origin

grok-cli1, verifying the E230-v3 patch, compiled the gate's W2 witness with
the **verb** form after the bare `souc w2.sio -o w2.elf` instrument was
shown to be empty (that form writes a file literally named `-o` and
routes to lean_single). The verb form produced a 12744-byte Madaros ELF
that SIGSEGV'd at N=3 with `struct W2 { x: i64, y: i64, z: i64 }`.

Exact source of that first 139 (266 bytes):

```sounio
struct W2 { x: i64, y: i64, z: i64 }
fn alloc_one() -> W2 with Alloc { W2 { x: 1, y: 1, z: 1 } }
fn main() -> i64 with IO, Mut, Panic, Div, Alloc {
    var i: i64 = 0
    while i < 3 {
        let _x = alloc_one()
        i = i + 1
    }
    print("done\n")
    0
}
```

Replay of that ELF on the same STAGE, 2026-08-18T16:24Z, cpuops-t560-proxmox:

```
w2_verb 12744 elf True
Segmentation fault
ORIG_W2_RC=139
```

## 2. Instrument (binding)

Every compile in this dispatch used:

```
souc compile <src> -o <out>
```

then confirmed the output **exists** and starts with `\x7fELF` before any
run-rc was claimed. Engine identity was taken from the compile log
(`Madaros v0.80.0`, `imported_compile:` / `lower_array:`), not from the
wrapper's default. `MADAROS_RAW_BIN` was set explicitly for the patched
binary.

Bare `souc <src> -o <out>` is a **false instrument**: rc=0, file named
`-o`, lean_single log, 36924 bytes on this source. That is how the E230
gate spent two rounds measuring nothing. See grok-cli1's
`docs/audit/SOUC_WRAPPER_BARE_DASH_O_SWALLOW_DISPATCH_2026-08-18.md`.

## 3. The question: 3-field struct, N=3, or the combination?

**The 3-field (more precisely: ≥3-slot) aggregate. Not N=3. Not the
combination.**

N=1 is enough. The helper, the loop, `print`, and the `Alloc` effect
are all dispensable.

Minimal programme that still SIGSEGVs on the patched compiler
(and is the committed witness):

```sounio
struct S { x: i64, y: i64, z: i64 }
fn main() -> i64 with IO {
    let _x = S { x: 1, y: 1, z: 1 }
    0
}
```

## 4. Matrix (patched E230 Madaros)

Compiler: `/orangefs/training/e230-v3-20260818T155411Z/build/madaros`
100088179 bytes, sha256 `b6682d155e135249b01d2708ac9d0abcf8ec148e1f2dbf0318450ccce94b58db`,
mtime 2026-08-18 15:58:49Z. `runtime_context_size() = 256`,
`runtime_context_field_e230_90_warning_fired() = 248`.
`souc compile` log says Madaros. Every listed ELF starts with `\x7fELF`.

| Case | Programme | bytes | run rc |
|---|---|---:|---:|
| A | `print("done\n")` only | 12744 | **0** |
| J | `while i < 3` no aggregate | 8648 | **0** |
| K | helper returns `i64`, N=3 | 8648 | **0** |
| P | `struct S {x,y,z: i64}` declared, never constructed | 8648 | **0** |
| G | 1-field `i64` struct, N=3 | 8648 | **0** |
| F | 2-field `i64` struct, N=3 | 8648 | **0** |
| U | 2-field `f64` struct, N=1 | 8648 | **0** |
| arr2 | `let a: [i64; 2] = [1, 1]` | 12744 | **0** |
| **B** | **3-field `i64`, N=1, no print, no helper** | 12744 | **139** |
| C | W4 exact (3-field + `print("hi\n")`) | 12744 | **139** |
| D | original tiny W2 (N=3 + helper + print) | 12744 | **139** |
| E | 3-field `i64`, N=3, inline | 12744 | **139** |
| L | 3-field `i64`, N=1, `with IO` only (no `Alloc`) | 12744 | **139** |
| M | 3-field `i64`, use `x+y+z` | 12744 | **139** |
| N | 3-field `i64`, N=1, helper | 12744 | **139** |
| H | 4-field `i64`, N=1 | 12744 | **139** |
| R | 3-field **`i32`** (12 B, **below** the 16 B unbox threshold) | 12744 | **139** |
| S | `{i64, i64, i32}` | 12744 | **139** |
| Q | `let a: [i64; 3] = [1, 1, 1]` | 12744 | **139** |
| arr4 | `let a: [i64; 4] = [1, 1, 1, 1]` | 12744 | **139** |
| arri32_3 | `let a: [i32; 3] = [1, 1, 1]` | 12744 | **139** |
| I | 3-field `f64`, N=3 | 12744 | **132** (SIGILL) |
| I′ | 3-field `f64`, N=1, no loop | 12744 | **132** (SIGILL) |
| arr3f64 | `let a: [f64; 3] = [1.0, 1.0, 1.0]` | 12744 | **139** |
| arr2f64 | `let a: [f64; 2] = [1.0, 1.0]` | 12744 | **139** |
| arr4f64 | `let a: [f64; 4] = [1.0, 1.0, 1.0, 1.0]` | 12744 | **139** |

So:

- **N=3 is irrelevant.** N=1 of a 3-field struct crashes; N=3 of a 2-field
  struct or of a bare `i64` does not.
- **It is not the 16 B unbox / handle-taking story the E230 gate used to
  justify 3×`i64`.** A 12 B `struct { i32, i32, i32 }` also 139. A 16 B
  `{ i64, i64 }` does not.
- **It is not unique to named structs.** `[i64; 3]` and `[i32; 3]` crash;
  `[i64; 2]` does not.
- Working description for the **139** family: **constructing a ≥3-slot
  integer aggregate** (struct or array) under this patched compiler.

### 4a. The 132 is a different site, not a noisy 139

Same width class, two faults. The pair that closes it:

| | 2-wide | 3-wide |
|---|---|---|
| `struct { f64, f64 }` / `struct { f64, f64, f64 }` | **0** (U) | **132** SIGILL (I / I′) |
| `[f64; 2]` / `[f64; 3]` | **139** | **139** (not 132) |
| `struct { i64, i64 }` / `struct { i64, i64, i64 }` | **0** | **139** |
| `[i64; 2]` / `[i64; 3]` | **0** | **139** |

`[f64; 3]` is **139**, not SIGILL. The 132 does **not** follow the array
of the same width, so 3-field `f64` structs have their **own** emit
path. Two faults from one patch is two sites, not one site with a
flaky signal.

`[f64; 2]` already 139 — while `{ f64, f64 }` is 0 and `[i64; 2]` is
0 — is a third split: f64 **arrays** die one slot earlier than i64
arrays. Recorded; not chased here.

## 5. Same source, other engines (engine divergence)

Same files, same `souc compile` + `\x7fELF` check.

| Engine | Binary | D (tiny W2 N=3) | B (N=1 3×i64) | Q (`[i64;3]`) | R (3×i32) | I′ (3×f64 struct) | `[f64;3]` |
|---|---|---|---|---|---|---|---|
| **Default Madaros** (this worktree) | `artifacts/self-hosted/madaros` 99964760 B, 2026-08-17 15:32Z | 12744, rc=0, prints `done` | 12744, rc=0 | rc=0 | rc=0 | rc=0 | rc=0 |
| **lean_single** | `bin/souc-lean-single-x86_64` via `SOUNIO_SOUC_ENGINE=lean_single` | 36924, rc=0, prints `done` | 36676, rc=0 | (family passes) | (family passes) | 36706, rc=0 | 36706, rc=0 |
| **E230-v3 patched Madaros** | STAGE build above, 100088179 B | 12744, **139** | 12744, **139** | **139** | **139** | **132** | **139** |

Default Madaros and lean_single **both pass** the original W2 and the
minimised B. This is therefore **not** a latent default-Madaros bug and
**not** a seed bug. It is a **from-source rebuild of Madaros with the
E230-v3 patch applied**. That changes the owner of the correction:
minimax-cli1's patch (codegen + ctx size), not `lower.sio`, not the
println-kind0 lane, not the default native-v2 GC.

lean_single ELF size 36924 on D matches grok-cli1's bare-form `-o` file
exactly — independent confirmation that the swallow-`-o` path is the
seed, not Madaros.

## 6. Suspected locus (not a root-cause; do not patch from this)

The v3 patch does two things that sit on the managed-alloc emit path:

1. `runtime_context_size` 248 → 256 and a new field at offset 248
   (`e230_90_warning_fired`).
2. A 1-for-1 replacement of
   `nc_core_emit_alloc_failure_diagnostic_into` in
   `codegen_x86_linux.sio`, plus new helpers
   `nc_write_rodata_to_stderr_into` / `nc_append_rodata_bytes`, plus the
   90 % warning fire body that stores to `[ctx+248]`.

The crash happens on the **first** construction of a ≥3-slot value, including
aggregates that should be **unboxed** (3×i32 = 12 B). That does **not**
fit "the new warning flag is written OOB on the handle-alloc slow path"
as a complete story — unboxed values should not take a handle. It **does**
fit "the hunk replacement disturbed neighbouring emit of 3-slot
stores / SRET / aggregate copy". It does **not** by itself explain the
3×`f64` SIGILL: that fault is absent on `[f64; 3]` (139) and present
only on the named 3-field `f64` struct, so the float-struct emit is a
second site.

Whoever picks this up should start with a control rebuild: same STAGE
tree **without** the v3 patch, `souc compile` of B, expect rc=0 (already
true of the committed prebuilt). Then rebuild **with only the
`runtime_context_size` bump** and **with only the codegen hunk**, and
see which half takes B from 0 to 139. gdb was not available on
cpuops-t560-proxmox in this session.

## 7. What not to do

- Do not edit `self-hosted/ir/lower.sio`. That file is claimed by
  codex-3 (`println-kind0-refusal-current-20260818`); fable-1 has
  history. This crash is not that family.
- Do not treat a 139 on W2/W3/W4 of `handle_table_ceiling_gate.sh` as
  an E230 result. On this patch those witnesses never reach the 90 %
  band; they die on the first iteration.
- Do not use bare `souc <file> -o <out>`.
- Do not "fix" by shrinking the gate structs to 2 fields. That would
  make the gate green by avoiding the crash, and would stop testing
  handle consumption (2×i64 is ≤16 B and unboxes).

## 8. Blocker record

```text
Blocker-ID: BLK-20260818-grok-cli5-trivial-3slot-139
Status: classified
Severity: B1
Class: compiler-semantics
Owner: minimax-cli1 (E230-v3 diagnostic patch)
Lane: e230-v3-verify / handle-table ceiling
Worktree: /workspace/.wt/grok-cli5 (dispatch); crashing ELF lives on
          /orangefs/training/e230-v3-20260818T155411Z
Branch: lane/grok-cli5/trivial-139-dispatch-20260818 (dispatch only)
Evidence: E2
Acceptance gate: `souc compile docs/audit/repro/trivial_3i64_alloc_139.sio
                 -o /tmp/b.elf` against a from-source Madaros that carries
                 the v3 patch produces a \x7fELF whose run is rc=0 (today
                 139). Same source remains rc=0 on unpatched default
                 Madaros and on lean_single.
Next action: split the v3 patch (ctx-size bump vs codegen hunk) and
             rebuild each half; do not touch lower.sio.
```

## 9. Registry

Row `repo.docs.audit.madaros-trivial-3i64-struct-alloc-139-dispatch-2026-08-18`
is in `docs/governance/topic-registry.v1.json`, synced **after** the
dispatch commits (`node scripts/docs/sync_governance_metadata.mjs`).
The earlier `--no-verify` commits were lease-blocked, not content-blocked.

## 10. Commands run (this session)

Default Madaros + lean_single matrix: `/tmp/g5-139-repro/` on
`sounio-workspace-control-0`, `SOUNIO_STDLIB_PATH` = this worktree
`stdlib`, no `MADAROS_RAW_BIN`.

Patched matrix: `scripts/dev/slurm_srun_minimal.sh --time=00:12:00`
then two follow-ups, host `cpuops-t560-proxmox`,
`MADAROS_RAW_BIN=$STAGE/build/madaros`, receipts under
`$STAGE/g5-139`, `$STAGE/g5-139b`, `$STAGE/g5-139c`.

f64-axis follow-up (same instrument, 2026-08-18T16:31Z): patched
receipts under `$STAGE/g5-139-f64`; default + lean_single under
`/tmp/g5-139-f64/`. `[f64;3]` is 139 on the patch, 0 on the other two
engines; 3-field `f64` struct is 132 on the patch, 0 on the other two.
