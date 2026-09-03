<!-- docs:meta
topic_id: repo.docs.audit.madaros-boxnew-sigsegv-2026-06-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-boxnew-sigsegv-2026-06-19
-->

# Madaros codegen SIGSEGV on `Box::new` — forensic dispatch (2026-06-19)

## Resolution note (2026-06-20)
The `Box::new(...)` prebuilt SIGSEGV described here is fixed on
`codex/madaros-boxnew-review` by the Madaros/lowerer changes and refreshed
`bin/madaros-linux-x86_64`. The `madaros-prebuilt-refresh.yml` full gate passed
for the refreshed binary, and `Box::new(7)` compiles through the default
`bin/madaros` path.

The later Slurm no-ulimit frame-fix validation is a separate remaining blocker:
Madaros rebuilds successfully on Slurm, but the freshly rebuilt raw compiler
still SIGSEGVs compiling the frame-fix reproducer under the default 8192 kB
stack limit.

Slurm validation is now split by contract:
- `slurm-jobs/madaros-frame-fix/submit_production_gate.sh` validates the
  production launcher (`bin/madaros`) and its stack policy on the compute node.
- `slurm-jobs/madaros-frame-fix/submit_stack_fix.sh` is a raw no-ulimit
  diagnostic that tracks the remaining frame-size blocker.

Production Slurm result:
- `RUN_ID=madaros-production-gate-20260620T122325`, Slurm job `4319`.
- Initial compute-node stack limit: `8192 kB`.
- `bin/madaros build` compiled and ran N=1/N=2/N=4/N=5 reproducer variants.
- Result: `PASS production launcher compiles and runs frame reproducer variants`.

## Symptom
The prebuilt `bin/madaros-linux-x86_64` (v0.80.0, refreshed on main @ `659492156`,
2026-06-19 14:43) **SIGSEGVs (rc=139) when compiling any source containing
`Box::new(...)`** via the `build` (native_v2) path.

```
# crashes:
fn main() with IO { let b = Box::new(7) }      # construction alone is enough
# works:
fn main() with IO { let v = 7  println(v) }
```

## Localization (all local, cheap — no SLURM needed)
| Path | Result |
|---|---|
| madaros `--check` Box::new | OK (parse + typecheck clean) |
| madaros `build` (native_v2) Box::new | **SIGSEGV** |
| madaros `build` plain (no Box) | OK (`hello.sio` builds + runs) |
| lean_single `build` Box::new | **OK** — emits ELF, runs, prints `7` |

So: the bug is at **runtime of the lean_single-built madaros**, specifically in the
native_v2 codegen/lowering of a `Box::new` expression. The bootstrap `lean_single`
itself compiles `Box::new` correctly.

## Crash forensics (from core dump `/tmp/core.148676`)
madaros is a static non-PIE EXEC (base `0x400000`, code `0x400000`–`0x5e34d74`),
fully stripped (no section headers).

- **Faulting RIP** = `0x3ebe4f2`, instruction `mov 0x0(%rdx),%rax` → **null/bad deref of `%rdx`**.
- Preceding context (vaddr `0x3ebe4a8`–`0x3ebe51c`):
  ```
  loop counter at -0x20(%rbp), init 0
  cond: counter < (*param1).field@0x10        ; field@0x10 = length
  body: rcx = (*param1).field@0               ; field@0   = data ptr
        rax = data[counter]   (mov (%rcx,%rdx,8),%rax)
        rdx = rax
        mov 0x0(%rdx),%rax    <-- CRASH       ; deref element[0].field0
        ...copies element fields @0,8,0x10,0x18 into locals -0xa8.. (>=4 i64 struct)
  ```
- **Call chain (stack-walked return addrs):** RIP `0x3ebe4f2` ← `0x3ebf3fe`
  (same/adjacent fn) ← `0xf81a40` ← `0x786f42`.

## Interpretation
The crashing function iterates a **list of pointers-to-struct** (`.data`@0, `.len`@0x10,
elements are ≥4×i64 structs). On the **first** iteration (index 0) the element pointer
is bad although `len > 0`. A list reporting `len>0` while `data[0]` is null/garbage is the
signature of the documented **lean_single nested-store miscompile** class
(`k.f1.f2[N]=v` two-level store silently discarded; `feedback_lean_single_miscompilations`),
**not** a logic error in this loop. Hypothesis: lean_single miscompiles the list/struct
construction inside madaros's `Box::new` lowering, leaving `len`/`data` inconsistent.

Candidate source sites (to confirm):
- `self-hosted/ir/lower.sio:5601 lower_box_new` (dispatch `:6297`) — IR Lowerer path
- native_v2 alloc emit: `self-hosted/native/codegen_x86_linux.sio` `nc_core_emit_alloc_into`
  (`:5918`), arg collection `self-hosted/native/machine_ir.sio:414 native_v2_collect_call_args`

## Why it matters
This is THE baseline SIGSEGV that (a) blocks the fixed point (madaros compiling main.sio →
gen2==gen3) and (b) poisoned the `slurm-jobs/madaros-frame-fix` stack-fix reproducer (every
N=1..5 crashed because the built madaros can't compile the Box::new-containing reproducer at
all — unrelated to the SRET frame fix being validated).

## Confirmed source (IR Lowerer path)
`self-hosted/ir/lower.sio:5601 fn lower_box_new` emits:
```
lo = lo.emit(ir_alloc(ptr_reg, 8))                       // heap-alloc 8 bytes
lo = lo.emit(ir_field_set(ptr_reg, 0, val_reg, ir_empty_name()))  // store value at ptr[0]
```
This is the **IR Lowerer** (`lower.sio`). Whether the **native_v2** path (`build`) routes
through this `Lowerer` or a separate lowering is the one remaining unknown to nail down
(search `native_v2_compile` → trace to where `ir_alloc`/Box::new is handled). The crashing
loop copies a list of ≥4-i64-field structs by value (`.data`@0, `.len`@0x10) — likely the
codegen iterating the IR instruction/reg list once the `ir_alloc` instr is present (hello.sio,
which has no `ir_alloc`, compiles + runs fine).

## Handoff for Codex — exact repro
```bash
cd /workspace/sounio
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
printf 'fn main() with IO {\n    let b = Box::new(7)\n}\n' > /tmp/boxtest.sio
./bin/madaros-linux-x86_64 build /tmp/boxtest.sio -o /tmp/box.elf   # rc=139 SIGSEGV
./bin/souc-lean-single-x86_64 /tmp/boxtest.sio /tmp/box_ls.elf && /tmp/box_ls.elf  # OK -> prints 7
```
- Core dump from the crash: `/tmp/core.148676` (~3.8 GB; static non-PIE EXEC base `0x400000`).
  Faulting RIP `0x3ebe4f2` = `mov 0x0(%rdx),%rax`; call chain `0x3ebe4f2 ← 0x3ebf3fe ← 0xf81a40 ← 0x786f42`.
- No gdb/strace/valgrind/symbols available locally; madaros is stripped (no section headers).
  Disassemble by file offset = vaddr − `0x400000` via
  `objdump -D -b binary -m i386:x86-64 --adjust-vma=0x400000 --start-address=... bin/madaros-linux-x86_64`.

## Next steps
1. Confirm exactly which source construction the crashing loop is (trace native_v2 pipeline).
2. If lean_single-miscompile: restructure the construction to avoid the two-level nested
   store (build in a local var, assign once) — the proven workaround pattern
   (`feedback_lean_single_miscompilations`).
3. Rebuild madaros via CI `madaros-prebuilt-refresh.yml` (proven, non-local) and re-test
   `Box::new(7)` + the `slurm-jobs/madaros-frame-fix` reproducer.
