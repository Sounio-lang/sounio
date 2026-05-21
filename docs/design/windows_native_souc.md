# Native `souc.exe` on Windows — status & roadmap

Goal: the self-hosted Sounio compiler running natively on Windows, validated by
running the cross-compiled PE under wine on the Linux pod.

## Validation harness

No Windows host or qemu on the pod. We validate by cross-compiling to
`x86_64-windows` PE32+ and executing under **wine 9.0** (installed on
`sounio-workspace-habitat-0`). Gate: `scripts/ci/windows_pe_smoke_gate.sh`.

## What works today (verified under wine)

The Linux→Windows cross-compile path is functional for the I/O primitives a
compiler needs:

| Capability | Mechanism | Status |
|---|---|---|
| Exit code | `ExitProcess` via trampoline | ✅ exit 42 confirmed |
| stdout (`print`) | `GetStdHandle` + `WriteFile` via RuntimeContext fn-ptr | ✅ "hello from windows" |
| File read/write | `CreateFileA`/`ReadFile`/`WriteFile`/`CloseHandle`, `os_id==3` path in `emit_*_syscall_for_target` (codegen.sio:1817+) | ✅ byte-exact round-trip |

The full 32K-line `self-hosted/compiler/lean_single.sio` **cross-compiles** to a
2.1 MB PE32+ with rc=0.

## What's broken: `souc.exe` crashes at runtime

Running the cross-compiled compiler under wine:
```
wine souc.exe prog.sio prog.elf
→ Unhandled page fault on EXECUTE access to 0x60DAA5  (exit 5)
```
(image base 0x400000, so fault offset ≈ 0x20DAA5 — near the end of the 2.1 MB
image, i.e. control jumped into the data/rodata region.)

### Confirmed gap #1 — argc/argv not initialized

`codegen.sio:5121`:
```
// No argc/argv on Windows PE minimal entry (would require GetCommandLineW + CommandLineToArgvW)
```
The Windows entry trampoline (`compile_to_pe_x86_64`, codegen.sio:5797+) sets up
VirtualAlloc heap, GetStdHandle stdout/stderr, and the file fn-ptr table, but
leaves `RuntimeContext.argc` and `RuntimeContext.argv_ptr` zero.

`get_arg`/`get_arg_count` (codegen.sio:2066-2087) read those fields, so the
compiler sees argc=0 and cannot locate its source/output path arguments. This is
a hard blocker for a CLI compiler regardless of the crash.

**Fix:** in the Windows trampoline, call `GetCommandLineA` (or
`GetCommandLineW` + `CommandLineToArgvW` + width conversion), build an
`argv`-style pointer array in the heap, and store base + count into the
RuntimeContext argc/argv_ptr fields. Add `GetCommandLineA` (and
`CommandLineToArgvW`/`LocalFree` if using the W path) to the .idata imports.

### Open: execute-access fault attribution

The fault is *execute* access, which is not the typical symptom of a NULL argv
read (that would be a *read* fault). Candidates, not yet bisected:
- corrupted return address / stack misalignment on some Win64 boundary call
  (Win64 requires RSP%16==8 at call sites + 32-byte shadow space);
- an indirect call through an uninitialized RuntimeContext fn-ptr;
- a consequence of argv=0 leading lean_single down an unintended path.

Do **not** assume argv is the whole fix. After implementing argv, re-run; if the
fault persists, bisect with a minimal arg-reading program before touching the
trampoline call sequence further.

## Roadmap

1. **argv setup in PE trampoline** (confirmed required) — GetCommandLine →
   argv array → RuntimeContext. Validate with a small `get_arg`/`get_arg_count`
   program under wine first, then re-test `souc.exe`.
2. **Resolve the execute fault** — minimal repro, single-step under
   `winedbg`, fix the offending ABI/pointer issue.
3. **Compile a real program with souc.exe under wine** — the milestone: PE
   compiler emits a working binary.
4. **Tier 2 cleanup** — audit `emit_exit` (encode.sio, hardcoded Linux syscall
   60) on Windows fallback paths; signal_handler.sio is POSIX-only (panic path).

## Reproduce

```bash
cd <worktree>
bash scripts/ci/windows_pe_smoke_gate.sh          # 6/6 PASS (I/O primitives)
./bin/souc compile self-hosted/compiler/lean_single.sio -o /tmp/souc.exe --target x86_64-windows
echo 'fn main() -> i64 { return 7 }' > /tmp/prog.sio
WINEDEBUG=-all wine /tmp/souc.exe /tmp/prog.sio /tmp/prog.elf   # currently faults
```
