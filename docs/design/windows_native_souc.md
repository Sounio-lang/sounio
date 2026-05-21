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

### Root cause (bisected with a minimal `arg_count()` probe)

A one-line program `fn main() -> i64 { return arg_count() }` cross-compiled to PE
faults under wine at a **read** of `0x10000000`. Disassembly of the emitted
`arg_count` builtin:
```asm
401000: push rbp; mov rbp,rsp; sub rsp,0x10
40100b: movabs rax, 0x10000000     ; absolute addr of the argc global (= ImageBase)
401015: mov    rax, [rax]          ; deref -> FAULT (rax=0x10000000 unmapped)
```

Two chained bugs, both attributed:

1. **Globals use absolute addresses keyed to ImageBase 0x10000000, but the image
   loads at 0x400000.** `pe_coff.sio:146,3020` declare ImageBase `0x10000000`,
   yet wine maps the module at `0x400000` (rip in the dump is `0x401015`). The
   compiler emits `movabs` of the *declared* base for global access; with no
   `.reloc` section the loader places the image elsewhere and every absolute
   global pointer dangles. (The earlier `souc.exe` execute-fault at `0x60DAA5`
   is the same class: a global/pointer computed at the wrong base, here landing
   in mapped-but-wrong memory and getting executed.)
2. **argc/argv are never initialized** (codegen.sio:5121). Even with a correct
   base, the argc global slot is zero because the trampoline never calls
   GetCommandLine.

Why the earlier exit/stdout/file tests passed: those builtins read the
RuntimeContext via **RIP-relative** loads (`emit_load_runtime_context_ptr_rbx`,
frame.sio:385 — `mov rbx,[rip+disp32]`), which are position-independent and
survive the base mismatch. The `arg_count`/`get_arg` globals do not; they use
absolute `movabs`. This is the real fault line for self-hosting.

## Roadmap

1. **Fix global addressing on Windows (root cause #1).** Two options:
   - (a) Emit RIP-relative loads for globals on PE targets (like the
     RuntimeContext path already does), eliminating absolute `movabs`; or
   - (b) Keep absolute addressing but emit a `.reloc` section so the loader
     fixes up pointers, and ensure the global data segment is actually mapped at
     ImageBase.
   Option (a) is cleaner and matches the working RuntimeContext path.
   Validate with the `arg_count()` probe under wine before touching `souc.exe`.
2. **Initialize argc/argv in the PE trampoline (root cause #2).** GetCommandLineA
   → build argv array in the VirtualAlloc heap → store base/count into the
   argc/argv globals (or RuntimeContext). Add GetCommandLineA to .idata imports.
3. **Re-test `souc.exe` under wine** — expect it to read its source/output path
   args and proceed.
4. **Compile a real program with souc.exe under wine** — the milestone: PE
   compiler emits a working binary.
5. **Tier 2 cleanup** — audit `emit_exit` (encode.sio, hardcoded Linux syscall
   60) on Windows fallback paths; signal_handler.sio is POSIX-only (panic path).

## Minimal repro for the root cause

```bash
echo 'fn main() -> i64 with IO, Mut, Panic, Div { return arg_count() }' > /tmp/argv.sio
./bin/souc compile /tmp/argv.sio -o /tmp/argv.exe --target x86_64-windows
WINEDEBUG=-all wine /tmp/argv.exe a b c          # read fault at 0x10000000
objdump -d -M intel /tmp/argv.exe | grep -A6 '<.text>:'   # movabs rax,0x10000000; mov rax,[rax]
```

## Reproduce

```bash
cd <worktree>
bash scripts/ci/windows_pe_smoke_gate.sh          # 6/6 PASS (I/O primitives)
./bin/souc compile self-hosted/compiler/lean_single.sio -o /tmp/souc.exe --target x86_64-windows
echo 'fn main() -> i64 { return 7 }' > /tmp/prog.sio
WINEDEBUG=-all wine /tmp/souc.exe /tmp/prog.sio /tmp/prog.elf   # currently faults
```
