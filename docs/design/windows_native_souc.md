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

1. **Global/BSS base mismatch.** The compiler (built from `lean_single.sio` via
   the Makefile bootstrap) hardcodes its global/BSS segment base at absolute
   `0x10000000` and emits `movabs rax, 0x10000000+off; mov rax,[rax]` for every
   global access:
   - `lean_single.sio:22857` `GL_BSS_BASE = 0x10000000 + 16`
   - `lean_single.sio:24059` `argc_addr = 0x10000000`, `:24060` `argv_addr = 0x10000000 + 8`
   - `:24129/:24301/:24332` BSS program header `p_vaddr = 0x10000000`

   But the emitted **PE maps `.bss` at VA `0x600000`** (ImageBase `0x400000` +
   RVA `0x200000`; confirmed via `objdump -h`). So every absolute global address
   points into unmapped memory. On Linux/ELF it works only because lean_single's
   own ELF emitter puts the BSS program header at exactly `0x10000000`, matching
   the hardcoded constant — that consistency is lost on the PE path.

   **Correction to an earlier note in git history (commit 810168e5): this is NOT
   a PE ImageBase relocation problem.** The PE's declared ImageBase IS
   `0x400000` and it loads there cleanly (rip=0x401015). `0x10000000` is the
   compiler's hardcoded *globals* base, unrelated to ImageBase. (`pe_coff.sio:71`
   `PE_SCN_MEM_SHARED = 268435456` is the same numeric value but an unrelated
   section-flag bit — a red herring.)

2. **argc/argv are never initialized** (codegen.sio:5121). Even with a correct
   base, the argc slot is zero because the trampoline never calls GetCommandLine.

Why the earlier exit/stdout/file tests passed:
- `print` materializes its string literal as **inline `movabs` immediates**
  pushed to the stack (verified: `movabs rax,0x6f646e6977206d6f` = "o mwind"),
  so it touches no global data segment.
- exit/stdout/file builtins read the RuntimeContext via **RIP-relative** loads
  (`emit_load_runtime_context_ptr_rbx`, frame.sio:385), position-independent.
- `hello_print.exe` *does* contain `movabs 0x100000a0` global accesses, but they
  sit in **un-executed branches** (int/float formatting that a pure-string print
  skips), so the bad base is never dereferenced. `arg_count` is the smallest
  program that puts a global access on the executed path.

## Root cause #1: FIXED + verified (2026-05-21)

`write_pe` (lean_single.sio) placed `.bss` at RVA `0x200000` (VA `0x600000`),
contradicting the global-access base `GL_BSS_BASE = 0x10000000`. One-line fix:
```
let bss_rva = 0x10000000 - image_base   // .bss VA == GL_BSS_BASE
```
**Proof:** rebuilt the compiler with the fix, recompiled the `arg_count()` probe
to PE, ran under wine:
- before: read fault at `0x10000000`;
- after: `.bss` VA = `0x10000000` (objdump -h), clean exit 0 (argc=0, since
  bug #2 still pending — exactly the predicted behaviour).

No regression: exit-code / stdout / file-round-trip PE programs still pass under
wine with the fixed compiler. Bug #2 (argc/argv init) is now the next blocker.

> **NOTE (2026-05-21):** `bin/souc-linux-x86_64` had not actually been
> rebuilt when this section was written, so the shipped binary still emitted
> `.bss` at `0x600000` and the probe faulted (exit 5). Bug #2's landing rebuilds
> and reinstalls the bootstrap binary (`make build` → install gen3), which is
> what makes both fixes live.

## Root cause #2: FIXED + verified (2026-05-21)

The Windows entry trampoline (`lean_single.sio`, `TARGET_OS == 3` branch) now
calls `GetCommandLineA` and parses the ANSI command line into argc/argv before
calling `main`:

- **`emit_pe_argv_init_x86()`** — emitted just after stack alignment in the PE
  trampoline. Calls `GetCommandLineA` (IAT slot 1), skips the program-name token
  (quote-aware), space-splits the remaining args, records each token start into a
  reserved BSS pointer table (`ARGV_TABLE_BSS_OFF`, 64 slots) and NUL-terminates
  it in place. Stores argc (real-arg count, program name excluded) at
  `GL_BSS_BASE-16` and argv base at `GL_BSS_BASE-8` — matching `emit_arg_count`/
  `emit_get_arg` and the Linux/macOS convention where `get_arg(0)` is the first
  real arg. All control flow uses rel32 jumps backpatched with `em32_at`.
- **`write_pe`** — `.idata` import table extended from 1 to 2 kernel32 imports
  (ExitProcess + GetCommandLineA): second INT/IAT thunk, a `GetCommandLineA\0`
  hint/name entry, DLL-name offset shifted to 124, `idata_total` 117→137. The
  trampoline's `call [rip+disp32]` for GetCommandLineA is patched to IAT slot 1.
- Stack: trampoline now does `and rsp,-16; sub rsp,32` (16-aligned + Win64 shadow
  space) so both `GetCommandLineA` and `main` are entered correctly aligned.

**Proof (wine, rebuilt + reinstalled compiler):**
- `arg_count()` probe + `get_arg` dump: `wine dump.exe alpha beta gamma` prints
  `3` then `[alpha] [beta] [gamma]`; quoted `"hello world" solo` → `2`,
  `[hello world] [solo]`.
- **Milestone (roadmap 3–4):** `wine souc.exe prog.sio prog.elf` reads both path
  args, compiles, and the emitted ELF runs (`return 7` → exit 7). souc.exe exits
  `0` on success, `1` on a missing-input failure.
- `scripts/ci/windows_pe_smoke_gate.sh` = 6/6 PASS; Linux self-host fixed point
  (`make build`) still holds (stage2 == stage3).

Known minor artifact (does **not** affect the compiler), characterized 2026-05-21:

- A `main` that returns a *runtime-derived* value **directly** as its process
  exit code, with **no** intervening effectful statement, reports an unstable
  small garbage value under wine (`return arg_count()` with argc=3 → exit 1;
  `let n=arg_count(); let m=n; return m` → 0). The same source returns the
  correct value on Linux, so the function body codegen is fine — the divergence
  is confined to the Windows main→ExitProcess hand-off for this degenerate shape.
- It is **not** about reading globals: `let n=arg_count(); return 42` → 42, and
  `return 99` → 99. Any arithmetic on the value (`arg_count()+100` → 103) or any
  intervening syscall-backed effect (`print_int`, `write_file`) makes the exit
  code correct. `get_arg`/`arg_count` values themselves are always correct
  (verified by dumping them to stdout and to a file).
- Real programs are unaffected: they compute their exit status and perform I/O.
  `souc.exe` exits **0** on a successful compile and **1** on a missing-input
  failure — both verified under wine. Root-causing the degenerate-main exit path
  is deferred (likely a tail-return materialization quirk specific to the PE
  trampoline boundary); it does not gate the milestone.

## Roadmap

1. ~~**Fix the global/BSS base on Windows (root cause #1).**~~ **DONE** (above).
   Options considered, simplest first:
   - (a) **Make the hardcoded globals base agree with the PE `.bss` VA.** Either
     emit the PE `.bss` at VA `0x10000000` to match `GL_BSS_BASE`, or make
     `GL_BSS_BASE` target-dependent (= the PE `.bss` VA, 0x600000). Smallest
     change; keeps absolute `movabs`. Risk: must stay consistent across every
     site that bakes `0x10000000` (argc/argv/r15 table/BSS phdr).
   - (b) Emit RIP-relative loads for globals on PE targets (like the
     RuntimeContext path), eliminating absolute addressing entirely. Cleaner and
     ASLR-friendly, but touches every global access site.
   Validate with the `arg_count()` probe under wine before touching `souc.exe`.
2. ~~**Initialize argc/argv in the PE trampoline (root cause #2).**~~ **DONE**
   (above). GetCommandLineA → parse into a BSS argv table → store base/count into
   the argc/argv globals. GetCommandLineA added to .idata imports.
3. ~~**Re-test `souc.exe` under wine**~~ **DONE** — reads source/output path args.
4. ~~**Compile a real program with souc.exe under wine**~~ **DONE** — milestone met:
   `souc.exe prog.sio prog.elf` emits a working ELF (`return 7` → exit 7).
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
