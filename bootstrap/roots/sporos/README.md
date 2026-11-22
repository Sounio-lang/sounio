# Bootstrap root S: the Sporos interpreter

The stage-1 interpreter of bootstrap root A (design decisions 24, 31–39):
a static Linux x86-64 ELF, written by hand byte by byte in format A, like
`seed-a`. No assembler, compiler or third-party code produces it.

| File | What it is |
|---|---|
| `sporos-stage0.hex` | Source of record: the annotated listing, in format A. Stage 0 is the loader: `open(argv[1])`, `mmap`, read-all with `mremap` growth, write to stdout. Exit 0/1/2. |
| `sporos-stage0` | The committed binary: exactly the bytes the listing describes (assembled by `seed-a` on the Linux gate; see below). |
| `sporos.hex` | Later slices: checker, dispatch table, handlers. One file per slice, same discipline. |

## Decisions → bytes

- 24: hand-written hex, format A. This directory.
- 31: the checker runs inside the interpreter (slice 2: `sporos.hex`).
- 32: binary bytecode, one opcode byte + fixed immediates (`formal/Sporos/Bytecode.lean`).
- 33: program by file (`argv[1]`, `open`), input by stdin. Stage 0 opens the file; later slices read stdin into the input buffer.
- 34: one Hoare triple per handler, over symbolic x86 state. Triples live in `formal/Bootstrap/SporosX*.lean` and follow the `seed-a` pattern (`entryCheck`/`iterCheck`/`decide`); they need instruction semantics over `X86Env` memory (queued behind `formal/Bootstrap/X86Env.lean` "Next"), so stage 0 ships with the differential story first.
- 35: no fixed ceiling. Stage 0 already grows with `mremap` (double on full); `ENOMEM` exits 2 and writes nothing more.
- 38: dispatch through a 256-entry jump table with one indirect jump (slice 2; needs `X86Min.jmpReg`, this branch).
- 39: a Sporos value is two 64-bit words (value + DAG node index); stacks move in 16-byte steps (slice 2+).

## Register use (all slices)

| Reg | Holds |
|---|---|
| `rbx` | base of the file/region being filled |
| `r12` | capacity of that region, in bytes |
| `r13` | bytes held so far |
| `r14` | input file descriptor |
| `rax` | syscall number, then its return |
| `rdi rsi rdx r10 r8 r9` | syscall arguments (Linux convention) |

## Exit statuses

| Code | Meaning |
|---|---|
| 0 | done; output is complete |
| 1 | refused input, failed `open`/`read`, or short `write` (fail closed) |
| 2 | the kernel refused memory (`mmap`/`mremap`); nothing more is written |
| 3 | reserved: all three copies differ (decision 49, TMR slices) |

## Stage 0 layout (`sporos-stage0.hex`)

ELF64 header + one `PT_LOAD` (read+execute), exactly like `seed-a` (load at
`0x400000`, code at file offset `0x78`). Entry: pop `argc`, require `argc ≥ 2`,
pop `argv[0]` and `argv[1]` (the path); `open(path, O_RDONLY)`; `mmap` 64 KiB
anonymous; read loop (`read` → append; partial reads accumulate; full region →
`mremap` double); `write(1, base, total)` with a length check (short write →
exit 1); `exit(0)`.

Only instruction forms the model decodes are used; the two forms stage 0 needs
beyond the seeds — register `add`/`sub` (`01`/`29`, `81`/`83` `/0` `/5`) and
nothing else — are added to `formal/Bootstrap/X86Min.lean` on this branch, with
differential cases owed in `bootstrap/model/x86diff_cpu.cpp`.

## Verification story

1. Static: every byte is commented; instruction boundaries, jump targets and
   ELF/`load` constraints are checked mechanically (throwaway script; the gate
   re-checks).
2. Differential: the binary runs on Linux against fixed vectors (empty file,
   one byte, exactly 64 KiB, 64 KiB + 1, missing file, `argc < 2`) and fuzz.
3. Proof: triples per handler once instruction semantics over `X86Env` memory
   lands; stage 0's loader triple is the first one.

## Slices

- Stage 0 (this file): loader. No ceiling, fail-closed statuses.
- Stage 1: SPR1 header check + bytecode decode + the Lean `Check` logic in hex; refuses with exit 1.
- Stage 2: 256-entry dispatch + echo-subset handlers (`getb putb exit` + stack plumbing), single copy.
- Stage 3: all 70 words, single copy, proved against `dstep`.
- Stage 4: TMR triplication (decisions 43–51): three mappings, vote after every step, triple output, reported bound.
