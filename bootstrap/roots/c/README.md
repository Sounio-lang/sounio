# Bootstrap root C: seed loader

`seed-c` is the third independent bootstrap seed loader. It is a 340-byte static Linux `EM_AARCH64` (`e_machine = 183`) ELF64 executable, written by hand byte by byte, separately from root A (`seed-a`) and root B (`seed-b`), in its own format and instruction set. Every byte is in `seed-c.bin` with what it does. No assembler, compiler, or third-party code produced it.

| File | What it is |
|---|---|
| `seed-c.bin` | The source of record: the annotated listing, in format C |
| `seed-c` | The committed binary: exactly the bytes the listing describes |
| `seed-c.hex` | The same bytes in format A, for the cross-check (`seed-a` assembles it) |
| `seed-c.oct` | The same bytes in format B, for the cross-check (`seed-b` assembles it) |
| `seed-a.bin` | `../a/seed-a`'s bytes in format C, for the cross-check (`seed-c` assembles it) |
| `seed-b.bin` | `../b/seed-b`'s bytes in format C, for the cross-check (`seed-c` assembles it) |
| `spr1-check.bin` | `spr1_check`'s bytes in format C (optional hook in `root_c_seed_gate.sh`; appended once `spr1_check` lands) |
| `seed_c_ref.cpp` | A reference model of the decoder, used by the gate as a test oracle only; it is not on the bootstrap path |

## Format C

- **Bytes.** A byte is a group of exactly eight contiguous binary digits, `'0'` or `'1'`, most significant bit first (`00000000` to `11111111`).
- **Groups are contiguous.** Nothing may stand between the eight digits of a group.
- **Between groups**, and only there: space (`0x20`), tab (`0x09`), newline (`0x0a`), or `':'`, which starts a comment that runs to the end of the line or end of input.
- **Everything else is refused:**
  - any byte outside `'0'`, `'1'`, space, tab, newline, `':'` when outside a comment (including `'2'`-`'9'`, letters, `'#'`, `';'`, carriage return `0x0d`);
  - a separator or `':'` inside a group;
  - an incomplete group at the end of input.

The exit status is 0 on success. It is 1 on a refused input, a failed read, or a short write. Output is streamed, so on exit 1 the bytes already written must be discarded.

## Independence from seed-a and seed-b

| | seed-a | seed-b | seed-c |
|---|---|---|---|
| Architecture | x86-64 (`EM_X86_64 = 62`) | x86-64 (`EM_X86_64 = 62`) | AArch64 (`EM_AARCH64 = 183`) |
| Base | 16, free pairs | 8, fixed three-digit groups | 2, fixed eight-bit groups |
| Separators | anywhere | only between groups | only between groups |
| Comments | `#` | `;` | `:` |
| Accumulator | sentinel bit (`1`) | 0 + 3-digit counter (`r13`) | 0 + 8-bit counter (`x20`) |
| Syscalls | x86-64 `0/1/60` (`syscall`) | x86-64 `0/1/60` (`syscall`) | Linux AArch64 `63/64/93` (`svc #0`) |

## Verification

- Formal model and universal correctness proofs: `formal/Bootstrap/FormatsC.lean`, `formal/Bootstrap/AArch64Min.lean`, `formal/Bootstrap/SeedC.lean`.
- Apple Silicon CPU differential: `bootstrap/model/aarch64_diff_gate.sh` (`MAP_JIT` native execution vs `AArch64DiffModel.lean`).
- Cross-platform seed gate: `bootstrap/model/root_c_seed_gate.sh` (native `aarch64_jit_exec` on Apple Silicon `arm64`, `seed-a`/`seed-b` cross-assembly on Linux `x86_64`, and Lean `aarch64run` differential fuzz on both).
