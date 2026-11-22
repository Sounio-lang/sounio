# Bootstrap root B: seed loader

`seed-b` is the first program of bootstrap root B, the assembly root. It is a 283-byte static Linux x86-64 ELF, written by hand byte by byte, separately from root A's `seed-a`, in a format of its own. Every byte is in `seed-b.oct` with what it does. No assembler, compiler or third-party code produced it.

| File | What it is |
|---|---|
| `seed-b.oct` | The source of record: the annotated listing, in format B |
| `seed-b` | The committed binary: exactly the bytes the listing describes |
| `seed-b.hex` | The same bytes in format A, for the cross-check (seed-a assembles it) |
| `../a/seed-a.oct` | seed-a's bytes in format B, for the cross-check (seed-b assembles it) |
| `seed_b_ref.cpp` | A reference model of the decoder, used by the gate as a test oracle only; it is not on the bootstrap path |

## Format B

- **Bytes.** A byte is a group of exactly three octal digits, `000` to `377`. The digits are 0-7, and the first digit of a group can only be 0-3.
- **Groups are contiguous.** Nothing may stand between the three digits of a group.
- **Between groups**, and only there: space, tab, newline, or `;`, which starts a comment that runs to the end of the line.
- **Everything else is refused:**
  - the digits 8 and 9, and any letter;
  - carriage return;
  - a separator or `;` inside a group;
  - a first digit above 3;
  - an incomplete group at the end of input.

The exit status is 0 on success. It is 1 on a refused input, a failed read or a short write. Output is streamed, so on exit 1 the bytes already written must be discarded.

## Independence from seed-a

The two loaders share no parsing logic: a defect in one cannot be mirrored in the other.

| | seed-a | seed-b |
|---|---|---|
| Base | 16, free pairs | 8, fixed three-digit groups |
| Separators | anywhere | only between groups |
| Comments | `#` | `;` |
| Accumulator | sentinel bit (starts at 1) | starts at 0, separate digit counter |
| Code layout | fail exit at the end, short jumps only | fail exit in the middle, loop closed by a near jump |

What they do share: the x86-64 Linux system-call interface (`read`, `write`, `exit`) and the ELF loader of the kernel.

## How the binaries are tied to the listings

`scripts/ci/root_b_seed_gate.sh`:

1. **Self-reproduction.** `seed-b < seed-b.oct` equals `seed-b`.
2. **Cross-check.** `seed-a < seed-b.hex` equals `seed-b`, and `seed-b < ../a/seed-a.oct` equals `seed-a`.
3. **Fixed vectors**, accepted and refused.
4. **All 256 byte values.**
5. **Differential fuzz** against `seed_b_ref.cpp`.

The two cross listings carry the original listing's annotations verbatim; only the byte tokens are rendered in the other base.
