# Bootstrap root A: seed loader

`seed-a` is the first program of bootstrap root A, the Forth root. It is a 271-byte static Linux x86-64 ELF, written by hand byte by byte. Every byte is in `seed-a.hex` with what it does. No assembler, compiler or third-party code produced it.

| File | What it is |
|---|---|
| `seed-a.hex` | The source of record: the annotated listing, in format A |
| `seed-a` | The committed binary: exactly the bytes the listing describes |
| `seed_a_ref.cpp` | A reference model of the decoder, used by the gate as a test oracle only; it is not on the bootstrap path |

## What it does

`seed-a < listing > bytes` reads a listing in format A and writes the bytes it describes.

**Format A**
- A byte is two lowercase hexadecimal digits (`0-9`, `a-f`). The two digits may be separated by whitespace.
- Whitespace is space, tab and newline.
- `#` starts a comment that runs to the end of the line.
- Everything else is refused: uppercase digits, carriage return, any other character, and an odd number of digits at the end of input.

**Exit status**
- 0 when the whole input was valid and every byte was written.
- 1 on a refused input, a failed read or a short write.
- Output is streamed, so on exit 1 the bytes already written must be discarded.

## How the binary is tied to the listing

`scripts/ci/root_a_seed_gate.sh`:

1. **Self-reproduction.** `seed-a < seed-a.hex` must be byte-identical to `seed-a`.
2. **Fixed vectors.** Inputs that must be accepted, with their bytes, and inputs that must be refused.
3. **All 256 byte values** round-trip.
4. **Differential fuzz** against `seed_a_ref.cpp`: the same exit status and the same bytes, on random valid, commented and arbitrary inputs.

## Where this goes

The design is in the decision record of 2026-11-13/14 (project notes, "Bootstrap root"):

- **Root B** has its own loader in octal, written separately. Each loader's listing will also be written in the other's format, so A assembles B and B assembles A (a cross-check at the base).
- The plan is a Lean 4 theorem, over our own minimal x86-64 semantics tested against the CPU: this loader turns every valid format-A listing into exactly the bytes it describes and refuses every other.
- The next stage of root A is a minimal Forth, written in format A and loaded by `seed-a`.
