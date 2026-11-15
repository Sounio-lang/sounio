Code whose arithmetic is modular by definition now says so with
`wrapping_add`/`wrapping_sub`/`wrapping_mul`, which ADR-031 R1 requires. Every
generator and hash produces the same bits as before; under ADR-031 the plain
operators trapped.

- **Generators.** xorshift128+ in `stdlib/random`, `stdlib/ssm`,
  `stdlib/molecular` and about 100 tests and examples; xorshift64*; the MMIX
  LCG; the unbounded `mnist` LCGs.
- **Hashes.** The `viz` frame and audit hashes; `djb2` in `pkg` and the
  bootstrap verifier.
- **Overflow checks.** `math::rational`'s overflow predicates no longer
  perform the overflowing operation they test for. The check is exact for
  every input, including `-1` with `i64::MIN`.
- **`theorem::{cdcl,smt}`.** The i32 hashes promote their operands to i64
  before multiplying, as the code's own note says the old semantics did.

Real defects the trap exposed are fixed rather than wrapped:

- `graphics::{epistemic,tiled_plot}` drew a constant band through
  `0/0 = NaN` (`NaN as i64` overflowed the row arithmetic). The band now sits
  on the baseline, as the center line already did.
- `issue11`'s byte reassembly computed a dead `2^64` factor after the last
  byte.

This closes the 44 new failures that the Madaros corpus regression measured
on `main`.
