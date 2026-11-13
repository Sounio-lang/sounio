<!-- docs:meta
topic_id: repo.docs.decisions.adr-031-integer-overflow-division-and-f64-text
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-031-integer-overflow-division-and-f64-text
-->

# ADR-031: Integer overflow, integer division by zero, and the text form of `f64`

**Status**: accepted (implementation pending)
**Date**: 2026-11-13
**Decided by**: the founder, choosing among the options presented in the 2026-11-13 decision session
**Builds on**: [ADR-015](adr-015-lexical-structure.md) (literals that do not fit are refused), [ADR-023](adr-023-checked-indexing-and-proof-erased-panic.md) (checked operations panic; a check is removed only when proven), [ADR-029](adr-029-bootstrap-provenance-root.md) (the committed seed is the provenance root)

## Context

Five open pull requests change what integer arithmetic and number printing do, and they do not follow one rule. Their state on 2026-11-13:

| PR | Change | Behaviour on overflow |
|---|---|---|
| #3435 (SIO-207) | Signed `add`/`sub`/`imul`/`neg` trap at run time on Madaros; `wrapping_add/sub/mul` builtins; E300 for constants out of range of `i8`/`u8`/`i32`/`u32` annotations | Signed: fails |
| #3505 (SIO-588) | Unsigned operations lower to new `OpUAdd`/`OpUSub`/`OpUMul`; dynamic `i32` results sign-extend the low 32 bits and `u32` results mask them | Unsigned: wraps to 0. Narrow widths: truncate silently |
| #3502 (SIO-428) | Decimal literals in (2^63, 2^64−1] in a signed position are refused (E282) | Compile time: refused |
| #3507 (SIO-78) | Type suffixes (`5u8`, `7i32`, `2.5f64`), out-of-range suffixed literals refused (E276); also refreshes `bin/souc-lean-single-x86_64` and its `SeedReceipt` | Compile time: refused |
| #3506 (SIO-396) | Madaros prints an `f64` with magnitude ≥ 2^63 as its exact decimal value, `%f` form | — |

Under these PRs as written, the same `MAX + 1` would fail for `i64`, become `0` for `u64`, and truncate for `i32`. Integer division by zero is undefined: #3435 leaves it as a follow-up. For `f64` values with magnitude ≥ 2^63, the two engines print different notations: Madaros with #3506 prints `10000000000000000000.000000`, `lean_single` prints `1.000000e19`.

## Decision

**R1 — Integer overflow fails, for every integer type.**
- **Run time.** For `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32` and `u64`, the operations `+`, `-`, `*` and unary `-` panic when the mathematical result is outside the range of the operand type. This holds with or without sign, and at every width: an `i32` or `u32` result that does not fit 32 bits panics; it is not sign-extended, masked or truncated.
- **Panic.** The panic has the shape of [ADR-023](adr-023-checked-indexing-and-proof-erased-panic.md): it names the operation, the type and the source span.
- **Compile time.** A constant expression whose value is outside its type is refused at compile time (E300 extended to all widths).
- **Modular arithmetic.** It is written explicitly with `wrapping_add`, `wrapping_sub`, `wrapping_mul` (and `wrapping_neg`) for every integer type.
- **Removing a check.** A check is removed only when the range is proven, by the mechanism of ADR-023 §4.
- **Engines.** The rule applies to both engines; Madaros first.

**R2 — Integer division and remainder fail on zero and on the unrepresentable quotient.**
- `/` and `%` on integers panic when the divisor is zero. With the same panic shape, signed division panics when the quotient does not fit, which is `MIN / -1`.
- A constant divisor of zero is refused at compile time.

**R3 — The committed seed changes only in its own pull request.**
- A change to `bin/souc-lean-single-x86_64` and its `SeedReceipt` is the whole content of a dedicated pull request with its receipt. It is never carried inside a language or compiler change.
- When a `lean_single.sio` change needs a new seed to pass the seed gate, the seed pull request is stacked on it and the two are merged in sequence.

**R4 — The text form of a finite `f64` is the exact decimal with six fraction digits.**
- `print`/`println`/`print_f64` of a finite `f64` produce the digits of C++23 `std::printf("%.6f")` for the same value, at every magnitude and on both engines.
- `lean_single`'s scientific notation for magnitudes ≥ 2^63 is a divergence to be corrected.
- The text forms of NaN and the infinities are outside this ruling.

## Consequences for the open pull requests

- **#3435** is the base of R1 for signed types. It needs a rebase onto `main` (conflicting on 2026-11-13). Its division-by-zero follow-up becomes R2.
- **#3505** changes:
  - Unsigned `OpUAdd`/`OpUSub`/`OpUMul` panic on carry or borrow instead of wrapping.
  - Dynamic narrow results are range-checked instead of sign-extended or masked.
  - Its witnesses `u64_wrap_arith` and `i32_dynamic_wrap` become panic witnesses, and their wrapping cases move to `wrapping_*`.
- **#3502** stays as written (refusal at compile time agrees with R1). It needs a rebase.
- **#3507** drops the seed refresh: R3 moves it to its own stacked pull request. It also needs a rebase.
- **#3506** stays as written for Madaros. `lean_single` follows in a separate change (R4).

## Release class

Under [ADR-018](adr-018-release-cycle-and-versioning.md):
- R1 and R2 are `semantics`: programs that wrapped silently, or divided by zero, now panic.
- The compile-time refusals are `refusal`.
- R4 changes printed text for magnitudes ≥ 2^63 on `lean_single`, so it is `numerical-correction`, with C++23 `printf` as the oracle.
- R3 is `internal`.
