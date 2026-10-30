# f128 and f256: Madaros vs lean_single

Moved verbatim from `CLAUDE.md` §13 (2026-10-30). Measurements are dated inline.

- **Dual-engine divergence.** `f128` is a working binary128 type on **both** engines, but they cover different surfaces. Both give 113 halvings to `1 + eps == 1` (binary64 gives 53) and print `1/3` as `3.33333333333333333333333333333333317e-0001` via `math::softfloat_f128_fmt::print_f128`.
  - **Madaros** computes `+ - * / %`, unary minus and IEEE compares, and supports params/returns, struct fields, methods, fixed arrays, exact 36-digit printing and exact literals. Measured: `7.5 % 2.0 = 1.5`; `2/3` through a struct field and an array element is correct to the last digit. It refuses, fail-closed:
    - `as` casts between `f128` and `f64` (**E248**, "wide-float casts fail closed");
    - values that flow through an `f128` global or a `type` alias over `f128` (V0-E.4.1 lowering refusal). The declarations alone build: an unused `let G: f128` or `type Wide = f128` compiles;
    - mixing an `f128` with an untyped literal or an `f256` (**E004**). Write typed constants, e.g. `let three: f128 = 3.0; x / three`, not `x / 3.0`.
  - **lean_single** (#2387/#2426) computes binary128 through libgcc for locals, direct-call params/returns, IEEE compares, and `as` casts in both directions. It refuses, fail-closed: struct fields, arrays, reading an `f128` global (an unused one builds; a `type` alias over `f128` works), tuples, methods, fn values, `println` of an `f128` (the stdlib `print_f128` works), and `%` ("modulo requires integer operands").
  - **Literal exactness differs by engine.** Measured 2026-09-26:
    - **Madaros** parses an f128 literal straight to binary128 (V0-E.5.9) and accepts the literals its parser can prove exact in binary128. Decimal digits accumulate in 128 bits (`self-hosted/parser/f128_literal.sio`), so a very long decimal is refused even when its value is exact, such as the full expansion of 2^200. `0.1` fails. `1e23` prints exactly, and so does the hex-float `0x1.0000000000000001p+0` (1 + 2⁻⁶⁴). The hex-float exponent sign is optional, as in C99: `…p0` and `…p+0` are the same literal (the unsigned form was refused with E012/E137 until the parser fix in `docs/audit/MADAROS_HEX_FLOAT_UNSIGNED_EXPONENT_DISPATCH_2026-09-26.md`).
    - **lean_single** widens through binary64 and refuses any literal not exact in binary64 (`f128_widen_refuses_inexact_literal` in `lean_single.sio`), so it also rejects `1e23` and that hex-float.
    - Portable constants: small dyadic decimals that are exact in binary64 (`0.25`, `3.0`), `f128_from_limbs(lo, hi)` (works on both), or computation (`one / ten`).
  - **Not implemented anywhere:** `Knowledge<f128>` and GUM over `f128` (KL-15).
  - **`f256`** on Madaros: declarations and some operators typecheck (an unused f256 enum field or global builds), but executable f256 value lowering is refused fail-closed (V0-E.4.1), never lowered as f64. lean_single refuses `f256` outright (KL-9). Only `stdlib/math/softfloat_f256.sio` add/sub over `F256Bits` computes f256 values (KL-15a); there is no f256 mul, div, fields, arrays or printing.
