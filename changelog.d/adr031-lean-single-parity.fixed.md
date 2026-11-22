lean_single now fails closed on integer overflow and division, like Madaros
(ADR-031 R1/R2). Before, the same program could wrap silently, or die on
SIGFPE (`/` or `%` by zero, `MIN / -1`), under lean_single while Madaros
refused it.

In lean_single's x86 emission, i64 `+ - *` and unary `-` check the overflow
flag and u64 `+ - *` check the carry flag (`mul` for the unsigned product).
A zero divisor and `MIN / -1` (also `MIN % -1`) are caught before
`div`/`idiv`. The failing path prints the same line as Madaros, for example
`sounio: integer overflow (i64 *)` or `sounio: division by zero`, and exits
1. A bare literal operand of unary `-` is exempt, so `-9223372036854775808`
still materialises. lean_single's own lexer accumulation and literal folds
now use `wrapping_*`, and the guarded lean_single reaches its fixed point.

Turning the guards on found code that relied on wrap. Where the wrap was
intended (hashes and RNGs in `random/`, `collections/`, `viz/`, and
test-local generators), it is now spelled `wrapping_*`. Where it was a bug,
it is fixed:
- `ir/egraph.sio`: the GUM sum of squares at the uncertainty clamp
  overflowed; it now saturates at 3037000499^2.
- `graphics/epistemic`: a constant band's NaN pixel row overflowed.
- `math/rational`: the overflow probes overflowed themselves.
- One test byte-weight overflow.

The aarch64 paths of lean_single are not changed. The seed is refreshed in
the stacked seed PR (ADR-031 R3).
