Madaros: integers wider than 64 bits (`i128`, `i256`, any `iN`/`uN` with
N > 64) now keep every limb across user-function boundaries, `let`
copies and assignment. Before this change, each of these silently kept only
limb 0:

- A wide parameter was bound as one machine word, so `fn
  f(q: i256, d: i256)` computed `q * d` in 64 bits.
- A wide return value travelled in `rax`.
- `let c: i256 = big` and `x = x * 2` on a wide `var` copied a single
  register.
- `(x >> 64) as i64` reported the operand's width, so the shift-and-cast
  spelling the KL-7 print refusal recommends was itself refused.

The 18 `lorenz_i256_*_tiny` certificates went through
`check_div_witness_i256`. They passed before ADR-031 only because both sides
of `q*d + r == n` agreed mod 2^64; after ADR-031 they trapped. The C++
`__int128` oracle shows all seven step-1 witnesses true, and Madaros now
computes the same mask (127; it was 1 once the trap was out of the way).

ABI. A wide parameter travels as the address of a caller-built `[i64;
limbs]` array; the caller extends a narrower literal argument. A
wide-returning function takes a hidden last parameter, a caller-owned
out-array, so nothing escapes the callee's region (ADR-010).

Taking a function with wide parameters or a wide return as a value is
refused, as is a wide parameter past the 15-parameter or 15-limb encoding,
or a return whose width cannot be established. All of these used to narrow
silently.

Pinned by `tests/run-pass/wide_int_fn_abi_high_limbs.sio`, whose values are
the inline computation plus `tests/oracles/wide_int_fn_abi_high_limbs.cpp`.
Main's compiler fails it.
