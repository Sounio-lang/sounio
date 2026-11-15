The aarch64-linux lane of the lean_single compiler now propagates GUM
first-order variance by the same rules as x86-64. Before this change, on
arm64:

- `+` and `−` used the blind `var(a) + var(b)`. Now they add the
  covariance `2·Σ s_k(a)·s_k(b)·u_k²`, so `e + e` reads `4·var(e)` and
  `x − x` reads 0.
- `/` carried neither variance nor sensitivity; `variance_of(e / 100.0)`
  read 0. It now applies the quotient rule to both, Hessian included.
- `sqrt`, `exp`, `ln`, `sin`, `tanh`, … returned the argument's variance
  unscaled. They now give `f'(u)²·var(u)`.
- `a·b` gained the correlation term, so `h·h` is `4h²σ²`.
- A loop-carried accumulator no longer correlates each iteration's fresh
  `measure()` with the last one. The pooled-mean case read 619.444 against
  an analytic 169.444.

Subtraction on arm64 uses `var(a) + var(b) − 2·cov(a, b)`. With only 8
channels, the channel-difference form x86 uses would drop the variance of
every input past the eighth.

On the 230 epistemic tests, 23 more now give the same output on aarch64
(under qemu) as on x86, and none got worse. Those 23 are pinned by the new
`a64_epistemic_variance_parity` gate, which the previous seed fails on all
23. Variance through calls to user functions still diverges on arm64,
because the lane has no interprocedural channels (KL-11).
