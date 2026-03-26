# Hyper Uncertainty Parenthesization: Octonions vs Sedenions

Reproduction command:

```bash
python3 scripts/research/generate_hyper_uncertainty_parenthesization.py
```

Artifact:

- `artifacts/research/hyper_uncertainty_parenthesization.v1.json`

## Main Result

The naive objective from the original idea,

- minimize `trace(Sigma_out)`

does **not** distinguish `(a*b)*c` from `a*(b*c)` for octonions under
first-order GUM propagation with full covariance tracking.

In the sampled model, octonion total variance is parenthesization-invariant,
while sedenion total variance is not.

## Why Octonion Trace Ties

For octonions, left and right multiplication are scaled orthogonal:

- `L_x^T L_x = ||x||^2 I`
- `R_x^T R_x = ||x||^2 I`

The artifact verifies this numerically using the exact same Cayley-Dickson sign
recursion as the compiler.

That gives the trace recurrence

- `trace(Sigma_(x*y)) = ||y||^2 trace(Sigma_x) + ||x||^2 trace(Sigma_y)`

and therefore

- `trace(Sigma_((a*b)*c)) = trace(Sigma_(a*(b*c)))`

because octonion norms are multiplicative:

- `||a*b||^2 = ||a||^2 ||b||^2`

So a compiler that extracts by total variance alone would see every legal
octonion bracketing as a tie.

## What Still Changes For Octonions

Although the traces tie, the covariance matrices do not.

The artifact includes deterministic examples showing:

1. A component variance can prefer the left bracketing.
2. Another component variance can prefer the right bracketing.
3. A scalar projection `q^T x` can strongly prefer one side.

That means octonion uncertainty optimization is still real, but the objective
must be tied to an observation:

- one output lane
- a real-part readout
- a norm-derived scalar
- a downstream measurement projection

In short:

- algebra determines what can be rewritten
- observation determines what uncertainty objective matters

## Why Sedenions Are Different

Sedenions are not a composition algebra, so the scaled-orthogonality argument
breaks. The artifact finds immediate left-win and right-win examples for
`trace(Sigma_out)` itself.

This is a strong compiler-design split:

- **Octonion**: optimize observation-specific variance, not total trace
- **Sedenion**: total trace is already a meaningful extraction objective

## Compiler Implication

The right future compiler objective is not:

- `min trace(Sigma_out)`

but rather something like:

- `min Var(h(expr))`

where `h` is the actual observation functional exposed by the program:

- component read
- comparison
- IO formatting
- FFI scalar extraction
- explicit `observe` boundary

That is the place where uncertainty-aware extraction becomes semantically
grounded rather than arbitrary.
