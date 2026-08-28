<!-- docs:meta
topic_id: repo.docs.dissertation.results.m5-gum-4th-order-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.m5-gum-4th-order-v1
-->

---
docs:meta: true
topic: dissertation-results
kind: quantitative-output
drug: rapamycin
model: PBPK28
status: implementation-complete
version: m5-v1
date: 2026-05-14
---

# M5 GUM Fourth-Order Cumulant Budget - v1

## 4.14.1 Why second-order saturates

M6 established the post-update PBPK28 rapamycin baseline under the adult
transplant lognormal prior:

| Quantity | Value |
|---|---:|
| `u_MC` | 0.357945 mg.h/L |
| `u_Hessian` | 0.295160 mg.h/L |
| `rel_Hess` | 0.175405 |

M2 tested whether the residual could be explained away by splitting the prior
into eta/epsilon levels. It could not: the conditional individual row remained
near 19% relative Hessian residual. The remaining discrepancy is therefore a
higher-moment problem. PBPK28 AUC is dominated by positive clearance and free
fraction uncertainties; once these are represented by lognormal inputs, the
input perturbations have non-zero skewness and excess fourth cumulant. A
quadratic Hessian correction assumes the normal fourth central moment structure
and cannot absorb the lognormal tail budget.

## 4.14.2 Fourth-order variance expansion

Let `X_i = theta_i - E[theta_i]` be independent centered input perturbations and
expand the scalar endpoint `Y=f(theta)` componentwise:

```text
f ~= f0 + c_i X_i + 1/2 d_i X_i^2 + 1/6 e_i X_i^3
```

where `c_i = df/dtheta_i`, `d_i = d2f/dtheta_i2`, and
`e_i = d3f/dtheta_i3`. Keeping variance terms through fourth central order gives

```text
Var(Y) ~=
  sum_i c_i^2 mu2_i
+ sum_i c_i d_i mu3_i
+ sum_i 1/4 d_i^2 (mu4_i - mu2_i^2)
+ sum_i 1/3 c_i e_i mu4_i
+ cross Hessian terms.
```

Writing `kappa4_i = mu4_i - 3 mu2_i^2` converts the diagonal quadratic term to

```text
1/4 d_i^2 (kappa4_i + 2 mu2_i^2).
```

For a normal input, `mu3=0` and `kappa4=0`, so this reduces to the usual
second-order GUM diagonal contribution `1/2 d_i^2 sigma_i^4`. M5 therefore
reuses the existing full PBPK28 Hessian budget, including off-diagonal
`1/2 H_ij^2 sigma_i^2 sigma_j^2`, and adds only the non-normal diagonal
corrections:

```text
Var_M5(Y) =
  Var_Hessian_full
+ sum_i c_i d_i kappa3_i
+ sum_i 1/4 d_i^2 kappa4_i
+ sum_i 1/3 c_i e_i mu4_i
```

This is an asymptotic cumulant budget, not a replacement for Monte Carlo. Its
role is to explain and reduce the Hessian-MC residual using explicit
higher-moment terms.

## 4.14.3 Lognormal cumulants

For a lognormal input with physical mean `m` and log-shape `s2`, where
`s2 = ln(1 + variance/m^2)`, the raw moments are

```text
E[X^r] = m^r exp(1/2 r(r-1)s2).
```

The centered third moment, fourth central moment, and fourth cumulant are:

```text
kappa3 = m^3 (exp(3s2) - 3 exp(s2) + 2)
mu4    = m^4 (exp(6s2) - 4 exp(3s2) + 6 exp(s2) - 3)
kappa4 = mu4 - 3 variance^2.
```

The launch prompt called the `mu4` expression `kappa4`; the implementation keeps
the two names separate because the Pébay accumulator finalizer returns the
actual fourth cumulant (`mu4 - 3 variance^2`), while the cubic derivative term
needs `mu4`.

## 4.14.4 Derivative extraction

This M5 branch intentionally does not cherry-pick or depend on the local D.2
autograd tape commit. Repo `origin/main` contains C/M6/D1/D5 but not D.2, and
the PBPK28 endpoint is already available as a scalar deterministic model.

Derivative extraction is therefore direct central finite difference:

```text
d1 ~= (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
d2 ~= (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)
d3 ~= (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / (2h^3)
```

The finite-difference step for PBPK28 uses
`h = max(1e-4*|mean|, 0.02*sd)` to keep the third derivative stable on the
current fixed-step PBPK runtime. The analytical validation uses
`AUC = D/(CL*V)`:

| Derivative | FD | Analytical | Relative error |
|---|---:|---:|---:|
| `dAUC/dCL` | -0.032518 | -0.032518 | 8.45e-13 |
| `d2AUC/dCL2` | 0.005245 | 0.005245 | 1.34e-9 |
| `d3AUC/dCL3` | -0.001269 | -0.001269 | 5.10e-5 |

Function pointers were not needed for the PBPK28 gate.

## 4.14.5 Pébay accumulator

`stdlib/darwin_pbpk/cumulants.sio` adds a single-pass central-moment accumulator
with fields `n`, `m1`, `m2`, `m3`, and `m4`. The update is the Pébay/West
generalization of Welford's online variance recurrence and finalizes to:

```text
mean     = m1
variance = M2/n
kappa3   = M3/n
kappa4   = M4/n - 3(M2/n)^2
```

The focused deterministic Gaussian-moment fixture has `kappa3 = 0.0` and
`kappa4 = 4.44e-16`. The lognormal CL prior has positive tail cumulants:

```text
sigma2_log(CL) = 0.134880
kappa3(CL)     = 125.007834
kappa4(CL)     = 1302.266361
```

## 4.14.6 Convergence study

Focused command:

```bash
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 \
  bin/souc run tests/run-pass/pbpk28_m5_gum_4th_order.sio
```

Output:

| Quantity | Value |
|---|---:|
| canonical `u_MC` | 0.357945 |
| `u_1st` on Hessian grid | 0.260506 |
| full `u_Hessian` | 0.295160 |
| M5 `u_total` | 0.378674 |
| `rel_Hess` | 0.175404 |
| `rel_fourth` | 0.057910 |
| dominant correction index | 0 (`CL_hep`) |
| skewness variance term | -0.050860 |
| excess fourth-cumulant variance term | 0.019447 |
| cubic derivative variance term | 0.087687 |

The fourth-order cumulant budget improves the residual from 17.54% to 5.79%,
which satisfies the dissertation 10% weakly nonlinear criterion for this M6
configuration. In absolute terms, the fourth-order estimate differs from the
canonical MC uncertainty by about 0.0207 mg.h/L. The correction is CL-dominated
because `CL_hep` carries the largest M6 variance share, appears in the dominant
inverse-clearance AUC relationship, and its cubic derivative is amplified by
the lognormal fourth central moment.

Gate marker:

```text
M5_GUM_FOURTH_ORDER_CUMULANT_BUDGET_PASS
```

## 4.14.7 Positioning

Wang and Iyer operationalize second-order GUM with symbolic derivatives, and
the R package `propagate` similarly stops at second-order propagation for
routine use. Mekid and Vaja derive higher-order expressions for very small
systems but do not provide a PBPK-scale operational cumulant budget. JCGM
Supplement 2 treats Monte Carlo as the standard route once the linear Taylor
regime is no longer adequate.

The M5 contribution is narrower and more concrete: for the PBPK28 dissertation
model, it implements an explicit fourth-order cumulant budget with stable online
cumulant checks, analytical lognormal cumulants, finite-difference derivative
extraction, and direct comparison to the canonical M6 Monte Carlo run.

## 4.14.8 Caveats

The v1 implementation is diagonal for non-normal third and fourth cumulant
terms and reuses the existing full Hessian only for the normal second-order
cross budget. Correlated prior cross-cumulants from M1 are therefore not folded
into this branch. The third derivative is finite-difference based rather than
autograd based by design, because D.2 was not present on `origin/main` for this
lane and is unnecessary for the scalar PBPK28 endpoint.
