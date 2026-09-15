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

Focused command, as originally recorded. It does not work as written, and how the recorded
output was actually produced cannot be established from this repository's history (see the note
after it):

```bash
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 \
  bin/souc run tests/run-pass/pbpk28_m5_gum_4th_order.sio
```

> **Reproduction status (2026-09-13, updated 2026-09-14).** `bin/souc` execs `SOUNIO_SOUC_BIN` with its arguments
> unchanged (it already did at the merge that added this file, `bebd78d74c`; earlier history is
> not in this clone). Measured on 2026-09-13 with `bin/souc-linux-x86_64`, before `bin/souc` began
> refusing the form: called that way, lean_single stopped at `error: no main` (a current-source lean_single shows
> why: it opens `run`, which does not exist, as a 0-byte source). `bin/souc` now refuses the form (exit 64). The pinned binary (sha256
> `3cbea2b4…`) is no longer in the repository.
> **Current state (2026-09-14, re-measured at HEAD `2a8e7ad145`).** Since `36f9b34cc7`, `main` declares `Epistemic`, the
> known-failure tag is gone, and the test carries `//@ timeout: 90`. The command below needs its
> `chmod +x` step: this lean_single build (sha256 below) wrote the ELF with mode `-rw-r--r--` under
> umask 0022, and without that step, as in the command this note gave before, running the ELF
> stopped at `Permission denied` (exit 126). Run through lean_single's raw interface, from the
> repository root because it resolves stdlib imports relative to the working directory,
> `cd "$(git rev-parse --show-toplevel)" && bin/souc-linux-x86_64 tests/run-pass/pbpk28_m5_gum_4th_order.sio /tmp/m5_gum_4th_order.elf && chmod +x /tmp/m5_gum_4th_order.elf && /tmp/m5_gum_4th_order.elf`
> (`bin/souc-linux-x86_64` sha256 `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`)
> printed `M5_GUM_FOURTH_ORDER_CUMULANT_BUDGET_PASS` and `PASS`, and all ten numeric values in the
> Output table below equal its output. The `(CL_hep)` label on dominant correction index 0 is not
> printed; the program prints `dominant_correction_idx: 0.000000`. The table cells are the original
> record and were not edited on 2026-09-14; the re-run matches their numbers with the current tree
> and does not establish how they were first produced. The test suite runs the test through
> `bin/souc run`, i.e. the default Madaros engine. Measured at HEAD `2a8e7ad145`, with a clean
> worktree and the prebuilt installed in `aaecebd878` (`bin/madaros-linux-x86_64` sha256
> `a1307ca6297963f89a12eec23f282b4bcff691be5c2cc6d1a00dbce4a30d8dd0`), from the repository root,
> `cd "$(git rev-parse --show-toplevel)" && bin/souc run tests/run-pass/pbpk28_m5_gum_4th_order.sio`
> exited 0 in 23 s. Its stdout begins with 49 lines of compiler progress log. The remaining 30
> lines print `M5_GUM_FOURTH_ORDER_CUMULANT_BUDGET_PASS`, `PASS` and the same ten Output values, and
> differ from the lean_single output in three lines, where lean_single prints a small magnitude in
> exponent form and Madaros prints `0.000000`: `Gaussian fixture kappa4` (lean_single
> `4.440892e-16`) and the `rel_err` of `dAUC/dCL` (`8.452175e-13`) and of `d2AUC/dCL2`
> (`1.337220e-9`). Those three values do not differ. A scratch copy of the test was made in which
> every `print_f64(x); println("")` also prints `f64_to_bits(x)`, `check_close` also prints the bits
> of its got and expected values, and `main` first prints three known constants
> (`1.0 / 2251799813685248.0`, 2000 times that, and `1.0 / 1000000000.0`); `diff` against the test
> shows no other changed line. Run at HEAD `0fc28e7fed` with the same two binaries (the worktree
> also held uncommitted edits to compiler and collections sources, none of them in the test's
> imports), it gave identical bit patterns for the three values on both engines. The constants also
> have identical bits on both engines but print in exponent form on lean_single and as `0.000000`
> on Madaros, so the three lines differ only in how `print_f64` formats small magnitudes. The probe
> prints the bits of 26 of the test's values. 10 of them differ between the engines but print
> identically to the six decimals shown, which is why they are not among the differing lines above.
> As relative differences from the lean_single value, computed from the bits: `var_1st` 3.4e-12,
> `u_1st` 1.7e-12, `u_2nd_hessian` 7.3e-11, `rel_hess_residual` 3.4e-10, `var_skewness` 3.5e-10,
> `var_kurtosis` 6.2e-10, `u_total` 6.7e-10, `var_total` 1.3e-9, `var_cubic` 2.1e-9 and
> `rel_fourth_residual` 1.2e-8. None of these differences changes a value recorded in this note:
> the tables are unchanged, and both engines print the same ten Output values. The other 16 are
> bit-identical: the two Gaussian fixture kappas, the three lognormal values, the `rel_err`, got
> and expected of each of the three derivative checks (nine values), `u_MC canonical` and the
> dominant index. Each engine reproduced its own bits on a second run.
> The first divergence is one input value. A second scratch probe, run at HEAD `0f6b452765` with
> the stdlib exported from that commit and the same two binaries, printed 5,274 bit records from
> three sites. The probe's `main` printed the 85 inputs of `hessian_pbpk28_auc`: the 71 PBPK
> parameters of `ep28_rapamycin_params()`, which returns `pbpk28_params_rapamycin()`, and the 14
> prior means and variances. It also printed the result's `var_first_order` and
> `var_second_order`. A traced copy of `h28_simulate_auc` printed the base simulation's initial
> concentration and, for each of its 1,681 steps, the blood concentration, the running AUC and the
> time. A print helper in `hessian_pbpk28_auc` (`stdlib/darwin_pbpk/epistemic_pbpk28_hessian.sio`)
> printed the base AUC, the 7 step sizes, the 14 first-order and 84 off-diagonal perturbed AUCs,
> the 7 sensitivities, the 7 diagonal and 21 off-diagonal Hessian entries, and the two variance
> sums. The `var_first_order` bits equal those of the first probe on both engines. Of the 85
> inputs, only `vasc_frac[10]` differs: the literal `0.041` on line 56 of
> `stdlib/darwin_pbpk/core/pbpk28_params.sio`, which lean_single turns into the bits
> `4586069543746904522` and Madaros into `4586069543746904523`. Exact integer arithmetic gives
> `4586069543746904523` as the correctly rounded binary64 value, so lean_single's value is one ULP
> low; for all 13 inexact `vasc_frac` literals the same check agrees with Madaros's bits.
> Downstream, the step sizes and the time grid are identical, and the rest differs: the base blood
> concentration first at step 6 and at 1,666 of the 1,681 steps, the running AUC at 1,668 steps,
> the base AUC, all 14 first-order and 83 of the 84 off-diagonal perturbed AUCs, all sensitivities
> and Hessian entries, both variance sums, `var_first_order` and `var_second_order`. Replacing that
> one literal with `41.0 / 1000.0`, a single division that both engines round to
> `4586069543746904523`, removes every difference: in scratch copies of the stdlib, all 5,274
> records of this probe and all 29 values of the first probe are bit-identical on the two engines,
> and Madaros's 29 values are unchanged. lean_single's source
> (`self-hosted/compiler/lean_single.sio` at `0f6b452765`) shows why the literal comes out low.
> When a literal's integer part is 0, the lexer moves the zeros right after the decimal point into
> the exponent. The code generator computes `int + frac / denom`, then divides or multiplies by
> `10.0` once per unit of the remaining exponent, rounding at each step, so `0.041` becomes
> `(41.0 / 100.0) / 10.0`. Evaluated for all 85 input literals, that computation misses the
> correctly rounded value for `0.041` only, which matches the single input difference the probe
> found.
> **Bit agreement after `10ac3eb3b3` (2026-09-15).** `10ac3eb3b3` makes lean_single lower each
> float literal to its correctly rounded binary64 bits, computed at compile time from the literal's
> text. It also refreshes the lean_single seed `bin/souc-lean-single-x86_64`, from sha256
> `e1d4eeb64d7d1f3ad22aba7f9d994111f4701fb79f7d3f391aa01a64657a8cb4` to
> `01c397f318f02a0c39d671bdca4db819f07f9200be3a90bf867f80f880aa4f76`. With the refreshed seed,
> `0.041` lowers to `4586069543746904523`, the bits Madaros produces. A third probe was made from
> the test like the first: it prints `f64_to_bits` beside every `print_f64`, but not the got and
> expected bits or the three constants, so it prints 20 bit patterns. It was run at HEAD
> `eb292c3674`, with `self-hosted/compiler/lean_single.sio` already holding the source committed in
> `10ac3eb3b3`, through three compilers: the previous seed, the refreshed seed built from that
> source, and the Madaros prebuilt (sha256 `a1307ca6…`). None of the five stdlib modules the test
> imports had uncommitted edits. The refreshed seed and Madaros gave identical bits for all 20
> values. The previous seed differed from Madaros in the same 10 values as above and in no others.
> For example, refreshing the seed moves `var_1st` from `4589554478734755281` to
> `4589554478734771891` and `u_total` from `4600493197977982340` to `4600493197982566279`. All three
> runs printed `M5_GUM_FOURTH_ORDER_CUMULANT_BUDGET_PASS` and `PASS`. The ten Output values print
> the same to six decimals, so the table is unchanged. The command above uses
> `bin/souc-linux-x86_64`, a separate binary that `10ac3eb3b3` did not replace. Run at HEAD
> `10ac3eb3b3`, it gave the same 20 bit patterns as the previous seed. To reproduce the agreement,
> run that command with `bin/souc-lean-single-x86_64` in place of `bin/souc-linux-x86_64`.
>
> The unmodified test at `2a8e7ad145`, run through the previous prebuilt, sha256
> `5cd3fdc228323b1f1baba9abd568d806af98461e5c97d33755daec553535dc24` (extracted from commit
> `2952a88fa2` and selected with `MADAROS_RAW_BIN`), exited 0 in 29 s with byte-identical program
> output. The test suite, also from the repository root,
> `cd "$(git rev-parse --show-toplevel)" && bash scripts/run_sio_test_suite.sh --filter-exact pbpk28_m5_gum_4th_order.sio --jobs 1`,
> reported `engine=madaros` and Pass: 1, Fail: 0, Total: 1, in 50 s wall time. The host is shared;
> its load average was about 21 during these runs, so the timings are not a benchmark.
> Before `aaecebd878` replaced that prebuilt, on 2026-09-14, the test run directly through it
> (refreshed in `054380db89`) printed the same markers in 17 s; the prebuilt before it (sha256
> `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`) took 102 s; and the suite run
> filtered to this test, with `//@ timeout: 90`, reported `Pass: 1` in 21 s wall time.
> **History.** On 2026-09-13, called through its raw interface on the test file, lean_single
> (`bin/souc-linux-x86_64`) stopped at `error: effect not declared in function signature at line
> 82`, and the default Madaros engine at `error[E035] … missing: Epistemic` in `main`: `main` did
> not declare `Epistemic`, while the `m5_pbpk28_convergence_budget` it calls is declared
> `with Mut, Div, Panic, Epistemic` (`stdlib/darwin_pbpk/cumulants.sio:442`). On 2026-09-14,
> before the fix, `cf42e812bd` marked the test `//@ known-failure`.
> The `canonical u_MC` row is a literal in the source, not a value this test computes: the test
> prints `print_f64(0.357945)`, and `m5_pbpk28_convergence_budget` passes the same literal as the
> `u_mc` argument of `variance_budget_4th_from_hessian` (`stdlib/darwin_pbpk/cumulants.sio:449`;
> parameter at `:386`), next to a comment citing `docs/dissertation/results/m6_prior_update_v1.md`.
> The recorded Monte Carlo runs do not agree on that value: re-run on 2026-09-13,
> `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio` printed a summary `u_MC` of
> 0.357945 mg.h/L, while `runs/m1_copula_sweep_v1.txt` records 0.549197 mg.h/L on the same summary
> line.

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
