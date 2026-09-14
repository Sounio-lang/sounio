<!-- docs:meta
topic_id: repo.docs.dissertation.results.m2-hierarchical-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.m2-hierarchical-v1
-->

---
docs:meta: true
topic: dissertation-results
kind: quantitative-output
drug: rapamycin
model: PBPK28
status: implementation-complete
version: v1
date: 2026-05-14
---

# PBPK28 M2 Hierarchical Eta/Epsilon Prior Decomposition - v1

**Harness**: `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio`
**Opt-in test wrapper**: `tests/run-pass/pbpk28_m2_hierarchical_prior.sio`
**Drug/configuration**: rapamycin, 5 mg IV bolus, AUC_blood(0->168 h), N=2000
**Zhang 2009 prior components**: `omega2 = 0.0566`, `sigma2 = 0.0894`

This document records the M2 decomposition of the PBPK28 lognormal prior into a
two-level eta/epsilon hierarchy while preserving the single-level MC baseline as
repo-local truth. The canonical single-level result remains:

| Baseline | u_MC (mg.h/L) | rel_Hess | Gate |
|---|---:|---:|---|
| Single-level LogNormal, E1 | 0.549197 | 0.155073 | `MC_CROSS_VALIDATION_PBPK28_LOGNORMAL_OUTPUT` |

That baseline is intentionally not reinterpreted as passing. It is the clean
origin/main value unless a separate M6 change supersedes it with new evidence.

## Algebra

For each positive PBPK28 prior parameter with nominal mean `mu`, M2 writes the
parameter draw as

```text
X = mu * exp(eta + epsilon - 0.5 * tau2)
eta     ~ Normal(0, omega2)
epsilon ~ Normal(0, sigma2)
eta independent of epsilon
```

The centering term preserves `E[X] = mu`.

Two comparison levels are reported:

```text
individual conditional path:
  eta = 0
  tau2 = sigma2
  Var[X | eta = 0] = mu^2 * (exp(sigma2) - 1)

population marginal path:
  tau2 = omega2 + sigma2
  Var[X] = mu^2 * (exp(omega2 + sigma2) - 1)
```

The GUM/Hessian propagation uses the same PBPK28 finite-difference machinery as
the single-level harness, but with the physical variances induced by the chosen
log-variance level:

```text
Var_GUM(y) ~= sum_i c_i^2 Var[X_i]
Var_Hess(y) ~= Var_GUM(y) + 0.5 * sum_ij H_ij^2 Var[X_i] Var[X_j]
```

This keeps the comparison aligned with JCGM first- and second-order uncertainty
propagation while the MC reference samples the corresponding lognormal hierarchy.

## Results

The numeric row below was recorded with this command. It does not work as written, and how the
recorded numbers were actually produced cannot be established from this repository's history
(see the note after it):

```bash
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 ./bin/souc run tests/run-pass/pbpk28_m2_hierarchical_prior.sio
```

> **Reproduction status (2026-09-13, updated 2026-09-14).** `bin/souc` execs `SOUNIO_SOUC_BIN` with its arguments
> unchanged (it already did at the merge that added this file, `bebd78d74c`; earlier history is
> not in this clone). Measured on 2026-09-13 with `bin/souc-linux-x86_64`, before `bin/souc` began
> refusing the form: called that way, lean_single stopped at `error: no main` (a current-source lean_single shows
> why: it opens `run`, which does not exist, as a 0-byte source). `bin/souc` now refuses the form (exit 64). The pinned binary (sha256
> `3cbea2b4…`) is no longer in the repository.
> **Current state (2026-09-14).** Since `2c4b0e7739`, `main` declares `Epistemic`. Run through
> lean_single's raw interface, from the repository root because it resolves stdlib imports
> relative to the working directory,
> `cd "$(git rev-parse --show-toplevel)" && bin/souc-linux-x86_64 tests/run-pass/pbpk28_m2_hierarchical_prior.sio /tmp/m2_hierarchical.elf && /tmp/m2_hierarchical.elf`
> (`bin/souc-linux-x86_64` sha256 `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`)
> printed `M2_HIERARCHICAL_PRIOR_OUTPUT` and `PASS`, and all twelve numeric entries in the table
> below equal its output. The table's "Hessian <=10%?" column (NO / informational) is not printed;
> the program prints `individual_hessian_criterion: OUTPUT (rel_Hess_individual > 0.10)`,
> consistent with the individual row's NO. The table cells are the original record and were not
> edited on 2026-09-14; the re-run matches their numbers with the current tree and does not
> establish how they were first produced. The test still carries `//@ known-failure`, with the
> reason on its line 2 updated in `2c4b0e7739`: the test suite runs it through `bin/souc run`, i.e.
> the default Madaros engine (`bin/madaros-linux-x86_64` sha256
> `5cd3fdc228323b1f1baba9abd568d806af98461e5c97d33755daec553535dc24`, refreshed in `054380db89`),
> which compiles the test and then exits 182 at run time after printing `madaros: handles full`,
> in 21 s; the previous prebuilt (sha256
> `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`) did the same after 337 s.
> **History (2026-09-13).** Called through its raw interface on the test file, lean_single
> (`bin/souc-linux-x86_64`) stopped at `error: effect not declared in function signature at line
> 12`, and the default Madaros engine at `error[E035] … missing: Epistemic`: `main` was declared
> `with IO, Mut, Div, Panic`, while the `mc28_hierarchical_selftest_main` it calls is declared
> `with IO, Mut, Div, Panic, Epistemic`
> (`stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio:599`). The test carried
> `//@ known-failure` from #2290.

| Level | u_GUM (mg.h/L) | u_Hessian (mg.h/L) | u_MC (mg.h/L) | MC mean AUC (mg.h/L) | rel_GUM | rel_Hess | Hessian <=10%? |
|---|---:|---:|---:|---:|---:|---:|---|
| Individual epsilon-only | 0.216951 | 0.275193 | 0.340487 | 1.076898 | 0.362821 | 0.191766 | NO |
| Population eta+epsilon | 0.281279 | 0.379208 | 0.470507 | 1.153652 | 0.402178 | 0.194044 | informational |

Gate marker:

```text
M2_HIERARCHICAL_PRIOR_OUTPUT
```

## Interpretation

M2 tests whether separating residual individual uncertainty from population
between-subject variability moves the conditional individual PBPK28 comparison
into the weakly nonlinear regime (`rel_Hess_individual <= 0.10`). The population
marginal row is reported separately because it answers a different question:
the variance seen by a population draw after both eta and epsilon are integrated.

The v1 implementation does not achieve the <=10% individual criterion. The
Hessian correction remains informative, but at these Zhang 2009 component values
the MC reference is still about 19% away from the second-order GUM estimate.

Safe dissertation wording should therefore keep three claims separate:

- the single-level E1 LogNormal baseline remains `rel_Hess = 0.155073` and does
  not meet the Hessian criterion;
- the M2 individual row is the conditional eta-fixed comparison and is the row
  eligible for the <=10% marker;
- the M2 population row is descriptive population-marginal evidence, not a
  replacement for the canonical single-level baseline.
