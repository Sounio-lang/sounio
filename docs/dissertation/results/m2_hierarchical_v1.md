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
> **Current state (lean_single: 2026-09-15, HEAD `0a786dea7d`, seed `bin/souc-lean-single-x86_64`;
> all other current-state measurements: 2026-09-14, HEAD `5f9ef80877`).** When HEAD was `5f9ef80877`
> (2026-09-14), the worktree had two uncommitted paths, `self-hosted/ir/lower.sio` (modified) and
> `tests/run-pass/madaros_struct_len_method.sio` (untracked), neither in this test's import closure.
> `main` declares `Epistemic` since `2c4b0e7739`. `2952a88fa2` changed the PBPK28 Crank-Nicolson
> step to write into caller-owned work storage, so the Monte Carlo samples no longer exhaust the
> Madaros handle table; the same commit replaced the test's `//@ known-failure` tag with
> `//@ timeout: 90`. `02593aede3` (`EpPrior28.kn` and `.n`, the `Pbpk28HessianBudget` fields) and
> `5f9ef80877` (`EpResult28.auc_blood_mean` and `.auc_blood_var`, `Cholesky7.l` and `.ok`) made
> `pub` the struct fields this harness reads from other modules, which the Madaros prebuilt
> installed in `aaecebd878` requires; after them the Madaros check of this test reports no
> `error[E259]`.
> - lean_single, through its raw interface, from the repository root because it resolves stdlib
>   imports relative to the working directory:
>   `cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/run-pass/pbpk28_m2_hierarchical_prior.sio /tmp/m2_hierarchical.elf && chmod +x /tmp/m2_hierarchical.elf && /tmp/m2_hierarchical.elf`
>   (the lean_single seed `bin/souc-lean-single-x86_64`, sha256
>   `9d7892132aa0a9cf839df4560bf968628dc30fda4af978a4efc4a99b4e8f89f5`, run verbatim at HEAD `0a786dea7d`)
>   printed `M2_HIERARCHICAL_PRIOR_OUTPUT` and `PASS`, and all twelve numeric entries in the table
>   below equal its output. Merge `09aedffafa` installed that seed, and its history includes
>   `10ac3eb3b3`, which changed how lean_single lowers float literals. `git status` at `0a786dea7d`,
>   taken alongside the run, listed one modified file, `self-hosted/ir/lower.sio`, and two untracked
>   tests, none in this test's import closure; between `5f9ef80877` and `0a786dea7d` that closure
>   differs only in `stdlib/numerical/linalg.sio`, where `min_pivot` became `pub`. Until 2026-09-15
>   this note gave the command with `bin/souc-linux-x86_64` (sha256
>   `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`) in place of the seed. The seed
>   run's 33 lines of stdout are byte-identical to the stdout recorded for the
>   `bin/souc-linux-x86_64` command at `5f9ef80877` and at `5a054a21a4`; only stdout was compared, not
>   the ELFs. The `chmod +x` step is needed: at `0a786dea7d`, under umask 0022 (which the command does
>   not set), the seed wrote the ELF with mode `-rw-r--r--`, and without the step the run stopped at
>   `Permission denied` (shell exit status 126); `bin/souc-linux-x86_64` did the same at `2952a88fa2`.
>   Earlier versions of this note omitted that step.
> - Madaros, the engine the test suite uses (measured 2026-09-14 with HEAD at `5f9ef80877`), also from
>   the repository root:
>   `cd "$(git rev-parse --show-toplevel)" && bin/souc run tests/run-pass/pbpk28_m2_hierarchical_prior.sio`
>   (`bin/madaros-linux-x86_64` sha256 `a1307ca6297963f89a12eec23f282b4bcff691be5c2cc6d1a00dbce4a30d8dd0`)
>   exited 0 in 40 s. Its stdout begins with 52 lines of compiler progress log; the remaining 33
>   lines, the test program's own output, are byte-identical to the lean_single output of that date,
>   from `bin/souc-linux-x86_64`, which the seed run above also matches.
> - The test suite, also from the repository root:
>   `cd "$(git rev-parse --show-toplevel)" && bash scripts/run_sio_test_suite.sh --filter-exact pbpk28_m2_hierarchical_prior.sio --jobs 1`
>   reported `engine=madaros` and Pass: 1, Fail: 0, Total: 1.
>
> The table's "Hessian <=10%?" column (NO / informational) is not printed; the program prints
> `individual_hessian_criterion: OUTPUT (rel_Hess_individual > 0.10)`, consistent with the
> individual row's NO. The table cells are the original record and were not edited on 2026-09-14;
> the re-runs match their numbers with the current tree and do not establish how they were first
> produced.
> **History (2026-09-14).** Before `2952a88fa2`, Madaros compiled the test and then exited 182 at
> run time after printing `madaros: handles full`, in 21 s with prebuilt sha256
> `5cd3fdc228323b1f1baba9abd568d806af98461e5c97d33755daec553535dc24` (the prebuilt before it, sha256
> `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`, did the same after 337 s), so
> the test carried `//@ known-failure`. Madaros handles are a counter that is never reclaimed, with
> a capacity of 4,194,304; padding the counter to overflow measured about 2,028 handles per Monte
> Carlo sample with the earlier step code. lean_single was not affected: its stdout at
> `5a054a21a4`, before `2952a88fa2`, is byte-identical to its stdout at `5f9ef80877`. At
> `2952a88fa2`, with prebuilt `5cd3fdc2…`, Madaros ran the test to the same program output. At
> `e35c96e0c3`, after `aaecebd878` installed prebuilt `a1307ca6…`, Madaros stopped the test at
> preflight with 38 `error[E259]` diagnostics ("struct field is private in its defining module"),
> while the `e35c96e0c3` tree built with prebuilt `5cd3fdc2…` still ran it to the same program
> output. At `02593aede3`, 6 of those diagnostics remained; `5f9ef80877` removed them.
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
