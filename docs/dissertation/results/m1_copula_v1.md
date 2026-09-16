<!-- docs:meta
topic_id: repo.docs.dissertation.results.m1-copula-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.m1-copula-v1
-->

# M1 Copula Sweep v1

## Scope

This sprint adds a Cholesky-Banachiewicz 7x7 factorization and uses it for a
Gaussian-copula sensitivity sweep over the existing PBPK28 LogNormal Monte Carlo
prior. The canonical independent LogNormal sampler remains the first MC path in
`stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio`; the copula sweep
is appended after the existing baseline summary.

Live stdout is captured in:

- `docs/dissertation/results/runs/m1_copula_sweep_v1.txt`

## Compiler Pin

Pinned compiler:

```text
SOUC_NATIVE=/workspace/sounio/bin/souc-linux-x86_64
sha256=3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93
```

Execution used the repo wrapper with `SOUNIO_SOUC_BIN=$SOUC_NATIVE` where the
command surface required `bin/souc run`; the pinned native binary itself exposes
the raw self-hosted compiler interface (`<source.sio> <output>`).

> **Reproduction status (2026-09-13).** The wrapper route above does not work as described:
> `bin/souc` execs `SOUNIO_SOUC_BIN` with its arguments unchanged (it already did at the merge
> that added this file, `bebd78d74c`; earlier history is not in this clone). Measured on
> 2026-09-13 with `bin/souc-linux-x86_64`, before `bin/souc` began refusing the form: lean_single
> stopped at `error: no main` (a current-source lean_single shows why: it opens `run`, which does
> not exist, as a 0-byte source). How the recorded values were originally produced cannot be
> established from this repository's history. `bin/souc`
> now refuses the form (exit 64). The working form is the raw interface, from the repository root,
> because lean_single resolves stdlib imports relative to the working directory:
>
> `cd "$(git rev-parse --show-toplevel)" && bin/souc-linux-x86_64 stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio /tmp/mc28.elf && /tmp/mc28.elf`
>
> The pinned binary (sha256 `3cbea2b4…`) is no longer in the repository. Re-run 2026-09-13 with
> `bin/souc-linux-x86_64` (sha256
> `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`): it printed
> `M1_COPULA_CHOLESKY_PASS` and `M1_COPULA_SWEEP_PASS`. The values below match
> `runs/m1_copula_sweep_v1.txt`, but **not the re-run**: `u_GUM` 0.317093 recorded against
> 0.228175 re-run, `u_Hessian` 0.464032 against 0.295160, summary `u_MC` 0.549197 against 0.357945,
> `rel_GUM` 0.422624 against 0.362543 and `rel_Hess` 0.155073 against 0.175405; each of `sweep_1`–`sweep_5` and `combined` also
> differs in mean AUC, `u_MC`, `rel_GUM` and `rel_Hess` (for example `combined` `u_MC` 0.407610
> against 0.254122). `N`, `seed`, `n_valid` and the rho settings are the same. The re-run's
> `rel_Hess` 0.175405 is the value `d6_full_integration_v1.md` records and the value the audit note
> below says the dispatch expected. The default Madaros engine,
> `bin/souc run stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio` with no
> `SOUNIO_SOUC_BIN` set (`bin/madaros-linux-x86_64` sha256
> `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`), exits 182 on this source.

## Results

Fixed settings:

- `N=2000`
- `seed=1729`
- `prior=LogNormal`
- `dt=0.5 h`
- drug: rapamycin
- `u_GUM=0.317093 mg.h/L`
- `u_Hessian=0.464032 mg.h/L`

| Scenario | rho(CL_hep, fu_plasma) | rho(CL_hep, CL_renal) | n_valid | mean AUC | u_MC | rel_GUM | rel_Hess |
|---|---:|---:|---:|---:|---:|---:|---:|
| independent baseline | 0.0 | 0.0 | 2000 | 1.204004 | 0.549197 | 0.422624 | 0.155073 |
| sweep_1 | -0.7 | 0.0 | 2000 | 1.133281 | 0.337081 | 0.059295 | 0.376618 |
| sweep_2 | -0.5 | 0.0 | 2000 | 1.153133 | 0.400060 | 0.207385 | 0.159906 |
| sweep_3 | -0.3 | 0.0 | 2000 | 1.173210 | 0.460379 | 0.311234 | 0.007934 |
| sweep_4 | 0.0 | 0.0 | 2000 | 1.204004 | 0.549197 | 0.422624 | 0.155073 |
| sweep_5 | 0.3 | 0.0 | 2000 | 1.235738 | 0.638226 | 0.503165 | 0.272936 |
| combined | -0.5 | 0.3 | 2000 | 1.156221 | 0.407610 | 0.222066 | 0.138421 |

The rho-zero copula row exactly matches the independent baseline in this pinned
run (`delta_mean=0`, `delta_u_MC=0`), so the default independent path is
unchanged.

Audit note: the dispatch text expected `rel_Hess(LogNormal)=0.175405`. The
current clean `origin/main` lane with the required pinned compiler produces
`rel_Hess(LogNormal)=0.155073`, matching the existing v2 dissertation result
documents. This sprint preserves that repo-local baseline rather than rewriting
it to an unsupported value.

## Self-Audit

- The Cholesky core is implemented in `stdlib/numerical/linalg.sio` as flat
  `[f64; 49]` storage plus a public wrapper with the requested signature:
  `cholesky(A: [[f64; 7]; 7]) -> [[f64; 7]; 7]`.
- Nested arrays compile in this repository. The wrapper is kept at the API
  surface; the numerical core remains flat because that matches the existing
  small-matrix style used by `stdlib/linalg/decomp.sio`.
- The nested-array return path uses explicit row literals. Direct nested
  mutation (`out[i][j] = ...`) did not persist under the pinned compiler in the
  focused test, so the literal conversion is the compiler-proven route.
- The Cholesky regression test constructs deterministic random SPD matrices
  `A=B*B^T+eps*I` and checks `||LL^T-A||_F/||A||_F <= 1e-10`; all five cases
  were below `1.3e-16`.
- Correlation support is local to
  `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio`, not
  `ep28_rapamycin_priors`, because this sprint is a dissertation sensitivity
  probe over prior dependence rather than a change to the canonical rapamycin
  prior definition.
- `M1_COPULA_CHOLESKY_PASS` is emitted only after the MC baseline completes,
  all copula rows complete with `n_valid > 1900`, and the rho-zero copula row
  reproduces the independent sampler.

## Section 4.10 Paragraph

The independent LogNormal prior is physiologically conservative as a baseline,
but it is not the only plausible joint prior: hepatic clearance, renal
clearance, and plasma unbound fraction can share upstream population structure
through physiology, assay context, and posterior updating. Krauss et al. (2015)
provide a precedent for treating PBPK population parameters as a joint
high-dimensional object rather than as independent marginals: their Bayesian
population PBPK workflow assumes independent prior random effects but then
allows posterior dependency structure and covariance-bearing multivariate
representations to characterize interindividual variability (PLOS One 2015,
DOI 10.1371/journal.pone.0139423; PMID 26431198; PMCID PMC4592188). I use that
paper only as methodological precedent for testing dependence, not as evidence
for the particular rapamycin correlations swept here. In the M1 sensitivity
sweep, negative dependence between hepatic clearance and plasma unbound fraction
compresses the Monte Carlo uncertainty: rho=-0.3 gives `u_MC=0.460379 mg.h/L`
and `rel_Hess=0.007934`, while stronger negative dependence at rho=-0.7 gives
`u_MC=0.337081 mg.h/L` and makes first-order GUM nearly adequate
(`rel_GUM=0.059295`) but worsens Hessian agreement because the fixed local
second-order budget now overstates the compressed MC spread. Positive dependence moves in the opposite direction
(`rho=+0.3`, `u_MC=0.638226`, `rel_Hess=0.272936`). The combined row
`rho(CL_hep,fu)=-0.5`, `rho(CL_hep,CL_renal)=+0.3` remains outside the 10%
Hessian criterion (`rel_Hess=0.138421`). The implication for M5 is that posterior
updating should estimate or constrain the joint prior structure, not merely
shrink one marginal CV: moderate correlation can move PBPK28 across the
Hessian/MC adequacy threshold, while stronger or mis-signed dependence changes
which approximation fails.

## Reproduction commands (2026-09-16)

The command in the 2026-09-13 note above is left unchanged: it invokes
`bin/souc-linux-x86_64` (sha256 `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`) and
contains no `chmod +x` step. How that run was actually invoked is not recorded. The current
form runs the lean_single seed (sha256
`9d7892132aa0a9cf839df4560bf968628dc30fda4af978a4efc4a99b4e8f89f5`, most recently changed by merge
commit `09aedffafa`) and adds the `chmod +x` step:

```bash
cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio /tmp/mc28.elf && chmod +x /tmp/mc28.elf && /tmp/mc28.elf
```

Run as written today, a line without `chmod +x` exits 126, with `Permission denied` reported
by the launcher: both binaries write the ELF with mode `-rw-r--r--` under umask 0022. How the
earlier runs recorded in this note were invoked is not recorded here, and earlier paragraphs
naming `bin/souc-linux-x86_64` describe those runs.

Measured 2026-09-16 at HEAD `169ac84463`, from the repository root, at the documented `/tmp`
ELF paths (none existed beforehand; each was removed afterwards). Each source below was
compiled and run with the seed and with `bin/souc-linux-x86_64`; every build exited 0, every
run exited 0 after `chmod +x`, and for each source the two binaries' stdout was byte-identical
(stdout only, not the ELF bytes):

- `stdlib/darwin_pbpk/validation/pbpk28_mc_cross_validation.sio` (ELF `/tmp/mc28.elf`): stdout contained `M1_COPULA_CHOLESKY_PASS`, `M1_COPULA_SWEEP_PASS`

Any other line in the blocks above — expected-output markers, `rg` searches, gate scripts —
was not run and is not part of this comparison. The values recorded elsewhere in this note
were not re-derived.
