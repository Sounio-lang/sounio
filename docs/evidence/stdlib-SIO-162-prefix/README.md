<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-162-prefix.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-162-prefix.readme
-->

# Canonical Beta prefix and imbalanced tails: SIO-162 follow-on

This increment addresses the measured large-shape accuracy gap in the [first bounded Normal/Beta façade](../stdlib-SIO-162/README.md), published at `c32c3c9d7917d1d86cf33dbb90a2652dd3b45404`. The earlier receipts remain unchanged. Both the public Beta density and incomplete-beta tails now consume one canonical checked logarithmic prefix.

For shapes at least 8, `beta_log_prefix_checked(a,b,x)` evaluates `log(x^a*(1-x)^b/B(a,b))` through centered deviance and a fixed nine-term Stirling correction. The correction follows the log-gamma expansion in [DLMF 5.11.1](https://dlmf.nist.gov/5.11.E1). Near the mean, a compensated 24-term series evaluates `log1p(u)-u` without subtracting its linear term. Canonical `math::dd64::two_sum/two_prod` form the residual `(a+b)*x-a`; a rounded binary64 mean caused additional errors for fractional shapes. Away from this centered region, original observation logarithms avoid amplifying mean rounding near support endpoints. Small or mixed shapes retain the shifted Lanczos path and its existing limitations.

The new public prefix result contains status and finite value. It requires finite positive shapes no larger than 1e6 and a finite observation strictly inside (0,1); invalid and unsupported inputs return explicit status. Density endpoints remain the façade's exact endpoint cases.

## Imbalanced CDF path and bounded work

A highly imbalanced reflected continued fraction still lost roughly 5e-11 in selected near-mean cases after the stable prefix alone was introduced. A guarded positive series now evaluates the suitably oriented CDF using the [hypergeometric identity in DLMF 8.17.8](https://dlmf.nist.gov/8.17.E8):

`I_x(a,b) = exp(prefix)/a * sum_{n>=0} (a+b)_n/(a+1)_n * x^n`.

The oriented small shape must lie in [8,64], the opposite shape must be at least 1024 times larger, x must be below 0.125 and (a+b)*x must be at most a+2. The symmetric orientation uses the original complement observation. Terms are positive and compensated. Because term ratios decrease in this region, termination bounds the remaining geometric tail. The result is used only when its direct probability is no larger than 0.9; the other probability then uses a stable complement. This path reports method 5. Continued fractions remain methods 3/4. The existing continued-fraction arithmetic is preserved outside the new series path.

The positive-series iteration count is charged to the caller's existing budget. Any defensive continued-fraction fallback receives only the remaining budget. A budget-depleted series returns status 3 with no available values. This is an iteration bound, not a floating-point accuracy certificate. Inverse budgets remain nested forward-evaluation budgets, as documented in the original façade.

## Paired executable evidence

Slurm 17129 completed on the t560 CPU worker with the retained d74 compiler, UID1000, stack 524288 KiB. The focused gate passed, exit0, including strict-seed checking, d74 checking/building, native endpoint/domain/nonfinite/underflow/quantile controls and four rejected output mutants. The compiler was not freshly rebuilt for this increment.

The C++23 Boost113-bit verifier consumed actual raw binary64 native output. Its 444 rows include the original 176 rows, large balanced/fractional/imbalanced near-mean fixtures, both sides of the centered-series threshold, shapes immediately below/equal/above 8 and 64, ratios immediately below/equal/above 1024, both sides of the guarded (a+b)*x boundary and budget-one depletion. Actual method selection and total forward iteration bounds are checked.

The same 444-row generated fixture was freshly compiled and executed against the preserved pre-repair façade/special kernels in source08. That native program exited0, but the stricter independent verifier rejected it with 424 numerical/branch-contract failures. Its central absolute log and relative value errors reached about 3.0943655e-9. The repaired program passed all 444 rows with zero verifier errors:

| Measured quantity | Repaired observation |
| --- | ---: |
| Maximum central absolute log error | 2.9438340724701076e-12 |
| Maximum central relative value error | 2.9438526318045341e-12 |
| Maximum absolute log error across the whole grid | 3.3657869034280321e-9 |
| Maximum scaled relative value error across the whole grid | 3.2571079500155804e-11 |

The central fixture accepts absolute log / relative value errors up to 5e-12 for densities and 2e-11 for tails. Outside that centered region, the prior bounded large-shape qualification remains; the observed maxima do not establish a universal bound. Large-magnitude log-tail ULP accuracy, mixed-shape cancellation and continued-fraction conditioning outside the new guard still require broader qualification.

The two impossible Beta inverse-SF residual targets remain unavailable. Independent nearest-float/neighbor checks are preserved; the repair does not turn an unrepresentable target into an apparently successful quantile.

The final native grid observed 0.07 seconds user CPU, 0.08 seconds elapsed and 1084 KiB maximum RSS. The whole 29-second Slurm job, including C++/Sounio compilation, allocated two logical CPUs and observed 768916 KiB batch maximum RSS. The paired before run is separately preserved from job17124. These are single resource observations, not a comparative speed/scaling claim.

`phase14/` contains repaired gate outputs and `phase13/` the freshly executed paired before case. Source/compiler/reference/ELF hashes, generated fixtures, exact native outputs, resources and Slurm accounting are retained; `FILES.sha256` covers this compact text evidence. Large executables and source snapshots remain remotely under `/scratch/sounio-stdlib-inference-scale-20261024/SIO162`. No quota or historical artifact removal was performed.

Documentation checks, whitespace and shell syntax checks are reported separately from scientific validation. Full corpus, fresh compiler rebuild, remote CI acceptance, human PR acceptance and merge are not claimed by this record.


## ContractsCore ratchet follow-through

The initial façade commit c32c3c9d79 failed [ContractsCore job121015566778](https://github.com/Sounio-lang/sounio/actions/runs/40141935813/job/121015566778) after its numerical exp-helper sweep passed. The static shrinking baseline still listed the deleted `stdlib/stats/distributions.sio dist_exp`: 113 helpers were found but 114 were pinned. The fix removes only that stale row from `scripts/ci/stdlib_exp_series_baseline.txt`, preserving the ratchet against introducing a new local exponential series.

Slurm17132 ran the unchanged full exp-helper gate with both the explicit d74 default engine and lean_single. Both native sweeps and static ratchets passed, exit0, with 70 helpers, 24 grid points and 113 found/113 pinned. `phase15/` retains the gate/compiler/baseline identity, exact outputs, exits, resources and Slurm accounting. The completed remote failure log was read directly; this diagnosis is separate from a fresh remote CI acceptance claim. New remote CI must run on the updated PR head.
