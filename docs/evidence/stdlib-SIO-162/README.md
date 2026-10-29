<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-162.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-162.readme
-->

# Checked Normal/Beta distributions: SIO-162

This package replaces the public Beta density's incorrect Stirling normalization and the Normal/Beta façade's local numerical approximations with the canonical checked gamma/beta and math kernels. Existing public constructors and scalar operation signatures remain available. New scalar operations expose log-density, survival function, log-tails, quantile and inverse survival function for both distributions.

The constructors remain pure parameter containers. Every evaluation validates the parameters. Scalar convenience calls assert status and availability; callers that need to recover from invalid parameters, exhausted work or unrepresentable scalar values use `stats::checked_distributions`.

## Checked API contract

`normal_density_checked(mu,sigma,x)` and `beta_density_checked(a,b,x)` return `CheckedDensity`: status, density, log-density and separate availability flags. A finite log-density survives scalar underflow or overflow. The scalar payload is unspecified when its flag is false. Exact support endpoints expose genuine IEEE infinities where the mathematical density or logarithm is infinite.

`normal_tails_checked(mu,sigma,x,budget,tol)` and `beta_tails_checked(a,b,x,budget,tol)` return both tails and their logarithms, each with its own availability flag. Normal tails use the smaller-tail identity `0.5 * Q(0.5,z*z/2)`, avoiding subtraction of a small survival probability from a rounded CDF. The Beta façade delegates directly to the canonical incomplete-beta pair. A positive tail below binary64 range has an unavailable scalar and an available finite logarithm; an unrepresentable logarithm of the opposite tail is also marked unavailable.

`normal_quantile_checked(mu,sigma,p,upper,budget,tol)` and `beta_quantile_checked(a,b,p,upper,budget,tol)` target CDF when `upper=false` and survival function when `upper=true`. The result contains availability, value, bracket, iteration count and an observed forward log-tail. Normal inversion uses the canonical gamma inverse and performs an additional forward check at the actual returned affine value. If affine rounding prevents the requested log-tail residual, it returns unsupported status rather than copying the target probability into the witness.

Status values are 0 for supported arithmetic/iteration completion, 1 for domain/options, 2 for nonfinite parameters or NaN, 3 for exhausted work, 4 for arithmetic failure and 5 for an unsupported/unrepresentable range. Status 0 is not a universal accuracy certificate. Availability flags must always be inspected.

## Supported work and numerical limits

Finite positive Normal scale is required. Beta shapes must be finite, positive and no larger than 1e6, matching the checked special kernel. Budgets are 1..1e6; tolerance is 1.7763568394002505e-15..1e-4. Infinite observation values have exact support-tail endpoints; NaN observations are rejected. Normal probability endpoints produce genuine infinite quantiles. Beta density endpoints distinguish shapes below, equal to and above one.

A forward result's iteration count and method come from its canonical special kernel. An inverse budget bounds forward evaluations; each forward evaluation separately receives a bounded inner kernel budget. Normal reserves one inverse-budget evaluation for its final transformed-value check. This is a bounded nested algorithm, not a single aggregate operation budget. Brackets are algorithmic diagnostics and do not certify that the exact mathematical quantile lies inside them at the last bit.

The canonical log-gamma combination can lose accuracy for large, nearly balanced shapes. The focused oracle therefore declares its tested error scales explicitly; it does not infer a uniform ULP bound or a statistical calibration guarantee from a finite grid. Unsupported standardized squares and affine values fail closed.

## Reproduction and evidence

Run `scripts/stdlib/checked_distributions_gate.sh OUTPUT_DIRECTORY` on an allocated Linux CPU worker with an explicitly identified `MADAROS_RAW_BIN`. The gate records compiler/source hashes, typechecks, native ELF hashes, actual binary64 output, GNU time observations and native exit status. It uses C++23 Boost with 113 significand bits and an extended exponent range to preserve extremely small reference tails. The verifier parses actual native output rather than comparing constants copied into the Sounio fixture. Corrupted values/statuses, missing completion and duplicate rows must be rejected.

The independent oracle measures absolute error for logarithms and relative error with a 1e-300 scale floor for values. Its declared grid limits are 2e-10 for ordinary shapes and 5e-8 when either Beta shape is at least 1000. These are finite-fixture acceptance limits, not advertised guarantees for every supported input. Native scalar failure probes cover invalid scale and an unavailable underflowed density.

Slurm job 17095 (base `a2a44df195bb8caf3fb07cbc9517275b29325e94`, preserved `source02`) executed the old public Beta-uniform regression: `beta_pdf(Beta(1,1),0.5)` returned **1.9915822398055509**, and the expected-one assertion exited 1. The corrected public façade returned binary64 bits `4607182418800017408` (exactly 1) and exited 0.

Slurm job 17113 completed the full focused gate, exit 0. Strict seed smoke checking and d74 checks/builds passed. All 176 native rows passed independent verification. The largest observed absolute log error was **5.8806410204348282e-9** and the largest scaled relative value error was **3.0943654122232811e-9**. These maxima remain within the explicitly bounded large-shape qualification; the canonical large-shape log-gamma cancellation remains a known accuracy gap, with stable log-beta and prefix/deviance evaluation required as follow-on work.

Two inverse-survival quantiles were deliberately unavailable. Independent Boost evaluation of the nearest binary64 quantile and both adjacent values found best achievable log-tail residuals of **2.8755709767685985e-11** for Beta(1,1), and **8.3996662576867427e-6** for Beta(0.5,0.5), at target probability 1e-6. Both exceed the requested 1e-12. The verifier accepts status 5 only after this independent representability check. Both scalar failure probes exited exactly 1. Four output mutants (status, available central density, missing completion and duplicate row) were rejected.

The native grid used 0.06 seconds of user CPU / 0.06 seconds elapsed and observed 1616 KiB maximum RSS. The whole 23-second Slurm job, including C++ and Sounio compilation, observed 766424 KiB batch maximum RSS and allocated two logical CPUs. These are single observations, not a throughput, scaling or comparative performance claim.

The retained compiler SHA-256 is `d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6`; it was **not freshly rebuilt for this package**. C++ compilation used g++ 13.3.0 with `-std=c++23`. Compact text evidence in `phase02/` and `phase08/` includes source/compiler/ELF identity, generated fixture, actual outputs, resources, parser-mutation diagnostics and Slurm receipts. Large executables and source snapshots remain in remote custody under `/scratch/sounio-stdlib-inference-scale-20261024/SIO162`; no historical deletion or quota change was performed. `FILES.sha256` covers the committed evidence.

Documentation registry/consistency checks, shell syntax and diff whitespace checks passed. Full corpus, fresh compiler rebuild, remote CI, PR review and merge are separate boundaries and are not claimed here.
