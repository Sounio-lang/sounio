# Checked channel and scalar information helpers

Module: math::channels. Results use math::probability::InformationResult.
All functions declare Mut,Div,Panic and use scalar state only.

| API | Quantity and units | Domain |
|---|---|---|
| bsc_capacity_bits(p) | Binary symmetric channel capacity, bits per channel use | finite p in [0,1] |
| awgn_capacity_nats(snr) | Real scalar AWGN capacity, nats per real channel use | finite SNR >=0 |
| pinsker_bound_nats(kl) | sqrt(KL/2), upper bound on total variation when KL is in nats | finite KL >=0 |
| cramer_rao_bound(info) | Reciprocal scalar Fisher information | finite info >0 |

## Meaning

BSC computes D(Bernoulli(p)||Bernoulli(1/2))/ln(2). In exact arithmetic this is
1-H_binary(p), with values1 at p0 and p1 and zero at p1/2. Near the zero-capacity point, a positive even series in t=2*p-1 avoids
subtracting entropies or first rounding the complement 1-p. Elsewhere the
divergence form is used.
p1 describes a deterministic invertible flip and retains capacity1.

AWGN computes ln(1+SNR)/2 for a real scalar, memoryless, additive Gaussian-noise
channel with the matching average-power constraint. Complex-channel conventions
or capacity in bits/s require explicit conversion and bandwidth assumptions.
No physical link budget, colored noise or fading model is inferred.

Pinsker does not cap the returned bound at1. It uses sqrt(KL)*sqrt(1/2), avoiding
loss of KL/2 at minimum-subnormal KL. The caller must supply an applicable KL;
the function does not infer distributions or verify statistical assumptions.

The reciprocal Fisher helper evaluates arithmetic only. Calling it a Cramer-Rao
bound assumes the caller has supplied appropriate Fisher information and met
the regularity/unbiasedness assumptions for that bound. No estimator efficiency,
identifiability or causal property is certified.

## Status

0 means available finite result; 2 nonfinite input; 3 invalid scalar domain;
6 unrepresentable or unexpected arithmetic result. All errors have value0
and infinite=false. Read status before value. Positive AWGN capacity rounding
entirely to zero returns6; SNR0 has a valid zero. A nonfinite reciprocal also
returns6. The probability-vector APIs have their separately documented size,
mass and KL-support statuses.

AWGN uses a 32-term atanh log1p series for SNR<=1/2, avoiding 1+SNR rounding
to1; other values use the shared checked logarithm. The smallest subnormal
input has an unrepresentable half-log and is unavailable, while twice that
input returns the minimum-subnormal capacity.

## Compatibility module

math::information retains all nine historical function names as public
compatibility wrappers. Shannon, collision/min entropy, KL, JS and AWGN return
nats; BSC now obeys its original documented bits/use contract. The former
private body returned an inaccurate nats-like expression and wrong endpoints.
The historical suffix _exact is retained for compatibility and is not a claim
of exact arithmetic or a proof.

Invalid compatibility calls return -1. KL support mismatch returns positive
IEEE infinity; use the typed API to distinguish semantic infinity and errors.
No rational surrogate logarithm, skipped positive-p/zero-q term, or unchecked
vector length remains in these compatibility bodies.

[Probability contract](PROBABILITY.md).
For the mathematical nonnegative extended KL convention, see the primary
[Mathlib KL documentation](https://leanprover.github.io/hex/docs/Mathlib/InformationTheory/KullbackLeibler/Basic.html).
The Sounio implementation has not been proved equivalent to that formalization.

Independent C++23 references and native Forge execution determine bounded
acceptance. These functions make no correctly-rounded, complete-stdlib,
sampling-calibration or comparative-performance claim.
