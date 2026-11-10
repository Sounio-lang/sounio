# Correlation inference and compatibility

Module stats::correlation_inference consumes the checked estimates from
[CORRELATION.md](CORRELATION.md). Coefficient validity and inference availability
are independent. Inputs are paired fixed256 arrays with n2..256.

## Explicit requests

pearson_inference(x,y,n,conf,assume_normal) requests normal-model inference only
when assume_normal=true. The caller declares iid paired observations from the
applicable bivariate-normal model; the library does not test or prove it.
The zero-correlation Student test uses df=n-2 under that model. Fisher-z CI is
an asymptotic normal approximation, not exact finite-sample coverage.

spearman_inference(x,y,n,allow_t_approx) always computes checked averaged-rank rho.
Only an explicit true flag requests the Student t approximation with n-2 df.
This is not a permutation/exact rank test or certified calibration for ties,
small samples, dependent pairs or nonstandard null distributions.
No Spearman CI is produced by this API.

CorrelationInference.status0 means coefficient available, not all inference
available. method1Pearson/2Spearman; calibration0unassessed/1Pearsonnormal-model/
2Spearmant-approx when a p-value is available.

## Status and payloads

Global status1..4 retain coefficient-core errors;5 invalid confidence
(nonfinite or outside0<conf<1). Invalid global results have coefficient0.
n preserves the request. All new unavailable numerical inference payloads are0;
read their status fields before using them.

| p_status | Meaning |
|---|---|
|0| Available under the explicitly requested calibration |
|1| Model/approximation not requested, or estimate unavailable |
|2| n<3 |
|3| Computed projection residual is zero; no finite t/p is fabricated |
|4| Arithmetic or tail unavailable |

| ci_status | Meaning |
|---|---|
|0| Available Fisher approximation |
|1| Not requested/assessed, or no Spearman CI contract |
|2| n<4 |
|3| Zero residual or rounded coefficient at/outside(-1,1) |
|4| Quantile outside the validated bounded-normal helper |
|5| Arithmetic unavailable |

For a valid conf outside the bounded normal-quantile contract, the coefficient
and Student p may remain available while CI is not. Quantile probabilities
must lie within1e-6..1-1e-6. A CI has its actual level only when available.

t is evaluated as r*sqrt(df)/residual_ratio without forming a raw covariance
product or subtracting rounded r² from1. A coefficient may round to1 while a
nonzero explicit residual supports a finite t and positive tail. Fisher CI is
unavailable if the rounded coefficient cannot be transformed inside(-1,1).

Small Fisher-z arguments use a local atanh series; larger values use checked
log differences. The inverse uses a local expm1 series near zero and negative
exponentials elsewhere, avoiding positive-exponential overflow. An exact
computed perfect fit does not trigger an invented huge t or a point CI.

## Compatibility surfaces

stats::correlation::pearson/spearman retain signatures and previous numerical
field names, adding status and availability/calibration fields. Pearson declares
the legacy normal-model assumption; Spearman opts into the legacy t
approximation. Unavailable compatibility p_value is -1; CI unavailable payloads
are0 and must be gated by ci_status. Existing inline examples are retained.

stats::inferential::spearman_rho/test retain fixed100 signatures and first
validate n2..100. They copy only a valid active prefix into the shared checked
core. No old selection-sort or threshold-to-zero inference body remains.
spearman_rho returns -2 outside the coefficient range on unavailable input.
SpearmanResult adds status,p_status,calibration; unavailable p is -1.
The unrelated t-test/ANOVA/effect-size implementation is preserved.

All arrays and helper storage are bounded; these adapters still incur fixed
compatibility copies. No measured execution/RSS gain is claimed.
Inference assumptions, sampling calibration, complete correlation-family coverage,
and exact transcendental rounding are outside native numerical acceptance.

Primary statistical descriptions:
[NIST average ranks and rank tests](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/rankcorr.htm);
[SAS Fisher transformation and confidence intervals](https://blogs.sas.com/content/iml/2017/09/20/fishers-transformation-correlation.html).
Independent C++23 quadmath references check df1/df3 Student closed forms and
normal-quantile/Fisher calculations; they are not a general Student oracle.

Adding result metadata expands the public struct layouts. Function argument
signatures are retained, but binary/foreign consumers need rebuilding and
external struct literals need the new fields. No external constructors were
found in the inspected repository; external consumers were not inventoried.
