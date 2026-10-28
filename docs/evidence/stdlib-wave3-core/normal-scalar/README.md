# Scalar standard-normal kernels: numeric and compatibility qualification

The four functions in `stats::normal_scalar` accept one `f64` and return one
`f64`: `normal_cdf_scalar`, `normal_sf_scalar`,
`normal_logcdf_scalar`, and `normal_logsf_scalar`.
NaN propagates; infinities have exact support endpoints. A probability may
underflow to zero while its logarithm remains finite. Arithmetic failure or
failure to converge returns NaN. Finite inputs whose squared magnitude
overflows have representational infinite logarithms. No all-real-input error
bound or correctly-rounded claim is made.

The scalar facade uses budget 2048 and tolerance
1.7763568394002505e-15 (8 binary64 epsilons). Checked APIs continue to use
their caller's original budget/tolerance/status/availability policy. This
different convergence choice is deliberate: the new scalar application asks
for tighter accuracy than the checked default. Existing checked inputs,
assertions, fixtures, tolerances and the 113-bit oracle are unchanged.

The implementation extracts the canonical compensated incomplete-gamma series
and continued fraction into `special::gamma_scalar_core`; checked gamma
wrappers call those same loops. The nonreflecting Lanczos body is shared via
`special::gamma::lgamma_positive_core`. The compensated small log1p branch
is shared via `math::scalar_log::log1p_small_scalar`.
`normal_tails_checked` and the scalar facade call the same
`normal_standard_logtail_into` core. There is no duplicate recurrence for
workers. Small tails are evaluated directly, and small logarithms near zero
use log1p; a small survival probability is never obtained by subtracting CDF
from one. Scalar status/work metadata uses references to caller-owned local
integers. These helpers do not construct records or allocate source-level
containers.

The gamma-half/erfc relation is documented by
[NIST DLMF 8.4.6](https://dlmf.nist.gov/8.4.E6), and incomplete-gamma
continued fractions by [DLMF 8.9](https://dlmf.nist.gov/8.9).
The independent C++23 oracle uses Boost cpp_dec_float_100 and erfc for the
application domain. Only extreme controls beyond |x|=1000 use a 16-term Mills
asymptotic expansion; see [DLMF 7.12](https://dlmf.nist.gov/7.12).
These controls do not substitute an asymptotic oracle inside the application
domain |x|<=40.

## Executed evidence

All compilation, execution and numerical checks below ran under Slurm on
dl380-dl380-proxmox. Raw compiler SHA256:
`e9ad6cde0328e1dbf249da726fb44a798b0dd575236d5ec4040cd309f558b03e`.
It is the qualified component compiler, not an assertion about a committed
or fully integrated compiler build.

| Job | Result and preserved boundary |
| --- | --- |
| 18847 | Setup failed on truncated archive, before any check. |
| 18854 | Private SDK download failed because the image lacked a CA bundle. |
| 18860 | Private Boost SDK extracted successfully; no host package/global trust change. |
| 18863 | Eight source checks, static compile/native, marker and source/compiler guards passed. |
| 18875 | Unchanged original checked gate passed before extraction; the new output adapter then failed native lowering on untyped inline bit printing. |
| 18878 | Typed bit-print helper fixed the adapter; 354 numeric cases and complete original checked gate passed. The original 444 output rows were byte identical before/after. |
| 18883 | Classification-only reporting confirmed that the large global relative error was subnormal quantization, with 0.325 ULP absolute error. |
| 18889 | The original scalar tolerance 1e-12 failed the additional application precision policy: 20 failures. Same inputs and already-built output were preserved. |
| 18891 | Tightening only the new scalar convergence tolerance to 8eps passed both original and application policies, all negative controls, and the full unchanged checked gate. |

Job 18891 completed with worker/gate status 0. Original checked output from
18875 and output after the final extraction/precision change are byte
identical (`CHECKED_BITWISE_STATUS=0`); all generated input rows are also
identical. This includes 444 original cases, invalid/underflow refusal probes,
the checked seed, and its existing mutant controls. It does not imply
exhaustive binary64 equivalence.

The new oracle checks 354 exact binary64 input words, four outputs per input:
a 257-point grid plus branch neighbors, symmetry, signed zero, subnormals,
infinities, NaN and extreme logarithmic controls. Probability acceptance is
2e-10 relative plus four minimum subnormals; log acceptance is 2e-10 absolute
with a relative condition on tiny logs. Extreme |log|>1e6 controls permit
eight ULP. The additional |x|<=40 application policy is 2e-12 relative plus
eight minimum subnormals and is combined with existing limits without
weakening them. Wrong complement, missing completion marker and duplicate
row controls all exit 1 for the required reason.

Final measured errors across this finite selection:

| Metric | Maximum |
| --- | --- |
| Relative probability error for normal nonzero outputs | 2.0875486399639348e-13 |
| Relative tiny-log error for normal nonzero outputs | 2.0875486399639348e-13 |
| Subnormal absolute error in minimum-subnormal units | 0.32463306444086731 |
| Absolute log error for |x|<=40 | 2.0874372060281046e-13 |
| Absolute log error including extreme controls | 3.3883158624702403e-11 |

The global probability relative metric 0.050774342988979607 includes
subnormal quantization and must not be interpreted as a normal-range error.

Final source-manifest SHA256:
`c8e3add5c340d321c3363e0269089b1fd77cf58c525b81bcf0a64b2125e07913`.
Static ELF:
`c9849340c7b18ec96b8fb0deb2a872d5b575cd6791ecd4359a06bcb93a30403d`.
Generated numeric ELF:
`9953eb04100363c1d30ba4796acb9b0bb7da7fe65e286b17535984c1d60873ea`.
C++23 oracle ELF:
`84593180dc8ffcf624585e85b856cd90af0a733ff6f32ed6b92697f06144847c`.
All 2091 source entries and 15653 Boost header entries passed final guards.

The private SDK is Ubuntu libboost1.83-dev 1.83.0-2.1ubuntu3.2 amd64,
package SHA256
`519ecf2c64308527e15b6582955681d192a832250baa4bc424967aaf7d02d68f`.
Header-manifest SHA256:
`e034035708b2a7df9d3dafccddc28c714a222a31207e02d2b630f04bba898a04`.
The package and all provenance/failure receipts are retained alongside the
numerical receipts. Reproduction requires C++23, explicit compiler hash,
Boost headers and Slurm; invoke `scripts/stdlib/normal_scalar_gate.sh` with a
fresh absolute output directory and its documented environment variables.

## Open integration boundary

This gate proves selected numerical concordance and original checked
compatibility. Source shape alone does not certify WorkerFn runtime safety:
ordinary compiler region prologues still exist. Worker admission requires a
separate source-bound final-IR proof of every reachable helper, local-stack
outref provenance, no escaping/captured references or managed runtime state,
and verified region elision. Four-pool execution, application ordering/RSS,
and integrated compiler gates remain separate receipts.


Durable receipt bundle: `/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/normal-scalar01`.

- `receipts-and-reconstruction.tar.gz`: SHA256 `599e08730876201bcdbc1c8aaf19088d300e8dc166b16a003395c0123447a03c`.
- `boost183-sdk-custody.tar.gz`: SHA256 `a0f043ae71f923eae70b07c071220b56d95166eb4ce157098ef08fee489d83b8`.

The first includes all listed job receipts, original failed attempts, source
archive/overlays, generated drivers, executable artifacts and guards; source
directories duplicated from those reconstruction inputs are excluded. The
second includes the SDK package, public CA input, hashes and successful/failed
download receipts; extracted headers are reproducible from the pinned package.
Archive hashes were verified after writing to durable storage.

