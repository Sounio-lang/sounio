<!-- docs:meta
topic_id: repo.docs.plans.stats-ks-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-ks-execution-2026-10-18
-->

# Checked KS execution receipt — 2026-10-18

## Scope and custody

Continuation of the statistics/scientific-visualization/mathematics/genomics/
physics sprint in `/workspace/sounio-sprint-stats-viz`.
Original isolated baseline `f5b2c5850be9f3d3995719ffcade44a6d851658a`.
Remote main rechecked at `a1844465373c1ae5a335de2d0a73819e5cec2230`.
Canonical checkout remains separate, observed branch fix/exp-product-mean,
with an uncommitted local compiler build that was not used by this acceptance.

No commit, push, PR, merge, current-source compiler certification or remote CI
is claimed. This is a pinned-compiler focused runtime closure.

## Changes

- Checked one-sample normal-reference KS and two-sample KS retain d/p/ne and
  add size/data/reference/arithmetic statuses, sample sizes, asymptotic and tie flags.
- All active sizes checked1..256 before array access. Finite data and valid
  finite normal mu/sigma required. Unavailable numerical payload is zero.
- One normal CDF requires externally fixed parameters; using fitted sample
  parameters does not become a valid ordinary KS normality test.
- Two-sample merge consumes each pooled equal-value run before comparing ECDFs.
  Rank cross-products remain exact integers before floating division.
- Shared in-place heapsort replaces local insertion sort; bounded one/two scratch
  arrays, no dynamic allocation or sequence-sized coordinate construction.
- Normal standardization handles finite opposite-sign subtraction overflow.
  Far-tail saturation prevents squaring unbounded standardized values. This
  private CDF is not certified as a general subnormal normal-tail kernel.
- Stephens-corrected asymptotic survival uses theta CDF for small lambda and
  direct alternating survival for larger lambda, each at most32 terms. The
  fixed100-term alternating policy is removed from the small-lambda path.
- Ties are flagged, including cross-sample equal values. D is meaningful for
  tied samples; the returned p remains an unadjusted continuous-null approximation.

[Public scientific contract](../../stdlib/stats/KS.md).

## Final native acceptance

- Candidate **15249** COMPLETED0:0,00:03:07: **60/60 passed**, zero failures,
  skips or errors. All previously accepted57 fixtures remain passing.
- Walk control **15250** FAILED1:0,00:03:03:58 pass + two numerical runtime
  failures (stats_ks_reference,status), after successful compilation.
- Tail control **15251** FAILED1:0,00:03:03:59 pass + one numerical runtime
  failure (stats_ks_reference), after successful compilation.
- New selected fixtures: stats_ks_reference,stats_ks_status,viz_ks_ecdf.

| Run | Job | Source archive SHA256 |
|---|---:|---|
| stats-viz-ks-20261018-v1 | 15249 | 5ec5b6c95e41be1fe142b89a28f91b30da7bb956db66c5ce23298246d1cd697a |
| stats-viz-ks-walk-20261018-v1 | 15250 | a78947a3ff7742468778a7f6262b5834364681e40df77a8331094a5ca01fcc26 |
| stats-viz-ks-tail-20261018-v1 | 15251 | 2cd321a328ef4d0781700ab40f4cc58f18677836646a9bbb973b75cfa3b58f7f |

Two controls isolate separate old policies in the new checked shell:
walk restores single-observation progression through a pooled tie group;
tail restores the fixed100-term alternating survival with absolute termination.
Both retain current validation, normal CDF, shared exp and result shape.
Neither is an untouched historical module. Each complete manifest differs from
the candidate only at stdlib/stats/ks_test.sio. Archive manifests were byte-compared
against their completed standalone package manifests.

Pinned committed Madaros v0.80 ELF SHA256:
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
Gzip SHA256:
`feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a`.
Native compilation and execution through Forge/Slurm only, GPUoff,
two CPUs,8192MiB ceiling,five-minute job limit.

## C++23 concordance

`tests/reference/ks_reference.cpp` compiled/executed on t560 with g++14.2,
`-std=c++23 -O2 -Wall -Wextra -pedantic`, exit0.
22 cases generated as a compact385-line Sounio fixture. The two-sample oracle
counts ECDFs directly at pooled support values (no merge) and uses raw long-double
erfc for the one-sample normal CDF. Kolmogorov identities/Stephens correction are
shared mathematical specifications; agreement is numerical, not calibration
accuracy in finite/tied/dependent samples.

Coverage: sizes1..256, unequal sizes, reversal, D0, complete tails after a sample
is exhausted, repeated values, extreme finite scales, subtraction overflow,
sigma1e-300, and interleaved/separated maximum samples. A heavily tied255/256
case has D=1/(255*256), exercising the theta/small-lambda path.
Smallest reference p approximately2.52233148426187875e-226.
D/ne use absolute+relative2e-12; p has relative3e-10 only, no masking absolute
floor. Oracle self-checks theta/alternating identity at lambda0.8 and exact-rational
ECDF cases. Boundary/status fixture additionally checks NaN/Inf, illegal mu/sigma,
out-of-range sizes, inactive poison, no caller mutation, equality under different
tie multiplicities and the existing module inline test selection.

## Scientific render

Synthetic X=[0,0,1],Y=[0,1,1],D=1/3. Complete step ECDFs include support tails.
A vertical connector is computed from returned D; D/p labels come from runtime
results. The explicit footer is 'Asymptotic continuous null; ties unassessed'.
This is an illustrative discrete sample, not evidence that continuous-null p is
valid for ties, nor a confidence band.

Production Canvas path asserts12nodes/3series and a known axis pixel. The same
scene emits920x450SVG. XML parsing and visual inspection of the generated SVG preview passed. The square
QuickLook preview crops the rightmost tick independently of the actual SVG; the
exported centered tick text ends at x780 inside its920-pixel viewport.
QuickLook thumbnail is an SVG preview, not a Canvas framebuffer capture.

## Parallel work and next domains

Subagents owned independent oracle, scientific figure/contract and read-only
domain audit. Root owned shared core, statuses, packager, controls, final review
and Forge acceptance. Independent static review found no critical core defect;
it identified the inherited normal gamma tail limitation recorded above.

[Domain source audit](SCIENTIFIC_KERNEL_AUDIT_2026_10_18.md) ranks checked streaming
k-mers first, then isothermal entropy/uncertainty and information-theory logs.
No genomic/physics/math kernel in that audit is claimed implemented by this KS
checkpoint. They remain implementation tracks; architecture and final decisions
remain with root.

## Local evidence

`/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/ks-2026-10-18`.
Retained selected code, method contract, C++ report, all final native test logs,
run metadata, JUnit and manifests for candidate and both controls, SVG/preview,
updated plan and source audit. Selected runtime files match the accepted package
manifest; DELIVERY.sha256 verifies local custody.

Root independently re-executed the C++ oracle with exit0, compared its TSV output
and emitted fixture byte-for-byte with retained reference/package copies.
Oracle executable SHA256:
`58f8dbb91aa07c8f5f62b90dbc960d70be4f3251196a2ac12acc63d036ee7cf2`.
C++ source SHA256:
`17524c03e0ac836d6d360120d70ebb78f2936306fc7729c4aede36d449f72fcb`.

No empirical runtime/RSS speedup, exact finite-n calibration, Lilliefors,
discrete-null correction, sampling-independence assessment, full stdlib suite
or whole-sprint completion is claimed.
