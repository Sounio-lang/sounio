<!-- docs:meta
topic_id: repo.docs.plans.math-information-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.math-information-execution-2026-10-18
-->

# Checked information theory and channel rendering — 2026-10-18

## Scope and source

Continuation of the scientific stdlib sprint: statistics, scientific visualization,
mathematics, genomics and physics. Worktree /workspace/sounio-sprint-stats-viz;
original isolated baseline f5b2c5850be9f3d3995719ffcade44a6d851658a.
Remote main refreshed at2b41c82671995278554214398b7caac01b89e524.
Canonical checkout observed on feat/two-readings at2b41c82671, with a local
compiler build; that artifact is excluded from acceptance. The sprint worktree
is not rebased onto the latest main. Changes remain uncommitted/unpushed.

## Implemented numerical contract

- math::probability has checked Shannon, collision and min entropy, KL and
  Jensen-Shannon divergence in nats, fixed eight-bin capacity.
- Size is validated before access; only the active prefix is inspected.
  Active probabilities must be finite/nonnegative, mass positive and within
  1e-12 of one; accepted inputs are normalized by compensated actual mass.
- Typed status, value and infinite flag distinguish an unavailable result,
  valid zero and semantic infinite KL. Positive P with zero Q is not skipped.
- Canonical checked logarithms replace rational surrogates. KL uses generalized
  nonnegative terms and a local phi series near equality. JS uses a direct
  difference even series near equality and maximum-scaled mixtures elsewhere.
- Shannon/min/collision use tail/cross-product formulations near point masses.
  Shannon/KL/JS accumulate tiny contributions in DBL_MIN units and restore scale
  once, retaining aggregate subnormal information lost by per-bin rounding.
- math::channels provides BSC capacity in bits/use, real scalar AWGN capacity
  in nats/use, Pinsker sqrt(KL/2) from nats and reciprocal scalar Fisher information.
  Domains, unavailable arithmetic and statistical/model assumptions are explicit.
- BSC endpoints0/1 retain capacity1. The centered parameter2*p-1 and a positive
  even series preserve capacity near p=1/2 without first rounding1-p.
  AWGN uses log1p; Pinsker avoids underflow in KL/2 before sqrt.
- All nine prior math::information names delegate to these kernels.
  Invalid scalar compatibility result is -1; infinite KL is positive IEEE infinity.
  Historical "_exact" names do not assert exact arithmetic or a proof.

[Probability contract](../../stdlib/math/PROBABILITY.md);
[channel/scalar/compatibility contract](../../stdlib/math/CHANNELS.md).

## Native evidence

Final **15320**, COMPLETED0:0, **3:29**: **73/73 pass**, zero failures/skips/errors.
Support-loss control **15321**, FAILED1:0,3:29:70pass+3 runtime numerical
failures after successful compilation (reference/status/legacy).
Endpoint-loss control **15322**, FAILED1:0,3:28:69pass+4 runtime numerical
failures after successful compilation (reference/status/legacy/figure).
Both controls preserve all previous69 fixtures.

New selected fixtures: math_information_reference, math_information_status,
math_information_legacy, viz_information_channels. All earlier69 cases are
retained in the selection. Builds/execution run through Forge/Slurm only.

| Run | Job | Source archive SHA256 |
|---|---:|---|
| stats-viz-information-20261018-v1 |15320|65a557e2211528cfae1e8f77ccf9d6634c3fbd3b33c7a1aaef1d68cef3497c55|
| stats-viz-information-control-support-20261018-v1 |15321|ebaeb6c88472e498440fd008d09fc4f02650547ca66a6cb210f921458906aa59|
| stats-viz-information-control-channel-20261018-v1 |15322|26f8aa862666ff87d5594ad7c15d5b4d6ead7ea732dda029e8579532d40c7959|

The support control replaces the typed support-mismatch return with successful
zero. The channel control returns zero at valid BSC endpoints. They retain
the new checked shell and are not untouched historical implementations.
Each final control manifest differs at only one controlled source file.

Pinned committed Madaros v0.80 ELF SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
Gzip SHA256:
feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a.
GPUoff,2CPUs,8192MiB ceiling,5minute limit. This is pinned compiler acceptance,
not a new compiler-source rebuild, full corpus, remote CI or merge.

## Independent C++23 reference and repair loop

g++14.2, -std=c++23 -O2, libquadmath on t560. __float128 arithmetic supplies
113-bit precision for the near-equal cases; inputs/output expectations are
binary64. No Python/Rust numerical oracle is used.

171 expected results:27 distribution pairs x5 methods plus36 scalar cases.
Includes uniform/deterministic/zero bins, normalization tolerance, malformed
domains/sizes, support mismatch, opposite ratios, near equality, permutations,
symmetry and seven-minimum-subnormal-bin aggregates.

One-ulp cases independently yield:
KL2.46519032881566189e-32; JS6.16297582203915473e-33.
Nonzero results require value>0 separately; tolerance is relative1e-11 against
the result itself plus two minimum f64 steps. Expected zeros/statuses are exact.
Subnormal literals use normal numerators/2^52 as in the prior entropy checkpoint.

Root reran the oracle with exit0 and byte-compared the generated fixture/report.
Executable SHA256:
819593cd862b4d5327da4becce05058b8500e0ccf723802ec8d1e2aa32a936b6.

Focused15316 passed legacy/status but failed the numerical reference after
compilation. Diagnostic15317 prints all failure identifiers and always finishes;
its green harness status is not acceptance. It isolated Shannon's three seven-bin
subnormal cases and one BSC scalar case. Shannon now accumulates tiny terms before
rounding; BSC now avoids its rounded complement near0.5. Tolerances/reference
values were not loosened. Focused15318 subsequently passed all four fixtures.

Three subagents delivered the probability core/contract, independent numerical
reference, and figure/reviews. Root owned channel helpers, compatibility adapters,
status fixture, integration, repairs, controls, native execution and final custody.
Independent source review found no blocking numerical/API/render issue.
All scalar domains/units/assumptions are documented in CHANNELS.md.

## Rendering and resources

129 BSC curve samples feed production Canvas/SVG with shared explicit axes.
The fixture checks capacity endpoints, symmetry, monotonicity, scene counts and
a known Canvas axis pixel. Labels identify memoryless BSC, bits/use, uniform
binary input and ideal coding limit, including deterministic flip at p1.
No uncertainty interval is invented for this deterministic functional.

SVG is parsed with xmllint and visually inspected through QuickLook. The
thumbnail's right-edge crop/white padding is not the SVG viewport. The actual
960x520 export retains the rightmost tick/legend. The preview is not a captured
Canvas framebuffer.

The probability APIs have a fixed eight-category bound and use scalar scratch,
no heap or working probability arrays. Collision entropy has at most28 pair
products. Legacy by-value array parameters remain small compatibility copies.
No measured speedup/RSS improvement, transcendental correct-rounding proof,
arbitrary-capacity distribution support or sampling-calibration claim is made.
Status0 does not prove every mathematically positive sub-f64 result is nonzero.

## Custody and next work

/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/information-2026-10-18.
Selected final sources, contracts, C++ report, native logs/metadata/JUnit/manifests,
source-archive hashes, SVG/preview and updated master are retained.
Archived manifests compare byte-for-byte to standalone packages.
Selected sources and DELIVERY.sha256 verify local custody.

Next: checked statistical correlation/rank semantics and associated scientific
diagnostic rendering, beginning with a source audit; broader resampling calibration,
genomics I/O/count consumers and physics covariance/invariants remain active.
No commit/push/PR/merge/publication or whole-sprint completion is claimed.
