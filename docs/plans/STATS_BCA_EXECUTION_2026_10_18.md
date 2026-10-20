<!-- docs:meta
topic_id: repo.docs.plans.stats-bca-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-bca-execution-2026-10-18
-->

# Checked mean BCa bootstrap — 2026-10-18

## Accepted native identity

Run `stats-viz-bca-20261018-v1`, Slurm **15063**:
COMPLETED **0:0**, elapsed **00:01:49**.
JUnit **37 tests, 0 failures, 0 skips, 0 errors**, retaining all previous 32.
Snapshot:
`b99eb040e271ed6a1d2785e7b683bb7cefa60735669a059ddeb16d99ff1bef56`.
The archived manifest matches the fully prepared source package; ten selected
code/reference/test/packaging files in local custody match that manifest.

2 CPUs, 8192 MiB, GPU off, five-minute limit, clean-v1, UID/GID 1000/1000.
Pinned committed Madaros v0.80 ELF SHA256:
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
This is not a compiler rebuilt from current sources. The canonical checkout's
separate local compiler build was not used.

Isolated worktree `/workspace/sounio-sprint-stats-viz`.
Original remote-main baseline `f5b2c5850be9f3d3995719ffcade44a6d851658a`.
Remote main rechecked `c4f1d3ce73987de9e2694af393711180f6a57fb1`.
The intervening stdlib edits are outside this focused closure; the latest
advance from 106f7a988d is site/docs only. Canonical active checkout untouched.
Changes remain uncommitted; no push, PR, merge or remote CI acceptance.

## Implementation

- `stats/bootstrap.sio`: one draw, Welford variance, heapsort and type-7 quantile
  kernel shared by the ordinary percentile and new checked mean-BCa APIs.
- BCa-specific anchored/scaled compensated means, midpoint tie convention with
  scale-relative tolerance, explicit bias probability/counts and adjusted tails.
- Mean jackknife acceleration through the centered-moment identity, with
  rescaled compensated second/third moments. No absolute-unit variance cutoff,
  variance cubing or stored n-value jackknife array.
- Explicit unavailable states: invalid requests/data, arithmetic failure,
  degenerate data/replicates, extreme bias probability, unsupported normal tails
  or a BCa denominator pole. No silent percentile fallback or bias clipping.
- Minimal empirical tail-support flag, including a deliberately unresolved
  eight-replicate case; not a precision or coverage guarantee.
- `stats/bca_normal.sio`: bounded positive-term normal integral and bracketed
  inverse, finite-domain statuses. Replaces the legacy omitted-sqrt(2) CDF
  argument convention and duplicated inverse coefficient body in this BCa path.
- `stats/epistemic/bootstrap.sio`: the public mean-BCa compatibility adapter
  delegates to the checked core, with explicit bounds. Return fields are public.
  Required effect declarations and missing pure test helper are repaired.
  Other legacy numerical routines are not certified.
- Real Canvas/SVG figure: percentile and BCa 95% intervals for the same sample,
  seed and 512 draws, with exact IR bounds, axis domain and method/budget title.

## Independent C++23 concordance

`tests/reference/bca_reference.cpp`, compiled with g++ 14.2.0,
`-std=c++23 -O2 -Wall -Wextra -pedantic`.
Reference uses long-double statistics, explicit leave-one-out mean recomputation,
two-pass bootstrap variance, std::sort, erfc-based normal CDF and a separately
bracketed inverse. Sounio uses the moment acceleration identity, Welford,
heapsort, positive-term CDF and 56-step inverse. RNG stream is intentionally
specified identically for pathwise comparison, not independent RNG-quality testing.

Nine mean-BCa cases include right skew; 1e-200 and 1e200 unit transforms;
a 1e15 common offset; values around 1e308; discrete ties; symmetry; n=256/b=1024
with i64-min seed; and b=8 with insufficient adjusted-tail support.
An additional independently searched two-replicate seed tests unavailable extreme
bias without zero replicate variance.

Assertions cover exact less/equal counts, normalized seed, resample count,
tail flag, mean/SE/bounds, z0, acceleration and adjusted probabilities.
Raw estimate/bounds use tolerance max(3e-9 * input range, two ULPs of the expected
raw value); SE relative tolerance 3e-10; BCa metadata absolute tolerance 3e-9.
Representative caller data entries remain unchanged.
These tolerances are executable comparison budgets, not uniform accuracy proofs.

Normal helper checks: 25 CDF points on [-6,6] at absolute tolerance 2e-14;
11 inverse probabilities from 1e-6 to 1-1e-6 at tolerance 3e-9.
Status tests cover invalid n/B/confidence, NaN/inf, constant data, unsupported
normal tails and unavailable payloads. Legacy tests cover finite high-range
bounds/SE and exact adapter-to-canonical equality for the same settings.

Reference source SHA256:
`a4c98972b0ba5ac346413e2b2c109b3110ead5d0d58e327b2606f87a831ce4b7`.
Reference executable SHA256:
`545f586e60b2a3916b32dc4e40b52650e096d6d14ace564b1cb6316b9700e4bb`.
Generated fixture SHA256:
`137377b202bb570ad11483605de58056bbf3d346629ea4752952d06dfe533eed`.

## Failing before / passing after

Control run `stats-viz-bca-20261018-before2`, Slurm **15066**:
37 tests, 35 pass, two runtime failures, no skips/errors;
FAILED **1:0**, elapsed **00:01:51**.
Snapshot:
`7433d78ae96a48fd5d1045bc336ce631599e63ca2fde7b2409457a76d55f8ec9`.

This is the **old numerical body**, with only effect declarations, public
result/field visibility and the missing pure test helper repaired so it compiles.
It is not an untouched historical checkout. The candidate/control manifests
differ only in `stdlib/stats/epistemic/bootstrap.sio`.
The same legacy and smoke fixtures pass typecheck and native compilation,
then fail their finite-result checks for data around 1e308. The accepted adapter
passes both and returns identical bounds/SE to the checked kernel.

Earlier job 15058 retained all 32 previous fixtures but failed the new smoke
at check time, exposing legacy effects/helper issues. First control 15064
failed check on private return fields. Neither is presented as numerical
regression evidence; 15066 establishes the runtime boundary.

## Resource and scientific limits

One 1024-double distribution buffer; no per-draw sample array, stored jackknife
array, duplicate sorted distribution or heap allocation in the canonical kernel.
The legacy adapter makes one fixed-capacity input copy. Complexity is a source
bound, not a measured runtime/RSS/stack speedup.

See [BCA.md](../../stdlib/stats/BCA.md). This covers mean BCa under ordinary iid
resampling, with assumptions unassessed. It does not establish empirical coverage,
correlation/Spearman or arbitrary-statistic BCa, regularity, dependent resampling,
general normal-tail accuracy, correctly rounded arithmetic or generator quality.
The minimal tail flag is not sufficient Monte Carlo precision.
Standalone SVG was extracted as a single XML element, parsed and visually inspected;
the preview is not a Canvas framebuffer export.

The five-domain standard-library sprint and execution/memory/artifact optimization
remain in progress. Next: assessed statistical assumptions and continued domain
numerical kernels, with broader resampling/coverage contracts still open.
