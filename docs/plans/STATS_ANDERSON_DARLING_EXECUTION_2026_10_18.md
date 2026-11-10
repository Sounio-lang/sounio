<!-- docs:meta
topic_id: repo.docs.plans.stats-anderson-darling-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-anderson-darling-execution-2026-10-18
-->

# Anderson-Darling execution receipt — 2026-10-18

## Scope and source identity

Continuation of the accepted statistics/scientific-visualization sprint in
`/workspace/sounio-sprint-stats-viz`. Original isolated baseline:
`f5b2c5850be9f3d3995719ffcade44a6d851658a`; changes remain uncommitted.

Remote main initially e00a5d537cd0c6d01762c38fcd70cc9739062710, refreshed to
`a1844465373c1ae5a335de2d0a73819e5cec2230` during final custody.
A fetch and path comparison from e00 to a184 found only
`stdlib/epistemic/claim_meet.sio` and
`tests/run-pass/knowledge_centre_seat.sio` in the inspected stdlib/test/compiler
binary surfaces; none are in this selected runtime closure. Upstream compiler
source changes are not certified by this pinned-binary execution.
No checkout/rebase/integration, commit, push, PR, CI or merge is claimed.

## Implementation

- Replace standalone AD's raw moments, low-accuracy erf/complement and fake
  p=1 invalid fallback with checked scaled compensated fitted-normal inference.
- A2 is the reverse-paired log-CDF/log-survival statistic, accumulated in one
  reindexed pass. Small tails are computed before complement and logarithm.
- One 256-double scratch array, shared O(n log n) heapsort, no dynamic core
  allocation or second normal-log vector. No measured RSS/speedup claim.
- Preserve A2/A2_star/p fields; add status/n/approximation/tie metadata.
- Statistic for complete samples n=3..256; approximate Stephens p only n>=8,
  Astar<10. Unsupported calibration retains the statistic, not a made-up floor.
- Replace private toy AD with public typed 100-value legacy adapter. Remove
  the unused p-to-Beta conversion, fake confidence type, moment/sort helpers
  and external sqrt/log/exp from the adapter module.
- Synthetic step ECDF/fitted-normal figure carries the computed A2/p and an
  explicit approximate/no-confidence-band footer through real Canvas and SVG.

See [method contract](../../stdlib/stats/ANDERSON_DARLING.md).

## Native acceptance

Final candidate: Slurm **15241**, run `stats-viz-ad-20261018-v3`.
Final control: Slurm **15242**, run `stats-viz-ad-control-20261018-v3`.
- Candidate COMPLETED, exit 0:0, elapsed 00:02:51: **57/57 pass**,
  zero failures, skips or errors in test-suite-junit.xml.
- Control FAILED, exit 1:0, elapsed 00:02:51: **53 prior pass + four new
  runtime numerical assertion failures**, zero skips/errors. Each failing
  native log shows successful compilation before execution.
- Four new fixtures: stats_ad_reference, stats_ad_status, stats_ad_legacy,
  viz_ad_ecdf. The former 53 fixtures remain selected and passing.

Candidate source archive SHA256:
`95261771888eb655be64299a258e1a55ee377b4bf06b2671ead914f0b6826169`.

Control source archive SHA256:
`a7c1990bb8c0248fca8c966444e47e05e2a8e8d4fbec7f6b44ea6bd8f52b3821`.

The complete content manifests differ only at
`stdlib/stats/anderson_darling.sio`. Control restores the old erf polynomial
and complemented normal tail (with shared exp), retaining the new validation,
scaling, sorting and calibration contract. This isolates the former CDF/tail
policy; it is not an untouched historical module. Both archived manifests
were byte-compared with their completed standalone packages before acceptance.

Pinned committed Madaros v0.80 ELF SHA256:
`125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f`.
Committed gzip SHA256:
`feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a`.
The provenance-warning workspace local build is not used.
All Sounio native compilation/execution occurs through Forge/Slurm,
two CPUs, 8192 MiB ceiling, five-minute limit, GPU off.

## Independent numerical reference

`tests/reference/anderson_darling_reference.cpp`, built with
`g++ -std=c++23 -O2 -Wall -Wextra -pedantic` on t560.
18 datasets, raw long-double centering/sample SD, standard-library erfc,
sorting/reverse pairing; no Sounio incomplete-gamma or scaled-moment algorithm.
Published calibration coefficients are shared deliberately; concordance of p
does not independently establish calibration accuracy.

Includes n=3,7,8,12,20,32,64,100,256; all four p pieces, ties/outliers,
large common offset, opposite-sign extremes and subnormal-scale samples.
A2 and Astar absolute bound 5e-10; positive p relative bound 5e-9
plus the minimum f64 subnormal step. Dedicated status fixture adds narrow
relative bounds for p approximately 2.84873788e-18, NaN/Inf, invalid/oversized
sizes, caller immutability, constants and an Astar approximately98.8424
with explicitly unavailable calibration. Legacy oversized i64 is checked
before narrowing/copying.

## Figure and custody

Final 920x450 SVG is XML-parsed and visually inspected via a QuickLook thumbnail.
The viewport was widened during visual review. Subsequent XML inspection
showed that the apparent final-tick crop belongs to QuickLook's square thumbnail:
the exported tick starts at x=700 with textLength=80, inside both original and
final SVG widths. The width change is a layout choice, not a proven SVG fix.
This is an SVG preview, not a Canvas framebuffer capture.
Canvas execution separately asserts an exact axis pixel and scene counts.
No CI/coverage band is drawn; the fitted CDF is joined at the observed coordinates.

Custody:
`/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/anderson-darling-2026-10-18`.
Selected source, reference generator/report, exact run metadata, per-case logs,
JUnit, manifests, final SVG and preview are retained. Selected runtime source
is checked against the accepted package manifest; DELIVERY.sha256 records custody.

## Preliminary runs and limits

- 15233/v1: 57 selected, 55 pass, two failures: generated integer literals
  failed Sounio type checking and truncated title length failed SVG text checks.
- 15234/control-v1: same generation issue plus old-tail numerical failures;
  not the final controlled comparison.
- 15235/v2: COMPLETED0:0,2:53,57/57. 15236/control-v2: FAILED1:0,2:56,
  53 prior pass + four new runtime assertion failures after compilation.
- v3 preserves v2's numerical implementation/reference and corrects only the
  figure canvas/export width and pixel-index stride.

These are 57 focused fixtures, not whole-stdlib validation or a claim of SOTA
superiority, uniform p accuracy, finite-sample calibration, independence, ties
validity, censoring support, KS/Lilliefors inference or full sprint completion.
Next method front is checked KS null/size/tie contracts and stable Kolmogorov
tails; genomics/physics and measured execution/memory/artifact work remain tracks.
