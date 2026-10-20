<!-- docs:meta
topic_id: repo.docs.plans.stats-correlation-permutation-execution-2026-10-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-correlation-permutation-execution-2026-10-18
-->

# Correlation permutation and seeded resampling execution — 2026-10-18

## Accepted scope

Final Forge/Slurm job **15363**: **80 tests, 0 failures, 0 skips, 0 errors**,
exit0:0, elapsed3:53. All77 previously accepted fixtures remain passing.
Run: stats-viz-correlation-permutation-20261018-v1.
Snapshot SHA256:
d4a1b246fe2d1ad98f7648038e6f7749a36294d87dc69e072a32a2c27cde93d5.

Worktree /workspace/sounio-sprint-stats-viz remains uncommitted at original
baseline f5b2c5850be9f3d3995719ffcade44a6d851658a. Remote main was freshly
checked at2b41c82671995278554214398b7caac01b89e524. Changed stdlib paths
between baseline and main are epistemic/claim_meet.sio, optimize/nelder_mead.sio
and polynomial/lib.sio; these do not overlap this block's touched code. This
does not establish execution of the candidate on the full current-main tree.
The main remote is the
reference; no rebase, commit, push, PR or merge is implied. The canonical
workspace's unrelated local compiler is excluded.

Committed Madaros v0.80 ELF SHA256:
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
Gzip SHA256:
feaaf7fe181f1ee3efd30e7fe23b1709937ea7e93b52e812276fb093b6ccf52a.
Each job requested2CPUs,8192MiB,6minute ceiling, uid/gid1000, noGPU.
Elapsed times are job measurements, not comparative kernel/RSS benchmarks.

## Library implementation

New stats::correlation_permutation supports Pearson and averaged-rank Spearman.
It exhaustively enumerates labelled permutations for n2..8 (including identity
and duplicate-value multiplicities), or runs1..100000 seeded Monte Carlo draws
for n2..256. Every draw resets the labels and uses full descending Fisher-Yates,
Park-Miller integer state and rejection-bounded index selection.

Preparation/ranking occurs once. Two f64[256] buffers and one reused i64[256]
index buffer have6144bytes array payload, excluding scalar records, compiler
temporaries and stack overhead. No heap or resampled-statistic history.
Spearman keeps exact bounded half-integer centered ranks. Pearson declares its
relative64epsilon floating tail policy; exhaustive enumeration does not turn
floating-point Pearson comparisons into exact real arithmetic.

Exact p=hits/n!. Monte Carlo p=(hits+1)/(B+1); Wilson95 and MC SE describe the
uncorrected hit fraction q=hits/B, not the correlation coefficient or corrected
p. Status/method/count/seed/assumption metadata have explicit bounded contracts.
The API leaves exchangeability_assessed=false. Random reassociation tests the
exchangeability/independence null; zero population correlation alone is
insufficient under general dependence. Studentized weak-null/blocked/cluster
procedures and coverage/power studies remain open.

The correlation core's previous checked public algorithms and moment/projection
body are preserved. Reusable preparation exposes borrowed output workspaces.

Shared percentile/BCa bootstrap and means-permutation normalization now guard
the negative remainder class mapping to absorbing state0. Seeds-2147483646 and
-4294967293 map to2147483646; other valid normalized streams are unchanged.
The new regression first checks a zero sample's normalized seed, failing safely
before unsafe draws in the previous body, then compares all result fields
against the positive canonical seed for all three consumers.

## Independent reference and controls

C++23, g++14.2.0 and libquadmath on t560 independently evaluate31cases.
Raw/high-precision Pearson moments and O(n^2) less/equal rank counting differ
from the library's preparation/heapsort. Integer hit/draw/seed comparisons are
exact; floating fields have explicit fixture tolerances.
Cases include n8 enumeration, tied multiplicity, extremes, offsets, subnormals,
inactive-prefix poison, n256, budgets1/100000, signed seed limits, zero/all hits,
and validation failures. The forced-rejection seed739806647 yields23/32 hits;
the oracle self-check proves modulo-only selection gives a different count.

The root re-executed the C++ oracle with exit0 and byte-identical31-row report.
Executable SHA256:
47c239e01beb53cb41ecdbda3c6546fd43c1ebf776538dcad78367091a9906b6.
Report SHA256:
92188846241c637e10345745cfd4f495daf3abae3aa73d9a5f5e1af2ca8f3547.

| Run | Job | Result | Elapsed |
|---|---:|---|---|
| Candidate |15363|80pass,0fail|3:53|
| Previous bootstrap body |15364|79pass,1 runtime assertion failure|3:54|
| Injected omitted RNG rejection |15365|79pass,1 runtime assertion failure|3:55|

Control15364 differs in exactly bootstrap.sio (plus derived manifest) and fails
only stats_resampling_seed_guard after successful native compilation.
Control15365 differs in exactly correlation_permutation.sio (plus manifest):
the rejection loop is disabled. It fails only the independent correlation
permutation reference after successful native compilation. This is a deliberate
fault-injection control, not a historical implementation. Both preserve all77
previous fixtures. No control uses the invalid zero-state shuffle path.

## Scientific rendering and custody

The new Canvas/SVG consumer uses five synthetic tied pairs, one exhaustive
120-label reference and MC budgets64/256/1024 with explicit seeds7/11/19.
The forest centers q, displays Wilson95 bounds on a common0..1 domain, labels
the corrected p separately and draws the conditional exact tail line.
Canvas axis pixel, scene metadata and formula assertions pass natively.
An additional independent C++23 checker recomputes the four figure datasets
and passes22checks against the actual native SVG: Wilson bounds/centers,
full0..1axis/ticks, colors, numeric corrected/exact p labels and assumption
texts. The exact reference projects to pixel217; fractional mathematical
coordinates are quantized by the renderer. Checker executable SHA256:
61effbe4b5dbdcb8f260c3ebc64a683279a08d8d7b97eb6722e87b2c5afa9e16.
This checker and final documentation/packager updates were added after the
native snapshot; the numerical sidecar ran separately in C++23 on t560.
The actual1180x450SVG parses as XML and was visually inspected. QuickLook
adds square padding/crops its right edge; the SVG retains the right-side p labels.
No interval-coverage assertion is imposed on a random MC realization.

Three subagents supplied core/seed repair, C++reference and figure/independent
review. Root integrated packaging, controlled executions, artifact and hash
custody, source/main identity and final acceptance. Darwin was consulted before
new Sounio; invalid generated candidates were corrected. No Sounio native
execution occurred outside Forge; numerical oracles use C++23/Sounio per ADR009.

Custody:
/Users/demetriosagourakis/Documents/Sounio-stdlib-sprint-2026-10-16/correlation-permutation-2026-10-18.
Selected sources, native logs/JUnit/receipts, manifests/archive hashes, C++ reports,
SVG and reviewed docs are retained. Full source archives/compiler builds/Git
history are not copied to the Mac. Source completion/native acceptance, remote
CI, merge and user acceptance remain distinct.

## Next major implementation front

A checked rectangular least-squares kernel using column-pivoted Householder QR,
compact reflectors, explicit rank/conditioning statuses and scale-safe residuals
can support multivariable statistics, genomic covariate adjustment and physics
calibration. The current private Matrix8 QR's absolute1e-15 cutoff is not such
a contract. Weighted fitting, numerical oracle and residual/uncertainty figures
must accompany that implementation; no normal-equation shortcut or SOTA
superiority claim is established here. The major multi-domain sprint continues.

