<!-- docs:meta
topic_id: repo.docs.plans.stats-viz-phase-profile-execution-2026-10-19
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.plans.stats-viz-phase-profile-execution-2026-10-19
-->

# Scientific stdlib validation: phase profiling and compact reference — 2026-10-19

## Accepted source and scope

Active remote worktree: /workspace/sounio-sprint-stats-viz-main2-20261019,
based on remote main 15a1983fd8d9bb4730f2fd6ca72e50f5e0c88d0f, rechecked this turn.
No commit, push, PR or merge was performed. Full compiler/repository CI is separate.

Forge/Slurm15613 accepted **101/101**, zero failures/skips/errors, exit0:0,
elapsed5:05. Snapshot a79744bdd0a0adbe29654ed3c9c3b480cf500bb967cff2f3e22e9298e2ae12f0.
Runtime manifest c8414e481dbfca15e77b728abfd619fd19a73d99c4da4217ca443750f52f4ecb.
The full corpus membership and scientific contracts remain unchanged.

Every stdlib .sio source is byte-identical to the prior101 snapshot. The filtered
production-source manifest is identical before/after, SHA256
fca076242ecef796a5b3f78ac76f98eeebbf5554acc47b0267f9b0a7ee96a014.
The optimization changes the reference generator/fixture and adds measurement tools.
It does not change a Poisson, QR, physics, genomics or rendering numerical kernel.

## Compact maximum-shape reference

The independent C++23 quad Poisson generator replaces the expanded256x8 case15
setup/check helpers with fixed-bound loops and24 directly rounded mean/predictor
reference-table entries. Eight coefficient and three offset constants are retained.
The table storage is472 bytes; there is no extra dense matrix or runtime log/exp
used to reconstruct expected values. The other25 cases, header and main are
byte-identical after excluding the replaced maximum-shape block.

C++ bit checks cover all2048 design entries,256 counts,256 offset values,
256 expected means and256 expected predictors, plus all24 table cells. Actual
native assertions still cover every mean/predictor, all8 coefficients, deviance,
convergence, rank/df, iteration and assumption metadata. The tiny coefficient
nonzero/relative guard and all tolerances remain unchanged.

Source SHA256:

- generator55dde8e085a998966109fd057676edf5fc856dee5c0324ed56b5d10fcab3a36b;
- compact fixture05682ab087ea431b5943e31cf97e114ba719bb79cee0a32472b5a2539edfac41;
- original fixture4e099afe29d28d3bd175a2cdbda57d3f95ffa335fcab5801ca1b243baf2ad998.

Root freshly compiled the C++23 generator with -O2 -Wall -Wextra -Wpedantic and
-lquadmath on t560; generated fixture matches exactly. The original generator and
fixture are retained. Independent source review found no coverage loss.

## Native guards and separate profiling verdict

| Stage | Slurm job | Outcome | Elapsed |
|---|---:|---|---|
| Full scientific corpus |15613|101/101,exit0:0|5:05|
| Last-row mutation |15614|0/1; compiles, numerical assertion fails|0:07|
| Phase measurement |15615|21 accepted phase rows,PROFILE_READY,exit0:0|1:15|
| Faulted phase measurement |15616|exit1:0,PROFILE_FAILED,no PROFILE_READY|1:19|

The last-row mutation changes only returned means[255] in the fixture before
checking. The unchanged expected value detects it. This is an injected coverage
control, not an alleged historic library bug. The faulted profiler uses the same
mutation: all three affected direct runs fail and batch runs are skipped. Other
entries still execute; the overall verdict propagates failure.

Profiling uses an explicit package-only dispatcher through Foundry corpus-smoke.
It declares corpus_acceptance=false; its pass is not a101-test result. The project
corpus dispatcher is unchanged. Original dispatcher hash and purpose are retained.
The Foundry runtime receipt supplies positive Slurm identity, uid/gid1000 and
clean-v1; SLURM_JOB_ID is not forwarded by the sanitized phase environment.

## Measurement method and attribution

Three paired before/after compilations run sequentially in one allocation, with
order before/after, after/before, before/after. A visualization entry supplies a
separate control. Initial committed-compiler materialization is outside timing;
canonical bin/souc startup and strict integrity rechecks are included in compilation.
The pinned decoded Madaros SHA256 is
125c25703ac4062fc6befb7acf77e945abb13405f0adb5ab09fbc5a8d4b5b29f.
No compiler source was rebuilt. External compiler paths are rejected.

Every accepted compilation produces a fresh ELF. That same ELF runs directly,
without recompilation, once and in a20-execution batch. All expected stdout markers
must be present; batch stdout is byte-identical to the initial accepted output.
ELF, compiler, inputs and runtime-receipt hashes are checked. Inputs are unchanged.
The parent stack ceiling is524288KiB, matching the native launcher policy; this is
an address-space limit, not a resident-memory reservation.

GNU time records wall/user/system seconds, maximum RSS and exit status. Wall has
0.01s output resolution. Direct-run timing includes timeout/process startup; the
batch includes one helper shell/receipt guard,20 startups and stdout writes.
Maximum RSS is the process/descendant maximum, not simultaneous tree aggregate.
See [GNU time](https://www.gnu.org/software/time/manual/html_node/Memory-Resources.html)
and [Linux getrusage](https://man7.org/linux/man-pages/man2/getrusage.2.html).

C++23 postprocessing separately computes CPU=user+system, medians/ranges, paired
comparisons and byte reductions. Source033e0412baa2288767687dc74c76268759bfc80e124cfd687b7dca187da191b2;
executable64aebe4f954dd2adbeb7bd3299b45d04928fd52c7553a7c03142c97df295918c.
Report d7e433e3bc309e05d4e45fc3db7d95174dda1c3f4c4b4e34a19c0ec63fedb7e1.
This postprocessor is a separate analysis sidecar, not part of measured compilation.
It rejects incomplete/failed rows, nonfinite measurements and zero/unavailable RSS.
Root rejection controls for missing row,NaN,unaccepted measurement and zero RSS all
exit nonzero without emitting a comparative report. The actual faulted profile
is also rejected.

## Observed results: three pairs on one host

| Metric for the26-case reference executable | Before | Compact | Reduction from C++ report |
|---|---:|---:|---:|
| Source bytes |170556|68952|59.572223%|
| ELF bytes, stable across trials |2150856|885192|58.844665%|
| Compile CPU median, s |7.59|5.38|29.117260%|
| Compile wall median, s |9.16|7.66|16.375546%|
| Compile maximum-RSS median,KiB |1800204|1698828|5.631362%|
| Direct-run wall median, s |0.03|0.03|0%|
| Direct-run maximum-RSS median,KiB |3072|1612|47.526042%|
| Batch20 wall median, s |0.62|0.61|1.612903%|

Wall ranges overlap: before9.06..10.94s; compact7.52..11.71s. The first compact
trial is slower in wall time. These are descriptive measurements of this workload,
not a confidence interval or generalized speedup claim. The batch time difference
is only one displayed clock step; no material execution-latency gain is asserted.
The numerical production algorithms are unchanged.

The visualization control records compile RSS3319308KiB and direct-run RSS8192KiB.
Its actual SVG is byte-identical between the prior scientific checkpoint, the new
full corpus and the profile: ed1f04d79d8af8a635d58c93eac84064c36c9f010febc597c2048e55bc2d85cc.
The independent24-marker C++ geometry checker still passes.

Whole-corpus observations are5:04 before and5:05 here, with current Slurm batch
MaxRSS7060932K. They do not demonstrate a whole-suite wall-time or peak-memory
reduction. Profiling isolates the heavy compile phase from the much smaller direct
execution footprint; that is the measured attribution for the next optimization.

## Custody, integration and next scientific work

Runtime packages and executable collections stay remote. The local review custody
contains changed sources, exact-before controls, raw timing/receipt manifests,
C++ reports and the actual SVG. No generated ELF collection or full source archive
is duplicated locally. Checksums bind the custody and selected runtime sources.

Post-acceptance receipt/master/status documentation is separate from the frozen
runtime source. The broad scientific sprint continues: robust/clustered covariance
and calibrated inference, bounded count-matrix/I/O, and separate negative-binomial
dispersion. Further validation/compile-once work should use these per-phase receipts
while concrete stdlib integrations remain the primary scientific track.
