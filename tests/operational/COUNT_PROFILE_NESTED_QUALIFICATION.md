# Nested profile-refit qualification orchestration

Preparation is separate from execution or statistical acceptance. Neither the
harness nor the job templates submit a Slurm job or fabricate a passing receipt.
Root may schedule the pilot only after the full final-source functional gate and
its independent C++ concordance pass. Existing frozen jobs/snapshots are not
modified. The numerical generator and summary use C++23; production fits and
sampling/refits use Sounio. All compilation/native/numerical execution is Slurm-only.

## Generator compatibility and entry policies

`count_profile_calibration --generate` retains the original generation function,
its source template, data, seed schedule, options and byte-level output. The
additive `--generate-nested` derives the new call from that same generated driver:
explicit two-row all-ones embedding for the declared intercept-vs-two-group
design, a reusable two-element warm-start buffer, checked nested workspace/query
and two appended arguments. The entry keeps the observed null search cold;
replica initialization policy belongs to the accepted production API.

`--generate-nested-traced` additionally creates one caller trace slot per isolated
dataset/process and calls the additive traced entry. It emits the same CAL_RESULT
and CAL_P schemas, plus actual counters:

```text
CAL_WORK index fit_attempts count_draws uniforms proposals profile_evaluations
CAL_TRACE index trace_status metric1 metric2 ... metric10
```

CAL_WORK comes directly from the returned bootstrap metadata. CAL_TRACE uses
scalar getters for actual callsite counters, in the module's declared order:
fit calls; IRLS entries; initial-rank QR; weighted QR; profile point evaluations;
paired ranking; tracked full log-PMF row calls; null cold searches; replica-null
seeded searches; alternative seeded searches. Metric7 includes the tracked profile and paired log-PMF rows and excludes PTRS proposal calls.
No counter is inferred from B,
RNG state or another counter. The default limit is i64MAX **per metric**, not a
global work budget. A latched trace error reports only a partial recorded prefix
and the traced entry returns unavailable status4/stage10. The summary preserves
those actual CAL_RESULT/P values; it never reconstructs or overwrites model
availability from trace values. Trace prefix sums use unsigned128 arithmetic.

B2 is supported only by the new generation/summary modes as a labeled functional
smoke: it retains the generated data/seeds and changes explicitly declared budgets
to2replicas/6profile attempts. The campaign harness accepts only pilotB31 and
qualificationB199. A B2 smoke is not a calibration or performance observation.

## Strict gate and campaign separation

The harness requires the unchanged seven-field
`stdlib-advanced-functional-v1` receipt, exact compiler/source-manifest/results
hashes and an identical canonical manifest over `find -L bin scripts self-hosted
stdlib tests examples`. No `.git` directory is required. Unknown/duplicate fields,
nonzero/malformed results, missing or duplicated required rows are rejected.

In addition to provenance, required CENTRAL rows cover the nested bootstrap,
nested contract, new paired likelihood, normalization adapter/status and both
trace fixtures, each with check/compile/native/stdout-marker. Independent C++
concordance requires the nested and nested-traced bootstrap modes, paired
likelihood, dispersion increment and normalization. Expected-negative controls
and `input-hashes unchanged0` / `compiler-hash unchanged0` rows are also mandatory.
Thus old full02 results rebound to a new manifest still fail before compilation.
No receipt is approved merely because its file is readable.

Campaigns use only `pilot_nested_B31` or `qualification_nested_B199`. They never
pool old results. Execution is serial: one compiled ELF per batch of16observed
datasets, one fresh process per dataset, then ordered stdout stitching and a
single final batch marker. The grid, independent generation seeds, bootstrap
seeds, options and B are unchanged. Summaries retain unavailable rows, report
availability and conditional rejection diagnostics plus the all-failures
sensitivity range. Availability selection can bias conditional rejection rates;
no global calibration or finite-sample validity follows from a partial round.

Round-robin batches, at most128datasets per cell, strict hash-based resume and
partial-batch preservation remain. The wall budget is operational and bounded
by86400seconds; templates require an explicit root-authorized **remaining** budget
so pilots/prior work are accounted for in the aggregate24h round. A hard kill
reserves the remainder conservatively; incomplete batches require review and are
not overwritten or silently discarded.

## Prepared jobs and operational controls

The two `.sbatch` templates are inert files until explicitly submitted. Arguments
are frozen source, output root, attested compiler, genuine final-gate receipt and
remaining wall seconds. Set durable Slurm chdir/stdout/stderr paths outside source.
The pilot allocation is15minutes; the round allocation is24hours. Root must cap
remaining seconds to the allocation and aggregate budget.

`count_profile_calibration_nested.sh` is a Slurm UID1000 operational test, not a
campaign. It compiles the baseline/current C++ generators, checks complete old-mode
byte identity across9cells, compares nested raw data/options/seeds, executes a
small generated B2 driver, verifies actual CAL_WORK and invalid environment
exit64, and rejects incomplete/missing-counter output. It constructs a deliberately
forged **negative control**, rebinding real old full02 rows to the current manifest
and compiler; the expected result is65 before generator compilation. That file
is explicitly named FORGED_OLD_ROWS and must never be used as a functional receipt.

Traced operational coverage additionally requires the accepted traced-source
snapshot. Source preparation, a focused smoke, a genuine combined gate, a pilot,
a full round and inferential calibration are distinct states.

The receipt also requires all four phases for the nonempty canonical-source glob
`tests/run-pass/stdlib_scale_count_*.sio`, plus the two normalization fixtures and
`stdlib_deep_stats_count_profile_nested.sio`. Rank, cached-normalizer, bootstrap-cache
and both GLM/increment trace-control C++ concordance labels are mandatory; rebinding
a pre-cache gate such as16933 to a newer manifest does not satisfy these rows.
