<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-169.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-169.readme
-->

# Dynamic right-censored survival curves and checked logrank: SIO-169

This increment implements ordinary right-censored Kaplan-Meier analysis with reusable caller-owned storage. One indexed heapsort orders the observations; one aggregated distinct-time walk produces pooled, group-0 and group-1 curves and the two-group logrank statistic. It removes the legacy logrank's repeated data scans and inline exp/log/Lanczos/incomplete-gamma copies. Cox models, delayed entry, left truncation, competing risks, weights, stratification and interval censoring are outside this API.

The branch depends on PR [3114](https://github.com/Sounio-lang/sounio/pull/3114) for reusable indexed sorting and its parent PR [3102](https://github.com/Sounio-lang/sounio/pull/3102) for checked Normal/special kernels. Its parent is `a5b2a7347a4cd2d09354198ba4094264a0d59176`. Incorporated remote main is `7ce5eddeef78c76c7e744150ed9c169c53a3d3b0`; its differences from compiler-source `c21b0e139830ca4949f80cba62887a8922a972d7` are workflow/changelog only, with unchanged compiler and numerical trees.

## Public API and statistical definition

`stats::checked_survival` exposes `survival_workspace`, `survival_checked` and `logrank_tail_checked`. Inputs are follow-up time, binary event (1 observed death, 0 right censoring), and binary group, with an active observation count. Times must be finite and nonnegative. Binary64 equality defines ties, including signed zero; no measurement-resolution tolerance or silent time repair is applied. At a tied time, deaths use the risk set before **any** deaths or censoring at that time are removed. Censor-only times are retained in the full output.

Kaplan-Meier uses the product of `(risk-deaths)/risk`, represented through a compensated log sum for stable log-survival. Greenwood accumulates `deaths/[risk*(risk-deaths)]`; the standard error is survival times the square root of that sum. The nominal confidence level is `1-alpha`; for example alpha=0.05 requests 95% pointwise limits. Pointwise log-log confidence limits use `log(-log(S))` and the canonical Normal critical quantile. These are pointwise asymptotic intervals, not simultaneous confidence bands or a finite-sample coverage certificate. Nelson-Aalen cumulative hazard accumulates `deaths/risk`. The [primary survival manual](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.formula.html) describes these estimator and confidence-transform choices; no R execution is used for concordance.

The logrank walk accumulates group-1 observed deaths, expected deaths `d*risk1/risk`, and the tied hypergeometric variance `d*(risk1/risk)*(risk0/risk)*(risk-d)/(risk-1)`. The score uses the exact integer contrast `d1*risk0-d0*risk1`, followed by one division by pooled risk and compensated accumulation. The declared observation bound makes these products exactly representable in i64 and binary64. Expected deaths for each group are independently accumulated, avoiding subtraction of a large expected total for the OE diagnostic. Variance is also compensated. Its p-value is the usual asymptotic chi-square-one-degree reference, not an exact randomization p-value; see the [primary logrank documentation](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survdiff.html). Zero variance and absent groups return explicit unavailability, not a fabricated null p=1.

The asymptotic two-sided tail is evaluated directly as canonical `Q(1/2,z*z/2)` with its log-upper value. It avoids both `1-P` cancellation in rare tails and subtract-ln2/add-ln2 cancellation near the null. A nonzero z whose squared argument is subnormal, unrepresented or nonfinite is refused explicitly. Scalar underflow can leave the logarithm available. Tolerance is an iterative stopping parameter, not a universal accuracy certificate.

The observed/expected ratio is a descriptive diagnostic. It is named `oe_ratio` with its own availability and must not be interpreted as a fitted Cox hazard ratio. The legacy `hr` field retains that original diagnostic meaning.

## Returned full curves

The caller supplies an integer sort-index buffer, a numerical curve buffer, and an integer count/flag buffer. Each pooled distinct time has three records: cohort 0 pooled, cohort 1 group 0, cohort 2 group 1. For row `j` and cohort `g`, record index is `3*j+g`.

| Numerical field within an 8-value record | Meaning |
| --- | --- |
| 0 | Follow-up time |
| 1, 2 | Survival and log-survival |
| 3, 4 | Greenwood sum and survival standard error |
| 5, 6 | Pointwise log-log lower and upper confidence limits |
| 7 | Nelson-Aalen cumulative hazard |

Integer records have stride 4: risk before removals, observed deaths, censoring, availability flags. Bits 1, 2, 4, 8, 16 and 32 respectively mark survival, log-survival, Greenwood, standard error, confidence limits and cumulative hazard. The counts and time remain meaningful when an estimate flag is absent; unavailable estimate slots are zeroed, and consumers must read the flags. Records beyond an absent/exhausted group's observed follow-up have flags zero, avoiding silent extrapolation beyond its final censoring time.

At S=1, Greenwood and standard error are zero, while the log-log interval is unavailable because its transform is singular. At a terminal all-death time, survival is exactly zero and log-survival is genuine negative infinity; the Greenwood/standard-error/log-log formulas are marked unavailable. A positive survival that underflows has independent log-survival availability. The implementation does not turn an unrepresentable confidence limit into an invented finite endpoint.

Each cohort has a median value, an availability flag, and a separate median status: 0 reached, 1 unreached, 5 numerically ambiguous. The first crossing of one half is classified with a reduced exact integer survival fraction while checked i64 products fit. Cross-cancellation happens before multiplication. After that size limit, positive IEEE divide/product intervals are widened at every step; only a decisive side is accepted. Overlap with one half is sticky ambiguity, preserving the nominal curve without claiming a later crossing was the first. This uses the current engines' binary64 arithmetic and explicit guard assumptions; it is not a Lean proof or a universal interval-arithmetic certification.

## Memory, work and refusal

The workspace query supports n up to 1,000,000 and reports a worst-case active payload: n sort indices, 24*n numerical values and 12*n integer count values, totaling 296*n bytes. It excludes the 24*n-byte time/event/group inputs, allocator capacity, virtual memory accounting and process RSS. Only actual distinct-time output records are written. Buffer ownership permits reuse and avoids per-observation clones or full record copies during sorting.

Status 1 is invalid shape/options, 2 nonfinite data/options, 3 work exhaustion, 4 unavailable arithmetic, 5 unsupported region and 6 payload cap. All data and shape preflight precedes writes. Invalid event/group/time, capacity, alpha and payload controls preserve both sort scratch and output buffers. Sort or critical-quantile refusal preserves output buffers, although sort scratch can change after sorting starts. Committed rows are recorded if an unexpected later arithmetic refusal occurs. `risk_visits` records the emitted prefix on that path, rather than a primitive-instruction count.

Critical-quantile and final-tail budgets have different units and are separate arguments. `critical_evaluations` counts Normal-quantile forward evaluations; each has its own inner cap inherited from the canonical inverse API, so worst-case work can be quadratic in that budget. `logrank.kernel_iterations` counts the final Gamma tail's iterations. These counters are never added or subtracted as though they shared a unit. Curves can succeed while the separate logrank result refuses its tail budget. A native witness exercises exactly that post-evaluation path and retains the performed work.

## Compatibility and bounded legacy correction

`stats::logrank::logrank` retains its fixed-array arguments and five original scalar fields. It now routes through the checked core, adds `Alloc`, and extends the result with log-p, variance, critical-evaluation and tail-iteration metadata. Invalid inputs, unavailable legacy scalar tails and undefined test/OE diagnostics fail closed. A tracked whole-Sounio import/call search found no external callers of this module; its inline module tests are updated and executed. This does not describe unknown downstream code outside the repository.

`medical::survival::make_event` additionally rejects nonfinite follow-up times. The private historical medical KM/Cox prototype is explicitly identified as unqualified: its old heuristic confidence is not a confidence interval. Its unrelated private algorithms are not promoted or rewritten by this change.

## Independent references and discriminating failures

The canonical executable implementations are C++23 and Sounio. C++ uses exact `cpp_rational` small risk tables and survival/median fractions, plus a 113-bit extended-exponent numerical reference for transforms and asymptotic tails. All returned rows are compared for ordinary cohorts. The 100,000-distinct-time stress case checks every integer risk-table record through a digest, a compensated full survival sum, and deterministic sampled numerical records; it does not pretend that sampling is an all-record numerical comparison. The 100,000-record tied cohort returns all distinct-time records for comparison.

The original legacy direct-tail counterexample has 256 uncensored observations, with one group dying before the other. The old implementation returns p=0 while the independent reference is `1.7546978631932102e-70`, log-p `-160.61865982521088`. Original code, native outputs and failure diagnostics are retained separately from the corrected run.

A development counterexample has times 1, 1+one ulp, 1+two ulps and 2. Its exact pooled survival reaches one half at the second event; a log-only threshold comparison delayed the median. The original failing output is preserved. Corrected tests include this equality and fractions 501/1001 above one half and 500/1001 below it.

A separate development counterexample uses 100,000 observations with one individual in group 0 and a block of 99,998 group-1 deaths. The subtract-large-floats score is `0.999979999993229284882545471191` instead of exact `0.99998`; the resulting log-tail error is `3.38547146457624545441903017515e-7`. Both engines reproduce the same failure in Slurm 17365. The exact-integer-contrast correction is evaluated against that same input without changing tolerances.

Two independently searched legal million-observation risk sequences exercise the explicit ambiguity path after exact i64 fraction overflow. Their exact final survival minus one half is `-20738890957/157764019889697048443206250` and `1119012944453/1918853757079427291879062500`. The required result is status 5 with the nominal curves retained, not an epsilon-based choice of either side. The search itself is C++23 exact rational arithmetic and is preserved as an arithmetic witness.

The independent Normal tail reference uses erf/log1p directly near zero, and a positive scaled Mills integral in rare tails to avoid the installed Boost 1.83 erfc cutoff. GK61 refinement/truncation checks qualify that reference. This prevents a second implementation's underflow from being accepted as scientific truth. Tail fixtures include signs, near-null values, scalar underflow with retained log, squared-argument refusal, nonfinite inputs and work exhaustion.

## Final qualification and resource observations

Slurm job **17367** completed with exit **0:0** in **00:02:31**, using immutable source packet 13, UID 1000 and a 524288 KiB stack. The exact result is `SURVIVAL_CHECKED_GATE_PASS`. Both the committed seed and source-built Madaros execute the fixtures; strict checks reject compiler diagnostics even if a compiler exits zero.

The pinned compiler artifact hashes are:

- Source-built c21 Madaros: `a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3`.
- Committed seed: `8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc`.

Both engines pass 29 ordinary cohorts, five scale cohorts (three of 100,000 records and two of 1,000,000), and 34 signed/near-null/extreme/refusal tail rows. The same legacy counterexample now passes independently. The legacy module tests, medical event caller and nonfinite medical-event refusal pass. A latent missing `Panic` declaration in the existing medical caller was repaired after an actual strict-Madaros E035; its assertions and inputs were preserved. Shape, event and absent-group legacy refusals have exact exit 1 and explicit before-call markers; the engines' native assertions are intentionally silent.

The focused gate also rejects six actual-output corruptions with **specific** verifier diagnostics and exact exit codes: risk set, confidence limit, availability mask, representable rare tail, full risk-table digest and missing completion marker. A subsequent invocation on the completed output directory exits 65 with `fresh output directory required`, while hashes of the completed gate result and compiler manifest remain unchanged.

The measured maximum absolute discrepancy across ordinary checked fields is `1.4402564651674005e-12`; across scale fields and aggregate checks it is `7.458086967177275e-12`. The largest scale log-tail discrepancy is `4.1146272839215863e-12`. Standalone tail rows have maximum absolute log discrepancy `2.4158566123247789e-12` and maximum relative log discrepancy `6.0293500615678877e-13`. These are observations on the retained grid, not all-input error bounds. Row comparisons use 5e-12 absolute plus 3e-13 relative, full-survival sums use 3e-10 absolute plus 3e-13 relative, representable p-values use 1e-10 relative, and log-tail comparisons retain separate near-null relative and ordinary absolute/scale envelopes. This slice qualifies the curve confidence calculation at alpha=0.05; other accepted alpha values inherit the canonical inverse kernel's checked availability and remaining accuracy limits.

For the same 100,000-record rare-group score failure, the corrected actual score is `0.999979999999999979998221988353`, with absolute error `2.00017780116469050505966774955e-17` and log-tail error `4.11462728392158631219720233962e-12`. The final job creates a derived protocol view by renaming only the already-verified full-family completion marker for the focused score reader. All numeric records are unchanged. Original full native output is committed; redundant projected numeric copies remain in retained scratch, with their hashes and exact transformation recorded in the job script. The scientific tolerance was not relaxed.

Observed per-executable resources on the allocated worker are:

| Fixture | Committed seed elapsed / max RSS (KiB) | Madaros elapsed / max RSS (KiB) |
| --- | --- | --- |
| Ordinary 29-cohort curve/inference fixture | 0:00.20 / 4108 | 0:00.08 / 1604 |
| Five scale cohorts, including two million-record ambiguity controls | 0:27.94 / 34352 | 0:31.93 / 31480 |

These are workload-specific observations. Scale output is sparse in time for the two million-record cohorts, while the 100,000-distinct-time cohort writes its entire table. Worst-case requested payload, virtual allocation and actual resident pages differ; the table is not a claim that a fully populated million-distinct-time curve requires only that RSS. Batch RSS also includes compiler execution and is recorded separately.

The deep-science runner registers the focused gate additively with a distinct child output directory. Its shell syntax is checked; the full shared runner and latest hosted CI remain separate integration gates. Registration changes no numerical source. Scientific source is commit `5464ff6672e9da98f1a257b292e0c5d1ba101844`; explicit native-refusal site markers are `cbe3e64f4118634cb7c7f3c72cb7c5971242394f`, and central registration is `7a77ce34c796dbecdf7372024bce230ba43c3b41`.

`FILES.sha256` covers the committed textual evidence. Original failures are kept under `legacy_before/`, `median_before/` and `score_before/`; exact-rational ambiguity search evidence is under `median_ambiguity_search/`; final outputs, work/refusal/mutation checks and source/ELF/compiler manifests are under `final/`. Executables and complete immutable source packets remain in retained scratch rather than Git. Million-observation fixtures are generated with compact loops. Files overlaid on hardlinked snapshots were unlinked before replacement; no shared quota was increased and no historical artifacts were removed.
