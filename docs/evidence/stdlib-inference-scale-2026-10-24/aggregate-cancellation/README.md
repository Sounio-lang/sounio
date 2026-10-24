<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.aggregate-cancellation.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.aggregate-cancellation.readme
-->

# Aggregate cancellation: failing-before and conservative availability

This folder preserves compact original receipts, exact replay sources as `.txt`,
compiler/source/ELF hashes, stdout/status/resources and the narrow production patch.
Full frozen source trees, native executables and compiled C++ binaries remain in
`/scratch/sounio-stdlib-inference-scale-20261024/aggregate_cancellation01`,
`aggregate_cancellation_analysis02` and `aggregate_guard_focused01`.
No compiler or benchmark source was modified by this audit.

## Observed defect, not a hypothetical accuracy claim

The Poisson count-zero identity is logP=-mean. Native input bits reconstruct exact
binary rational contrasts; the reference never substitutes analytic fitted means.
For three rows with first means [1e-300,2,1] and second [2e-300,1,2], the true mean
contrast is approximately -3.33333333333333341686363945e-301. Both compiler92d and
fresh d74 reported status0/valuebits0 in full, rank and cached-rank aggregates.
Reversing row order retained the negative result. A separate two-row exact
cancellation produced available0. These observations are in Slurm16996.

The initial16996 auxiliary boolean classification was inconsistent with its own
raw output and exact-rational decimal report. It is NOT used as an acceptance
classification. C++-only retry16997 preserved the SAME native stdout and ELFs,
freezing the exact numerator-nonzero and raw-native-bit-zero predicates BEFORE
presentation. It classified all three tiny-first routes as false available zeros.
Both versions and their raw receipts remain here; STATUS0 for those diagnostic
jobs means processing completed, not that the original numerical result passed.

## Authorized narrow policy

The only production change is a guard in the shared mean-finish engine: if
`scale>0` but the computed scaled sum is0, return existing unavailable status5 and
cleared value0. All floating operations/order for nonzero sums remain unchanged.
All six full/rank/cached and traced/untraced entries share this finish engine.

Mixed nonzero rows with a genuinely exact zero sum are deliberately unavailable
under this policy as well. The implementation has no exact-sum certificate to
separate them from a lost tiny contribution followed by cancellation. Pointwise
zero rows (`scale==0`) remain available0. This is conservative availability,
not exact summation, an interval bound or a certificate for every nonzero sign.
The snapshot/API shapes, RNG/fits, compiler code and benchmark frozen1132/48
packets remain unchanged. The known consequence is stricter numerical availability.

## Focused result16998

All15 selected fixture check/compile/native/marker phases passed. The new fixture
reports24 rows across four cases and six APIs, and checks slot reuse, invalid-size
clearing, detached snapshot, public cache reset and independent trace status.
Its exact C++23 oracle reconstructs rational values from native input bits and
checks the nonzero result against the exact midpoint rounding cell of adjacent
binary64 values, including ties-to-even. There is no absolute tolerance floor.

Existing comparators also passed: three61-scalar/three-original-aggregate paired
outputs, cached and legacy216-case full likelihood outputs, exact Poisson-zero
bits, eight far-fallback cases, and the three B2 traced/cache/nested outputs.
Two legacy stdout byte comparisons passed. The same final new fixture compiled
against frozen pre-guard1132 code but failed natively with exit1 at its first
status assertion; guarded execution and the exact C++ oracle exited0.

The new oracle's --before mode separately requires the original16996 false-zero
witness and its reversed/exact-cancellation controls. This is witness validation,
not a passing scientific output from the old library. The focused pass does not
claim a full suite, performance target, campaign calibration or general error bound.

## Custody and replay

SOURCE_MAP.tsv links original text receipts to scratch paths; ORIGINAL_FILES.sha256
hashes those original bytes. These paths precede optional lossless gzip compression
or `.cpp`/`.sio` to `.txt` evidence renaming. FILES.sha256 is the authoritative final
copied-file manifest. Existing per-run manifests can name full source/ELFs omitted
here; they remain original receipts, not standalone bundle manifests.

The patch transforms likelihood1132d30ab527980de547f1449ca484236f251e9eaecacb2043d907c5c0b48402
to7d3fd3f8603fdb23bce3e442795e26e869327f0547a90fab7ceda2edb25aafc2.
The final fixture SHA is71532629a2b8c8f95793c42a68703991f22e7d4c3be6a310367f3774b46c9b7f;
reference SHA is9fc2da0bd53063816970390a49527f7dcb470faf1adb1ecf2f136ce5d00b28ff.
Use the preserved focused dispatch/job for compilation and execution only under
Slurm. Assembly of this custody folder reran no numerical/native work.
