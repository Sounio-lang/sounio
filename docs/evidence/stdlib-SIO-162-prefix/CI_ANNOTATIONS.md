<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-162-prefix.ci-annotations
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-162-prefix.ci-annotations
-->

# Hosted test-discovery follow-up

The PR3102 run 40149022452 Contracts Core job [121036707381](https://github.com/Sounio-lang/sounio/actions/runs/40149022452/job/121036707381) failed the test_annotation_coverage_gate step because stdlib_beta_uniform_baseline.sio and stdlib_checked_distributions.sio had no mode annotation in their first 20 lines. PR3110 inherited the same two missing annotations in run 40149862516, job [121039266305](https://github.com/Sounio-lang/sounio/actions/runs/40149862516/job/121039266305). The Contracts aggregate jobs reported Core failure rather than a separate numerical defect.

Both fixtures were already compiled and executed by the explicit focused gate, including current-c21 Slurm17204. The ordinary Full Test Suite passed while skipping these unannotated files; that pass therefore did not supply discovery-based coverage of these two fixtures. The fix adds only run-pass modes and exact actual stdout markers to their first lines. It changes no scientific implementation or assertion. The previously executed native markers are BETA_UNIFORM_BITS 4607182418800017408 and CHECKED_DISTRIBUTIONS_SMOKE_PASS.

[Raw CI log tails](ci_annotations/162-core-tail.log) and [the inherited failure tail](ci_annotations/167-core-tail.log) are retained separately from [the local annotation-coverage result](ci_annotations/coverage_gate.stdout). The annotation gate is structural, with its own refusal controls; it does not execute numerical verification. A subsequent hosted run must establish that the newly discovered tests and the remaining Contracts steps pass. No hosted CI pass is claimed by this local correction. Existing native scientific receipts remain immutable.
