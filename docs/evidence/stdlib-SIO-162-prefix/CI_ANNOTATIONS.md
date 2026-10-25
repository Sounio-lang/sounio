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

[Raw CI log tails](ci_annotations/162-core-tail.log) and [the inherited failure tail](ci_annotations/167-core-tail.log) are retained separately from [the local annotation-coverage result](ci_annotations/coverage_gate.stdout). The annotation gate is structural, with its own refusal controls; it does not execute numerical verification. Both newly selected fixtures were also strict-checked, independently native-compiled and executed with the committed bootstrap seed SHA-256 8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc. Slurm 17256 completed 0:0; both exact stdout markers matched and both check/compile diagnostic scans were clean. The [bootstrap-native gate result](ci_annotations/seed_native/gate.result), [source/compiler hashes](ci_annotations/seed_native/compiler.sha256), executable hashes and [accounting](ci_annotations/seed_native/sacct.txt) are retained. This is actual bootstrap runtime qualification, distinct from earlier Madaros runtime qualification. An initial launcher control, Slurm17254, incorrectly sent the verb check directly to the raw positional CLI and failed with source=check/no-main; it never compiled the fixture and is excluded from qualification. Its original logs are preserved in launcher_control. The corrected launcher uses the lean_single wrapper engine and verifies the exact same 895 seed bytes. A subsequent hosted run must establish that the newly discovered tests and the remaining Contracts steps pass. No hosted CI pass is claimed by this local correction. Existing native scientific receipts remain immutable.
