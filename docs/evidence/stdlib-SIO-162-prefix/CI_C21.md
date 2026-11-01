<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-162-prefix.ci-c21
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-162-prefix.ci-c21
-->

# Current-main compiler repeat of SIO-162

The reviewed SIO-162 implementation was synchronized with origin/main c21b0e139830ca4949f80cba62887a8922a972d7 in merge 7bf6f07c665e84b4877b3e7bd5db833cb56c1f7e. Changelog fragments were committed at 20ad2fbf3c. No scientific algorithm changed in this sync.

Slurm 17204 completed with exit 0:0 in 29 seconds on the allocated CPU worker, UID 1000, stack 524288 KiB. The focused gate rebuilt and executed the actual native fixtures, all 444 C++ reference rows, the two unavailable scalar probes (each expected exit 1), and four corrupted-output rejection controls. Strict seed diagnostics were scanned independently of process exit status and were empty. See [compact receipts](c21/gate.result), [actual C++ concordance](c21/reference.verify.stdout), [source hashes](c21/inputs.sha256), [compiler hashes](c21/compiler.sha256), and [Slurm accounting](c21/sacct.txt).

The executed compiler was /scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready, SHA-256 a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3, independently rebuilt from source c21. Root verified its 5295-file input catalog against the source archive. Upstream rebuild custody: run 40144792765, Witness job 121024346990, artifact 12580970584; archive SHA-256 e3eaa0099eb039094c47afdcc3f3e22a90a6bf8a2432d0b5d4e867c486feb211. This repeat does not relabel the earlier retained d74 compiler as a fresh rebuild.

The bounded work regions, representability refusals and observed error envelopes in the parent evidence remain applicable. Neither status 0 nor an iteration tolerance is a universal accuracy certificate. This receipt closes the focused current-compiler repeat; it does not claim completion of shared full-science integration or hosted PR CI. Native executables remain in remote scratch; Git contains hashes and compact textual evidence.
