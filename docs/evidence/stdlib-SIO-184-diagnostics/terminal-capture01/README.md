<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-sio-184-diagnostics.terminal-capture01.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-sio-184-diagnostics.terminal-capture01.readme
-->

# Terminal integrity of completed diagnostic captures

> **Status**: Beta | **Last validated**: unvalidated | **Source**: fake-compiler operational controls

Parent239682aa004a386bd44e22ce96eaad063bb99f03 retains the original20-file custody and26-control source catalog in Git history. The top-level SOURCE_SHA256 catalog now binds all current operational sources; this directory also binds the helper/test terminal-integrity delta. Original26 SELFTEST receipts remain labelled historical, alongside this current28-control receipt. Wrapper and runner command branches are unchanged.

Before complete.json is issued, the helper now traverses every started check/run phase, requires exactly regular start/result/raw files, checks matching case/phase/command/source identities and status, and hashes every completed raw output again. Empty selection explicitly completes with captured_phases0. An unfinished or changed earlier capture refuses with65 and no fresh suite JSON.

Actual28 fake-compiler controls pass. The added two-selected-case control lets the first check complete and pass its immediate verification, then makes the later check mutate the first raw log. Both commands return0 and create completed phase receipts, but finalization detects the changed hash and returns65; neither complete.json nor fresh schema-v1 results is emitted. LATE_MUTATION_ACTUAL.json preserves actual status/stdout/stderr, both completed receipts and changed raw bytes in base64. This is output-receipt integrity, not a security or universal process-trace claim. All prior default/capture rows/stdout/status equivalence, raw-byte, path, source and metadata-failure controls still pass. No real compiler/native science/Python or new job runs in this regression.
