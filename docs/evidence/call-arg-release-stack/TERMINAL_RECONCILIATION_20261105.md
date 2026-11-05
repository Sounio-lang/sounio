<!-- docs:meta
topic_id: repo.docs.evidence.call-arg-release-stack.terminal-reconciliation-20261105
authority: repo_only
audience: users
last_validated: 2026-11-05
validated_by: Slurm accounting and raw worker receipts recovered on 2026-11-05
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.call-arg-release-stack.terminal-reconciliation-20261105
-->

# Terminal receipt reconciliation — 2026-11-05

The prior receipt recorded batch 20996 and the ordinary rebuild as running.
The terminal accounting and worker files are now recovered.

- Batch 20996: COMPLETED, exit 0:0, elapsed 01:01:45.
- Ordinary rebuild: BUILD_STATUS=0; SHA-256
  a08c5660182d10d9d91ff6cf3551e70362437046d480b05a260dca20494c561f.
  It is byte-identical to the mapped compiler used by the targeted gate.
- Targeted stack gate 20996.4: COMPLETED 0:0.
- Full stdlib step 20996.6: CANCELLED, 0:15, elapsed 00:32:28.
  No whole-stdlib completion or success is established.
- The auxiliary STDLIB_FULL_STATUS file contains 0. That value conflicts with
  cancellation and is not an acceptance receipt. The script's EXIT-only status
  capture could retain a zero status during termination.
- The full suite directory has before-manifests and retained partial outputs,
  but no completed suite summary was recovered.

The earlier full-suite launch as an overlapping step in the compiler-build
allocation was inadequate: finishing the parent batch ended the full-suite
step. Future full acceptance must own a suitably bounded allocation, and its
status writer must map TERM/INT to nonzero status. The new baseline batch uses
explicit signal traps and awaits its build and gate in the foreground.

Raw evidence: results/slurm-final-20996.tsv and results/final-20996/.
The original partial stdlib matrix and refusals remain preserved. They must
not be relabelled as completed acceptance merely because the parent batch
returned 0.

## Integration work now

Current inspected main: a232a6b1d5b7dc8205aabe9064887700d7033ec2.
Its release helper still recurses and already has three arguments.

main-a232-integration-unapplied.patch is a source proposal against that exact
parent. It keeps the three-argument signature and changes only traversal. It
has NOT been applied, built, or accepted. check.sio is held by the active
devpass--stdlib-compiler36-integration lease. A handoff request was sent on
the coordination bus; no claim was removed or overridden.

An independent clean-main baseline was staged from the committed tree and
submitted as Slurm 22425, using main-baseline-20261105.sbatch. BUILD_STATUS=0;
the clean-main compiler SHA-256 is
adb912b2f17d80bf444ea251a541a20db450ac71acf9016a724f897e69a80207.
The overall diagnostic job ended FAILED 139:0: declaration/call/check/build/run
passed for 24/25/26/32/48, declaration64 passed, and call64.check exited 139.

Independent control job 22426 records the same call64 check as 139 at effective
soft/hard 8192 KiB and 0 at soft/hard 16384 KiB. This is a diagnostic control,
not acceptance at a larger stack. The current-main failure is preserved before
any loop change. Only an applied and qualified exact integration SHA can close
that gate.

Main also still uses inline BorrowEnv arrays, while the historical sprint
snapshot used the paged representation. The proposed diff does not edit
borrow.sio or change the target representation. The historical paged-snapshot
acceptance must not be transferred to this clean-main parent.

The source proposal passed patch --dry-run against the exact extracted parent.
No Sounio source was changed. The owner has been requested to hand off
check.sio; until that happens the rebase/application is blocked by the active
lease, not by a missing user approval.

PR #3405 remains the canonical review target:
https://github.com/Sounio-lang/sounio/pull/3405 .
Its existing head/base and red CI are not reclassified by recovery of these
older specialized receipts. After handoff, reconcile on that same branch,
qualify the resulting exact SHA, and retain human control of merge into main.

The clean-main binary, private build cache, diagnostic inputs and outputs are
archived separately under artifacts/SIO586-main-baseline-22425/. The archive
has its own SHA-256 receipt. The published evidence directory contains the
small accounting/manifest/matrix extracts; raw artifacts remain in that archive.
