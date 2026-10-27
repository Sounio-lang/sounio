Blocker-ID: BLK-20260906-pireus-host-memory-observation
Status: owned; stale-memory predicate corrected, full recovery still blocked
Severity: B1
Class: platform-resource
Owner: codex-pireus
Lane: continuity-20260906 / M2-M4 live acceptance
Worktree: /workspace/.wt/pireus-integration-20260906
Branch: codex/pireus-inkling-cycle-20260906
Files-Owned: tools/pireus/continuity/**; registered continuity/recovery concept docs and governance entries
Files-Read-Only: /workspace/.wt/pireus-spark-pair-arbiter frozen controller, policy, backend, native executable
Do-Not-Touch: original dirty worktree, unrelated lanes, protected host services, device barrier, memory floor
Repro: SOUNIO_SPARK_PAIR_HOLDER=codex-pireus-continuity-20260906 /workspace/.wt/pireus-spark-pair-arbiter/scripts/dev/spark_pair_arbiter.sh recover
Observed: REFUSE action RECOVERY_DETACH_SLURMD reason HOST_MEMORY_FLOOR; frozen backend combines fresh host memory with stale Slurm FreeMem=815 MiB on Spark 8e54
Expected: recover from already-fenced, drained, job-free hosts using current verified observations and restore Slurm ownership without lowering the host floor
Acceptance-Gate: canonical recovery PASS; lease slurm-owned; both host grants valid SLURM; both workers restored and Slurm IDLE; fresh two-node material canary PASS
Evidence-Level: E4
Evidence: validation/pireus-fence-recovery-3.log; validation/pireus-fence-recovery-4.log; validation/pireus-current-host-8.txt; validation/pireus-current-slurm-8.txt; validation/recovery-detach-live-1/
Fallback-Path: none; no model replacement, threshold reduction or GPU grant bypass
Legacy-Kept: yes
LLM-Offload: logged:reviews/recovery-initial-* and reviews/recovery-followup-*
Next-Action: isolate the host watchdog/commit interleaving that may invalidate Slurm activation; preserve the reviewed observer revision, 32768 MiB floor and device barrier

The initial cause of the host fence during Inkling weight loading remains an
investigation item. It is not proven to be host OOM or the memory predicate.
All checkpoint hashes passed on both nodes. No Inkling inference completed.
The third/fourth canonical recovery attempts reproducibly isolate the current
memory observation barrier; this is distinct from the initial fence trigger.

Bounded recovery detachment stopped the worker recreation race. Both exact
node selectors were removed after native Sounio ALLOW with live owned lease,
zero jobs/allocations, fresh memory above the unchanged floor, and fenced
hosts. The adapter preserved per-node CAS effects and postconditions.
No pair-wide atomicity, restored Slurm ownership or new GPU execution is claimed.
The unchanged predecessor then refused HOST_MEMORY_FLOOR again.

Slurm documents FreeMem as the OS-reported free memory field:
https://slurm.schedmd.com/scontrol.html
It is distinct from the fresh host MemAvailable observation captured here.
The current frozen bootstrap migration also requires UNINITIALIZED; it cannot
be repurposed to rebind this RECOVERY_REQUIRED lease without a new contract.


2026-09-06 observer revision update:
- Frozen backend revision 2daf6df41434cd9b9d9a723b7675c9d2d2eda46850d2e66d0a67817f4b8e828d
  uses both host memory bits. The fresh watchdog predicate remains required.
- Native observer rebind ALLOW and journal/lease CAS postconditions passed:
  validation/recovery-observer-migration-live-1. Source/executable/host policy,
  minimum memory and barrier authority were unchanged.
- New freeze 452289a8a9c201271f6631973c65cbc428cbe3a7f7228e2483e9768a875fb46b
  passed verification and canonical recovery reached GRANT_HOST_SLURM at epoch 11.
- Activation verification failed; canonical transaction proved both hosts
  FENCED again. See validation/recovery-observer-live-1/recover.log.
- A deterministic mocked interleaving reproduces stale FENCED enforcement
  after a concurrent SLURM commit. This is a source-level possible race,
  not proof of the live incident cause. See runtime/reproduce_watchdog_interleaving.sh
  and validation/watchdog-interleaving-repro.log.

The controlled second recovery also refused at epoch 12, exit 42.
Both grant files were observed in SLURM before returning to FENCED within
seconds. Raw observations, failed initial observer transport (missing Python
in the observer image / pod replacement), receipts and transition summary
are retained in validation/recovery-observer-live-2. This supports the
concurrency investigation; it does not identify the exact failing predicate.

Recovery update2026-09-07: content-address correction, exact observer migration and epoch15 canonical recovery passed; Slurm ownership restored. Fresh serving-capacity blocker is BLOCKER_TP2_CAPACITY.md. Historical failures above remain evidence.
