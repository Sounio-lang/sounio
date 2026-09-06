Blocker-ID: BLK-20260906-pireus-host-memory-observation
Status: owned
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
Next-Action: implement and verify a versioned recovery observation/migration contract for stale Slurm memory, preserving fresh host memory >=32768 MiB and the device barrier

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
