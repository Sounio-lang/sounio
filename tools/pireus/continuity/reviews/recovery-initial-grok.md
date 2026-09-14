BLOCKER

Do not apply. Native allow/refuse is narrower than the live effect, and the transport can change pair state without the conditions you listed still holding.

1. Effect can evict the fence. The only mutation is deleting `slurmd_selector_key` on both nodes. Nothing proves host-fence, watchdog, device-barrier, or protected services use a different selector/nodeset. If they share that label, DETACH_FENCED_WORKERS disables the controls the decision requires to stay intact. Before any live run, pin and record the fence/workload selectors and refuse if they intersect.

2. Two-node apply is not atomic and has no rollback. Node 1 can lose the label, node 2 can fail (stale RV, lease RV, expiry ≤15s). The pair is then split, evidence can show a crash, and nothing restores the first label. Bound this to both-or-neither (pre-refresh both RVs, abort if either patch fails, write PARTIAL, do not continue to canonical recover).

3. Decision-time facts are not effect-time facts. Sounio only sees a one-shot vector. Apply rechecks lease RV, lease remaining >15s, and CAS on node uid/RV/label. It does not re-read RECOVERY_REQUIRED, holder, epoch, FENCED, grant_valid=0, watchdog/barrier/protected, slurm_mask, consumers, or MemAvailable. A fence flip, new alloc, or watchdog lapse in that window still patches. Re-observe both hosts immediately before the CAS loop and abort on drift.

4. Vacuous authority slot. `checks[2]` is hardcoded `True` (`values[3]`). Sounio still demands it equal 1, so it is not a control. That is either a dropped check (UIDs/names/cgroup-empty/predecessor) or a silent bypass of the cgroup-empty proof this extension exists to skip. Put the skip in the native contract as an explicit named fact, or remove the slot. Do not smuggle a tautology.

5. Slurm mask is not closed. Sounio only requires `(mask & 30) == 30` and `mask ≤ 511`; bits 0,5–8 never refuse. If any ignored bit means jobs, allocs, not-drained, or error, ALLOW is wrong. Encode drained/zerojobs/zeroallocs as separate booleans, or require `mask ==` the exact safe value.

6. Transport does not enforce the printed safety contract. After ALLOW it does not assert `gpu_grant==false`, `resume==false`, `claim_ready==false`, memory floor 32768, or return code 0 matching the JSON. Post-apply it never read-backs both nodes, lease, and fence reports. Require those asserts and a postcondition bundle: selector gone on both nodes, lease RV/annotations unchanged, both hosts still FENCED, watchdog/barrier/protected still 1.

Also missing from this review bundle: `recovery-detach-lock.json` (must hash this `.py` and `.sio`), bit definitions for `slurm_mask`, and proof `cfg["slurmd_selector_key"]` is what the recreating init/sidecars actually match. Canonical recover remaining mandatory does not fix a partial or fence-evicting detach.

No execution claimed.
