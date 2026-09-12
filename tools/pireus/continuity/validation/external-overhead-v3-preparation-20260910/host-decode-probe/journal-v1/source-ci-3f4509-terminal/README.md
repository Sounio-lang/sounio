# Current-source CI terminal evidence: 3f4509b51d

Run 34480508853, attempt 1, job 102882321904 failed at 2026-09-10T14:15:10Z.
The log reports runner shutdown and exit 143. CI Decision 102906513469 failed.
Contracts passed. This source is not qualified for the journal diagnostic.

The observed gen2 log reached Merged IR: 13269 functions. Its last complete
resource sample reported host MemAvailable 53240 KiB and SwapFree 4 KiB.
All sampled host oom_kill values were zero; cgroup memory.events was unavailable.
These observations do not establish an OOM termination or the shutdown cause.
No explicit 75-minute deadline annotation was emitted for this attempt.
The final meminfo sample is partial and is retained unmodified in job.log.

The current-source artifact upload was skipped. There is no collected gen2
executable or terminal compiler success receipt from this run. Existing artifacts
belong to other jobs and cannot fill that evidence gap.

The separate DL380 job 11991 completed the frozen compiler workload under its
declared CPU/memory/time envelope. It does not qualify this CI run or establish a
same-host causal comparison. A declared CI resource-envelope qualification is the
next engineering task; no automatic CI retry or Inkling submission was made.

Raw API responses and the complete available job log are preserved here.
manifest.json records SHA-256 and byte size for every other file in this packet.
