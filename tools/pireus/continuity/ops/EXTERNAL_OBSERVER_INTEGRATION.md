# External observer: container integration, 2026-09-09

The publisher executes the target only after the external observer has bound its
identity and recorded a valid first sample. The supervisor wraps the unchanged
memory guardian; the guardian remains responsible for stopping its child.
The container launcher and lifecycle target are preserved byte-for-byte by the
builder. Source commit: 2c6245294a93bbf16bf33c060aa71372a9fbe555.

## CPU control 11974: partial result, attempt closed

Slurm completed 0:0 on both exclusive Spark nodes, 20:13:01–20:13:18
(raw scheduler timestamps). The pinned Apptainer image was verified on both
workers. This used a CPU-only anonymous 64 MiB allocation, not Inkling inference.
The allocation reserved both nodes and exposed devices through the existing
--nv launcher; it does not qualify CUDA or loaded-model overhead.

Both ranks produced 35 samples. The first sample preceded target execution.
Nonce, entry hash, job/rank, worker UID, boot ID, PID/starttime, namespace and
actual task-cgroup binding checks passed. PID survived exec. Both observers
detected target exit and stopped attribution (TARGET_INVALIDATED, metrics null,
exit 3). Both guardians returned 0, with minimum available host memory
122337464320 and 122270679040 bytes, above their unchanged 33 GiB threshold.

The original memory-control oracle required PSS rise/release >=48 MiB and
cgroup memory.current rise/release >=32 MiB. **It failed on the total-cgroup
rise in both ranks.** The collector's initial assertion failure is preserved as
a failed criterion, not converted into qualification. No retry occurred.

| Median change, baseline to allocation (bytes) | rank 0 | rank 1 |
|---|---:|---:|
| Process PSS | 67108864 | 67108864 |
| Cgroup memory.current | 1130496 | 995328 |
| Cgroup anon | 67256320 | 67215360 |
| Cgroup file | -65826816 | -66097152 |

PSS also fell exactly 64 MiB after release in both ranks. During allocation,
pgscan_direct and pgsteal_direct increased by 16111 pages in rank 0 and 16175
pages in rank 1. These are post-hoc observations: anonymous growth coincided
with file-cache reduction and direct reclaim. The 512 MiB per-node Slurm
request and approximately 510–511 MiB observed task totals make this control's
resource envelope relevant. A specific enforcement limit was not sampled, so
these receipts alone do not prove which limit triggered reclaim.

The post-hoc decomposition explains why total memory.current was not a clean
allocation witness here. It does not change the original pass threshold or
establish the cause of Inkling job 11972's guard stop.

## Evidence and next transition

Raw attempt files, worker identity snapshots, source copies, journals, accounting
and qualification.json are in
../validation/external-observer-integration-20260909/cpu-attempt/.
qualification.json retains original_memory_control_pass=false.
verify_cpu_packet.py pins the packet and recomputes the phase medians and
original pass/fail criteria offline; altered packets are refused.

Eight handoff/builder controls and twelve existing observer controls pass.
The generated runtime manifest remains unqualified; the CPU handoff result is
recorded separately in status.json. No source CI or full runtime freeze is
claimed for this integration.

Next: predeclare a revised CPU control that records memory.max/high/events and
uses process PSS plus anonymous-memory accounting for the allocation witness,
while retaining total and file-cache counters as separate observations. Qualify
that control under an explicit resource envelope before a separate loaded-model
overhead experiment. Do not replay 11974 or resume the frozen pilot.
