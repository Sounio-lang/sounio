# Lifecycle diagnostic11971/11972 — terminal review

The declared two-arm diagnostic attempt is closed. The first arm completed;
the second stopped at the unchanged memory guard. Paired inference and full
diagnostic coverage did not complete. Root cause and instrumentation overhead
remain unqualified. No retry or memory optimization was applied.

Source:0cbf8267d42c4e8d2ea27ce4f18734259cb6b042, CI34334985264 SUCCESS.
Execution freeze SHA256:
6510a1fc1e9fb35131efedc6424ccedcb0dba2ecac38b3a3c178ebc063ba50e2.
The same16 frozen inputs and101-file snapshot were verified before launch.
The only runtime replacement was the separately generated observer-bearing
offline_generate.py. Both launches had fresh host preflight and exclusive
TP2 Slurm ownership. Context16384, physical cache6144, SWA896, output4096,
guard33GiB and protected floor32GiB stayed unchanged.

## Terminal evidence

| Arm | Job | Slurm | Complete proposals | Output tokens | Journal coverage |
|---|---|---|---|---|---|
| without feedback |11971|COMPLETED0:0|8/8|943|both start/end; eight full cycles|
| with feedback |11972|FAILED75:0|1/8|121|both start; one full cycle; partial request1; no end|

11971 ran2026-09-09T10:51:49–11:12:21; extern completed11:12:28.
11972 ran2026-09-09T11:32:51–11:42:22; extern completed11:42:31.
Timestamps are the raw scheduler timestamps. Job identity, paired node set,
worker UID before/after collection, raw log custody and response hashes are
preserved in validation/lifecycle-diagnostic-inference-20260909.

11971 token validation passed against the pinned diagnostic snapshot, with
byte-identical responses across ranks. Its guardians exited0 with minima
35,854,704,640 and35,596,685,312 bytes. All eight per-request lifecycle sequences
were checked. These qualify this bounded diagnostic arm, not general serving.

11972 reached MODEL_READY and saved request0 on both ranks; the saved121-token
response bytes and log hashes agree. Rank0 then stopped during request1:
available35,433,025,536 bytes, exactly454,656 bytes (444KiB) below the33GiB
guard, still1,073,287,168 bytes above the protected32GiB floor.
There is one MEMORY_GUARD_STOP receipt, not two successful guardian exits.
Both completion receipts and requests1..7 are absent. The collector records
all16 missing files; it does not synthesize empty receipts.

## What the observations establish

At request0 in both jobs and ranks, CLEANUP_BEFORE → CLEANUP_AFTER changes
CUDA allocated by0bytes. CLEANUP_AFTER → REFERENCES_RELEASED reduces it by
807,936bytes. CUDA reserved remains unchanged across those cleanup boundaries.
This distinguishes cleanup from release of remaining Python references. It
does not prove that cleanup is defective or that empty_cache would help.

In11971, post-release CUDA reserved is constant across all eight requests at
81,897,979,904bytes; allocated is81,697,782,784bytes except request5, which is
1,024bytes lower. Persistent CUDA allocation growth at these boundaries is
not observed. Eight finite requests do not establish indefinite stability.

At the matched first release,11972 has50,331,648bytes (48MiB) more CUDA reserved
than11971 on both ranks, but only5,632bytes more allocated. This differs from
the72MiB difference observed in the previous uninstrumented pair11969/11970;
those separate profiles/runs must not be combined as one fixed overhead.

From request0 DECODE_ENTRY to CLEANUP_BEFORE:

| Job / rank | Host MemAvailable decline, bytes | Process PSS growth, bytes | CUDA reserved growth, bytes |
|---|---:|---:|---:|
|11971 /0|22,872,064|44,068,864|2,097,152|
|11971 /1|457,428,992|43,978,752|2,097,152|
|11972 /0|1,197,154,304|70,496,256|2,097,152|
|11972 /1|992,169,984|70,311,936|2,097,152|

The views diverge; they are not additive on shared-memory Spark. Do not
subtract these numbers to assign an unexplained residual to a particular
process, page cache, CUDA driver or leak. The rank causing the guard stop
also changed from rank1 in11970 to rank0 in11972.

The1Hz observer did not capture the guardian's exact threshold crossing.
11972 rank0's lowest journal MemAvailable is35,492,184,064bytes, above33GiB,
while the50ms guardian captured the lower stop sample. The final rank0 host
sample occurs5.216s after its last process/CUDA hook; rank1's interval is6.546s.
There are no further decode-sample hooks, cleanup events or OBSERVER_END for
request1. Missing synchronized process/device evidence at the stop is still
missing. Monotonic timestamps are compared only within a rank.

Maximum host sample gaps were1.145/1.208s in11971 and1.207/1.061s in11972.
No host/process/owned-child read errors were recorded in the captured rows.
Cgroup/device accounting was declared unavailable, not measured as zero.
CUDA peak counters are lifetime values and cannot be labeled decode peaks.
Hook durations exclude scheduling/GIL and other observer effects; a diagnostic
pass does not measure or qualify instrumentation overhead.

## Reproduction and next decision

Run:
python3 tools/pireus/continuity/ops/review_lifecycle_diagnostic.py --output NEW_PATH

The reviewer pins both collection manifests and verifies every collected
artifact before projection. review.json contains the exact hook rows,
counter deltas, partial coverage and claim boundaries. Three controls pass:
actual review reproduction; modified journal refusal; modified collection
identity refusal. No model or GPU replay is involved in these controls.

The next investigation should identify the host/device-accounting view that
moves during the drop, and determine whether an independent bounded sampler
can capture process and device/cgroup information through the final decode
interval. Inventory supported counters and validate their meaning and sampling
cost first. Keep expensive observation outside the guardian's50ms decision
path. This is a prerequisite investigation, not authorization to change the
frozen profile or launch another GPU attempt automatically.

No cache reduction, allocator flushing, prompt/output reduction or guard
relaxation is justified by this packet alone. Any selected intervention needs
a new execution identity and separate qualification.

Original pilot remains1/9cells,32/288,zero gain-qualified;11956 is immutable.
M5/M6 remain NOT_STARTED; V13/V14 remain OPEN. No HTTP/general16K serving,
GDR, native gain or novelty acceptance is promoted.
