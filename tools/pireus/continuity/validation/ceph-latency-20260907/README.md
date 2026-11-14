# OSD0 real-load latency investigation

The real PostgreSQL restores reproduced fresh BlueStore slow-operation events.
This supersedes the earlier historical-only interpretation for production
cutover, while preserving the earlier capacity and volume-canary results.

Both T560 NVMe drives report SMART passed and no media errors. Device-level
samples nevertheless show OSD0 write latency reaching seconds while OSD1
typically remains near milliseconds. This observation does not by itself
identify a physical, firmware or flash garbage-collection fault.

The SSD and its LVM device advertise discard support, but the kernel showed
zero completed discards and BlueStore had discard disabled. The installed
Ceph19.2.3 executable and pinned upstream source both report runtime support
for bdev_enable_discard and bdev_async_discard_threads.

At 2026-09-07T04:51:57Z, only OSD0 received:
- bdev_async_discard_threads=1
- bdev_enable_discard=true

There were no prior explicit OSD0 overrides for those settings. The guarded
script retained the previous values and removes its overrides on application
failure. To return to the prior configuration after this completed application,
remove these two OSD0 overrides; review current config first to preserve any
later operator changes.

The kernel subsequently reported completed discards, confirming that Ceph
issued requests for allocator-released extents. This is not raw-device discard
or an offline trim of the whole free-space map. No OSD restart, warning mute,
warning-threshold change or direct block-device write was used.

**The experiment has not established stable latency.** Fresh slow events
continued after activation in the retained profile, although no new stalled-read
event was observed. The setting remains enabled during the explicitly isolated
bulk-loading rehearsal. It must not be presented as hardware repair or
production storage acceptance.

Profile samples span changing workloads and are diagnostic observations, not
a controlled performance comparison. The timed-v2 restore also crossed the
configuration-change boundary and is retained as a timing rejection.

Evidence: profile.jsonl, profile.py, discard-change.json and the guarded
enable-managed-discard.py. Additional live rehearsal profiling remains on
T560 under /var/tmp/pireus-ceph-latency-20260907 until archived.

Primary implementation reference:
https://github.com/ceph/ceph/blob/v19.2.3/src/common/options/global.yaml.in

Blocker-ID: BLK-20260907-pireus-relocation-io
Status: classified
Severity: B1
Class: platform-resource
Owner: codex-pireus
Evidence-Level: E3
Acceptance-Gate: complete restore/application/endpoint procedure within the
authorized pause and no fresh slow/stalled I/O events under that workload
Next-Action: finish bounded bulk-loading experiment and reassess I/O; keep
source database authoritative until every production gate passes
