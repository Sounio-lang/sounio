# Ceph capacity relief and isolated target qualification

User authority: explicit RESOLVA following the authorized PostgreSQL relocation.
The source database remained authoritative and serving throughout this operation.

## Applied correction

The replicated_ssd CRUSH rule uses distinct hosts and the rbd_ssd pool retains
size3/min_size2. OSD11 on R770 was89.08% full, while OSD23 on the same host
had about939GiB free. Three approximately126–128GB PGs dominated OSD11's
data. The upmap balancer was active with no further optimization found.

A single replica of PG5.b was moved from11 to23:
`ceph osd pg-upmap-items 5.b 11 23`.
There was no previous upmap entry for this PG. The desired mapping changed
from [0,11,20] to [0,23,20], preserving SSD class and three separate hosts.
No data was deleted by an operator; Ceph retired the superseded replica after
recovery. No full/nearfull thresholds, replication factors or OSD weights changed.

The operation ran in host tmux. The balancer was temporarily disabled to avoid
concurrent mapping changes and restored to its prior enabled state in finally.
It completed at2026-09-07T03:23:51Z with up=acting=[0,23,20], active+clean.
There was a transient peering phase during final mapping convergence.
The old replica's space was reclaimed asynchronously afterwards.

Final captured utilization: OSD11 **59.36%**, OSD23 **54.61%**.
OSD_NEARFULL and all14 POOL_NEARFULL alerts cleared.
The balancer is again active in upmap mode. Other historical PG remaps are
not claimed resolved by this single-PG intervention.

An undo would remove the5.b upmap exception, but would put the large replica
back on the previously nearfull OSD. Do not perform that undo without a fresh
capacity/availability assessment.

## I/O investigation and honest health boundary

Both T560 NVMe devices report SMART passed, no critical warning and zero
media/data-integrity errors. No NVMe timeout/reset/I/O error matched the
kernel-log inspection. These observations do not establish lifetime hardware
reliability or prove that an intermittent fault cannot recur.

The remaining Ceph summary is HEALTH_WARN:
BLUESTORE_SLOW_OP_ALERT and DB_DEVICE_STALLED_READ_ALERT. Both warning
lifetimes are86400 seconds. Latest observed events:
- OSD0 slow operation:2026-09-06T21:59:25.907-0300.
- OSD1 slow operation:2026-09-06T17:53:31.433-0300.
- OSD0 stalled reads:2026-09-06T11:59:46.242/243-0300.

No new corresponding events appeared across the recovery and volume probe.
The first volume-probe attempt refused because log rotation changed raw
current-file counts. Reading current logs plus the rotated gzip restored the
exact baseline totals (OSD0:5942 slow/2 stalled; OSD1:6 slow/0 stalled).
That refusal is retained; it was not mistaken for fresh hardware failure.
No alert was muted, warning lifetime shortened or daemon restarted.

The original inspect_relocation_storage.py is a conservative preliminary
health-flag screen and still reports STOP for retained I/O alerts. The additional
live log/SMART, loaded-recovery and volume-probe evidence classifies those
observed alerts as historical for this isolated rehearsal; it does not turn
HEALTH_WARN into HEALTH_OK or establish production cutover acceptance.

## Actual volume qualification

The approved64GiB PVC pireus-pg-relocation-data bound through
ceph-rbd-ssd-checkpoints with PV reclaim policy Retain. On R770:
1. A writer pod wrote1GiB with fsync, verified its SHA256 and ran pg_test_fsync.
2. A second pod mounted the volume read-only, verified the SHA256 and exact
   file size, and emitted FRESH_POD_INTEGRITY_PASS.
3. The postcondition checked unchanged I/O event totals and SMART again.

Both pods succeeded. Approximate write/fsync/close measurements were17–19ms.
This proves observed PVC provisioning, normal remount/read integrity and
filesystem synchronization behavior. It is not a power-loss test or a
guarantee of application-level latency or restore duration.
Volume UIDs, image IDs, logs and pre/postconditions are in the sibling
relocation-storage-canary-20260907 directory.

The isolated PostgreSQL target is now running on R770, cron disabled, using
the pinned AMD64 image and private bootstrap credentials outside Git.
Real per-database snapshot backup and restore rehearsal remain distinct gates.
The source address/port and protected host services were not switched.

## Blocker checkpoint

Blocker-ID: BLK-20260907-pireus-relocation-storage
Status: review-ready
Severity: B1
Class: platform-resource
Owner: codex-pireus
Lane: continuity-20260906
Evidence-Level: E3
Acceptance: capacity relief and isolated PVC I/O qualification passed;
production cutover remains gated by the real restore/data/application rehearsal.
Next-Action: finish the real snapshot restore and measure the complete paused
procedure against the approved15-minute maximum. Preserve the historical
health alerts and recheck for fresh events before any cutover.

## Sources

- https://docs.ceph.com/en/squid/rados/operations/upmap/
- https://docs.ceph.com/en/squid/rados/operations/health-checks/
- https://www.postgresql.org/docs/16/pgtestfsync.html
