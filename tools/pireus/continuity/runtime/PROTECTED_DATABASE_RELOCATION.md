# Proposed Beagle memory database relocation to unblock Inkling TP2

Status: AUTHORIZED; execution precheck blocked on target storage health.
User authorization: AUTORIZADO. Read-only discovery completed2026-09-07.
This is a capacity proposal, not proof that the full Inkling load fits.

## Observed source and destination

Source: Docker container beagle-memory-pg-pdb on spark-3c59, published5433.
Data bind: /home/demetrios/beagle-memory-pg-pdb/data.
PostgreSQL16.14 ARM64, about6,981,099,626 bytes across databases.
Image index paradedb/paradedb@sha256:556edd8c7500d5ab1bc9c9be1ae97a582cbceb261f8d551386715eb755ab3dcf.
Resolved linux/amd64 child:
sha256:ea4e8267016c929924a15df03eaf2ad9bb2cda9d194169e03eca232303898cc7.
Resolved linux/arm64 child:
sha256:e0fa7d3a1686e15bdd224f378c0f2c49c8d625def6b5d57b98ccc72cfdd50ad7.
The common immutable index is verified. Target executable extension
compatibility is NOT yet verified and must pass before cutover.

Installed extensions in the inspected database:
pg_search0.24.2, pg_ivm1.13, vector0.8.2, postgis3.6.4,
postgis_topology3.6.4, fuzzystrmatch1.2, postgis_tiger_geocoder3.6.4,
pg_stat_statements1.10, plpgsql1.0.
Preloads: pg_search,pg_cron,pg_stat_statements.
shared_buffers12GiB; work_mem128MiB; maintenance_work_mem2GiB.
wal_level=replica; archive_mode=off;10 senders/slots.

Candidate: r770-proxmox. Fresh /proc/meminfo: MemAvailable78,946,132kB.
Local /scratch free64,553,119,744 bytes is staging capacity only.
Proposed durable database volume:64GiB RBD using an existing Retain class;
provisioning, pool health and actual capacity must be proven before rehearsal.
Target workload proposal: namespace beagle, pin r770-proxmox, no GPU,
CPU request2/limit8, memory request24GiB/limit48GiB. Recheck real capacity
and existing reservations before scheduling. Preserve source DB settings
initially; stress/rehearsal decides acceptance, not this resource estimate.

## Authorized execution gates

1. Create isolated target with immutable amd64 image, durable volume and
   private credentials. Verify PostgreSQL and every extension version in
   every source database. No source endpoint/client or service change.
   Target cron disabled during rehearsal to prevent duplicate jobs.
2. Inventory all databases, globals/roles/ACLs, sequences, large objects,
   extension-owned objects, scheduled jobs and client paths. Keep credentials,
   dumps and data out of Git, logs and review prompts; private artifacts0600.
   Preserve original data directory and its device/inode.
3. Create a logical backup and restore rehearsal. Use pg_dump custom-format
   per database plus protected globals export; pg_restore --exit-on-error.
   ARM-to-x86 physical data-directory copying is not the migration path.
   Rebuild extension indexes as required. Compare schema/ACLs, sequence state,
   counts and content digests from matched snapshots, large objects, and
   application-level vector/BM25/PostGIS/IVM behavior. Record timings.
4. Cutover is permitted only after full rehearsal success and a bounded
   maintenance procedure whose measured duration fits a15-minute maximum.
   If rehearsal cannot support that maximum, STOP before service disruption
   and report the revised measured requirement.
5. During the approved write pause, drain client transactions and disable
   scheduled writers. Isolate the source database from clients, take the
   final logical snapshot and restore it. A live rehearsal snapshot cannot
   be promoted while later source writes remain uncopied. Do not use logical
   replication as a shortcut: current wal_level is replica, and DDL,
   sequences, large objects and extension behavior need explicit handling.
6. Preserve the existing Spark address/port5433 through a lightweight TCP
   proxy to the new private target; preserve authentication and TLS behavior.
   Bind/reachability/health and existing-client discovery must be validated.
   Stop the source DB only as part of approved cutover; do not restart
   docker/containerd/kubelet/vxlan or remove/replace the original data path.
   Restore jobs and client writes only after database/application acceptance.
7. Recheck canonical host protected-resource evidence. Frozen protection
   covers service identities and original PostgreSQL data directory inode;
   it must continue to pass, without resetting its baseline.
   Run fresh exclusive TP2 memory qualification, measure weight-loading
   peak/cache budget and only then serving. Reserve32GiB on both Sparks.
   Migration itself is not Inkling acceptance.

## Rollback and stop conditions

Before target writes: source data remains intact; withdraw proxy, start the
original container and restore original endpoint/writers. No divergent writes
are allowed on both copies.
After target writes: never point clients at a stale source. Pause writers
and reverse-export/restore the new authoritative data before returning.
A failure must preserve both data copies and custody, not discard target writes.

No data copies, target workloads or source changes have been performed.
The explicit user approval now covers this Beagle service migration and its
bounded write pause, subject to the gates above.

## Sources

PostgreSQL16 SQL dump documentation explains cross-architecture logical
transfer, per-database snapshot consistency and separate global objects:
https://www.postgresql.org/docs/16/backup-dump.html
Logical replication restrictions for schemas, sequences and large objects:
https://www.postgresql.org/docs/16/logical-replication-restrictions.html

## Authorized execution update — 2026-09-07

The user explicitly replied AUTORIZADO to this proposal. Authorization covers
the gated migration and at most 15 minutes of write pause; it remains valid.
The current stop is storage readiness, not missing user approval.

The read-only precheck found the actual target pool rbd_ssd flagged nearfull,
with percent_used 0.8972344994544983 and max_avail 477850009600 bytes.
Nominal free bytes exceed the requested64GiB, but do not resolve the pool
nearfull condition or the concurrent BlueStore slow operations (osd.0/osd.1)
and BlueFS DB stalled reads (osd.0). Their exact relationship to a future
volume's PG placement is not established. No storage health PASS is claimed.
The CSI identity can read health/df but osd df returned EACCES; no wider
credentials or permissions were installed.

All available approved RBD Retain classes use this target pool. The local
R770 runtime class had no existing PV; /scratch remains staging only.
No PVC, target database, logical backup, endpoint change, source stop or
write pause was performed. No Ceph rebalancing, deletion, threshold changes,
or daemon restarts were performed.

Source catalog inventory:6 databases,5 connectable,16 roles,3 memberships,
83 sequences,0 large objects and0 catalogued cron jobs across connectable
databases. This is a sequential catalog observation, NOT a matched-snapshot
backup or proof of application consistency. External scheduled writers and
client paths remain to be inventoried. pg_cron1.6 is installed in another
database and must be included in the eventual compatibility rehearsal.
Private details are under
/workspace/.cache/pireus-continuity/protected-db-relocation-20260907/
with directory0700 and source-catalog-private.json0600; no role passwords or
application rows were queried. The committed summary contains aggregate
counts and extension versions only.

Reproduce the storage observation from this worktree:
`python3 tools/pireus/continuity/runtime/inspect_relocation_storage.py`
It returns2 for an observed storage stop; a zero result only means additional
provisioning and I/O validation may proceed, never full migration acceptance.

```text
Blocker-ID: BLK-20260907-pireus-relocation-storage
Status: classified
Severity: B1
Class: platform-resource
Owner: codex-pireus
Lane: continuity-20260906
Worktree: /workspace/.wt/pireus-integration-20260906
Branch: codex/pireus-inkling-cycle-20260906
Files-Owned: tools/pireus/continuity/**
Do-Not-Touch: source database data; protected host daemons; Ceph placement/thresholds
Repro: python3 tools/pireus/continuity/runtime/inspect_relocation_storage.py
Observed: STOP; target_pool_nearfull; BlueStore slow ops; BlueFS stalled reads
Expected: healthy durable target storage followed by provisioning and I/O acceptance
Acceptance-Gate: approved plan storage prerequisite, then isolated restore rehearsal
Evidence-Level: E3
Evidence: relocation-storage-20260907.json; relocation-inventory-summary-20260907.json
Fallback-Path: none
Legacy-Kept: yes; original source database remains authoritative
LLM-Offload: not-required; operational observations, no semantic or mathematical changes
Next-Action: resolve target pool capacity/I/O alerts in the storage operations scope, then rerun precheck before creating the isolated target
```

After storage acceptance, resume gates1–7 above. Approval is already present;
do not repeat the migration approval request. Inkling serving and the eight
actual LLM proposals remain pending, as does fresh post-migration memory proof.
