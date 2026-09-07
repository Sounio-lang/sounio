# Proposed Beagle memory database relocation to unblock Inkling TP2

Status: REVIEWABLE PROPOSAL; not executed; authorization required for
protected-service migration. Read-only discovery completed2026-09-07.
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

## Authorized execution would follow these gates

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

No migrations, credentials, data copies, target workloads or source changes
have been performed. Approval must explicitly cover this Beagle service
migration and its bounded write pause; ordinary Pireus recovery authority did
not authorize changing a protected application.

## Sources

PostgreSQL16 SQL dump documentation explains cross-architecture logical
transfer, per-database snapshot consistency and separate global objects:
https://www.postgresql.org/docs/16/backup-dump.html
Logical replication restrictions for schemas, sequences and large objects:
https://www.postgresql.org/docs/16/logical-replication-restrictions.html
