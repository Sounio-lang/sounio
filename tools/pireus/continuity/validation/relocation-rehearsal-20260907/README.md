# Real logical restore rehearsal

This checkpoint is not production migration acceptance. The original Spark
database remains authoritative; no source write pause or endpoint switch has
occurred. Private archives, row fingerprints, credentials and catalog details
remain outside Git under the protected relocation cache.

The typed-v1 rehearsal restored five connectable databases and compared all
189 tables plus large-object contents and ownership. It found zero data
mismatches, but took 1010.91 seconds (16 minutes 51 seconds), exceeding the
15-minute maintenance ceiling even before endpoint work. WAL sizing changed
on the isolated target during that diagnostic run; it is not a fixed-configuration
performance baseline.

Real-load findings and implemented corrections:

- Increase target shared-memory volume from 2 GiB to 16 GiB while preserving
  the 48 GiB pod memory limit. The initial restore failed on /dev/shm allocation,
  not PVC capacity. Restore workers are limited to four.
- Initialize the replacement PGDATA with the source bootstrap role and restore
  its attributes/password privately. All 16 roles, three memberships and
  extension owner/version observations matched after this correction.
- PostGIS extension installation changes database search_path during restore.
  Capture source database settings and reapply them after extension creation.
- Capture and restore all 83 sequences, including extension-owned sequences
  omitted from pg_dump's SEQUENCE SET entries. Sequence capture is explicitly
  not MVCC-atomic while the source accepts writes; final capture requires the
  approved write pause.
- Compare schema dumps while normalizing only random psql restrict tokens;
  compare ACL grants in canonical order, not their incidental array ordering.
- Compute a SHA-256 for each full typed record inside PostgreSQL, then hash
  the ordered digest stream. Preserve full row coverage and counts. This
  avoids transferring every large record twice. Normalize PostgreSQL output
  settings and order by discovered primary keys, with a full-record fallback.
- Publish source inventory/fingerprint JSON atomically for concurrent readers.
- Restore destination databases sequentially to avoid shared checkpoint
  contention; source backups still overlap the destination pipeline.
- Increase target max_wal_size to 8 GiB and min_wal_size to 2 GiB.
  fsync, full_page_writes and synchronous_commit were not relaxed.

A real PostgreSQL control confirmed stable exported snapshots after a concurrent
write and detected a changed array lower bound with unchanged row count.
Its first v2 invocation completed those assertions but hit a timeout during
test-database cleanup; that process is not reported as a clean test pass.

The v2 rehearsal includes data, schema, roles, ACLs, database settings, extension
ownership and captured sequence comparisons in its timed pipeline. Application
functional tests and endpoint behavior remain separate gates.

## Storage stop discovered by real load

The earlier volume canary and PG migration had no fresh slow/stalled events
within their observation windows. Real PostgreSQL restores subsequently
triggered thousands of new BlueStore slow-operation events on OSD0.
The remaining warnings can no longer be classified as merely historical
for production cutover. Capacity relief remains valid.

A ten-second disk sample observed approximately 129 ms average write latency
on nvme0n1 versus 1.6 ms on nvme1n1, both on T560, with SMART media errors zero.
This locates an I/O latency problem but does not establish physical failure.
No health warning was muted or its threshold changed.

Blocker-ID: BLK-20260907-pireus-relocation-io
Status: classified
Severity: B1
Class: platform-resource
Owner: codex-pireus
Lane: continuity-20260906
Evidence-Level: E3
Acceptance-Gate: no new slow/stalled events under the complete rehearsal,
full data/metadata/application acceptance and measured procedure within 15 minutes
Next-Action: investigate and correct OSD0 write latency, repeat the full gate
Legacy-Kept: original source database and data directory remain authoritative
LLM-Offload: not-required; operational work only

## Typed-v2 result

The complete data and metadata rehearsal passed: five databases, 189 tables,
83 captured sequences, all schemas, ACLs, roles/memberships, database properties,
settings and extension owners/versions matched. The source remained live, so
its later sequence values are not claimed identical to the captured values.
The restored-database BM25/vector/PostGIS/IVM functional transaction passed and
rolled back in 33.26 seconds.

Core elapsed time was 1270.84 seconds (21 minutes 11 seconds), with target
metadata verification taking 12.15 seconds. This is a timing rejection.
Managed asynchronous discard was enabled on OSD0 during the diagnostic run;
that change did not establish stable I/O acceptance. Raw device discard and
OSD restart were not used. Details are in ../ceph-latency-20260907/.

An additional isolated bulk-loading rehearsal is in progress: prepare empty
databases before the source maintenance clock, use zstd archives, and load
pre-data plus data in one transaction under wal_level=minimal with archive
mode off and zero WAL senders. fsync/full_page_writes/synchronous_commit stay
on. Normal wal_level=replica must be restored, the target restarted, and
functional acceptance repeated before any endpoint switch. This is not yet
a validated production procedure.
