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

## Bulk and endpoint controls

V3 core restore/metadata took 859.28 seconds and normal-runtime restoration,
ANALYZE and functional controls took another 29.76 seconds. It lacked a
measured complete endpoint/controller margin and crossed the OSD0 replica
relief; it is not a stable post-repair acceptance run.

V4 stopped during pg_dump on a Kubernetes WebSocket stream i/o timeout.
The source stayed online. Its partial archive is retained privately.
The rehearsal now detects an exited backup producer immediately instead
of waiting for a missing artifact until the global deadline. Both the
failed-producer negative control and completed-artifact control passed.

V5 explicitly selected KUBECTL_REMOTE_COMMAND_WEBSOCKETS=false. All189 table
digests, 83 captured sequence states and schema/ACL/role metadata passed.
Core time was866.22 seconds; normal restart/ANALYZE/functions added42.51
seconds and network controls7.08 seconds:915.80 seconds total, excluding
source pause/controller overhead. This exceeds900 seconds and is rejected.

The forwarding controls passed authenticated queries through the Spark
probe port15433, a Kubernetes client path, IPv4/IPv6 and unchanged SSL
negotiation. Cilium remote-node ingress is paired with PostgreSQL HBA
restriction to10.100.100.59/32; other TCP sources are rejected by HBA.
Both a missing-network-policy negative and explicit HBA rejection passed.
The temporary Cilium policy was removed and the owned probe unit stopped.
Production5433 and the source database were never switched.

The new target volume excludes OSD0 and passed independent volume controls.
Only the isolated target was switched; the original Retain PVC remains.
Cron stays off through a reloadable configuration-file setting, with its
bootstrap command-line override removed. This avoids an extra restart
when activation is eventually authorized by all migration gates.
V6 is the first full rehearsal on that volume and remains pending.

## Isolated-pool V6 acceptance

V6 completed data/metadata restore in509.75 seconds, normal restart/ANALYZE/
functional controls in63.47 seconds, and endpoint controls in7.33 seconds:
580.54 seconds total. All189 tables,83 sequence states and metadata passed.
The dedicated pool's nine serving OSDs had zero fresh slow/stalled events
over30 samples spanning the entire run. Every PG kept clean, three-host
placement and excluded OSD0. The destination storage/timing blocker is
resolved for this scoped path; the global OSD0 fault remains open.

Closed template0 was audited using source/target temporary clones with
matching schema, properties and typed content: zero user tables/sequences/
large objects. Both owned temporary databases were removed, and original
template0 was never opened. This was an explicit transient administrator
catalog operation; it did not pause source application writers.

Fresh source-shape checks found no replication slots/origins/clients,
prepared transactions, foreign tables, publications, subscriptions,
custom tablespaces, collation-version mismatch, existing IMMVs or cron jobs.
These guards must run again immediately before final maintenance.
