# PostgreSQL relocation qualification — 2026-09-07

The user authorized the gated database relocation, then explicitly authorized
resolving its storage blocker ("RESOLVA"). No source database writes, client
cutover or write pause were performed during these qualification jobs.

## Actual AMD64 execution

The pinned child image
`paradedb/paradedb@sha256:ea4e8267016c929924a15df03eaf2ad9bb2cda9d194169e03eca232303898cc7`
executed on r770-proxmox. PostgreSQL reports16.14. Extension control defaults
match the source inventory: pg_search0.24.2, pg_cron1.6, pg_ivm1.13,
vector0.8.2, postgis/postgis_topology/postgis_tiger_geocoder3.6.4,
fuzzystrmatch1.2 and pg_stat_statements1.10.

The isolated functional jobs initialize disposable local data, with TCP
listening disabled and cron.launch_active_jobs=off. They do not attach the
proposed durable PVC or copy source data. They exercise a BM25 index/search,
vector distance, PostGIS distance, and actual incremental view maintenance.

## Restore negative control — retained failure

A pg_ivm1.13 IMMV created before pg_dump is restored as a table, but inserting
a new row into its base table does not update the restored IMMV. pg_restore
itself exits successfully; the behavioral assertion fails:
`Restored IVM no longer maintains itself`.

Evidence: extension-custody.log and extension-custody.json.
This is a real limitation of the tested restore path, not a success or a skip.

Read-only queries of pgivm.pg_ivm_immv found0 entries in each of memory,
memory_test, paradedb and template1. PostgreSQL's other inspected database
does not install pg_ivm. Thus the current source has no existing IMMVs to
migrate. The source-matching synthetic control removes its IMMV before the
dump, restores, creates a new IMMV, inserts a third row and verifies the
incremental update and BM25 behavior. It passes:
`ZERO_PREEXISTING_IMMV_RESTORE_PASS`.

Evidence: source-shape-custody.log and source-shape-custody.json.

**Mandatory migration condition:** repeat the source IMMV inventory for both
the rehearsal and the final paused snapshot. If any existing IMMV is found,
stop this restore path until view definitions, metadata, dependencies,
ownership and behavior are explicitly reconstructed and tested. Do not
claim general pg_ivm logical-restore support.

## Evidence custody and scope

The cluster removes completed pods quickly. The first version-check job's
completion status was retained, but its pod was already collected before
archival. The functional positive and negative controls were rerun with
streamed log capture; their pod UIDs, image IDs and exit states are retained.
The image job initially needed the two exact R770 scheduling tolerations
added to its pending pod; the checked-in manifest includes them.

These tests prove executable extension behavior on the pinned AMD64 image
and the limited synthetic restore scenario. They do not prove a restore of
the real databases, source/target data parity, storage durability, endpoint
compatibility, the15-minute cutover limit or Inkling serving.

The durable target and storage-canary manifests are prepared separately.
Apply them only after the corresponding storage prerequisites have been
assessed; server dry-run is not runtime acceptance.

## References

- https://www.postgresql.org/docs/16/backup-dump.html
- https://github.com/sraoss/pg_ivm
- https://www.paradedb.com/blog/paradedb-0-20-0
