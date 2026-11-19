# Deadline and authority-controlled PostgreSQL cutover

The user authorized this migration and storage repair. The isolated-pool
full rehearsal passed in580.54 seconds, leaving319.46 seconds under the
900-second ceiling. The new pool excludes OSD0; global repair is not claimed.
The original source data directory and the old rehearsal Retain PVC persist.

Run prepare and execute in remote tmux via relocation_cutover.py with one
new private --output directory. Preparation requires V6 data, metadata,
runtime, network, pool and closed-template0 evidence. It reruns fresh source
shape and protected-host checks, creates one temporary credential-bearing
maintenance-client pod, checks the original endpoint/client IP, prepares
empty isolated target databases, and stages an owned source-host controller.
Credentials, configuration backups and transaction tokens remain private.

No source configuration or endpoint change occurs during preparation.
Before execution the controller must be active and enabled in systemd.
The installed controller hash and maintenance-client pod UID/IP are checked.

Execution first records FENCING durably, then installs owned HBA/peer-map
and PostgreSQL configuration fences. Source cron is disabled and the
default becomes read-only. Existing client backends are terminated; their
uncommitted transactions roll back. New TCP clients are explicitly rejected,
while only the OS postgres maintenance identity maps to the memory role.
This is a database maintenance pause: normal client reads and writes are
unavailable during the fence.

A fresh final backup, complete typed content/schema/ACL/sequence/role
comparison, normal WAL restoration, ANALYZE and rolled-back functional
controls run under that fence. The source-host watchdog independently
starts rollback at840 seconds, leaving60 seconds of the approved ceiling
for source recovery. It survives workspace disconnects and is enabled at
host boot. A changed boot also triggers pre-commit recovery.

Before authority transfer, the old Docker database is stopped and an owned
systemd proxy binds the original5433 endpoint. The proxy initially permits
only a fixed maintenance-client pod and host loopback; ordinary clients
remain blocked. The maintenance connection must reach the verified target
pod/data directory, ordinary access must fail, IPv4/IPv6 SSL behavior must
match, pool I/O observations must remain fresh and protected-host preflight
must pass after the original database is stopped.

The host durably records TARGET before publishing the proxy activation
marker. That record is the irreversible authority boundary for automatic
rollback. A lost RPC or activation-file failure after TARGET is reconciled
toward the target; the old source is never reopened. The target scheduler
is then enabled through SIGHUP, without an additional database restart.

Before TARGET, failure or deadline expiry stops/disables the owned proxy,
restores only source configuration bytes owned by this transaction,
starts the original container if needed, and verifies readiness/settings.
Unexpected config or identity changes are preserved and reported, not
overwritten. After TARGET, reversal requires a new write pause and reverse
export/restore of authoritative target data; stale-source rollback refuses.

Validation includes eight controller state/failure cases, a real TCP
activation/half-close test, and a disposable PostgreSQL16.14 cluster.
The real cluster proved peer-only maintenance, read-only/cron fencing,
draining of an uncommitted writer, actual stop/restart and exact config
restoration. Its injected deadline restored the original service in1.16
seconds. These fixture tests do not claim production/systemd acceptance.
The live execution receipt remains separate and mandatory.

After acceptance, a persistent target-authority annotation and production
ingress block accidental reuse of the destructive rehearsal preparation.
Do not apply the bootstrap target manifest to an authoritative database:
its initial cron-off override is deliberately removed by the preparation
workflow. The source controller state, fenced old data directory, target
PVC and private final archives are retained for custody and recovery.

## Accepted production migration, 2026-09-07

The protected cutover completed with TARGET authority at 07:32:19 UTC.
Endpoint and scheduler activation took 559.823 seconds (9m19.82s), within
the authorized 900-second limit. All 189 tables matched; 83 sequences,
schemas, ACLs, roles and database properties passed final comparison.
Normal WAL/durability settings and rolled-back extension controls passed.
The original 5433 endpoint now forwards to the R770 target. Normal-client
activation, IPv4/IPv6 SSL behavior, target cron and protected-host preflight
passed. The old source container is stopped; its original data directory
and private final backups are retained. The temporary maintenance client
was removed after UID verification.

The destination pool had no fresh slow/stalled events during final migration.
Global OSD0 repair remains open. Do not rerun destructive rehearsal or
bootstrap manifests on the authoritative target. Stale-source rollback is
refused after TARGET; reversal requires a new protected reverse migration.

Evidence: ../validation/relocation-production-20260907/summary.json.
Inkling serving and eight actual model proposals still require fresh memory
qualification and execution; database migration is not model acceptance.
