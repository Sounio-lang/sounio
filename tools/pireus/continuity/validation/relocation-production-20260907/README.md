# Production PostgreSQL migration evidence

Final source fence: 2026-09-07T07:23:01Z.
Durable TARGET authority: 2026-09-07T07:32:19Z.
Full endpoint/scheduler activation: 559.823452 seconds.
Acceptance is recorded in summary.json; the stage receipts have narrower
scope and deliberately do not independently claim production acceptance.
The metadata helper's source_write_pause=false and sequence_snapshot_atomic=false
are generic stage declarations: it does not own the source fence. The final
orchestrator and host controller bind this run to the drained, peer-only source
fence; final-data-summary.json records that binding and all captured/live
sequence comparisons passed. No MVCC-atomic sequence snapshot is claimed.

Only aggregate counts, timings, booleans and public controller state are
retained here. SQL archives, row fingerprints, credentials, configuration
backups and transaction token remain in protected runtime storage outside Git.
Global OSD0 latency remains unresolved; destination-pool I/O acceptance is scoped.
