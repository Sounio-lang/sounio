# Journal collection and process binding controls

The supplemental collector now reads each rank's journal after a terminal base collection. It preserves exact bytes and explicitly records missing files, with worker UID/node snapshots before and after collection and the observed boot identity. Existing output directories are rejected.

The inspector requires the prepared journal runtime bytes and a pinned base collection. It checks the supplemental manifest, runtime receipt binding, lifecycle PID and job/rank, observation timestamps inside the lifecycle interval, native metric shape and the ordered first-fifteen-decode window. Truncated or malformed records are retained by collection and rejected by inspection; empty or record-boundary partial journals cannot qualify a complete window.

Seven new synthetic controls exercise complete transport, missing/partial records, malformed writes, corruption/PID substitution, worker replacement, boot/chronology mutation and rejection of the historical stdout runtime. The full host-decode suite passed as recorded in tests.txt. No model allocation was performed.

The base full source/freeze inspector must still run as part of a prospective orchestration adapter. This supplement deliberately returns source_qualified=false, full_diagnostic_custody_qualified=false, loaded_model_qualified=false, timing_eligible=false and pilot_acceptance=false. A complete journal window alone does not establish these broader claims. The next step is the new protocol/freeze and adapter, followed by independent source acceptance before any hardware attempt.
