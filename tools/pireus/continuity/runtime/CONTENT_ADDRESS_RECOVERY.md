# Canonical host script identity and partial installation recovery

Epoch 13 refused action 29 before installing the new host watchdog:
the ConfigMap name was generated from an awk extraction ending in two
newlines, whereas YAML literal clipping and bridge transport produce one.
The corrected script has one final newline and is named
pireus-spark-host-fence-b6ae432c7ec9. Its executable shell statements are
unchanged. The content-address test compares parsed YAML, extraction,
transport and the three admission references, with negative controls.

The prior serialization freeze remains preserved at
/workspace/.wt/pireus-serialization-frozen-20260907 (commit 1ecd6ec3ad).
The actual predecessor host revision is preserved at
/workspace/.wt/pireus-observer-frozen-20260906 (commit da73523e52).
The correction changes exactly three frozen material hashes: host manifest,
policy, and admission. The backend and native authority remain unchanged.

recovery_content_address.py uses the unchanged native recovery migration
engine. Its lock pins Lease UID and epoch 13, host epoch 12 and predecessor
freeze, and all three verified roots. This is a specific failed-install
state; it does not accept arbitrary stale host grants. The grant must be
FENCED and invalid, with current watchdog/barrier/protected-service evidence,
no worker selectors/pods/jobs, and at least 32768 MiB available on each host.
The adapter performs fresh native admission twice and CAS on journal then
Lease. It neither grants GPUs nor resumes scheduling. Canonical recovery
must independently establish Slurm ownership afterwards.

Executed local and live observations are stored separately in validation.
A review is not execution evidence. In particular, the Qwen review asserted
post-migration and partial-replay success before those were executed; those
claims are not accepted as evidence.
