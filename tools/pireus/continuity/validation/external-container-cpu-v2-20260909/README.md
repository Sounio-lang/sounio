# CPU container memory control v2

This prospective protocol is frozen before a new Slurm submission.
Job 11974 remains closed with its original failed total-cgroup oracle.

The target still touches/releases 64 MiB, with baseline 2 s, allocated 3 s,
released 2 s. Keep the 512M per-node request and guardian 33 GiB unchanged.
The new observer adds task memory.max/high/swap.max; it does not change limits.
PSS and anon must each rise and fall 48–80 MiB. Total, file and reclaim
remain observations. Require stable readable limits, no new OOM events,
at least three settled samples per phase and sample gaps <=500 ms.

Protocol.json pins runtime and oracle bytes. The publisher/supervisor identity
and custody checks remain a separate mandatory gate; the numerical oracle
alone does not establish launch provenance. Raw journals, worker UID/boot
identity before/after, entry hashes, acknowledgements, result hashes and
unique completed Slurm accounting must all match the new job and source.

Execution requires a fresh pair preflight, empty pair queue, exclusive worker
directories, one Slurm attempt from remote tmux, and no automatic retry.
No job has been submitted for this v2 packet. Synthetic limits used in unit
tests do not supply missing observations for 11974.

This qualifies only the declared CPU control if both ranks pass every gate.
It cannot qualify loaded Inkling overhead, serving, a model profile, or pilot
resumption. Ancestor cgroup limits are not inferred from task-local limits.
