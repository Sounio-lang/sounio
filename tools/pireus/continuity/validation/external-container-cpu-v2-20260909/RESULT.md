# CPU v2 result: job 11975 PASS

Source 845b51ed0c, prospective protocol committed before submission.
Slurm COMPLETED 0:0, 2026-09-09 20:37:32 to 20:37:52 (raw scheduler times).
One attempt, no retry. The fresh preflight accepted the pair; queue was empty.

Both ranks pass the frozen numerical oracle and separate handoff/custody checks:
entry begins after first sample; target PID survives exec; hashes, nonce,
job/rank, worker UID/boot and actual cgroup agree. Both observers invalidate
the exited target with metrics null. Both guardians exit 0 above 33 GiB.
36 samples per rank; maximum gaps 205.448 / 205.642 ms.

PSS rises/falls exactly 67108864 bytes in both ranks.
Anonymous memory rises 67178496 / 67166208 bytes and falls 67092480 bytes
in each rank. File cache falls 66142208 / 66174976 bytes during allocation.
Total current rises only 1101824 / 962560 bytes: the legacy v1 criterion
would still fail and remains recorded separately, without affecting v2.

All sampled task memory.max/high/swap.max values are 'max'; task events and
events.local have zero deltas. These observations do NOT establish absence
of ancestor limits. Direct reclaim increased, but this packet does not
identify its triggering limit or explain the earlier Inkling guard stop.

28 local tests pass. Tests using historical rows with synthetic limit fields
are unit fixtures only; qualification here uses new 11975 observed values.
The packet verifier pins raw custody and repeats the frozen numerical oracle.

The generated protocol/manifest retain their original pre-execution state.
This result and status.json supply execution state without rewriting the freeze.
No Inkling, loaded-model overhead, general serving or pilot acceptance is claimed.
Next is a separately frozen loaded-model observer overhead diagnostic, with
exact-source CI and complete runtime identity before any model execution.
