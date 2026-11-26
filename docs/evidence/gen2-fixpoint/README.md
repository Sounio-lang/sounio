<!-- docs:meta
topic_id: repo.docs.evidence.gen2-fixpoint.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.gen2-fixpoint.readme
-->

# gen2 fixed-point receipts

One file per compiler source. The name is the git tree hash of `self-hosted/`
(`git rev-parse HEAD:self-hosted`), and the file records the Slurm job that
compiled that source to a fixed point (gen2 == gen3).

`scripts/ci/gen2_receipt_gate.sh` runs in Fast Checks, so CI Decision is red
for any head whose `self-hosted/` tree has no valid receipt here. A merge of
main that changes only docs or CI keeps the tree hash, so it keeps its receipt.
Any change under `self-hosted/` needs a new job and a new receipt.

Why the rung left the PR run: GitHub-hosted runners kept losing the ~2 h 50 min,
~40 GiB self-compile (#3741, #3746). It runs on the DARWIN cluster instead.

## Format

`key=value`, one line per key, every key exactly once:

| key | meaning |
|---|---|
| `tree` | the 40-hex tree hash, same as the file name |
| `commit` | the commit the job checked out; its `self-hosted/` tree must be `tree` |
| `slurm_job` | Slurm job id |
| `node` | Slurm node name |
| `rung` | `fixpoint` |
| `gen1_sha256`, `gen2_sha256`, `gen3_sha256` | at least 12 hex; gen2 must equal gen3 |
| `maxrss_kb`, `wall_s` | MaxRSS and wall time of the gate step |
| `date` | `YYYY-MM-DD` |
| `receipt_doc` | where the full receipt (prediction, whether it held) lives |

The gate checks that the receipt belongs to this source and records a fixed
point. It cannot re-run the job. The signed-off commit that adds a receipt is
the statement that the job ran as recorded.
