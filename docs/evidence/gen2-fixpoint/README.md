<!-- docs:meta
topic_id: repo.docs.evidence.gen2-fixpoint.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.gen2-fixpoint.readme
-->

# gen2 fixed-point receipts

One file per compiler state. The name is the gen2 inputs key: sha256 over
"<path> <git object id>" for every path in `scripts/ci/gen2_receipt_inputs.txt`,
plus its `param` lines. The file records the Slurm job that compiled that state
to a fixed point (gen2 == gen3).

`scripts/ci/gen2_receipt_gate.sh` runs in Fast Checks, so CI Decision is red
for any head whose inputs key has no valid receipt here. A merge of main that
touches none of the listed inputs keeps the key, and with it the receipt. A change
to any input (compiler, stdlib, bootstrap ELFs, the gate, the build script or
their helpers) needs a new job and a new receipt.

Why the rung left the PR run: GitHub-hosted runners kept losing the ~2 h 50 min,
~40 GiB self-compile (#3741, #3746). It runs on the DARWIN cluster instead.

## Format

`key=value`, one line per key, every key exactly once:

| key | meaning |
|---|---|
| `inputs_key` | 64 hex, the same as the file name |
| `commit` | 40 hex: the commit the job checked out. Its inputs key, recomputed there, must equal `inputs_key` |
| `slurm_job` | Slurm job id |
| `node` | Slurm node name |
| `rung` | `fixpoint` |
| `gate_rc` | `0` |
| `ir_parity` | `pass` |
| `gen1_sha256`, `gen2_sha256`, `gen3_sha256` | 64 hex each; gen2 must equal gen3 |
| `maxrss_kb`, `wall_s` | MaxRSS and wall time of the gate step |
| `date` | `YYYY-MM-DD` |
| `receipt_doc` | where the full receipt lives (the prediction, and whether it held) |

When the commit is not in the clone, the gate fetches it from origin. If that
fails, the gate fails.

## What the gate establishes, and what it does not

This is an **attested** receipt, checked for consistency, not a re-execution.
The gate establishes three things: the receipt is well formed, it records a
fixed point with rc 0 and IR parity, and the commit it names has exactly this
state's inputs. It cannot establish that the job ran. That statement is made by
the DCO-signed commit that adds the receipt. The Slurm accounting
(`sacct -j <slurm_job>`, including the `r-…` step names that carry the
digests) is the record it can be checked against.
