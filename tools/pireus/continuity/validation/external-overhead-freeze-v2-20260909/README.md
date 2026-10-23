# External overhead screening v2: prospective parity correction

No model attempt used v1. Its immutable freeze remains historical.
V2 supersedes it because responses embed Slurm job IDs: cross-job raw bytes
cannot match. Within a job, both ranks must still produce identical bytes.
Across jobs, decode the response and exclude exactly job; every other field
must match, including output IDs, execution profile, revision and input hash.

The frozen evaluator checks request coverage, lifecycle ordering, external
identity/coverage/metrics/OOM events, guardian completion and the prospective
256 MiB memory / 1.10 decode ratio limits. Five freeze and ten evaluator tests
pass. Synthetic metric/timing fixtures are unit controls, not model evidence.

The evaluator deliberately does not confer runtime/collection custody.
Its result keeps loaded_model_overhead_qualified=false until a separate
collector verifies source/runtime/input hashes, unique terminal scheduler
identity, worker UID/boot continuity, completion receipts and external
handoff/acknowledgement hashes. That integration remains required.

Both runtimes and inputs are frozen under the private root in validation.json.
The source CI run34402455439 remains pending in the archived observation.
Do not push to the PR and cancel that exact-source run, allocate the pair, or
promote prior CPU success into loaded-model acceptance.
