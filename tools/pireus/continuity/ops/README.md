# Pilot transport recovery

The frozen74789 pilot passed its required integration checks and completed its
first deterministic cell (32 admitted and benchmarked plans, zero gain eligible).
Tokenizer job11955 completed32 requests on both ranks with exit0. The coordinator
then refused its immediate scontrol completion query. Slurm retains completed jobs
there for only300 seconds; later recovery must use durable accounting.

resume_pilot.py leaves original manifests, code dependencies, requests, CI acceptance
and completed experimental cells unchanged. This separately versioned operations
helper adopts an unfinished stage only after its frozen intent/input and unique
paired completion match a successful top-level Slurm accounting record for the
same job and both Sparks. It records controller UID, query, raw output, timestamps
and helper SHA; both recovery receipt and completed marker enter the existing
custody journal. Missing pairs, failed/active/duplicate accounting, another node
pair or changed inputs refuse recovery. No job is submitted by the adoption step.

The helper invokes the original pilot driver after adoption. If another immediate
completion query races final Slurm state, it can adopt that completed stage using
the same checks and resume. Failures elsewhere stop; partial model generations
are never automatically resubmitted. All native admission, parity, gain and memory
criteria remain owned by the original frozen pipeline. Operations-helper controls
are separate from the74789 integration acceptance and hardware measurements.

Run in remote tmux:
python3 tools/pireus/continuity/ops/resume_pilot.py --run /absolute/frozen/pilot

The operation uses its own nonblocking lock. Native pilot execution retains its
original pilot lock. Synthetic refusal tests do not establish model acceptance.
