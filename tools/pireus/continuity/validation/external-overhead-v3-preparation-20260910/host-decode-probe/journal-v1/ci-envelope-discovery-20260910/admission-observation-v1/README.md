# CPU36 admission observation

The frozen dispatch remains distinct from source qualification. See timestamped
job, runner and group snapshots; an online runner is not job assignment.
No queue cause is confirmed. No dispatch, JIT regeneration, permission change
or inference was performed during this observation.

The read-only observer polls the existing job once per minute in remote tmux
until completion or the original runner lifetime deadline. It records remaining
lifetime and whether the complete 150-minute job budget still fits. It never
cancels, retries or extends the attempt. Its private output is retained under
/workspace/.cache/pireus-continuity/ci-runner-execute-v1-20260910.

packages.txt was collected from the running execute pod, not reconstructed
from the image or preflight. This does not freeze future apt repository state.
