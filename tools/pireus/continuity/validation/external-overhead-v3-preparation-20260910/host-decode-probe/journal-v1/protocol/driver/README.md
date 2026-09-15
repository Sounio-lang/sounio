# One-shot journal diagnostic driver

driver.py and contract.json are byte-identical to the private run directory /workspace/.cache/pireus-continuity/host-decode-journal-v1-run-20260910. The driver checks itself, Slurm configuration, frozen package and orchestration hashes before loading the adapter. --check passed without source-CI requests or hardware submission.

--execute requires remote tmux, an unused attempt marker and successful exact-source readiness. The pinned launcher then performs its own fresh exclusive-pair preflight. After a terminal launch with a unique job identity the driver collects base and per-rank journals and invokes the composed inspector. Refusal, collection failure and inspection failure are recorded; none resubmit the model.

Six synthetic controls passed for check-only behavior, missing tmux, an existing attempt, source-check rejection, launch refusal, collection failure and changed driver bytes. The public snapshot is prepared only: no waiter or model execution has started. Source 342b3f4e36 must be published and obtain the exact required checks before execution.
