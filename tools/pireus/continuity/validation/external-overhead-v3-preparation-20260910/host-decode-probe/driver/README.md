# Persistent diagnostic driver

The private entrypoint is /workspace/.cache/pireus-continuity/host-decode-diagnostic-run-20260910/driver.py. Its --check mode verifies the packet, source identity, orchestration and Slurm configuration without checking CI or allocating hardware. It passed locally.

The --execute mode requires remote tmux and fresh exact-source CI, records exclusive attempt entry, and calls the frozen launcher with its fresh pair and queue gates. There is no wait/retry loop. A terminal stage is collected and inspected separately, including unsuccessful model execution. No attempt has been entered or submitted.
