# BLK-20260907-pireus-tp2-host-capacity

- Severity: B2
- Class: runtime-capacity
- Evidence: E3 collective measurements; serving allocation unmeasured
- Owner: codex-pireus / continuity-20260906
- Worktree: /workspace/.wt/pireus-integration-20260906
- Branch: codex/pireus-inkling-cycle-20260906
- Status: OPEN

Canonical recovery passed at epoch15; both hosts and scheduler returned to
Slurm ownership. This blocker concerns serving capacity.

Exclusive TP2 InfiniBand canaries11869,11870,11871 emitted PASS for both
ranks. Job11871 was observed COMPLETED ExitCode0:0. Baseline used64 NCCL
channels. Bounding channels to8 and buffer1MiB recovered about1.4GiB of
host MemAvailable per node. One channel with Ring/Simple did not improve
the limiting node further. These were temporary diagnostic job environments.

Rank0 MemAvailable after collectives was107.150GiB baseline and108.569/
108.565GiB in the bounded/minimal variants. Half the serialized text tensors
is77.361GiB. Subtracting that optimistic partition and the unchanged32GiB
floor leaves -2.211/-0.791/-0.796GiB. This estimate is not measured runtime
weights: repacking, replication, caches and loading transients are additional
unknowns. It does not justify another full load under the current budget.

The protected beagle-memory-pg-pdb container on3c59 was observed at4.654GiB.
Configured shared_buffers is12GiB; resident shared pages can grow with
database activity. No query text, private records or credentials were
collected in this database inspection. No protected service was changed.

Acceptance: fresh two-node memory qualification must support the unchanged
checkpoint/runtime and32GiB floor including loading peak and caches, then
actual TP2 serving and generation. Do not accept serialized-half arithmetic
or successful NCCL as serving evidence.

Next action: establish viable capacity while preserving the protected
database, or prove a supported runtime memory reduction with the exact
checkpoint. Maintenance or relocation of the protected service requires a
concrete reviewable change and explicit authorization. Do not lower the
floor or silently change the model/quantization.

Independent progress: deterministic-live-baseline-20260907 completed native
admission, materialization, pair parity and30-block paired timing. All8
candidates passed material parity; native decisions were8 NO_GAIN,0 eligible.
The requested8 real Inkling proposals remain pending.
