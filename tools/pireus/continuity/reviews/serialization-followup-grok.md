**BLOCKER** (not CLEAR). Do not deploy. Migration freeze shape is still acceptable; the host lock is still wrong.

## 1. `flock --close FILE CMD` drops the lock (fatal)

```bash
/usr/bin/flock --close --exclusive --wait "$wait_seconds" \
  --conflict-exit-code 75 \
  "$HOST_ROOT/var/lib/pireus-spark-pair/host-transition.lock" \
  /usr/bin/timeout --kill-after=2s 45s \
  ...
```

util-linux `/usr/bin/flock FILE CMD` acquires, then **`close(fd)` + `execvp`**. `flock(2)` is released when the last FD on that file description is closed. There is no surviving flock parent, no FD for the subshell, and **no lock during the critical section**.

The comment (`flock` parent outside timeout’s group, sole owner, descendants cannot retain the lock) does not match this invocation. `--close` here is the opposite of a parent-held lock: it is the “don’t leave the lock on the daemon you exec” flag.

Consequences:
- `commit` / `fence` / `enforce-notify` / `emergency-fence` interleave again (original live failure).
- Escaped-session / post-timeout reuse / emergency-outwait tests still pass: they only prove wait-for-acquire and timeout bounds. After acquire the lock is already gone.
- The **only** test that would catch this is locked interleaving (`committed` must not appear while `active_enforcement_cycle` is in `device_barrier_attached`). That is not evidence it was run.

Required shape: the **already-subshell** function must keep an FD, flock that FD, and close it only in the `timeout` child, e.g. `exec {lockfd}<>lock`; `flock -x -w N "$lockfd"`; `timeout ... {lockfd}>&-`. Then timeout’s process-group kill cannot reap the holder, and `setsid`/host children cannot inherit the FD. `flock --close FILE timeout ...` cannot do that.

Until that is true, serialization is inert. Same class of SLURM-then-unfenced window, including ExecStopPost.

## 2. Grok #1 budget still not met (watchdog fail-open remains)

Operation cap is `45s` + `--kill-after=2s`. Paths still hold the lock across `docker stop -t 30` **and** a full `status_once` / `enforce_fenced_compute_state` tail (`fence_and_status`, FENCED `active_enforcement_cycle`). `commit_grant` still does **two** `status_once` plus barrier detach. `docker_host` is still unbounded except by that 45s wrapper.

A legitimate 30s grace plus status work can still return 124. Any nonzero skips `watchdog_ping`. `WatchdogSec=60` / fresh 55 then cannot absorb `~47s` failure + `sleep 2` + another docker-30 cycle. `Restart=on-failure` + `StartLimitBurst=5` / `StartLimitIntervalSec=60` is unchanged. That is still fail-open if the unit is disabled while a GPU container is running.

The new test is `sleep 1; sleep 30; sleep 1` under a 36s join — it does not run `stop -t 30` plus `status_once`. Slack vs 45s is unproven.

Need one of: budget **above** docker 30 + the real status/commit tails (and that the following ping still fits in 60s), drop the lock before docker grace, or cut `-t 30`.

## What did get fixed (not sufficient)

- `TimeoutStopSec=120`, `KillMode=control-group`, boot `TimeoutStartSec=60`.
- Emergency `--wait 60` > a *bounded* owner (~47s hold, not 10+45+2; wait is not hold).
- `capture_protected_baseline` on the daemon path is locked.
- Any nonzero `cycle_rc` skips heartbeat (not only 75/124/137).
- Observer freeze: only `host_fence_manifest_sha256` + `material_policy_sha256` + `admission_manifest_sha256`; policy only `host_fence_configmap`; admission bytes only three content-address substitutions. Native 19-field authority, barrier, 32768 floor unchanged. Journal CAS then lease CAS stays sequential; `journal_partial` replay only.

Those do not activate a lock that `flock --close FILE CMD` immediately releases.

Fix (1) with a parent-owned FD, then re-prove (2) with a real docker-grace + status path and watchdog arithmetic — not another `sleep 32`. Do not install this freeze until then.
