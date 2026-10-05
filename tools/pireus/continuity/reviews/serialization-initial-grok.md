**BLOCKER** (not CLEAR). Migration custody is sound; the new host lock is not.

## 1. Lock timeout vs `docker stop -t 30` vs skipped heartbeats (fatal)

`with_host_transition_lock` caps the **entire** critical section at `timeout 30s` (`--kill-after=2s`). Locked paths still call `enforce_fenced_compute_state` → `stop_gpu_docker_containers` → `docker stop -t 30` with **no** `host_ns` bound (`docker_host` is raw docker). `fence_pair`, `fence_invalid_local`, and every FENCED `active_enforcement_cycle` take that path.

So a legitimate fence can run to the 30s wall, get 124/137, and **not** finish the stop. `enforce-notify` then skips `watchdog_ping` on `75|124|137`.

Unchanged unit: `WatchdogSec=60`, `WATCHDOG_FRESH_SECONDS=55`. One 124 is survivable; **two consecutive 30s timeouts** (~62s with `sleep 2`) starve the systemd watchdog. `Restart=on-failure` + `StartLimitBurst=5` / `StartLimitIntervalSec=60` can then **disable the unit**. That is fail-open: GPU container still running, fence watchdog gone.

`commit_grant` is the same budget with **two** `status_once` runs (many 3s `host_ns` calls). A slow-but-valid commit is SIGTERM’d after `COMMITTING_*` / barrier detach — the race you already hit, now induced by the wrapper.

## 2. `flock` without `--close` (lock leaks across process-group kill)

```bash
/usr/bin/timeout ... /usr/bin/flock --exclusive --wait 10 ...
  /usr/bin/env ... /bin/bash -c 'source "$1"; shift; "$@"' ...
```

No `flock --close`. The lock FD is inherited by docker / `chroot` / `nsenter -t 1 -p` children.

`timeout` (no `--foreground`) only signals **its** process group. `host_ns` uses `nsenter -p` (host PID ns); those tasks are not reliably in that group. A grandchild that keeps the FD **pins `host-transition.lock` after 124/137**. Later `commit`/`fence`/`emergency-fence` get 75; watchdog again skips heartbeats. Tests never spawn `nsenter`/docker children, so they cannot see this.

## 3. ExecStopPost vs host-global lock vs unchanged systemd kill policy

`emergency-fence` now does `flock --wait 10` + full `fence_invalid_local`. Unit text is unchanged: `TimeoutStopSec=8`, `KillMode=mixed`, `ExecStopPost=... emergency-fence`.

- Mixed: SIGTERM **main only**. The lock holder is the timeout/flock/bash **child**, still running.
- Lock is **host-global**; holders include DaemonSet `kubectl exec` (different cgroup). Stop cannot kill them.
- `--wait 10` already exceeds `TimeoutStopSec=8`.

Stop/restart therefore often SIGKILLs the service **without** a completed emergency fence, while a pod-side `commit` may still hold the lock and write `SLURM`. Same SLURM-then-unfenced window, now on the deadman path.

`capture_protected_baseline` at the start of `enforce-notify` is still unlocked (only the CLI path was wrapped). Secondary, same class of omission.

## What is actually good

- Per-host exclusive lock around grant mutation **and** `status`/`report` is the right shape; the unlocked interleaving test matches the live failure mode.
- While a live `commit_grant` holds the lock, `COMMITTING_*` is no longer observable; that branch becomes crash recovery only.
- Skipping ping on 75/124/137 is the correct *intent* (do not forge a cycle).
- `write_grant_record` remains tmp+`mv`.
- Observer rebind: 19-field native ALLOW, `gpu_grant`/`resume` false, journal CAS then lease CAS, RV/UID tests, replay only on exact `journal_partial`, scheduler spec untouched, only `host_fence_manifest_sha256` may move. **No migration BLOCKER.** This freeze still does not install the new fence; serialization is inert until canonical recovery.

## Required before this freeze is installable

- Budget the lock timeout **above** `docker stop -t 30` + both `status_once` paths, or stop holding the lock across docker grace, or cut `-t 30`.
- `flock --close` so only the flock parent owns the FD; ensure timeout kills **that** parent.
- Align `TimeoutStopSec` / `KillMode` with ExecStopPost, or make emergency fence **non-blocking** on this lock (holders outside the service cgroup will always exist).
- Treat **any** wrapper failure other than a completed cycle as non-heartbeat (127/143/mkdir/`timeout` missing currently still ping).
- Extend tests to: inherited FD + leftover child, `docker stop -t 30` vs 30s cap, ExecStopPost vs pod-held lock, two 124s vs `WatchdogSec=60`.

Do not deploy this revision until those are fixed. The rebind script can wait; it does not activate the lock.
