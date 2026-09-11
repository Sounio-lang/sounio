CLEAR

The fixes address all initial blockers:
1. **Lock timeout handling**: The new `flock --close` ensures the lock FD is not inherited, and `--kill-after=2s` within a 45s operation timeout prevents zombie locks. The `docker stop -t 30` fits within the 45s budget, and systemd `TimeoutStopSec=120`/`KillMode=control-group` ensures emergency fences complete.
2. **Lock leaks**: The `--close` flag and process-group kill in `timeout` prevent FD inheritance by children, validated by tests with escaped sessions.
3. **Systemd alignment**: `ExecStopPost` now uses `with_host_transition_lock_wait 60` to outwait normal owners, and `KillMode=control-group` ensures all cgroup processes (including escaped children) are killed during service stop.
4. **Heartbeat logic**: Non-heartbeat pings on failures (75/124/137) are enforced, with explicit test coverage for two consecutive 30s docker timeouts.

All required changes (lock timeout budgeting, `flock --close`, systemd alignment, heartbeat suppression, and test coverage) are implemented and verified. The migration freeze now uses sequential CAS for journal/lease updates, and the real-process tests validate the fixed paths. No remaining blockers exist.
