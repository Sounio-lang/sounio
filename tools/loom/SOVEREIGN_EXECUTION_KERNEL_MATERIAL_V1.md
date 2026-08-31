# Sovereign Execution Kernel Material V1

This is the transitory Linux/C++20 material realization of frozen Sounio action
9042. It is a canary, not production authority.

The HostGuardian stores one grant as an internal state machine. The client sends
only `CONSUME`; it receives no token, handle, grant descriptor, or release
capability. The Guardian accepts the transition only after binding the Unix
peer to the preregistered PID, UID/GID, pidfd, start tick, executable, harness
ancestry, and operation.

Six interface surrogates (`GUI`, `TUI`, `CLI`, `Pod`, `tmux`, and coordinator)
attempt `RELEASE` as same-UID processes and remain alive until the material
witness begins. A seventh same-UID hostile process attempts `CONSUME`. All are
refused while the material-start count remains zero. After legitimate atomic
consumption, every surrogate and the client transport die while the
HostGuardian-owned witness completes exactly once.

A separate Guardian-death treatment arms `PR_SET_PDEATHSIG` before release.
Killing the true Guardian must extinguish the material process, erase the
in-memory grant with the Guardian address space, and leave the completion marker
affirmatively absent.

Two compile-time sabotage twins are required:

- removing only the exact peer PID/start-tick principal binding must admit the
  hostile same-UID spoof;
- removing only `PDEATHSIG` must allow the material child to survive Guardian
  death until the test kills it.

Both twins must fail while the unsabotaged source passes the frozen Sounio
treatment, revocation, and production-prerequisite decisions.
