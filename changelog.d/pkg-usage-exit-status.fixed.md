`souc pkg` and `souc install` now exit non-zero on a usage error instead of
reporting success. Before this change the three usage-error paths printed a
diagnostic and returned, so the process exited 0 on a command that did nothing
(the silent-success class the wrapper-hardening campaign refuses):

- `souc pkg` — `souc pkg: missing subcommand` exited 0, now exits 1.
- `souc pkg <unknown>` — `souc pkg: unknown subcommand: <unknown>` exited 0,
  now exits 1.
- `souc install` — `souc install: missing package name` exited 0, now exits 1.

The diagnostics are unchanged; only the exit status is. The three paths now
end in `panic` (rc=1), the file's idiom for a failing command — `exit_process()`
is a no-op under the lean_single seed (E200). The deliberate registry stubs
(`pkg help`, `pkg self-test`, `pkg info`, `install <name>@<version>`) keep
exiting 0.

New gate `scripts/ci/pkg_dispatch_usage_gate.py` drives `bin/souc` with the
compiler under test pinned via `MADAROS_RAW_BIN`: the three refusals must exit
non-zero with the diagnostic intact, and the four honest paths must still exit
0.
