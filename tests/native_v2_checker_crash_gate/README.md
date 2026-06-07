Single-module checker crash gate.

This gate records small programs that should exercise the self-hosted
typechecker through `mc --check` without crashing. `valid/*.sio` must be
accepted with `check: OK`; `invalid/*.sio` must be rejected cleanly and must
not segfault or be accepted.

The initial corpus was chosen to cover known release-wall risk areas:
closures, tuple/Option match patterns, nested if-let, large by-value struct
returns, methods, and generics.

Active `valid/` coverage is limited to cases that are both well-typed and
currently accepted by `--check`. Known remaining false positives or crashes are
kept under `deferred/` with a short classification so the gate stays green while
preserving the release-wall map.

Fixed in this lane:

- plain binary expressions such as `x + y` and `p.x + p.y` no longer SIGSEGV;
  the checker uses an in-place binary operator/unit tail for non-unit operands.
- `impl` item checking no longer goes through the by-value `check_impl_item`
  bridge; method tests now fail cleanly with E011 instead of SIGSEGV.
