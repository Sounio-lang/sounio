`bootstrap/stage0.c` now refuses what it cannot compile (exit 2, no output
file) instead of writing a silently wrong binary (ADR-004).

Before, it dropped what it could not handle and exited 0:
- unknown identifiers, enum variants and constants compiled as `0`;
- calls to unknown functions compiled as `call +0`;
- every variant or literal `match` arm was compiled as taken;
- non-zero global initializers were dropped (the global started at zero);
- code past 1 MiB, functions past 1024 and the other fixed tables were cut
  off; unknown characters, type tokens and expression tokens were skipped;
- a method sharing a bare name with another function bound to the first;
- `expect` syntax errors were printed but the binary was still written.

Two of these are fixed rather than refused: a struct field store now
writes the named field (it always wrote field 0), and call sites are
patched from a recorded list instead of a byte-pattern scan of the whole
code. Built that way, `lean_single.sio` came out as a 1 MiB ELF that
segfaulted; it is now refused at its first non-zero global initializer.
`bootstrap/boot1.sio` still compiles to the identical ELF.

Pinned by `scripts/ci/stage0_fail_closed_gate.sh` (Contracts Core,
core-b): seven probes that compiled with exit 0 before the fix.
