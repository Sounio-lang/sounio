New gate `scripts/ci/a64_epistemic_variance_parity_gate.sh`, wired into the
core-c shard. It runs each case in
`tests/selfhost/aarch64_compile/epistemic_variance_parity_cases.txt` as an
aarch64 binary under qemu and requires its stdout and exit code to equal
the x86-64 run's. It is the first gate that executes the arm64 lane rather
than only compiling it. The lean_single seed was refreshed to a fixed point.
