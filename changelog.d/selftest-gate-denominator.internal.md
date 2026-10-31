# Self-test gate anchors its denominator to the source tree

`scripts/ci/madaros_self_test_gate.sh` now requires the
`compiler/main tests: N/M passed` denominator to equal the
`passed = passed + 1` site count in `self-hosted/compiler/main.sio`.
A truncated or fake runner printing a self-consistent small fraction
(e.g. `1/1 passed`) no longer emits `MADAROS_SELF_TEST_GATE_OK`.
Follow-up to the P2 thread on #2937.
