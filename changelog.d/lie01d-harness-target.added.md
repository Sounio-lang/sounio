The test harness accepts a cross-compile target via `SOUNIO_TEST_TARGET`
(e.g. `SOUNIO_TEST_TARGET=aarch64-macos` with a multi-target Madaros), so
run-pass tests can execute through `scripts/dev/run_sio_test_suite_v2.sh` on
a Mac instead of only directly. The raw positional compiler interface ignores
`--target`, so the wrapper switches to the `build` verb when the knob is set
and ad-hoc signs Mach-O outputs on Darwin; temporary files moved to
per-invocation private dirs so parallel jobs no longer collide on a fixed
mktemp template in shared /tmp.
