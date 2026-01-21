# Test annotations quick reference

Canonical source: `tests/README.md`.

## Common directives

- `//@ run-pass` — should compile and run
- `//@ compile-fail` — should fail to compile
- `//@ error-pattern: <text>` — expected substring in diagnostics
- `//@ ignore` — skip this test
- `//@ check-only` — for `run-pass`, only `check` (skip `run`)
- `//@ timeout-ms: <n>` — override timeout for `check` and `run` (milliseconds)
- `//@ run-timeout-ms: <n>` — override timeout for `run` (milliseconds)

## Where to look for similar tests

- Effects: `tests/run-pass/effects.sio`
- Async: `tests/run-pass/async_basic.sio`
- FFI: `tests/ffi/test_raw_pointers.sio`
- Diagnostics: `tests/ui/`
