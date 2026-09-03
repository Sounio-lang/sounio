# `tests/parity/` — Golden-regression + parity harness

## Why this exists

The shipped compiler `bin/souc-linux-x86_64` is built from the single
file `self-hosted/compiler/lean_single.sio`.  The modular tree at
`self-hosted/{lexer,parser,check,ir,native}/` is a future target kept
in lock-step by discipline, not by a test.  See
[`docs/compiler/KNOWN_LIMITATIONS.md`](../../docs/compiler/KNOWN_LIMITATIONS.md),
section *"Single-source build path"*, for the full context.

This directory exists so that:

1. **Today** — any semantic regression in `lean_single.sio` (or in the
   shipped wrapper) is caught by diffing current compiler output
   against captured golden outputs.
2. **Tomorrow** — when the modular build path produces a working
   alternate binary, running `run_parity.sh --parity PATH` will
   immediately surface any operational divergence between the two.

## Layout

- `MANIFEST.txt` — curated list of `.sio` programs that exercise the
  full Wave 9 type gates (E201–E206) and the `run`-pass examples whose
  numbers appear in the Wave 9 papers.  Comments at the top of the
  file describe the line format.
- `run_parity.sh` — the harness itself.  Three modes: `--capture`,
  `--check` (default), `--parity PATH`.  Run with `-h` for details.
- `golden/` — captured `stdout`, `stderr`, and exit code (`rc`) for
  each manifest entry, as produced by the current compiler.

## Typical use

```bash
# Default: CI-style regression check against the golden set.
./tests/parity/run_parity.sh

# After an intentional, reviewed semantic change: refresh golden.
./tests/parity/run_parity.sh --capture

# Once the modular build produces an alternate binary, diff the two.
./tests/parity/run_parity.sh --parity ./bin/souc-modular
```

## Known issues this harness already surfaced

### 1. `bin/souc check` swallows the self-hosted compiler's exit code

The launcher's `check` command invokes the self-hosted compiler, prints
the error (including `error[E20X]` and `typecheck: failed`), but the
wrapper always returns exit code 0.  This is a long-standing defect in
the wrapper, not a semantic bug in `lean_single.sio`: the binary itself
does detect and report the error correctly.

Until that is fixed upstream, the harness accepts a `fail` entry as
correctly failing if *either* `rc != 0` *or* the output contains
`typecheck: failed`.  Relevant code path in the wrapper:

```294:297:bin/souc
  check)
    shift
    compat_check "$@"
    exit 0
    ;;
```

The correct behaviour would be to forward `compat_check`'s return
value instead of unconditionally exiting zero.  We deliberately have
not patched this here because fixing the wrapper is a separable
maintenance task that should be landed in its own commit with its own
review.

## Adding a new entry

Append to `MANIFEST.txt`, then run `--capture` to freeze the expected
output.  Commit the new golden files alongside the manifest edit.  If
the new entry needs a behaviour invariant beyond
`rc == 0 && marker in output`, extend the harness rather than encoding
it ad hoc in the program itself.
