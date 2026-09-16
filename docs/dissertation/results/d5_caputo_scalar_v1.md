<!-- docs:meta
topic_id: repo.docs.dissertation.results.d5-caputo-scalar-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.d5-caputo-scalar-v1
-->

---
docs:meta: true
topic: dissertation-results
kind: numerical-method
model: scalar-fractional-ode
status: implementation-complete
version: d5-v1
date: 2026-05-14
---

# D5 Caputo Scalar v1

## Scope

D5 adds a scalar Caputo fractional-derivative helper for the special-functions
stdlib. The implementation is intentionally narrow: fixed-grid L1 discretization
for `0 < alpha < 1`, scalar `f64` samples, and a 512-sample test surface. It is
not a distributed-order solver, a variable-step fractional integrator, or a PBPK
model claim.

## Implementation

- `stdlib/special/caputo.sio` implements L1 weights
  `b_j = (j+1)^(1-alpha) - j^(1-alpha)`.
- `caputo_l1_derivative` evaluates the weighted backward differences with Kahan
  compensated summation.
- `mittag_leffler_e_alpha` provides a bounded scalar series helper for
  regression tests and fractional-decay witnesses.
- `fractional_decay_ml` evaluates `c0 * E_alpha(-lambda t^alpha)`.

The scale factor is:

```text
1 / (Gamma(2 - alpha) * dt^alpha)
```

## Validation

Pinned compiler:

```text
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64
sha256=3cbea2b475e79737046f8ccf463c07d22cd5fb678fd479a032ee04bd8e19da93
```

> **Reproduction command corrected (2026-09-13).** The command first recorded here was
> `SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run <test>`. `bin/souc`
> execs that override with its arguments unchanged (it already did at the merge that added
> this file, `bebd78d74c`). Measured on 2026-09-13 with `bin/souc-linux-x86_64`, before `bin/souc`
> began refusing the form: lean_single stopped at
> `error: no main` (a current-source lean_single shows why: it opens `run`, which does not exist,
> as a 0-byte source). How the recorded values were originally produced cannot be established
> from this repository's history. `bin/souc` now refuses the form (exit 64). The command below uses the
> ELF's raw `<source.sio> <output>` interface. Run it from the repository root: lean_single
> resolves stdlib imports relative to the working directory. The pinned binary (sha256
> `3cbea2b4…`) is no longer in the repository; the re-run used `bin/souc-linux-x86_64`, sha256
> `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`.
> Re-run 2026-09-13: lean_single printed `D5_CAPUTO_SCALAR_PASS`. Only that marker was
> re-checked; every other value in this file is the original record. The default Madaros engine
> printed it too, run as `bin/souc run tests/stdlib/special/test_caputo_scalar.sio` with no `SOUNIO_SOUC_BIN` set (the Madaros
> ELF behind it: `bin/madaros-linux-x86_64`, sha256
> `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`).

Focused test (from the repository root):

```text
cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/special/test_caputo_scalar.sio /tmp/caputo_scalar.elf && chmod +x /tmp/caputo_scalar.elf && /tmp/caputo_scalar.elf
```

The test surface covers:

- constant derivative equals zero,
- power witnesses `t` and `t^2` at `t = 1.0`, `dt = 0.01`, and
  `alpha = 0.7, 0.8, 0.9`,
- Mittag-Leffler decay identity,
- fractional-decay L1 solve-vs-analytical checks over `t in [0, 24h]`,
  `dt = 0.1h`, and `CL/V = 0.1 h^-1` for `alpha = 0.7` and `alpha = 0.9`.

## Caveats

The L1 method is first-order to `2-alpha` order under the usual smoothness
assumptions and degrades near weakly singular initial behavior. The current
tests use finite tolerances appropriate for a small fixed-grid stdlib witness,
not a high-precision fractional calculus benchmark.

The linear power witness is exact to floating-point tolerance. The quadratic
power witness is below `1e-3` at `alpha = 0.7`; at `alpha = 0.8` and
`alpha = 0.9`, the mathematically expected fixed-grid L1 truncation error at
`dt = 0.01` is about `1.7e-3` and `2.9e-3`, so the committed regression uses a
`3e-3` guard for those two points rather than masking the discretization error.

## Reproduction commands (2026-09-16)

The `bin/souc-lean-single-x86_64` line above runs the lean_single seed (sha256
`9d7892132aa0a9cf839df4560bf968628dc30fda4af978a4efc4a99b4e8f89f5`, most recently changed by merge
commit `09aedffafa`) and includes a `chmod +x` step. Until 2026-09-16 that line invoked
`bin/souc-linux-x86_64` (sha256
`a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`) and contained no `chmod +x` step.

Run as written today, a line without `chmod +x` exits 126, with `Permission denied` reported
by the launcher: both binaries write the ELF with mode `-rw-r--r--` under umask 0022. How the
earlier runs recorded in this note were invoked is not recorded here, and earlier paragraphs
naming `bin/souc-linux-x86_64` describe those runs.

Measured 2026-09-16 at HEAD `169ac84463`, from the repository root, at the documented `/tmp`
ELF paths (none existed beforehand; each was removed afterwards). Each source below was
compiled and run with the seed and with `bin/souc-linux-x86_64`; every build exited 0, every
run exited 0 after `chmod +x`, and for each source the two binaries' stdout was byte-identical
(stdout only, not the ELF bytes):

- `tests/stdlib/special/test_caputo_scalar.sio` (ELF `/tmp/caputo_scalar.elf`): stdout contained `D5_CAPUTO_SCALAR_PASS`

Any other line in the blocks above — expected-output markers, `rg` searches, gate scripts —
was not run and is not part of this comparison. The values recorded elsewhere in this note
were not re-derived.
