<!-- docs:meta
topic_id: repo.docs.dissertation.results.ml-negz-fix-v1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.ml-negz-fix-v1
-->

---
topic_id: repo.docs.dissertation.results.ml-negz-fix-v1
title: Mittag-Leffler Negative Real Fix v1
doc_type: dissertation_result
status: active
owner: phase-d
last_updated: 2026-05-15
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.ml-negz-fix-v1
---

# Mittag-Leffler negative real fix v1

Gate target: `ML_NEGATIVE_Z_FIX_PASS`.

## Root Cause

The consolidated D.5 implementation did not contain a separate
`stdlib/special/mittag_leffler.sio`; the active implementation is
`stdlib/special/caputo.sio::mittag_leffler_e_alpha`.  Before this lane,
that function used the direct power series for every real argument.  For
large negative inputs, for example `alpha = 0.7`, `z = -50`, the direct
series forms huge alternating powers and relies on cancellation that is not
available in f64 arithmetic.  D.8 therefore observed catastrophic overflow
and sign loss:

```text
alpha=0.7, z=-50
Sounio before: -1.222688e+89
D.8 reference: 0.022762834959846902
```

## Fix

For `0 < alpha < 1` and `z <= -5`, the implementation now avoids the direct
series.  It routes to a stable negative-real branch based on the completely
monotone density after the substitution `s = r^alpha`:

```text
E_alpha(-x^alpha)
  = sin(pi alpha)/(pi alpha)
    * integral_0^infinity exp(-x s^(1/alpha))
      / (s^2 + 2s cos(pi alpha) + 1) ds
```

The substitution removes the endpoint singularity and the implementation
uses compensated Simpson summation on `[0, 4]`, which is sufficient for the
D.8 `x >= 5` negative-real grid.  The special case `alpha = 0.5` uses the
large-negative asymptotic for `exp(x^2) erfc(x)` so that the classical
`E_{1/2}(-x)` check remains stable.

The existing direct series remains unchanged for `z > -5`, including the D.6
fractional PINN operating range.

## Focused Results

> **Reproduction commands corrected (2026-09-13).** This replaces an engine-dependency note
> dated 2026-08-17, which found the recorded command
> `SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 bin/souc run <test>` failing with
> `error: no main`. The cause: `bin/souc` execs that override with its arguments unchanged
> (it already did at the merge that added this file, `bebd78d74c`), so lean_single never
> reached a real source (a current-source lean_single shows why: it opens `run`, which does not
> exist, as a 0-byte source). The same `error: no main` was measured again on 2026-09-13 with
> `bin/souc-linux-x86_64`, before `bin/souc` began refusing the form. How the recorded values were
> originally produced cannot be established from this repository's history. `bin/souc` now refuses the
> form (exit 64). The commands below use the
> lean_single ELF's raw `<source.sio> <output>` interface. Run them from the repository root:
> lean_single resolves stdlib imports relative to the working directory. The pinned binary
> (sha256 `3cbea2b4…`) is no longer in the repository; the re-run used `bin/souc-linux-x86_64`,
> sha256 `a63ca2c960183aafcdca56e57a0c2da88b5a2005db9df5c4f2dc6a6434b8a694`.
> Re-run 2026-09-13: lean_single printed `ML_NEGATIVE_Z_FIX_PASS`, `D5_CAPUTO_SCALAR_PASS` and
> `D5_CAPUTO_TENSOR_PASS`. Only those markers were re-checked; the values in the table below are
> the original record. The default Madaros engine (`bin/souc run <test>` with no `SOUNIO_SOUC_BIN`
> set; `bin/madaros-linux-x86_64` sha256
> `7ba4e70b6fd3a073697c629b5f17c68041afe604ebf6bb630e7c78e11e31eedb`) also printed the first
> two, and rejected `test_caputo_l1_tape.sio` with `error[E037]` in `stdlib/tensor/ops.sio`.

Command (from the repository root):

```bash
cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/special/test_mittag_leffler_d8_grid.sio /tmp/ml_d8_grid.elf && chmod +x /tmp/ml_d8_grid.elf && /tmp/ml_d8_grid.elf
```

Result:

```text
ML_NEGATIVE_Z_FIX_PASS
```

Key values covered by the grid test:

| alpha | z | reference | status |
|---:|---:|---:|---|
| 0.7 | -50 | 0.022762834959846902 | PASS, relative error < 1e-8 |
| 0.7 | -100 | 0.013738939227872674 | PASS, relative error < 1e-8 |
| 0.5 | -10 | 0.056140992743822586 | PASS, relative error < 1e-8 |
| 0.8 | -50 | 0.010076920355356178 | PASS, relative error < 1e-8 |
| 0.9 | -50 | 0.003272422290569466 | PASS, relative error < 1e-8 |

The test also covers `alpha in {0.5, 0.7, 0.8, 0.9}` and
`z in {-10, -20, -50, -100}`.

## Regression Coverage

Available consolidated-main regressions:

```bash
cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/special/test_caputo_scalar.sio /tmp/caputo_scalar.elf && chmod +x /tmp/caputo_scalar.elf && /tmp/caputo_scalar.elf
# D5_CAPUTO_SCALAR_PASS

cd "$(git rev-parse --show-toplevel)" && bin/souc-lean-single-x86_64 tests/stdlib/tensor/test_caputo_l1_tape.sio /tmp/caputo_l1_tape.elf && chmod +x /tmp/caputo_l1_tape.elf && /tmp/caputo_l1_tape.elf
# D5_CAPUTO_TENSOR_PASS
```

The prompt references an older S1 `tests/stdlib/special/test_mittag_leffler.sio`
777-case audit.  That file is not present on consolidated `main`; this lane
therefore does not claim the historical `D2_MITTAG_LEFFLER_BROAD_PASS` marker.
The new D.8 grid test is the executable replacement for the large-negative
real blocker.

## Downstream Note

A temporary run of the D.8 Sounio emitter after this fix produced the expected
large-negative values and still emitted `D8_SOUNIO_CROSSVAL_EMIT_PASS`.  The
D.8 Python comparison script may still need a precision-output cleanup because
its current CSV parser receives Sounio values through six-decimal `print_f64`
formatting.

## Reproduction commands (2026-09-16)

The 3 `bin/souc-lean-single-x86_64` lines above run the lean_single seed (sha256
`9d7892132aa0a9cf839df4560bf968628dc30fda4af978a4efc4a99b4e8f89f5`, most recently changed by merge
commit `09aedffafa`) and include a `chmod +x` step. Until 2026-09-16 those lines invoked
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

- `tests/stdlib/special/test_mittag_leffler_d8_grid.sio` (ELF `/tmp/ml_d8_grid.elf`): stdout contained `ML_NEGATIVE_Z_FIX_PASS`
- `tests/stdlib/special/test_caputo_scalar.sio` (ELF `/tmp/caputo_scalar.elf`): stdout contained `D5_CAPUTO_SCALAR_PASS`
- `tests/stdlib/tensor/test_caputo_l1_tape.sio` (ELF `/tmp/caputo_l1_tape.elf`): stdout contained `D5_CAPUTO_TENSOR_PASS`

Any other line in the blocks above — expected-output markers, `rg` searches, gate scripts —
was not run and is not part of this comparison. The values recorded elsewhere in this note
were not re-derived.
