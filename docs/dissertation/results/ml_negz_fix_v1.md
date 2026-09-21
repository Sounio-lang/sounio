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

> **Engine dependency (verified 2026-08-17).** `SOUNIO_SOUC_BIN=<elf>` is documented in
> `bin/souc`'s own header comment as "exec that ELF directly (**raw, lean_single CLI**)" — the
> command below is silently pinned to the lean_single-family binary, undisclosed. Worse, as of
> today that exact command **fails**: `error: no main` (the raw lean_single CLI misparses this
> invocation shape). The plain, undocumented `bin/souc run tests/stdlib/special/test_mittag_leffler_d8_grid.sio`
> (no override, default Madaros) **succeeds** and prints the same `ML_NEGATIVE_Z_FIX_PASS`
> marker. The prescribed reproduction path below is both wrongly attributed and currently broken;
> use the plain default-engine command instead until this is corrected.

Command:

```bash
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 \
  bin/souc run tests/stdlib/special/test_mittag_leffler_d8_grid.sio
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
SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 \
  bin/souc run tests/stdlib/special/test_caputo_scalar.sio
# D5_CAPUTO_SCALAR_PASS

SOUNIO_SOUC_BIN=/workspace/sounio/bin/souc-linux-x86_64 \
  bin/souc run tests/stdlib/tensor/test_caputo_l1_tape.sio
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
