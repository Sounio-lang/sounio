# molecular::lj — measured outcome of the two drivers

Madaros v0.80.0, Linux x86-64, 2026-11-13. Eight independently seeded runs per quantity; the error is the standard error over those eight. Both drivers were run to completion in the form committed here.

## `lj_nist_parity.sio` — exit 0, `LJ_NIST_PARITY_OK`, about 30 minutes

Reference: NIST Lennard-Jones benchmark tables, T* = 0.85, cutoff 3 with long-range corrections.

| Quantity | This code | NIST | Difference | 3 se |
|---|---|---|---|---|
| Liquid, N = 500, rho* = 0.776: U*/N | -5.511167 ± 0.002275 | -5.5121 ± 0.000455 | 0.000933 | 0.006959 |
| Liquid: p* | 0.007619 ± 0.012595 | 0.0067714 ± 0.00177 | 0.000848 | 0.038158 |
| Vapour, N = 500, rho* = 0.009: U*/N | -0.093750 ± 0.000267 | -0.093973 ± 0.000129 | 0.000223 | 0.000889 |
| Vapour: p* | 0.007173 ± 0.000004 | 0.0071641 ± 0.00000224 | 0.000009 | 0.000014 |
| Grand canonical, V = 512, ln z = -4.7788: rho* | 0.009640 ± 0.000005 | 0.0096404 ± 0.00000094 | 0.000001 | 0.000014 |

The liquid pressure is the weakest line: its standard error is larger than the value, so it shows only that the virial route is not grossly wrong. The vapour pressure sits at 2 se.

## `lj_widom_finite_size.sio` — exit 0, `LJ_WIDOM_FINITE_SIZE_OK`, about 30 minutes

| Check | This code | Predicted | Difference | Status |
|---|---|---|---|---|
| W1 dilute gas, -ln m against 2 B rho | -0.00268998 ± 0.00002731 | -0.00262885 | 2.2 se | holds, predicted before the run |
| W2 N = 500, ln z | -4.775871 ± 0.000142 | -4.776089 | 1.5 se | explained after the fact, not counted |
| W3 N = 250, ln z | -4.773849 ± 0.000158 | -4.774095 | 1.6 se | holds, predicted before the run |

The first prediction for W2 was the tabulated -4.7788 and it failed by 20 se. The driver header gives the two finite-size terms that were missing. W2 and W3 both sit about 0.0002 above the corrected model; two values at 1.5 se do not establish a further term.

## Found on the way

- A three-field tuple result from the exchange move exhausted the Madaros handle table after 4.2 million trials: a function region returned its garbage only at return, and the caller's loop had none. Fixed in the compiler, not here: loops now run in a region reset at every iteration end (ADR-010, loop regions; `madaros_loop_region_reclaim.sio` drives 10M such tuples through one loop). The move returns `(n, dU, kind)` again, kind +1 for an insertion and -1 for a deletion, so acceptance can be counted per move type.
- Per-move acceptance, ideal limit (`test_lj_core.sio`, z = 1e-3, V = 8000, 4e5 trials): detailed balance predicts both rates equal to sum_N Poisson(8; N) min(1, 8/(N+1)) = 0.860413. Measured 0.859432 (insertion) and 0.859768 (deletion): 0.1% under the ideal value, the size of the 2 B2 rho interaction shift, and 3.4e-4 apart.
- Cost on Madaros: about 80 ns per pair distance.
