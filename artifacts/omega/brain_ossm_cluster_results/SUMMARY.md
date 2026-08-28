# Brain O-SSM Cluster Results Summary

> ⚠️ **CORRECTION NOTICE (2026-07-16) — read before citing any "O-SSM advantage" below.**
> These Slurm runs (job-52 / job-53, 2026-04-08) were executed with an octonion
> multiplication table that had a sign error in `e2·e5` (`-a2*b5+a5*b2` instead of
> `+a2*b5-a5*b2`). That table fails alternativity and composition — it is **not** the
> octonion algebra — so any non-associativity numbers here (e.g. Fractal-G2 v3, the
> "non-assoc gap", the "small overall O-SSM advantage" reading) are artifacts of the broken
> table, not octonion results. The run-log figures are **left unchanged** as a historical
> record; they have **not** been re-run on the corrected algebra and should not be treated
> as current. The repo's corrected record is negative on non-associativity (A/B re-audit
> NEGATIVE, ABIDE associator null, `zd_bptt` ZD-advantage → +0.00pp). Corpus fix: PR #1024.

This directory contains two real Slurm runs copied from OrangeFS into the workspace:

- Baseline multi-seed run: `job-52.log`
- Fixed-seed replay with `SEED=42` override in a temporary staged snapshot: `job-53.log`

## Run Metadata

| Run | Job | Host | Run root |
| --- | --- | --- | --- |
| Baseline | `52` | `gpuorangefs-r770-proxmox` | `/orangefs/training/sounio/brain-ossm-runs/brain-ossm-prebuilt-20260408T122242Z` |
| SEED=42 | `53` | `gpuorangefs-r770-proxmox` | `/orangefs/training/sounio/brain-ossm-runs/brain-ossm-seed42-20260408T170929Z` |

## Fractal-G2 v3

| Metric | Baseline | SEED=42 |
| --- | --- | --- |
| O-SSM overall | `11.760000±0.640812%` | `9.400000±0.000010%` |
| H-SSM overall | `9.680000±0.123126%` | `9.000000±0.000000%` |
| Overall gap | `+2.080000pp` | `+0.400000pp` |
| O-SSM non-assoc | `1.462687±0.376285%` | `0.623053±0.000000%` |
| H-SSM non-assoc | `0.208955±0.084957%` | `0.934579±0.000000%` |
| Non-assoc gap | `+1.253731pp` | `-0.311526pp` |
| ListOps mean | `10.733333% vs 10.666667%` | `12.400000% vs 12.400000%` |

Reading:
- The baseline multi-seed run supports the stronger Fractal-G2 claim.
- The single-seed replay at `42` still keeps a small overall O-SSM advantage, but it loses the non-associative subset advantage.

## Brain Classifier

| Metric | Baseline | SEED=42 |
| --- | --- | --- |
| O-SSM range | `23.333333%` to `40.000000%` | fixed at `46.666667%` |
| H-SSM range | `33.333333%` to `90.000000%` | fixed at `66.666667%` |
| Associator norm | nonzero in all O-SSM seeds | `2.783339` |

Reading:
- The negative result remains: H-SSM stays ahead on the toy brain classifier.
- `SEED=42` gives O-SSM a somewhat better single-seed outcome than the weaker baseline seeds, but not enough to beat H-SSM.

## Native Algebra

| Task | Baseline O-SSM | Baseline H-SSM | SEED=42 O-SSM | SEED=42 H-SSM |
| --- | --- | --- | --- | --- |
| ListOps | `28.700000%` | `32.900000%` | `25.700000%` | `24.000000%` |
| Bracket | `56.500000%` | `67.500000%` | `71.800000%` | `59.800000%` |

Reading:
- The baseline favored H-SSM.
- `SEED=42` flips both tasks toward O-SSM, especially `Bracket`.
- This is a useful reminder that single-seed wins are not enough for the main claim.

## Multi-Head Unit Benchmark

| Metric | Baseline | SEED=42 |
| --- | --- | --- |
| MH-Oct ListOps | `27.100000%` | `32.800000%` |
| MH-Quat ListOps | `33.800000%` | `35.200000%` |
| MH-Dense ListOps | `34.400000%` | `39.300000%` |
| MH-Diag ListOps | `54.000000%` | `43.800000%` |
| Non-assoc gap `(MH-Oct - MH-Quat)` | `-2.985075pp` | `-1.246106pp` |

Reading:
- `SEED=42` narrows the Oct-vs-Quat gap but does not reverse it.
- The diagonal baseline is still very competitive.

## Direct Associativity Probe

| Metric | Baseline | SEED=42 |
| --- | --- | --- |
| O-SSM non-assoc accuracy | `5.373134%` | `4.361371%` |
| H-SSM non-assoc accuracy | `5.373134%` | `6.230530%` |
| Gap `(O-SSM - H-SSM)` | `+0.000000pp` | `-1.869159pp` |

Reading:
- This benchmark does not show a robust non-associative win.
- The result strengthens the current narrative that the science is novel, but the killer application still has to come from real temporal data.

## Recommended Interpretation

- Use the baseline run as the default reference for the current cluster benchmark suite.
- Treat `SEED=42` as a deterministic spot check, not as the headline result.
- For paper-grade claims, keep emphasizing multi-seed behavior and move the decisive test to ABIDE-scale temporal data.

## Files

- Baseline outputs: `*_results.txt`
- Fixed-seed outputs: `*_seed42_results.txt`
- Slurm logs: `job-52.log`, `job-53.log`
