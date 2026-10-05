<!-- docs:meta
topic_id: repo.docs.audit.kaxi-ptx-golden-drop-not-mismatch-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.kaxi-ptx-golden-drop-not-mismatch-2026-08-19
-->

# kaxi_ptx 0/318 is DROP, not MISMATCH

> **Status**: measurement receipt | **Last validated**: 2026-08-19 | **Source**: `scripts/ci/kaxi_ptx_golden_gate.sh` live run, not a golden recapture

**Do not bisect `kaxi_to_ptx`.** The published 318/318 is a May receipt (#1921). The live gate is 0/318 because the emitter never starts. That is an instrument failure, not byte drift. `BLK-20260510-lane1-golden-drift` stays closed.

Criterion was written to `.scratch/` at 2026-08-19T00:12:00Z, before the first emit. `.scratch/` does not survive. This file is that criterion plus the receipt.

## Question

#1921 dated the 318/318 claim honestly and fixed nothing. The gate is not workflow-reachable, so 318 → 0 went unnoticed. The deciding question is the same one as the silent-XPAS lesson: is 0/318 a **regression** or an **instrument** that never measured?

cursor-2's budget row (`docs/audit/CI_GATE_BUDGET_2026-08-18.tsv` line 43, unread here as write) already said `0/318 PASS, 318 DROP rc=1` and marked `measured=yes`. DROP is not DIFF. If the 318 never produced PTX, the red is vacuous.

## Gate vocabulary (from the script)

`scripts/ci/kaxi_ptx_golden_gate.sh` lines 83–105. Population: 53 patterns × 6 modes = 318. Goldens at `tests/golden/kaxi_ptx/<mode>/<pattern>.{ptx,sha256,unsupported}`. Emit: `./bin/kretikos kaxi-emit-ptx <pattern> -o <tmp> --no-ptxas [mode-flag]`.

| Label | Condition | Meaning |
|---|---|---|
| PASS | rc=0 and nonempty tmp and sha256 == golden | byte-identical emit |
| DROP | expected supported and (rc≠0 or empty tmp) | combo never produced PTX |
| DIFF | nonempty tmp and sha256 ≠ golden | MISMATCH — the only byte-drift class |
| REGR | golden is `.unsupported` and rc=0 | was unsupported, now emits |
| MISS | no `.ptx` and no `.unsupported` | missing golden, not an emit verdict |

A single shared compile-driver abort that the gate labels DROP 318 times is still one instrument failure, not 318 independent emitter defects.

`ptxas` is irrelevant: the gate always passes `--no-ptxas`.

## Verdict

**INSTRUMENT. 0/318 is DROP, not DIFF.**

Measured 2026-08-19T00:12:01Z–00:13:56Z on this worktree. GPU driver + gate were byte-identical to `origin/main` (diff 0). Re-checked against `origin/main` `6bce611b7b` when this file was written.

`bin/kretikos kaxi-emit-ptx` compiles `self-hosted/gpu/kretikos_kaxi_to_ptx.sio` *before* any pattern/mode dispatch. That compile aborts:

```
error: unreadable import: self-hosted/gpu/erdos90_hc_smoke_emit.sio
```

The wrapper then prints `failed to compile K-AXI→PTX driver` and exits 1 with an empty output file. The gate calls that DROP.

Madaros `souc compile` (correct form, not bare `souc`) dies on the same hole: `unresolved import in authoritative closure: gpu::erdos90_hc_smoke_emit`.

Live probes, all rc=1, size=0, same stderr:

- `default/exit_only`
- `default/vec_add`
- `--epistemic`
- `--f32`

No combo reached `sha256sum`. DIFF = 0. PASS = 0. The full 318 was not re-run: they share one driver compile. Replaying the gate would be 318 copies of the same parse failure (that is the 80 s on cursor-2's row).

## What is missing / what is not

| Candidate | Status |
|---|---|
| `self-hosted/gpu/erdos90_hc_smoke_emit.sio` | **absent on `origin/main`** |
| `tests/golden/kaxi_ptx/` | present (318 `.sha256`, 0 `.unsupported`) |
| `bin/kretikos` | present (wrapper) |
| `self-hosted/gpu/kaxi_to_ptx.sio` | present; last main edit `2828d89c27` (2026-06-17) |
| `ptxas` | absent, not the cause |

The missing module is imported on `origin/main` by `kretikos_kaxi_to_ptx.sio:18` and `kretikos_emit_kaxi.sio:34` (`5c85634f3d`, 2026-06-15). The `use` is live: both drivers call `kaxi_emit_erdos90_hc_smoke_build` for pattern `erdos90_hc_smoke` (not one of the 318 goldens). The file exists only on the off-main snapshot `b8828063d6` (2026-06-17, "chore: snapshot erdos and compiler WIP artifacts").

## Owner

Not `kaxi_to_ptx`. Not `BLK-20260510-lane1-golden-drift` (that blocker was 38 commits of emitter drift with PTX actually produced and sha different). The owner is the dangling `erdos90_hc_smoke_emit` import: the instrument cannot emit.

This lane did not restore the WIP file, did not recapture goldens, and did not change the gate.

## `measured=` on the budget row

For the published 318/318 **byte-identity** claim: **`measured=no`**.

The gate ran (cursor-2: 80 s, rc=1 on `665f412bd9`). Those 80 seconds are the cost of **not** measuring PTX. Request sent on the coord bus (`msg-1787098477-1355405-5593`, lane `claim-root-budget-20260818`) asking cursor-2 to flip `measured=yes` → `no`. Her TSV was not edited from this lane.

## Consequence for the 11 claim-numeric gates

cursor-2's own table, family `claim-numeric`, unread here as write. On her polarity before this correction:

| Class | n | gates |
|---|--:|---|
| `measured=yes` | 7 | `fo_pk_struct_auc_thalf_driver`, `functor_f_g2_covariance`, **`kaxi_ptx_golden`**, `san_imagenet_fpga_dl380`, `sedenion_phi_injectivity`, `sounio_direct_driver_support`, `windows_pe_smoke` |
| `measured=no` | 4 | `kretikos_kaxi_lse8` (no ptxas), `kretikos_kaxi_phase_y` (no libcuda), `kretikos_kaxi_sinkhorn16` (no ptxas), `native_v2_cpu_compiler_umbrella` (180 s cap) |

Flipping kaxi `measured=yes` → `no` moves the 11 from **7 measuring / 4 empty** to **6 / 5**. Almost half of the published-number gates did not measure the number they advertise. This file does not re-audit the other ten; it only reclassifies kaxi.

## Not done

- No fix of the dangling import.
- No golden recapture.
- No bisect.
- No edit of `docs/audit/CI_GATE_BUDGET_2026-08-18.tsv`.
