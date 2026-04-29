# Phase 2 Status — 2026-04-13 (end-of-session)

## What's landed this session

All Phase 2 code artifacts per `PROTOCOL_PHASE2.md`:

| Commit | Artifact | Purpose |
|--------|----------|---------|
| `88672cf3` | `associator_field_full.sio` | 200-ROI per-subject associator field + quickselect p95 |
| `88672cf3` | `null_permutations.sio` | 1000 S_8 channel permutations per subject |
| `88672cf3` | `zero_divisor_full.sio` | Experiment C: 168-class sedenion ZD distance |
| `88672cf3` | `preprocess_zd_supports.py` | JSON → `sedenion_zd_168.bin` one-shot |
| `88672cf3` | `analysis_phase2.py` | covariate regression + Holm-Bonferroni + figure |
| `88672cf3` | `sedenion_zd_168.bin` | 168 pair records, 8072 bytes |
| `d01c4746` | `submit_phase1_pilot.sh` | SLURM submitter for 10-subject pilot |
| `d01c4746` | `submit_phase2_full.sh` | SLURM submitter for 1034-subject full cohort |
| `d01c4746` | `slurm-jobs/non-assoc-connectomics/README.md` | operator-facing docs |

**Verification in-session (synthetic data, since real data absent):**
- `associator_field_full.sio`: 200-node α=0.5 synthetic → mean=8.13, p95=31.31. Quickselect correct on 1.3M f64 array.
- `null_permutations.sio` (5 perms): p95 ∈ [30.89, 36.91] for α=0.5 synthetic. Non-identity shuffles don't collapse to observed.
- `zero_divisor_full.sio` with 16D decaying synthetic feature → d_ZD = 0.922. Algorithm matches hand-derived formula.
- `preprocess_zd_supports.py` → 168 pairs, 8072 bytes.
- `analysis_phase2.py` imports cleanly (numpy + optional scipy).

## What's blocked

### Phase 1 real pilot — blocked on `frames.bin`

`/orangefs/training/sounio/abide-data/` currently holds only four 64-feature manifest TSVs from a different pipeline. The `frames.bin` schema our code expects (7-eigenvector × 200-ROI, produced by `scripts/research/abide_preprocess.py`) does not exist.

`submit_phase1_pilot.sh` correctly detects and refuses to submit:
```
frames.bin missing at /orangefs/training/sounio/abide-data/frames.bin (size=0)
run scripts/research/abide_preprocess.py on cluster first
```

### `abide_preprocess.py` cannot trivially run on current cluster

Probed 2026-04-13:
- **Login pod** (`slurm-pilot-login-slinky-*`): `/usr/bin/python3` present, but `numpy`, `scipy`, `pandas` not installed.
- **Worker nodes** (`cpuops-t560-proxmox`, `gpuorangefs-r770-proxmox`): same — no numpy/scipy/pandas.
- **Network from worker**: SSL cert verification fails for `https://s3.amazonaws.com` (`CERTIFICATE_VERIFY_FAILED` on ABIDE public bucket). Needs `ca-certificates` package or a `--trust-store` shim.

Running `abide_preprocess.py` needs one of:

1. **Container image with deps baked in** — the `infra/beagle-sounio/Dockerfile` already has pandas, numpy, scipy pinned. But the current submission pattern stages a repo snapshot into OrangeFS, not a container. Adding container-runtime SLURM would be a cluster-side change (pyxis/enroot or similar).
2. **Pre-install deps to `/orangefs/training/sounio/python-userbase/`** — shared user-base on OrangeFS so `PYTHONUSERBASE` env var works across nodes. Requires a one-time `pip install --user numpy scipy pandas` with cert override, from a node that CAN install.
3. **Push frames.bin pre-built from elsewhere** — Demetrios's canonical dev machine (`/home/demetrios/RustroverProjects/sounio`) may already have a frames.bin. `kubectl cp` or `scp` it to OrangeFS.

Option 2 is cleanest for reproducibility; option 3 is fastest for getting Phase 1 running.

## Next session — ordered next steps

1. **Get frames.bin on OrangeFS.** Pick option from above (recommend option 2). Validate schema: `od -An -tld -N16 /orangefs/training/sounio/abide-data/frames.bin` should show two integers ~450 and ~580 (n_asd, n_td).
2. **Run `submit_phase1_pilot.sh`** — ~10 min wall.
3. **Collect pilot.csv; run `analysis.py` locally.** Record Cohen's d + 95% CI.
4. **Gate decision** per `PROTOCOL_PHASE2.md § Precondition`:
   - `|d| > 0.15` AND CI excludes 0 → **proceed** to step 5.
   - Otherwise → halt, update `PILOT_NOTES.md`, revise PROTOCOL_PHASE2 per § Post-freeze amendments.
5. **Run `submit_phase2_full.sh PHASE1_GATE=1`** — ~4 hr wall at parallelism 128.
6. **Collect + analyze** per README. Emit `artifacts/research/non_assoc_connectomics_phase2.{json,png}`.
7. **Update `PILOT_NOTES.md`** with observed effect sizes, CIs, Holm-Bonferroni verdicts.
8. **Decide Phase 3** per `PROTOCOL_PHASE2.md § Stopping rule`: writeup if ≥1 hypothesis rejected; null-result paper otherwise. Either way, commit + push.

## Open scientific-design questions (do not touch without logged amendment)

- **Frames.bin v1 (7 eigenvectors) vs v2 (8)**: Phase 1 and Experiment B use 7. Experiment C needs 8. If Phase 1 pilot runs on v1, Phase 2 Experiment B also runs on v1 (subset), but Experiment C cannot — `zero_divisor_full.sio` detects and aborts. Options:
  1. Run v1 preprocessor, then separately re-run with 8 eigs to a v2 file, use v1 for A/B and v2 for C. Cleanest scientifically but two preprocessing passes.
  2. Run v2 from the start; A.sio and B.sio already only use the first 7 eigs, so v2 is a compatible superset.
  Recommend (2) — update `abide_preprocess.py` to export 8 eigs before first run. One-line change at `eigenvectors[:, 1:8]` → `eigenvectors[:, 1:9]`.

- **Motion scrubbing threshold**: `PROTOCOL_PHASE2.md § Subject inclusion` specifies `mean_FD ≤ 0.5`. `abide_preprocess.py` currently doesn't filter; `analysis_phase2.py` needs a phenotypic join + filter step BEFORE the group test. TODO: add `--fd-threshold 0.5` CLI flag.

- **Phenotypic CSV location**: the download is in `abide_preprocess.py` line 73 as `phenotypic.csv` in CACHE dir. Ensure the Phase 2 submit scripts either pull this to a canonical path or analysis_phase2.py knows the right location.

## Commits

```
d01c4746  [infra] Phase 2 SLURM submit scripts + operator README
88672cf3  [experiments] Phase 2 code — full-cohort associator + null + ZD + analysis
```
