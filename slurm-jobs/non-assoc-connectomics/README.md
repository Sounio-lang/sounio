# Non-Associative Connectomics — SLURM Submission

Cluster-side execution for the non-associative epistemic connectomics
experiment. See `experiments/non_assoc_connectomics/PROTOCOL.md` (Phase 1)
and `PROTOCOL_PHASE2.md` (Phase 2) for the frozen scientific design.

## Prerequisites

### Cluster access

This pattern submits via `kubectl -n slurm-pilot exec <login-pod> -- sbatch`.
Verified from the Sounio workspace with service account
`system:serviceaccount:beagle:default` (has `pods/exec create`).

Login pod name (as of 2026-04-13):
`slurm-pilot-login-slinky-5ffb48b759-5c84k`. If that changes, the submit
scripts auto-resolve via `kubectl get pods -l app.kubernetes.io/name=login`.

### ABIDE frames.bin

**Phase 1 pilot AND Phase 2 require** a pre-built `frames.bin` at
`/orangefs/training/sounio/abide-data/frames.bin` with schema:

```
Header:     [n_asd: i64_LE, n_td: i64_LE]               (16 bytes)
Per-subj:   7 eigenvectors × 200 ROIs × f64_LE          (11200 bytes/subj)
Order:      all ASD subjects first, then all TD subjects
```

If only the 64-feature manifests are present (e.g. `abide_roi_manifest.tsv`
from a different pipeline), re-run `scripts/research/abide_preprocess.py`
on the cluster to build frames.bin. The script requires `numpy`, `scipy`,
`pandas` and network access to S3 (public ABIDE bucket).

For Phase 2 Experiment C, `frames.bin` must be **v2** (8 eigenvectors, not
7). Update `abide_preprocess.py:57` to export `eigenvectors[:, 1:9]` then
re-run. `zero_divisor_full.sio` detects the schema and aborts if v1.

## Scripts

### submit_phase1_pilot.sh

10-subject pilot (5 ASD + 5 TD). Runs `associator_field.sio` — Phase 1's
30-ROI code, scaled on the real ABIDE eigenvectors. Completes in ≤ 10 min.

```bash
cd /workspace/sounio
bash slurm-jobs/non-assoc-connectomics/submit_phase1_pilot.sh
```

Collect results when SLURM reports complete:

```bash
LOGIN_POD=$(kubectl -n slurm-pilot get pods -l app.kubernetes.io/name=login \
  --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
kubectl -n slurm-pilot exec "$LOGIN_POD" -- \
  cat /orangefs/training/sounio/phase1-pilot-results/<RUN_ID>/obs/pilot.csv \
  > /tmp/phase1_pilot.csv
python3 experiments/non_assoc_connectomics/analysis.py /tmp/phase1_pilot.csv
```

The analysis reports Cohen's d and 95% bootstrap CI. **Gate for Phase 2**:
`|d| > 0.15 AND CI excludes zero` → proceed. Otherwise revise per
`PROTOCOL_PHASE2.md § Post-freeze amendments`.

### submit_phase2_full.sh

Full-cohort Phase 2. ~4 hours wall at parallelism 128. Refuses to submit
unless `PHASE1_GATE=1` is set (must acknowledge Phase 1 pilot passed).

```bash
cd /workspace/sounio
PHASE1_GATE=1 bash slurm-jobs/non-assoc-connectomics/submit_phase2_full.sh
```

Submits TWO job arrays with dependency:
1. Main: array 0..n_total-1 at parallelism 128. Per subject: observed
   p95 (`associator_field_full.sio`) + 1000 null permutations
   (`null_permutations.sio`). ~30 min per subject.
2. ZD: single task after main completes. All-subjects Experiment C
   (`zero_divisor_full.sio`). ~30 seconds total.

Collect + analyze:

```bash
LOGIN_POD=...  # as above
kubectl -n slurm-pilot exec "$LOGIN_POD" -- \
  tar -C /orangefs/training/sounio/phase2-results/<RUN_ID> -cf - . \
  | tar -C /tmp/phase2 -xf -

python3 experiments/non_assoc_connectomics/analysis_phase2.py \
  --obs  /tmp/phase2/obs  \
  --null /tmp/phase2/null \
  --zd   /tmp/phase2/zd   \
  --pheno /orangefs/training/sounio/abide-data/phenotypic.csv \
  --out  artifacts/research/non_assoc_connectomics_phase2
```

Output: `artifacts/research/non_assoc_connectomics_phase2.{json,png}` with
Cohen's d, 95% CI, KS test, bootstrap p, Holm-Bonferroni rejection decisions.

## Resource profile

| Step | Array size | Parallelism | Per-task | Wall |
|------|------------|-------------|----------|------|
| Phase 1 pilot | 1 | — | 4 CPU, 4 GiB | ≤ 10 min |
| Phase 2 main | n ≈ 1034 | 128 | 4 CPU, 8 GiB | ~4 hr |
| Phase 2 ZD | 1 | — | 2 CPU, 4 GiB | ~30 s |

CPU-only throughout (Experiment B is scalar polynomial arithmetic; GPU
offload is Phase 3 scale-up territory if n grows).

## Monitoring

```bash
LOGIN_POD=...
kubectl -n slurm-pilot exec "$LOGIN_POD" -- squeue -u $USER
kubectl -n slurm-pilot exec "$LOGIN_POD" -- \
  tail -f /orangefs/training/sounio/phase2-results/<RUN_ID>/logs/<JOB_ID>_0.out
```

## Rollback

```bash
LOGIN_POD=...
kubectl -n slurm-pilot exec "$LOGIN_POD" -- scancel <JOB_ID>
kubectl -n slurm-pilot exec "$LOGIN_POD" -- \
  rm -rf /orangefs/training/sounio/non-assoc-runs/<RUN_ID> \
         /orangefs/training/sounio/phase2-results/<RUN_ID>
```
