# Pilot Notes — Phase 1

**Date**: 2026-04-13
**Mode**: Real ABIDE-I data, n = 10 (5 ASD + 5 TD)

## Gate status (updated 2026-04-13 with real ABIDE data)

| Gate | Status | Value |
|------|--------|-------|
| `sim_ground_truth.sio` monotone α-recovery | PASS (2026-04-12) | p95(α=0)=0.000, p95(α=0.5)=3.61, p95(α=1.0)=74.14 |
| `associator_field.sio` real-ABIDE pilot | **RAN** | 10 subjects processed from S3, CSV emitted |
| `zero_divisor_proximity.sio` stub | PASS | d² = 0.89 on synthetic |
| `analysis.py` Cohen's d + CI | PASS | d = −0.934, 95% CI = [−2.29, −0.17] |
| β⁴ compiler variance | PASS | octonion associator Var matches GUM truth exactly (commit `f5167f99`) |

## Real ABIDE-I pilot — observed statistic

```
n_asd = 5, n_td = 5
mean(p95 | ASD) = 0.00146
mean(p95 | TD)  = 0.01767
Cohen's d (ASD − TD) = −0.934
95% bootstrap CI      = [−2.29, −0.17]
CI crosses zero       = False
KS two-sample D       = 0.600,  p = 0.357
```

**Subjects processed** (all 10 from site CMU_a — see site-confound caveat below):
```
ASD: CMU_a_0050642, CMU_a_0050646, CMU_a_0050647, CMU_a_0050649, CMU_a_0050653
TD:  CMU_a_0050656, CMU_a_0050659, CMU_a_0050660, CMU_a_0050663, CMU_a_0050664
```

## Interpretation — honest

### What passed

- **Pipeline works end-to-end on real ABIDE data.** `abide_preprocess.py`-equivalent Python fetched 10 subject .1D files from `s3.amazonaws.com/fcp-indi`, extracted Laplacian eigenvectors, serialized to `frames.bin`. `associator_field.sio` loaded the frames, computed per-subject p95 associator norm² over C(30,3)=4060 triples, emitted CSV. `analysis.py` computed Cohen's d + 10k-bootstrap 95% CI.
- **|d| = 0.934 > 0.15** and **95% CI excludes zero** — the numerical form of the `PROTOCOL_PHASE2.md § Precondition` gate passes.

### What doesn't pass — why Phase 2 should NOT be triggered on this pilot alone

1. **Effect direction is opposite to H1.** PROTOCOL.md and PROTOCOL_PHASE2.md hypothesize that **ASD > TD** on the p95 associator statistic. Observed here: ASD < TD (d = −0.934). Proceeding with the original one-sided H1 on Phase 2 would be confirmation-bias territory. The correct response is one of:
   - Amend the protocol (per PROTOCOL_PHASE2.md § Post-freeze amendments) to a two-sided H1 ("ASD ≠ TD"), document the reason, re-freeze before Phase 2.
   - Abandon H1 and designate the observed direction as an exploratory finding, requiring independent replication to claim.

2. **Site-confounded pilot.** All 10 subjects are from the CMU_a site — a direct consequence of selecting by FILE_ID lexicographic order. ABIDE has 17 sites; subjects are not randomly distributed across ASD/TD within a site, and CMU_a has particular scanner/protocol characteristics. Any group-level effect at n=10 from a single site is plausibly a site artifact.
   - Mitigation: refit with site-stratified sampling (one or two subjects per site, balanced on DX_GROUP).
   - Phase 2's covariate regression (site dummies) handles this at the full-cohort level but is underpowered for n=10.

3. **Sample size underpowered for direction inference.** At n=5 vs n=5, the 95% CI [−2.29, −0.17] spans an order of magnitude. The true d could be anywhere from "small" to "very large." Effect-sign inference from n=10 is weak.

### Recommendation

**Do NOT run `submit_phase2_full.sh PHASE1_GATE=1` based on this pilot.**

Instead:
1. Fetch a **site-stratified** pilot — one ASD + one TD from each of 10 ABIDE-I sites, 20 subjects total.
2. Re-run the pipeline; re-compute Cohen's d + CI on the stratified pilot.
3. If the **direction flips or the CI includes zero on the stratified sample**, the CMU_a result was artifact — re-specify H1 or abandon.
4. If the stratified pilot still shows **|d| > 0.15 + CI excludes zero with consistent sign**, amend PROTOCOL_PHASE2 to two-sided, re-freeze, proceed to Phase 2.

## Infrastructure notes

### Local pilot preprocessor (2026-04-13)

`/tmp/abide_pilot/fetch_pilot.py` is a throwaway one-off that does what the
cluster couldn't: fetches ABIDE subjects from S3 directly in this workspace
(which has numpy + scipy + pandas + SSL trust). Not committed to the repo
because it's a workaround for the cluster's missing Python deps + SSL
verify failure (see `PHASE2_STATUS.md § What's blocked`).

Canonical approach for Phase 2 is still: fix the cluster's Python
environment (option 2 in PHASE2_STATUS), then use
`scripts/research/abide_preprocess.py` as intended. Do not scale the
local fetcher to 1,034 subjects.

### frames.bin location

Local: `/workspace/sounio/artifacts/research/abide/frames.bin` (112 KB, 10 subjects).

To push to OrangeFS for future cluster pilots:
```bash
kubectl -n slurm-pilot cp artifacts/research/abide/frames.bin \
  slurm-pilot-login-slinky-<hash>:/orangefs/training/sounio/abide-data/frames.bin.pilot10
```

(Keep the `.pilot10` suffix to avoid overwriting a future full-cohort frames.bin.)

## Commits

- `b7506fd3` — Phase 1 synthetic pipeline
- `88672cf3` — Phase 2 code
- `d01c4746` — Phase 2 SLURM submit scripts
- `6b14c20a` — PHASE2_STATUS.md
- (this commit) — PILOT_NOTES.md update + pilot artifacts
