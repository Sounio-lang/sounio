# Phase-D Hadwiger-Nelson campaign — cluster fan-out (384 CPU / ~1 TB)

Scales the **compute half** of the χ(plane) ≥ 6 search: fan many candidate
unit-distance graphs across the Slurm cluster as a SLURM array, SAT-testing
5-colourability of each. Built on the sound, validated Phase-D engine.

## Honest framing (unchanged)
- No 6-chromatic unit-distance graph is known; brute enlargement of the de Grey ring is
  *known* to stay 5-colourable. **Expected outcome of this sweep: all negative.**
- The cluster supplies compute (many candidates in parallel; hard near-threshold instances
  cracked faster). It does **not** supply the **ML/FunSearch generator** — the variants
  here are systematic enlargements/rotations, the known-negative direction. A genuine
  χ≥6 attempt needs a generator proposing *non-de-Grey* constructions; this fan-out is the
  verified **evaluator** such a generator would call.
- **Any `kcolorable:false` is only a CANDIDATE.** It is NOT a result until its CNF/DRAT is
  verified by `drat-trim` **and** re-proved by the Lean `bv_decide` pipeline (cf.
  `formal/lean4/SounioHeule510NotColorable.lean`). The worker keeps the proof artifacts on
  a hit and the submit script flags them; never claim χ≥6 without that chain.

## Pieces
- `scripts/research/cnp_campaign_worker.py` — self-contained, exact: closure under the
  certified unit directions of HeuleGraph510, COMPLETE edges (floor-grid + float screen,
  validated == all-exact: 11553 @ 2600), 5-colourability via a SAT binary (kissat/cadical,
  emits DRAT) or python-sat. One candidate `(size, rcap, variant, colors)` per call.
- `submit-cnp-campaign.sh` — stages worker + the certified `510.vtx/510.edge` (from
  Heule's CNP-SAT) to `/orangefs`, writes a SLURM array (one config per task), submits via
  the login pod. Edit the `CONFIGS` grid to taste.

## How to submit
This session could **not** submit: the BeagleCockpit MCP tools (`cockpit_submit_job`, the
canonical path) were not loaded, and direct `kubectl exec` into the shared login pod is
gated. Submit one of two ways:
1. **Cockpit MCP (canonical)** — in a fresh session with the cockpit tools, submit
   `run.sbatch` / this job (never hand-roll YAML).
2. **Run the script** — from a shell with login-pod access:
   `bash slurm-jobs/chromatic-campaign/submit-cnp-campaign.sh`
   (set `PARTITION`/`NODELIST` to a real CPU partition first; verify with `sinfo`).

**Untested from the workspace** (no cluster access this session): syntax-checked and the
worker is validated locally, but the sbatch/login-pod path needs a live cluster to confirm.

## Next (v2, when a hard instance dominates)
Per-instance **cube-and-conquer** (march_cu split → parallel cadical leaves) for the
near-threshold sizes — the 20k local solve ran >50 min single-core, exactly the case
cube-and-conquer across many cores is built for.
