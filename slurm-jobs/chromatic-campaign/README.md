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

## Cube-and-conquer (parallel SAT per hard instance) — `cnp_cube.py`, `cnp_prep.py`
Single-threaded SAT is the wrong tool near the 5-colouring threshold (20k took ~50 min/1
core). Cube-and-conquer (Heule's method) splits ONE instance across many cores, soundly:
- **Cubes = the proper k-colourings of a clique** in the graph. Every proper colouring of
  G restricts to one, so the cubes cover all cases: **G is k-colourable ⇔ some (CNF+cube)
  is SAT; ALL cubes UNSAT ⇔ G is NOT k-colourable**. A triangle gives `k·(k-1)·(k-2)=60`
  cubes (k=5); each cube fixes 3 vertices, pruning the search hugely.
- **Soundness self-test** (`python3 cnp_cube.py selftest`): K₆ (χ=6) → all 60 cubes UNSAT
  → "NOT 5-colourable" ✓ ; K₃ → SAT ✓.
- **Pipeline:** `cnp_prep.py` builds the closure graph, dumps `edges_<n>.json`, finds a
  clique, writes `cubes_<n>.jsonl`; a SLURM array runs `cnp_cube.py solve` per cube;
  aggregate — any SAT ⇒ k-colourable, all UNSAT ⇒ χ≥k+1 (then verify with drat-trim +
  the Lean `bv_decide` pipeline before any claim).
- Run 20260525: 20k instance (150224 edges) → clique [84,90,96] → 60 cubes fanned across
  cpuops-t560 / 5860 / r770 — verdict in minutes vs ~50 min single-core.

Note: bounded by the cluster's ~20 allocatable CPUs (DYNAMIC CfgTRES caps), so cubes run
in ~3 waves; still a large speedup and the correct architecture if a χ≥6 candidate ever
needs its UNSAT cracked + certified.
