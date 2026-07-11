# `scripts/research/` — campaign harness (NOT the Sounio science path)

**Boundary (read first).** Everything in this directory is the **research-campaign
harness**: Python/shell tooling that *prepares manifests, gates data quality,
orchestrates runs, parses model output, computes decision verdicts, and stamps
provenance*. It is **not** the Sounio science path.

The actual scientific model — the octonionic O-SSM dynamics, uncertainty
propagation, associator geometry — is implemented in Sounio and runs through
`bin/souc` on `.sio` sources (e.g. `examples/brain_ossm_abide.sio`,
`examples/cognitive_ossm/`). Per CLAUDE.md Principle 4, the science stays in
Sounio; these drivers only build the scaffolding around it and read the
`STATE_TRACE` / benchmark output the compiled `.sio` model emits.

Do not migrate numerical *science* into this directory. If a computation is a
scientific result, it belongs in `.sio`.

## What lives here

A NeuroDyn / Brain-O-SSM computational-psychiatry campaign:

- **Manifest & null generators** — synthetic paired non-associative trajectories,
  temporal-arrow / Fano-line / permutation-null manifests, with `SHA256SUMS`.
- **Diagnostics** — hidden-state separability, orientation / associator-vector
  readouts, Fano-orbit geometry, temporal-arrow polarity, all with permutation
  null envelopes.
- **Campaign gates** — ABIDE dynamic-FC (`abide_dynamic_fc_switching_*`) and
  ADHD-200 dimensional (`adhd200_*`) target → gate → decision chains, each
  emitting a bounded, no-promotion verdict.

Convention across drivers: `argparse` CLIs with `__main__`, deterministic
JSON + Markdown + `SHA256SUMS` outputs, and explicit **no-clinical-claim**
boundaries in the decision gates.

## Data & reproducibility

- **Public data only.** ADHD-200 access (`adhd200_s3_bootstrap.py`,
  `adhd200_data_access_audit.py`) uses the **anonymous public** `s3://fcp-indi`
  bucket / PCP mirrors over unauthenticated HTTPS. No credentials, no PHI, no
  private buckets are used or stored.
- **External inputs are not versioned here.** Drivers consume `STATE_TRACE`
  CSVs, ROI `.1D` timeseries, and manifests produced elsewhere; reproducibility
  depends on that upstream data provenance.
- **Cluster-coupled outliers.** A few scripts (`neurodyn_scale14x14_missing_roi_rerun.py`,
  `neurodyn_orangefs_not_required_gate.py`, the Slurm smokes) hardwire this pod's
  k8s / CephFS / partition defaults and invoke `kubectl` / `singularity` / `srun`;
  they are recorded for provenance but will not run off this cluster.

## Dependencies

Mostly Python standard library. Third-party: see [`requirements.txt`](requirements.txt)
(`numpy`, `scipy`). Install into a throwaway venv; nothing here is part of the
Sounio toolchain build.
