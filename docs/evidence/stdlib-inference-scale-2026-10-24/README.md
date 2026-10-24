<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-inference-scale-2026-10-24.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-inference-scale-2026-10-24.readme
-->

# Stdlib inference and scientific rendering: evidence ledger

This sprint extends the native Sounio computation-to-figure path from remote
`main`. C++23 is the independent numerical reference; Sounio is the production
implementation. Raw receipts retain failed and superseded attempts alongside
accepted focused runs. A successful focused run does not establish a full corpus
pass, merge acceptance, calibrated inference or a universal performance result.

## Implemented and measured

- **Paired NB2 likelihood:** checked local mean/dispersion increments preserve
  signed differences lost by subtracting separately rounded objectives. Three
  original unavailable campaign cases have independently negative fixed-point
  likelihood ratios. The implementation retains their sign; it does not clamp
  negative likelihood ratios to zero or omit failed replicates.
- **Nested profile bootstrap:** explicit finite embedding validation, separate
  warm-start storage and ordinary fit convergence checks coordinate null and
  alternative fits. Exact binary64 design reconstruction is an operational
  witness, not a proof of real model nesting or of prediction associativity.
  Full B2 seed/data concordance and fixed-alpha/incomplete-output controls are
  distinct from B199 qualification.
- **Genomics normalization:** reusable result storage, checked log status and
  strided views add explicit capacity/validation semantics. Three paired
  20000-by-512 trials show no measured speed or RSS improvement: median elapsed
  35.50s before and 35.69s after, median RSS 80384KiB for both.
- **Streaming line reduction:** caller-owned state commits only after a chunk
  passes preflight, can drain/reuse output buffers, and preserves global point
  indices. Independent geometry checks cover chunk boundaries and gap runs.
  The selected 2000001-point alternating-gap experiment took 1.79s/452608KiB
  before and 1.39s/62464KiB after. Both versions succeeded; these are measurements
  of one workload, not a general speedup guarantee or an old arena failure.
- **Scientific overlays:** opt-in two-layer panels render actual fit diagnostics
  through SVG and CPU Canvas, with shared series colors, clipping, text bounds
  and all-input preflight. Native numeric coordinates precede pixel projection;
  C++ checks geometry and rejects perturbed bindings. No pixel parity or human
  acceptance claim follows from structural checks alone.

## Performance and qualification boundaries

The selected n256/alpha0 B31/maxfits64 paired pilot improved from 154.51s to 138.28s
before seeded null-replica initialization. A later paired pilot comparing the
original path to the seeded null-replica candidate measured 156.49s to 128.64s.
Each pair preserved its recorded scientific result and RNG state. These are
separate pilot pairs, not an isolated causal comparison between the two new
candidates. B199 three-pair medians and the 2x target remain pending.

The historical B199 campaign retains unavailable cases and incomplete cells.
It cannot be pooled with B31 pilots or the new nested/traced implementation.
`assumptions_assessed`, `global_optimality_assessed` and
`finite_sample_validity_assessed` remain false. Neither seeded search nor a
stable local contrast certifies a global likelihood optimum.

## Custody and replay

- `inference/` contains immutable diagnostics, C++ concordance, normalization
  receipts and paired pilot receipts. Its own manifests bind the retained files.
- `reducer/` binds the streaming reducer experiment and independent controls.
- `viz/` binds the scientific overlay render/geometry controls.

Source and compiler identities, exit statuses, resource receipts and Slurm IDs
inside each bundle identify the actual execution. Generated ELFs and full source
archives remain on cluster scratch; they are deliberately excluded from Git.
Later integration, CI, tracing, campaign-report and qualification receipts must
be recorded separately rather than rewriting these historical results.
