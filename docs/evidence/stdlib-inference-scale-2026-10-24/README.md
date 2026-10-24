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

## Current-main compiler refresh

`compiler-main29ce/` records source commit
`29ce2abb281e7842881f7570a47934f890de70d2`, Slurm16873 and the fresh native
compiler hash `8c630110db238a38ad345fd782585d525b3a8d438276795cf5b253373824d31a`.
The build completed in7:18.69 wall time, with735920KiB maximum RSS. Two positive
native borrowing/callback probes completed; seven negative borrowing/private-call
probes were rejected as expected. The negative rows retain actual exit1, rather
than a transformed success code. This compiler differs from the earlier
`ee0f341e...` controlled-pilot compiler. Its focused compiler probes do not
establish scientific integration acceptance; that requires separate gate receipts.

`benchmark-drivers/` preserves the original before/after B199 drivers, the two
paired B31 pilot drivers, and the original transformation recipe. The B199 after
driver is preserved input, not evidence that an after B199 benchmark was executed.

## Complete numerical correctness gate

`functional-correctness4f421/` records Slurm16870 on the immutable source commit
`4f4210232bd57e3058651967e158d09c916b5508`, using the controlled `ee0f341e...`
compiler. Its final STATUS is0: native fixtures, independent C++ concordance,
mutant rejection and final source/compiler hash checks passed. This repeats the
previous gate after correcting the missing quadmath link and annotation order.
The complete phase table is readable directly; compressed receipts retain stdout,
stderr, resource measurements, executable identities and hash verifications.
This gate covers that numerical branch, including normalization and legacy
rendering; it does not include later overlay, tracing or marker-size extensions,
and is not B199 qualification. The separate combined-current-main gate remains
its own acceptance boundary.

## Combined inference, normalization and rendering gate

`combined-current-main/` records Slurm16875: final STATUS0 on the integrated
source archived at commit prefix761c2f71ca and compiled by the freshly rebuilt
main29ce compiler8c630110.... The actual input manifest, phase table and compressed
receipts bind that snapshot. All recorded native fixtures, C++ concordances,
negative controls and final source/compiler hash checks passed, including the
streaming reducer, layered scientific diagnostics, canonical bitmap rendering,
count/ODE workflows and the new numerical/normalization contracts. This is the
combined functional gate; later tracing and marker-sizing code require their
own renewed acceptance. Neither this gate nor the earlier B31 performance pilots
establish calibration or the B1992x target.

## Record owner, validated instrumentation and readable campaign figure

`compiler-record-owner/` retains a discriminating compiler regression and its
fix. A nested-if binding lost the nominal record owner; inline field projection
could read an adjacent boolean as alpha, while a typed borrowed helper read the
correct unchanged record. Saving the incorrect inline alpha was a diagnostic,
**not a scientific fix**. The recursive owner join now requires compatible known
layouts from both arms. Original failing witnesses pass with compiler92d2108e;
Slurm16899 runs the original unannotated traced/untraced B2 programs and checks
both against independent C++. Slurm16913 passes the composed current-source f64
compiler gate, including the permanent owner witnesses and incompatible-type
rejection. Failed invocations and superseded diagnostics remain labelled.

The ten original work metrics count actual fit/IRLS/QR/profile/rank and full-PMF
API callsites. They are not a flop count or an inferred sampler cost. The B2
original passes equality off/on, resets, bounded failures, snapshots and actual
RNG checks. OpsTraced16903 and final-summary16910 validate the traced generator,
its work rows, unavailable bounded traces, old-receipt rejection and unchanged
legacy generated datasets. They do not manufacture a passing final CENTRAL
receipt or constitute a completed qualification campaign.

`campaign-report/` binds the frozen original seven READY first batches, missing
cell masks, the portable raw asset, native scientific coordinates, SVG and actual
Canvas PNG. Count classifications and elapsed-time medians are independently
checked before projection. The graphic states partial/unassessed status and
includes early termination in unavailable-class times. The opt-in sized scatter
API accepts square radii1..4; the report uses radius3. The shared conservative
edge guard fixes valid marker footprints previously discarded before integer
projection. The failing-before witness, exact pixel/geometry controls, unchanged
radius1 legacy replay, zero-output preflight failures and three report mutants
are retained. This is execution plus agent visual inspection, not human approval
or calibration.
