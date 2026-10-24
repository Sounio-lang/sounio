# Stdlib inference-to-document sprint — 2026-10-23

This ledger distinguishes executable availability, numerical concordance,
resource measurements and statistical calibration. New calibrated-inference
flags remain false. A finite bootstrap result is not a calibration certificate.

## Source and execution boundary

Work started from remote main and refreshed through
81466d023a4c855db4863b80661f6d02387a6bcd. Heavy execution runs under Slurm on
cpuops-t560-proxmox. The compiler was built from main
6869753568c92253dc133877ffc7da00519061f4 in job 16388:
SHA256 ee0f341e6e710d35c162084023cbb10ce72d2b7ca29726ea6e3a5994f080eded.
The later main refresh changes documentation/site files. Compiler build status
and native results are separate from GitHub checks and merge-queue execution.

Raw scratch custody is /scratch/sounio-stdlib-advanced-20261023 on t560.
Full source manifests, ELF identities, stdout, stderr and exit codes remain in
that custody. This folder retains compact reviewable receipts and actual images;
it intentionally does not duplicate compiler ELFs or historical source archives.
No shared storage quota or compiler arena limit was increased.

## Delivered contracts

- Profile-refit ML NB2 LR bootstrap refits beta and alpha for observed and
  simulated null/alternative fits. Positive range edges and failed replicas
  remain unavailable; Poisson boundary metadata is explicit.
- Dynamic OLS HC3/CR0/CR1S covariance and influence-based contrasts use QR and
  caller storage, including more than the previous fixed row/column limits.
- Checked logarithms, scaled products and CountGLM/NB2 internals reuse scalar
  scratch. Exact count-zero Poisson log probability is -mean after validation,
  independently of deviance-square availability.
- Keyed count data keeps integer responses raw and aligns offsets with design
  rows. The count example performs normalization, actual profile fits and
  figure generation with visible failed/edge/Poisson statuses.
- Scientific documents borrow two panels and validate data, text and reference
  lines before either backend writes output. Count residuals use fitted means.
- The ODE observer uses actual accepted ROS4 states, records quantity drift and
  sampled event brackets, and feeds the same document renderer.
- Canonical bitmap data fixes the mixed glyph orientation exposed by actual
  Canvas images. See the visualization evidence and pinned public-domain source.

## Preserved failing controls

The original 16-dataset B31 monoprocess pilot (qualification01, job16411)
completed indices0..10 then exited181 with `madaros: arena full (allocation of
48 bytes)`, at MaxRSS1917016KiB. It was not a Slurm OOM. The identical generated
driver against the reusable-scratch implementation (pilot_replay_hot01,
job16420) completed all16, status0, MaxRSS205936KiB, preserving the completed
prefix byte for byte. This supports the allocation fix; it does not establish
constant runtime memory for arbitrary B or statistical calibration.

The original font passes source/backend parity but fails canonical orientation;
the corrected native font matches all96 glyphs and6832 checked pixels. The
fixed-alpha bootstrap mutant is rejected. Missing Canvas coverage is rejected
before output files are opened.

The first complete integration run (full01, job16453) preserves STATUS1:
460 recorded phases included one C++ compile failure caused by a malformed
stream separator in the new exact-Poisson reference. The native exact-bit cases
and the other phases passed. The source was corrected and full02 is a fresh
rerun; full01 is not presented as a passing gate.

## Qualification and remaining boundaries

The harness requires a passing, source/compiler-bound functional receipt. It
uses B31 for the separate pilot and B199 for a nine-cell n/alpha qualification
matrix, with deterministic C++ seed scheduling, per-dataset native receipts,
explicit unavailable outcomes, conditional rejection summaries and uncertainty.
Interrupted batches remain intact and require review. Pilot and qualification
samples are not pooled. Wall time is capped at24h for a qualification round.

Final gate, resource-repeat and qualification results are recorded in their
own receipts as they complete. No claim of universal calibration, full compiler
corpus coverage, comparative scientific superiority, or human figure acceptance
follows from these focused checks.

## Actual visualization artifacts

[Scientific Canvas document](images/scientific-document.canvas.png) and
[CountGLM Canvas diagnostics](images/count-diagnostics.canvas.png) are
byte-identical copies of export04/job16422, not separately redrawn previews.
The corresponding [document SVG](svg/scientific-document.svg) and
[count SVG](svg/count-diagnostics.svg) are extracted from the same native
stdout streams. Exact filenames and all retained receipts are listed in
CONTENTS.sha256; check that manifest from this directory. Source and native
identities are in receipts/SOURCE.sha256 and receipts/NATIVE.sha256.

The two panels contain2050/2056 total marks for the generic/fitted examples;
24 text-bound checks and the late-invalid zero-output control passed. PNGs
were visually inspected for readable orientation, negative ticks, titles and
reference lines. Both SVGs passed structural XML validation. These checks are
separate from numerical concordance and human scientific acceptance.

## Final resource repeat

[Resource ledger](resources/RESOURCE_LEDGER.md) retains all three paired trials
and the failed/completed pilot controls. Resource04/job16479 completed0:0 with
RESOURCE_COMPARE_PASS. C++ analyzer medians: QR3.16→3.13s and
106496→8192KiB; CountGLM .5912→.6336s and42508→2788KiB. The unchanged
median time limit is1.10; CountGLM's median ratio is1.07171853857 while its
third individual pair is1.12855209743. This is lower measured memory with a
modest median time increase for CountGLM, not a blanket speedup claim.

## Complete functional gate

[Full02 results](functional/full02/RESULTS.tsv), Slurm16476, completed with
STATUS0:461 recorded phases all have status0. These are phase records, not461
independent scientific tests. This includes native fixtures, independent C++23
concordance, the fixed-alpha mutant control, all96 canonical font glyphs and
6832 checked pixels, and final unchanged source/compiler hash verification.
The original [full01 results](functional/full01/RESULTS.tsv) retain STATUS1 and
the malformed C++ stream-separator compile failure. No failing row was removed
or reclassified to create the passing full02 result.

The final qualification job16488 started only after full02 passed. Its first
control rejected full01 with exit65. A lossless label-normalization ledger binds
the receipt required by the harness to all original rows and can reconstruct
the raw TSV. The final isolated pilot and B199 campaign are separate execution
states; consult their actual receipts before making a calibration claim.

Independent read-only review found no material blocker in the modified scratch
reset/lifetime paths, exact P0 guard ordering, document/font/export code, or
count/ODE integration. Static review does not replace native/backend evidence.
