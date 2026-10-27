# Native fMRI fixture path

The prior pipeline printed numeric constants after checking file existence and
reading only motion text. Its exact source, the original NIfTI reader, and the
original test are retained in before_*.sio. The v1 golden JSON remains unchanged
as a historical witness; it is not the independent oracle for this path.

The new path reads actual bytes through NativeFile and owned ByteBuffer,
including embedded NUL. All files, descriptors, buffers, and platform contexts
are explicitly closed. The reader retains the existing 65536-byte file and
4096-sample bounds. It accepts single-file uint8 NIfTI-1/2 in either byte order,
checks dimensions, payload extent, and integral voxel offsets, and rejects
nonidentity scaling. NIfTI scaling and offsets follow the official format
[header definitions](https://github.com/NIFTI-Imaging/nifti_clib/blob/master/niftilib/nifti1.h).
This path does not decode compressed or other voxel datatypes.

For matching four-dimensional images with 2..500 frames, calculations are:

- Motion FD uses translation in mm, rotation in radians, a 50 mm head radius,
  mean over successive frame pairs, maximum, and count above 0.2 mm / frame count.
- Temporal SNR uses each voxel's temporal mean / sample standard deviation.
  The median spans all spatial voxels. A zero-variance voxel makes this metric
  unavailable; it is not silently removed.
- DVARS is the temporal median of RMS voxel differences over successive frames.
- Connectivity is the Pearson correlation between the two images' global
  mean time series. A constant global signal is explicitly unavailable.
- The 95 percent Fisher-z interval uses 1.96/sqrt(n-3); perfect correlation
  takes the limiting zero width. Fewer than four frames cannot supply this interval.
- Atlas coverage and a composite uncertainty index are unavailable without
  an atlas input and a defined index. Neither receives an invented numeric value.

Motion TSV requires the six ordered tx/ty/tz/rx/ry/rz columns and at most nine
fractional decimal digits. Malformed width, unsupported precision, or a row
count different from the images fails. This is a bounded paired-image QC path;
spatial registration, atlas parcellation and preprocessing are separate operations.

The existing main and two malformed-header probes retain their bodies and
assertions. Their effect rows now declare allocation for native owned I/O and
the checked parser's mutation/divergence/panic effects.

Slurm 18314 qualified binary reading against both unchanged fixtures (384 and
576 bytes, NUL bytes, payload edges, dimensions and offsets). Slurm 18336 ran
the original test body with the effect migration and calculated the metrics.
The independent C++23 reference matched at the unchanged 1e-11 threshold.
Slurm 18349 preserved the first overprecision failure; 18355 then passed all
13 input cases, including changed voxels/motion, a constant signal, extended
offset, big endian, exact refusals and rejected output mutation. The followup
header and result-reuse controls have separate receipts.

Numerical acceptance uses decoded f64 bit rows and an independent C++23
long-double/std-math calculation from the actual input files, rather than the
six-decimal display or a copied golden value. The original fixtures' temporal
SNR, DVARS and motion metrics agree with the old values; the two global signals
are perfectly correlated, whereas the old emitter printed a different constant.

Slurm 18365 completed the final 17-case gate and FMRI_NATIVE_GATE_OK. Additional
cases reject unsupported dimensions, nonidentity scaling in both header
versions, and a NIfTI-2 data offset inside its fixed header. Canonical binary
reader and reused-result tests passed check/compile/native. A missing Alloc
effect was rejected with E035. Source inventory, compiler/runtime, all generated
ELFs, derived files and actual dynamic-library hashes verified at completion.
All three original test function bodies and assertions are retained; only the
documented effect rows and explanatory header comment changed.

## Explicit atlas and index configuration

The user superseded the initial three-input API with a mandatory atlas and
explicit index configuration. The 17-case receipts above apply to the earlier
API only. The expanded API is being qualified separately; those old receipts
do not certify its new call signature or calculations.

The candidate call is
run_fmri_pipeline_gate(image_a, image_b, motion, atlas, &config).
The caller constructs FmriIndexConfig with fmri_index_config_new(weights,
scales, coverage_mode, expected_regions, alignment_policy, formula). Each of
weights and scales contains five f64 values in this exact order: FD mean,
DVARS median, inverse temporal SNR, Fisher-z interval width, uncovered fraction.
All fields are mandatory; the library supplies no implicit weights or scales.

Version 1 computes sum((w/max(w)) * feature/scale) / sum(w/max(w)).
Weights must be finite and nonnegative, at least one must be positive, and
every scale must be finite and positive. Intermediate overflow and nonzero
values lost to underflow are refused. An unavailable tSNR or interval makes
the index unavailable only when its corresponding weight is positive. The
output key uncertainty_score is retained for compatibility, but this configured
scalar is not a calibrated probability or a clinically validated uncertainty
estimate. It is not restricted to the range zero to one. The fixture's explicit
coefficients test the mechanism and are not production defaults.

The required uint8 atlas has one timepoint and exactly the images' spatial
dimensions. Zero denotes background; positive integer labels must be within
1..expected_regions, with expected_regions in 1..255. Coverage mode 1 is the
number of labeled spatial voxels divided by all spatial voxels. Mode 2 is the
number of distinct positive labels present divided by expected_regions.
The pipeline also emits both counts. Required alignment policy 1 is an
explicit caller assertion of voxel-index alignment: this bounded path does
not infer registration or validate an atlas against world-space transforms.

Formula 1 and the two coverage modes are explicitly versioned. The public
pipeline returns 16 for an invalid or unreadable atlas and 17 for an invalid
configuration or unrepresentable index arithmetic. Missing required component
metrics produce required_metric_unavailable rather than a fabricated number.
The low-level calculation resets all prior result fields on every call.

Slurm18448 generated the binary atlas from the pinned NIfTI fixture and wrote
its nine-row configuration TSV using C++23. Slurm18466 compiled both the old
and expanded independent C++23 references, reproduced every original mutation,
and verified that the first17 cases retain byte-identical image/motion inputs
and the same expected native exits. All47 expanded configuration/atlas cases
matched the C++ input acceptance/refusal contract; every syntactically valid
configuration emitted a constructor, including semantic-invalid cases that
must reach the Sounio validator. Source, executable and generated-file guards
passed. This is a reference/generator result, not expanded Sounio acceptance.

The first expanded Sounio check and compile passed (18456), then the native
program signaled132 before metrics. Stage diagnostics18457 localized the
boundary to reference-valued struct field access. Standalone original
regression18458 failed its unchanged p.b.x assertion while direct leaf(&b)
worked. Diagnostic18459 printed the erroneous chained value and showed that
forwarding leaf(p.b) still reads the correct value. The compiler repair and
full47-case Sounio/C++ replay remain required; the failed witnesses are retained.

Slurm18531 completed0:0: all47 expanded fMRI cases passed their exact expected
native exits, every successful metric row matched C++23 at the unchanged1e-11
threshold, malformed inputs/configurations were refused, and a wrong numeric
output was rejected. Binary reader, result reuse and missing-Alloc controls
also passed their expected outcomes. All source/runtime/compiler/ELF/derived
and dynamic-library custody guards verified, with FMRI_NATIVE_GATE_OK.
This replay pins raw222c478c... from the isolated reference-field proposal.
That artifact still had a separate broader alias regression, so this is an
actual fMRI behavior receipt and not a compiler-wide release claim. The later
reference/alias source tranche is separately qualified by18561; the mandatory
fMRI gate must run again on the final coherent source-built compiler.
