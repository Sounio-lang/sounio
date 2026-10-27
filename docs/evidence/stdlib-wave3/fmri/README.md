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
