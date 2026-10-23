# Native foundation receipt — 2026-10-23

Slurm job16114 executed the indexed source tree
`714b99224bb9726d1258265cf2eded05a9024395` on cpuops-t560-proxmox.
All registered checks, expected negative exits, native executions, stdout
markers, C++ references, and input/compiler hash verifications passed.
RESULTS.tsv and STATUS preserve the recorded result. This is a native/manual
gate receipt; remote GitHub CI and merge remain separate states.

The source-built Madaros ELF was
`1148eabd3905ec76911d027c578fb8d6b2bd2dde2c186ee3c02c4c3b97733029`.
It is the protected build from compiler PR3044. The shipped prebuilt was not
updated. The source snapshot and executable artifacts remain under
`/scratch/sounio-stdlib-deep-20261023/foundation3`; the full source/input
manifests were checked there. The tracked receipt is deliberately compact.
Endpoint hashes cover listed files, not every environmental input, tool or
possible compiler dependency. The wrapper may materialize its prebuilt marker.

Independent review identified an eigen workspace phase hole. Job16116 compiled
the final regression with the preceding eigen implementation, then observed
native exit1; identities and the exit are retained. The same final regression
passed with the phase-state fix in16114. The direct dense/tridiagonal range
tests and descending-diagonal ordering witness also passed.

The structured8192x64 QR fixture ran3.16s at106496KiB peak RSS in16114, compared
with43.50s at106496KiB before optimization. This is a single structured fixture,
not a general dense-performance guarantee. Native SVD reconstruction covers
tall/wide cases and a structured1024x32 case; LU/Cholesky references include
strided reconstruction and solves; exported QR is compared against a C++ Gram
identity using actual native R/scales/permutation.

Lean4.33 `lake build SounioNumericBuffer` passed separately in Slurm16052.
Its source is unchanged in this packet. These are selected structural proofs,
not allocator/compiler/IEEE verification.

The gate itself is versioned and fails closed on unknown oracle modes,
non-exact expected negative exit, missing completion marker, manifest errors,
input/compiler changes, timeout, compilation or runtime failures. It requires
a new output directory and bounded C++/Sounio steps. The caller is responsible
for supplying an independently attested current-source compiler; the script's
file/hash checks alone do not attest how the compiler was built.
