# Complete native UCA18 root conformance, 2026-10-27

Slurm array 18450: all eight tasks COMPLETED 0. Complete collector 18486:
COMPLETED 0, UNICODE_COMPLETE_MANIFEST_OK files=2307 rows=1180750 partitions=8.

This qualifies the DUCET18 / CLDR49 root Sounio algorithm against the sealed
kind-4 fixture manifest. It does not qualify production CLDR locale tailoring,
BIDI versus ICU, generic HeapVec, the primitive source-string NUL ABI, or the
whole stdlib.

The original 1,176,850 serialized oracle rows remain byte-for-byte identical
to the previous corpus prefix. C++23 prefix witness 18444 checks every original
row; no original code point or expected ordering was replaced. Additional rows
bring the full manifest to 1,180,750, including option parity and adjacent
surrogate code-point controls.

Every original official code-point sequence enters the lower-level
IntegerBuffer UCA key API directly. For every valid-scalar operand, the same
native harness also derives a public Utf8Text key and requires identical
weights under the same settings. Surrogate code points remain separate raw
code points in the low-level API; the strict UTF8 owner rejects their encoding.
The original chunk 413 row 39 failure witness is retained and passes unchanged
after the explicit domain correction. Focused qualification: Slurm 18446.

Compiler: real-blob source dc04dc3b506f09682a5b4c241ab78f4db8f7918f,
raw ELF SHA256 a2c91344ce5ada4d68e9db4521342248b0dc8511e442c021d0f7140d35344915.
Native conformance ELF SHA256:
70fd483dc981fc0b7443114c6b56cd325689bc012dcce3f468496f521e638fe2.
Platform ABI112 shared library SHA256:
de01597868c09725c2c26651d23203043c65fdc04a3e5e96f45dfe4a51d494c4.
Manifest SHA256:
0686152cf10c85f00576fa32fd87f93c8f6090af80923e75e9096e41c7a367bd.
Fixture hash-list SHA256:
100bb1d193628303bedab967769902e70d4e5ad7f643433f268c2a9cd5ae3e70.

All eight partition workers verify the same qualified Sounio source, compiler,
runtime, ELF, manifest, hash list and every fixture byte before and after their
runs. The complete collector additionally rechecks the entire fixture byte
manifest before and after combining exactly one result for each chunk, with
the expected single ELF identity and exact row counts.

Receipt roots on DL380:
- /scratch/sounio-stdlib-wave3-20261027/core/uca-native-full03
- /scratch/sounio-stdlib-wave3-20261027/core/uca-native-collect02
- /scratch/sounio-stdlib-wave3-20261027/core/uca-fixtures03
- /scratch/sounio-stdlib-wave3-20261027/core/uca-codepoints-native01

The first complete collector (Slurm 18479, uca-native-collect01) failed before
collection because its working directory did not match the relative paths in
the fixture hash list. Its original error and worker remain intact. Collector
18486 changes only that metadata working directory and verifies the unchanged
fixtures and original native receipts.
