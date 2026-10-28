# Runtime bytes and byte slices: qualified stdlib API

Slurm **19557 completed 0** in 00:01:51, maximum RSS 5857068 KiB.
The complete candidate matrix passes against both immutable compiler ELFs:

- predecessor `root-integrated01/results/madaros`, SHA-256
  `11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4`;
- converged `root-integrated02/results/madaros`, SHA-256
  `8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932`.

Both paths are under
`/scratch/sounio-stdlib-wave3-20261027`. No compiler bootstrap or compiler
source change is part of this library change. These are focused public
check/compile/native and C++23 concordance results, not official CI or a
whole-stdlib verdict.

## Public contract

`text::utf8_bytes`, also exported through `text`, `text::mod` and
`text::lib`, provides:

```sounio
pub struct Utf8Bytes {owner:&Utf8Text,start:i64,size:i64}
pub fn utf8_text_as_bytes(owner:&Utf8Text)->Result<Utf8Bytes,i64>
pub fn utf8_bytes_len(view:&Utf8Bytes)->i64
pub fn utf8_bytes_get(view:&Utf8Bytes,index:i64)->ByteValue
pub fn utf8_bytes_slice(view:&Utf8Bytes,start:i64,end:i64)->Result<Utf8Bytes,i64>
```

The fields are private. Construction retains the original `Utf8Text` owner;
these functions copy no payload bytes and expose no raw pointer. They may
construct the language's ordinary result/view metadata; this is not a claim
that those carriers allocate no runtime metadata.

Every stored byte is visible, including embedded U+0000. Length and indices
are bytes. The half-open slice may end or begin inside a multibyte scalar,
and may be empty. Bounds failures use status 4; an invalid text owner returns
its existing status. `Utf8View` remains the distinct scalar-boundary API.

The compiler rejects owner close and mutation while the returned view is
used later, including a view forwarded through another function's Result.
Both check and compile refuse each negative with E038 and produce no ELF.
The after-last-use positive checks, compiles and runs 0. Literal and text
fixtures forward `&Utf8Bytes` into `probe_dump_bytes`, which forwards it
again into the public accessors. The owned fixture reads a negative index
and then valid bytes, checks empty/full/invalid slices, and reads a slice
inside an encoded scalar. All run on both compilers.

## Four original runtime regressions

The original files remain byte-for-byte under
`tests/fixtures/runtime_bytes_contract/original/`. Their original unsupported
primitive methods/dynamic slice syntax still fail check and compile with
status 1 and no ELF, as recorded in each matrix. They are preserved as the
before case; assertions and expected data were not weakened.

The four normal runtime regression paths now exercise the actual available
ownership APIs:

- binary: NativeFile/BufferedReader to ByteBuffer, then ByteView;
- literal: Utf8Text constructor, then Utf8Bytes;
- text: original motion TSV bytes to Utf8Text, then Utf8Bytes;
- dynamic slice: ByteView's first `min(8, size)` bytes.

All retain their original START/LEN/AFTER markers and original default data.
They now print and independently verify every resulting byte and check
bounds. Native file reads use seven-byte refill windows so the fixture
covers refills across scalar boundaries. Owners are explicitly closed on
success and error paths.

The C++23 oracle reads the original binary/text input independently and
requires contiguous indices, exact values, exact length and ordered unique
markers. Its generated fixtures cover all 256 byte values, NUL-containing
UTF-8, a cut inside an encoded scalar, malformed UTF-8 and empty input.
The file/text error paths and extra-argument exits are checked separately.
Malformed UTF-8 is rejected as status 5; missing binary input preserves
errno 2, with no success marker.

Wrong-byte, wrong-index, missing-row, duplicate-row, wrong-length and
missing-AFTER controls are all generated successfully and rejected by the
oracle. Wrong kind and missing/unknown argument controls are also checked.
The C++ compiler uses C++23 with warnings treated as errors.

## Preserved compiler failure: escaping nested-field borrow

The first candidate returned `ByteView` from
`byte_view_new(&text.data, 0, count)`. An immediate native prototype passed
in Slurm 19509. Stronger testing in 19547 exposed SIGILL after a negative
access followed by a valid access, and SIGSEGV while forwarding that view.
That candidate is **not qualified** and is not the implementation shipped here.

The isolated controls in 19550, 19551, 19552, 19554 and 19556 retain the original source and compiler
outputs. Exact immediate prototypes and two successful accesses pass, while
the nested-field-backed negative-to-valid case fails by signal 4.
Direct ByteBuffer and direct ByteView-owner controls pass the same sequence.
`SOUNIO_NO_REGION_RECLAIM=1` does not repair the failure and was diagnostic
only. This evidence localizes the boundary to the escaped subfield-backed
view; the precise compiler cause remains open. No lowerer/checker fix or
general nested-field-borrow guarantee is claimed.

The accepted `Utf8Bytes` stores `&Utf8Text` directly, like the existing
`Utf8View`. It returns no reference to a helper's field slot and passes the
stronger controls above. All failed inputs and logs remain in the durable
bundle. Job 19549 was a malformed diagnostic shell wrapper, retained
separately; its corrected successor is 19550. A test helper's early-return
linear-consumption error in 19547 was corrected by consuming its path buffer
on both branches; 19553 records the intermediate component replay.

## Reproduction and custody

`scripts/stdlib/runtime_bytes_gate.sh RAW SOURCE ORACLE PLATFORM_LIBRARY FRESH_OUTPUT`
requires a Slurm allocation. It does not rebuild the compiler. The exact
worker is [qualified/worker.sbatch](qualified/worker.sbatch). Results,
manifests, borrow diagnostics and individual C++ oracle outputs are in
[qualified/results](qualified/results). Each compiler, platform library,
oracle, source file, original/generated datum and native ELF is hashed before
use and verified again at the end. The worker also compares the complete
stdlib file manifest to the candidate worktree manifest, preventing omission
or mixed unqualified library files.

The accepted source is
`/scratch/sounio-stdlib-wave3-20261027/core/runtime-bytes03/source`.
Its immutable inputs and complete results are retained beside it.

Slurm **19558 completed 0** and verified a complete archive readback:

`/orangefs/training/sounio-foundry/runs/stdlib-wave3-20261027/core/runtime-bytes01/runtime-bytes-source-raw-receipts-v1.tar.gz`

SHA-256:
`eba6b04a558d19113ddc4be5b8704d1d124a18c62e86fa6af21b336f9d685c1a`.

The archive includes both raw compilers, platform library, the prototype,
failed candidate, accepted full stdlib inputs, all diagnostic inputs/native
ELFs/logs, original data and scheduler receipts. Per-file manifest readback
passes. [durable](durable/) records the archive and worker receipts. Generic
HeapVec/HeapDeque, nested-field compiler repair, primitive string method
syntax and global compiler/stdlib acceptance remain outside this result.
