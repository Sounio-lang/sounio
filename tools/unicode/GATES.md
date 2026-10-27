# Wave3 core execution gates

`scripts/stdlib/wave3_core_gate.sh OUT PHASE ...` requires a fresh absolute
output directory and pins the compiler ELF and platform runtime. It accepts a
Slurm job, or an explicitly identified GitHub Actions environment with
CI=true, GITHUB_ACTIONS=true, numeric run ID, repository, workflow and40-hex
source SHA matching the checkout HEAD. Authority metadata is recorded. An
environment-parser simulation is not an official CI receipt.

Compiler modes:

- `WAVE3_COMPILER_MODE=raw` (default) requires MADAROS_RAW_BIN and
  WAVE3_COMPILER_SHA256. It selects the actual ELF through the public wrappers,
  records wrapper identity, and rejects an incorrect digest.
- `WAVE3_COMPILER_MODE=strict-wrapper` materializes the tracked committed
  gzip/digest pair, requires its declared digest, selects the exact committed
  ELF path, and enables SOUNIO_REQUIRE_COMMITTED_MADAROS=1. A foreign raw
  override is rejected. It never copies a new compiler into the committed
  slot or labels an uncommitted candidate committed.

Both modes reject SOUNIO_SOUC_BIN, SOUNIO_MADAROS_BIN and a non-Madaros engine.
The platform filename must be libstdlib_platform.so; its full digest is checked
before and after execution. Wrapper scripts, version metadata, all stdlib/core
test/tool sources and assets are hashed before/after. Each phase records status;
a success marker with nonzero STATUS is not acceptance.

Phases are separate claims:

- `identity`: wrapper/ELF/runtime/source identity only; no runtime library test.
- `smoke`: thirteen checked/compiled/native fixtures plus the unchanged
  original text formatting suite. It covers owned UTF8, source literals,
  exact-page access, interpolation/formatting, properties,91-row Unicode,
  DUCET root, locale contracts and both text facades. It does not close a
  complete official corpus.
- `chunks ELF FIXTURES PART PARTS`: execute one declared partition of a pinned
  prepared manifest. A chunk part cannot be reported as complete acceptance.
- `collect ELF FIXTURES PARTDIR...`: compile the C++23 complete collector and
  require every declared file/row exactly once. PARTDIR arguments are the
  chunk-output directories, whose parents retain EXECUTABLE.sha256 and
  NATIVE_STATUS metadata. Missing/duplicate parts and mismatched ELF identity
  fail. Collection does not infer that unrelated corpus families passed.

All phases require WAVE3_COMPILER_SHA256, WAVE3_PLATFORM_SO and
WAVE3_PLATFORM_SHA256. Corpus phases also require:

```text
WAVE3_CORE_ELF_SHA256
WAVE3_FIXTURE_MANIFEST_SHA256
WAVE3_FIXTURE_HASH_FILE
WAVE3_FIXTURE_HASH_ROOT
WAVE3_FIXTURE_HASH_SHA256
WAVE3_CORE_SOURCE_MANIFEST
WAVE3_CORE_SOURCE_ROOT
WAVE3_CORE_SOURCE_MANIFEST_SHA256
```

The last three pin the exact extracted source used to compile the supplied ELF,
including embedded data, independently of the gate's current checkout.
WAVE3_FIXTURE_FAMILY=unicode (default) selects the root Unicode/UCA fixture
protocol; =locale selects the explicit CLDR kind5 protocol. The root runner
does not accept locale rows as a tolerance or implicit alternate family.

## Closed adapter receipts

Slurm18631 passed the earlier eleven-fixture smoke plus original format witness,
actual raw a2c/compiler/runtime custody, actual strict committed-wrapper identity
for shipped e33dcf87f3a7cbca50b2c5c008da2a06153bfa552c73475b01239aff87415514,
and explicit rejection controls: outside authority, malformed/mismatched GitHub
source SHA, wrong raw digest, engine/ELF bypass and foreign strict override.
The GitHub dispatch positive is a synthetic Git repository inside Slurm, labeled
a parser simulation. It is not a GitHub Actions result.

Slurm18633 passed the complete locale default collector through this gate:
763 files/390251 rows/all8 partitions of final-format ELF
9b26bf5f358cdfa8ee3e6fdc4b2e4d5fb4f29f33cf37836b9d9f3a12659c1b8e.
Wrong ELF-source manifest, fixture manifest, fixture family and missing-part
controls all failed. Original byte guards and per-part receipts were retained.

The earlier18627 archive omitted version metadata needed by bin/souc --version;
its status2 packaging failure and logs remain preserved. The adapter did not
turn that failure into a passing compiler identity.

Slurm18636 passed the final thirteen-fixture gate plus the unchanged original
format suite, including both canonical text and explicit text::lib facades.
Their dependency is the qualified BIDI source c9fcb10f39fd948d5a66bb476d80d64895994bd4d04662fba6402b9b680632de; no BIDI algorithm or oracle verdict was changed. The final gate repeated all authority/strict-wrapper identity and rejection controls.

Final coherent base Unicode,
UCA, CLDR, BIDI and compiler/generic ownership acceptance remain independent;
neither identity nor focused smoke substitutes for their complete manifests.

