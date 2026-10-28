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
  chunk-output directories, whose parents retain EXECUTABLE.sha256 and a
  successful outer STATUS. Missing/duplicate parts and mismatched ELF identity
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

Complete collection additionally requires the outer receipt format. The default
`WAVE3_PART_RECEIPT_FORMAT=gate` requires each parent STATUS to be0 after all
source/compiler/runtime guards. The explicit `legacy-worker` adapter requires
both WORKER_STATUS and NATIVE_STATUS to be0 for the preserved custom locale
workers. Missing status is never inferred as success. The gate hashes these
outer receipts, exact ELF identity files, part STATUS and RESULTS.tsv before
collection and checks them again afterward.

Slurm18639 qualified this tightening on the unchanged763-file/390251-row final
locale default manifest. A private copy of an otherwise valid part with only
WORKER_STATUS changed to1 was rejected. Wrong receipt format, source manifest,
fixture manifest, family and missing-part controls were also rejected. The
actual prior valid corpora and original failed receipts were preserved.

Slurm18654 additionally exercised the actual public strict wrapper, beyond
identity: committed e33 checked, compiled and natively ran the minimal
STRICT_WRAPPER_DISPATCH_OK witness. Setting only the raw override to the
foreign a2c artifact under SOUNIO_REQUIRE_COMMITTED_MADAROS=1 was refused with
status78 by the wrapper itself. This is minimal dispatch qualification for the
old committed compiler; it does not claim that e33 supports static blobs or the
new Unicode APIs.


The explicit `WAVE3_FIXTURE_FAMILY=locale-metadata` selects kind6 and the
72-file/9190-row CLDR49 constructor metadata protocol. It uses the dedicated
run_locale_metadata_chunks/collect_locale_metadata_chunks adapters.
A metadata result does not imply a collation-key corpus result; a locale
key corpus does not imply complete metadata coverage.
The retained public harness is
`tests/stdlib/wave3-core/unicode_locale_metadata.sio`; its inputs use
`tests/stdlib/wave3-core/locale-meta-data/current.bin` inside each run directory.

## Public guarded metadata replay

Slurm18714 completed0:0 in3m30s using the tracked
`tests/stdlib/wave3-core/unicode_locale_metadata.sio` harness. Check, compile,
first-chunk native execution, all72 fixtures/9190 rows through the public
`locale-metadata` gate, and complete collection all passed.
Executed ELF:
97e111445bc2409501a6be89feb468aa0c176e3c88d0c8b87e23820a8147221d.
Compiled-source manifest:
f76f007592f3873fb3864e54a217cd8aedbee3b2c03ce5e267483e53741297a7.

The unchanged earlier gate rejected the new metadata family with status64;
the new adapter completed the same declared metadata protocol.
Duplicate part, wrong family, wrong compiled-source identity and a private
part copy whose outer STATUS alone was changed to1 were each rejected.
All original source, compiler, runtime, executable and archive guards passed.
The isolated source snapshot includes the already-qualified root
byte_buffer_budget dependency without editing the services worktree.
This is full declared constructor-metadata acceptance, not a new full
collation-key or BIDI result.

## Generic compiler compatibility smoke

Slurm18738 completed0:0 in1m22s with the same e9ad6cde compiler that executed
the public generic core fixtures. All13 core fixtures checked, compiled and
ran natively; the original formatting suite also compiled and ran unchanged.
Both text facades, owned UTF8/NUL, exact-page accesses, properties, the91-row
Unicode smoke, DUCET root and CLDR constructor/key contracts passed.

The exact snapshot used qualified root dependencies:
byte_buffer423a294a229d41723a666bc6ac07ac722b64b2e1b2a406bd74db888b8edf2327
and BIDIc9fcb10f39fd948d5a66bb476d80d64895994bd4d04662fba6402b9b680632de.
Gate source manifest SHA256:
3c781883e9a046cbc9dc9519d042367c6251bf6eb48337e9d90bd51de5af727d.
All gate source/compiler/platform/executable guards passed before/after.

This is early compatibility evidence for the generic compiler. It does not
retag the earlier complete Unicode/UCA/CLDR executions as runs on this compiler,
and does not close BIDI, persistent owner loans or the final integrated suite.


## Explicit coherent-ELF corpus invocation

WAVE3_COHERENT_KIND is optional. The default preserves the original no-argument
ELF invocation. In chunks/collect phases, set 1 for family unicode, 2 for locale,
or 3 for locale-metadata. Other kinds, mismatched family and identity/smoke
phases are refused 64 before creating output.

The runner passes exactly that one argument to the actual ELF and writes
ELF_INVOCATION.tsv. The collector validates its kind/argc binding for every
partition, and the public gate pins those receipt bytes together with the
ordinary partition, status and executable identity records.
ELF_ARGUMENT_KIND records the declared invocation mode. Existing STATUS and
RESULTS.tsv protocols retain their legacy schemas.

Slurm 19291 completed 0 in 3m38s: the same original no-argument Sounio harness
works through both old/new gates; the old gate refuses the coherent harness;
all three explicit protocols pass part+complete-prepared-manifest collection.
Wrong kind, wrong family, wrong phase and tampered/missing kind/argc records
are refused. These private one-chunk manifests qualify the adapter only.
The comprehensive corpus replay is a separate receipt.

The first 19289 attempt retained a source guard failure: the wrapper updated
its copied binary verification cache after the manifest was frozen. 19291
primes both wrapper copies before freezing that manifest. No cache/source
guard was removed, and the original failed receipt is retained.
See docs/evidence/stdlib-wave3-core/coherent-text for source and artifact pins.
