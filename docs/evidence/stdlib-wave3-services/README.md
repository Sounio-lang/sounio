# Stdlib service-wave evidence

Base: `a95536ac032cb0ff1b3418c3232fa8c07fba09ac`.
Branch: `devpass/stdlib-wave3-services-20261027`.

## Arrow oracle migration

The canonical oracle is `tests/reference/arrow_cpp_interop.cpp` compiled as
C++23 against official Arrow 26.0.0. The required preservation map is
`tests/reference/arrow_legacy_case_inventory.tsv`. Legacy Python files are
read-only historical inputs; no Python judge is part of acceptance.

Slurm 18105 preserved the initial C++ compilation failure. Slurm 18109 built
the corrected judge with `-Wall -Wextra -Werror`, confirmed
`ARROW_CPP_VERSION 26.0.0`, and recorded its executable and bound libraries.
Receipt root on DL380p:
`/scratch/sounio-stdlib-wave3-20261027/services/arrow-cpp-02/results`.

This is build and SDK binding evidence. Full producer/oracle interoperability
requires a separate completed gate receipt; none is asserted by the build.
The SDK was independently provisioned and qualified by the root lane
(Slurm 18103). No system packages were installed by this lane.

## Full Arrow interoperability receipt

Slurm **18126** completed with status 0 on DL380p. It required all five
producer markers, all 30 inventory case groups, the SDK-absence refusal,
`ARROW_CPP_INTEROP_OK`, `ARROW_CPP_ID_MUTANT_REJECTED` and
`ARROW_CPP_GATE_OK`.

Receipt root:
`/scratch/sounio-stdlib-wave3-20261027/services/arrow-full-02/results/gate`.
`INPUTS.sha256`, `COMPILER.sha256`, `EXECUTABLES.sha256`,
`BOUND_LIBRARIES.sha256` and `ARTIFACTS.sha256` have successful final
verification files. `FIXTURE_RELOCATION.json` records each original/derived
source hash, literal path mapping and byte-exact reverse check.
`EXPECTED_CASES.txt` and `OBSERVED_CASES.txt` agree exactly.

The raw compiler was built from base a95536ac and pinned to SHA-256
`7df87ae9d9b4b3c0cefcd18b892512b6bf7e5c6fde6690ca5223669a9a1a2639`.
The initial full job 18123 passed the semantic checks but failed inventory
closure because the inventory had an extra blank row. Job 18126 used the
same assertions, inputs and oracle after removing that blank row and passed
the complete gate. This is a focused Arrow receipt, separate from the
whole-stdlib acceptance run.

## Strict CSV receipt

Slurm **18130** completed with status 0. The identical
`test_csv_strict.sio` fixture checked and compiled against the base reader,
then failed natively (status 1). With the changed reader it checked, compiled
and printed `CSV_STRICT_OK`. The offset/numeric-continuation fixture printed
`CSV_STRICT_OFFSETS_OK`; existing checked-numeric and native-CSV tests also
checked, compiled and ran successfully.

Receipt root:
`/scratch/sounio-stdlib-wave3-20261027/services/csv-strict-03/results`.
Source, compiler/runtime and executable hashes have final rechecks.
Job 18124 lacked the runtime library search path for the native-CSV control;
job 18127 had a worker-script newline error before test execution. Both
attempts are preserved. Job 18130 uses the qualified platform library.

The string reader validates syntax before returning any rows and exposes
status22 plus a zero-based byte offset. EOF inside a quoted field reports
the byte length. Doubled quotes, quoted CR/LF and CRLF terminators remain
accepted. Bare CR outside quotes, stray quotes and bytes after a closing
quote are rejected. Existing ragged-row handling and numeric-failure row
continuation remain unchanged. BOM, UTF-8 and strict width policy are
separate from lexical validation.

## Linux systems wrapper receipt

Slurm **18129** completed with status 0 on DL380p. New PollSet, task-reap and
deadline-I/O tests plus existing poll-executor and poll-socket tests each
checked, compiled and ran natively: 15 phases passed. New success markers
are `POLL_SET_OK`, `POLL_REAP_OK` and `DEADLINE_SOCKET_OK`.

Receipt root:
`/scratch/sounio-stdlib-wave3-20261027/services/systems-01/results`.
`RESULTS.tsv` records every phase; source, compiler/runtime and executable
hashes have final verification files. The linked platform library was the
root lane's separately qualified ABI100–111 library from Slurm 18125.
This lane does not alter the shared C++ platform implementation.

PollSet registers stable logical socket IDs with interests and caller tags.
A wait polls all registrations against one absolute monotonic deadline;
no events returns count0/status 0. Closed IDs yield flag16. Owners remain
responsible for socket lifetime. A close during a blocked wait need not wake
the wait before an event or deadline. Callers serialize registration, wait
and event reads; the three event-field ABI reads are not an atomic concurrent
snapshot. Registration payload budget is 24 bytes per entry; native container
and bounded snapshot overhead are separate.

TCP helpers retain the absolute deadline across partial transfers and expose
completed byte counts on errors. Read-exact appends into the caller buffer;
early EOF returns status61 (ENODATA) and the retained count. A timeout returns
status 110. UDP destinations are explicit IPv4/port values; zero-length
datagrams remain datagrams and sender address/port describe the consumed
packet, including truncated packets.

Reaping removes terminal task storage without reusing its ID. Pending tasks
must be canceled or completed first; reaping a pending task reports22,
and a missing or already-reaped ID reports4. The regression repeatedly
reuses a two-slot storage budget while IDs continue increasing and verifies
that compaction preserves other live tasks.

## Incremental CSV receipt

Slurm **18138** completed with status 0. The streaming fixture, existing byte
buffer fixture and existing native-file fixture each checked, compiled and
ran: nine phases passed. `CSV_STREAM_OK` confirms refill sizes 1 through 17
produce identical BOM, doubled-quote, quoted-newline, trailing-empty-field
and UTF-8 results. Controls reject malformed EOF, stray/post-quote bytes,
bare CR, width mismatch, malformed UTF-8 per field and field-budget overflow.
A 257-byte field passes without the legacy 64-byte convenience cap.

Receipt root:
`/scratch/sounio-stdlib-wave3-20261027/services/csv-stream-01/results`.
Source, current compiler/qualified runtime and executable hashes have final
verification files. New owners are `io::buffered_reader::BufferedReader`,
`data::csv_stream::CsvStream` and `CsvOwnedRow`; all have explicit close
functions. The caller reuses or closes each owned row. Field access is checked
and field-copy appends unescaped bytes into a caller-owned buffer.

The stream returns its first record, so callers may treat it as the header.
Expected width 0 learns the first record's width; subsequent records must
match. Only an initial UTF-8 BOM is stripped. No normalization or implicit
case folding occurs. Syntax and budget errors are terminal and hide the
incomplete row; EOF is status 1. UTF-8 is validated per field, preventing
a lead byte in one field and a continuation byte in another from combining.

## Incremental JSON writer receipt

Slurm **18143** completed with status 0. The new writer fixture and existing
owned DOM fixture each checked, compiled and ran: six phases passed.
`JSON_WRITER_OK` covers output chunks 1 through 7, nested objects/arrays,
i64 minimum/maximum, exact exponent tokens, UTF-8 and embedded NUL, quote,
backslash and newline escaping. The independent existing DOM parser reads
the produced files and recovers the original values.

Controls reject NaN/Infinity, leading-zero/incomplete/space-padded number
tokens, malformed UTF-8, depth overflow, missing object values, a second
root and unfinished documents. Explicit finish flushes the bounded buffer;
explicit close releases both owners and reports earlier errors. An I/O or
validation error can leave a partial file and remains sticky.

Receipt root:
`/scratch/sounio-stdlib-wave3-20261027/services/json-writer-01/results`.
Source, compiler/qualified runtime and executable hashes have final rechecks.
API: `json::writer`. Exact number lexemes are validated by the owned DOM
parser and emitted unchanged; the writer makes no round-trip f64 formatting
claim. Existing scientific JSON paths remain available.

## Incremental Arrow C++/Sounio concordance

Slurm **18152** completed with status 0. It repeated all five legacy producers,
all 30 required C++ groups and the logical-mutant control, then qualified
100 batches of 16 i64 rows per stream in both directions. Each stream exceeds
8192 bytes. Sounio read the official SDK producer with refill sizes 1 and 17;
the SDK read the Sounio producer and validated all 1600 values. Missing EOS
bytes and trailing bytes were rejected. The Sounio fixture also retains its
standalone self-produced stream check; the cross-language gate additionally
requires `ARROW_NATIVE_CPP_READ_OK`.

Receipt root:
`/scratch/sounio-stdlib-wave3-20261027/services/arrow-incremental-03/results`.
The legacy gate receipt is under `gate`; `incremental` contains reversible
path relocation, original/derived hashes, SDK/native files and execution
logs. `INCREMENTAL.sha256`, `PLATFORM.sha256` and input custody have final
rechecks. Required markers: `ARROW_NATIVE_STREAM_OK`,
`ARROW_INCREMENTAL_CPP_OK`, `ARROW_INCREMENTAL_BIDIRECTIONAL_OK`.

Job 18146 passed the legacy gate then lacked the SDK dependency search path
for its later C++ producer invocation. Corrected job18148 passed both
directions; job18152 repeated final-source qualification after making the
default fixture independent of optional external fixture files. No numeric
inputs, producer assertions, legacy case groups or SDK version changed.

## Separate genomic semantic receipts

Slurm **18161** completed with status 0 for the new GFF graph fixture and
existing dynamic-GFF fixture: six check/compile/native phases. Coverage:
forward Parents, multipart IDs, percent-decoded scalar/list boundaries,
UTF-8 exact identities, no implicit normalization, incompatible multipart
rows, missing Parents, self/multi-node cycles, duplicate reserved attributes,
scalar-ID lists, anonymous leaves and index/graph/comparison budgets.
Cycle errors include representative rows for an actual edge on a cycle.

Receipt:
`/scratch/sounio-stdlib-wave3-20261027/services/gff-graph-04/results`.
Initial jobs 18153/18154 exposed linear-owner control-flow/projection issues
in the new module. The implementation now consumes resources in explicit
branches and uses borrowed helper access. Job 18155 passed before the
additional cycle-edge witness; 18161 qualified that final witness.

Slurm **18160** completed with status 0 for new VCF document/REF validation
plus existing VCF core, VCF header and dynamic FASTA: twelve phases.
Coverage: empty document, repeated POS, contiguous CHROM blocks, reopened
CHROM, global/within-row duplicate IDs, missing terminal LF, Unicode
identity separation, index/comparison budget, FASTA window bounds,
case-folded concrete bases, literal N, mismatched/absent/ambiguous references,
and failure-preserving window append.

Receipt:
`/scratch/sounio-stdlib-wave3-20261027/services/vcf-document-02/results`.
Job 18156 preserved a parse failure from a reserved local identifier; 18160
uses the corrected name with the same test cases and passed. Source,
compiler/runtime and executable hashes have final verification files in
both final receipt roots.

Public contracts and limits:
`stdlib/genomics/io/SEMANTIC_VALIDATION.md`. These are separate semantic
layers. Ontology membership, genomic random-access indexing, variant
normalization and SV/breakend resolution are not inferred.

## Durable Arrow gate and dictionary-state receipt

Slurm **18172** completed with status 0 running the versioned
`scripts/arrow_ipc_gate.sh` entrypoint. It now requires all original
30 case groups and five producers, the logical mutant, both 100-batch
cross-language paths, and a native dictionary stream with an initial
three-value dictionary followed by a two-value delta. The one-byte-refill
reader preserves all five dictionary values; official C++26 verifies both
logical batches independently.

Receipt root:
`/scratch/sounio-stdlib-wave3-20261027/services/arrow-durable-04/results/gate`.
The gate requires `STDLIB_PLATFORM_LIBRARY` to identify the exact
`libstdlib_platform.so`, places its directory first in the loader search,
hashes it in `PLATFORM.sha256` and rechecks it afterward. Native reader,
file, byte-buffer and platform-loader source dependencies are included in
the input manifest. Existing source/compiler/executable/SDK-library/artifact
rechecks remain mandatory.

The path-only relocator retains the original 20-output legacy audit and has
a separate five-output native mode. Both record original/derived hashes and
reverse-substitution byte equality. No assertions, case markers or numeric
inputs are changed. Jobs 18165,18166 and18171 passed earlier durable-gate
iterations;18172 qualifies the final loader-path and dependency-custody
version. The gate itself now emits
`ARROW_INCREMENTAL_DICTIONARY_CPP_OK` and
`ARROW_INCREMENTAL_BIDIRECTIONAL_OK`, so future callers cannot omit these
adapter checks while obtaining `ARROW_CPP_GATE_OK`.
