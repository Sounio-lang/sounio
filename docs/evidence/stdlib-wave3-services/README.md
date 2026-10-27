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

## Atomic UDP endpoint receipt

Slurm **18175** completed with status 0: all 15 systems wrapper phases passed
against the root lane's separately qualified ABI112 platform library.
`native_udp_sender` now loads one packed IPv4/port endpoint atomically.
`native_udp_receive_from` takes an exclusive typed socket borrow across
the receive and endpoint read and returns `UdpReceive` with byte count,
IPv4, port and status.

The packet source binds to 127.0.0.2 while the receiver binds to127.0.0.1.
Tests verify its actual source address/port, a truncated datagram
(status90, no published prefix), and a zero-length datagram. Existing
pollset, reap, deadline, executor and socket controls also pass.

Receipt:
`/scratch/sounio-stdlib-wave3-20261027/services/systems-02/results`.
Runtime SHA-256:
`de01597868c09725c2c26651d23203043c65fdc04a3e5e96f45dfe4a51d494c4`.
The platform implementation and its concurrent packed-endpoint qualification
(Slurm18169) belong to the root lane. This commit changes only Sounio
wrappers/tests/docs.

An atomic latest-endpoint getter by itself does not associate two separately
called operations with a particular packet if another receiver runs between
them. Preserve the exclusive owner/serialization discipline across receive
and sender retrieval. Low-level ABI users must provide the same external
serialization. On truncation the datagram is consumed and metadata is
available, but the incomplete payload is not appended to the public buffer.

## Native-qualified API source ports

Fresh-compiler before replay Slurm **18177** preserved all six original
selected check failures on raw SHA-256
`9919d03afb60e418dc111797895860e27e93a22ca52a9cc51cd0e0edac1571b2`.
Compiler source/qualification: sealed a95536ac plus resolver54230f,
ownershipe611bac andcd332933, qualified by the compiler lane in18157.
The baseline source snapshot and all original fixtures remain unchanged.

Slurm **18191** completed with status0 for the native-qualified subset:
original `test_gp_e2e.sio`, original `test_database_core.sio`, new
active/PAC control and new database control. All twelve
check/compile/native phases passed. C++23 independently checks confidence
scores at0/250/1000, a concrete precision-weighted posterior, retained
credibility and three PAC ceiling points. Both public active APIs reject
confidence -1 and1001 after successful check/compile; each executable exits1
before its acceptance marker. The C++ judge also rejects an altered
posterior-confidence output.

Receipt:
`/scratch/sounio-stdlib-wave3-20261027/services/api-refusals-native-qualified-01/results`.
Source/compiler/executable manifests have final rechecks; negative-template
and derived-input hashes are recorded. `ACTIVE_CPP_CONCORDANCE_OK` and
`API_NATIVE_PORT_CONTROLS_OK` are required.

The active adapter replaces stale BetaConfidence/provenance assumptions
with canonical flat `Epistemic`. It divides the validated integer
credibility by1000 for a continuous exploration score. Precision-only
updates preserve prior credibility because observation variance supplies
no new credibility evidence; no float-to-confidence quantization occurs.
PAC formula implementations are unchanged. This bounded concordance does
not establish a general numerical error bound or prove PAC assumptions.
The unsupported trait-associated Model item becomes the explicit concrete
`DecisionStumpModel` alias.

The larger replay18189 remains a recorded failure: readonly-array
concatenation prevents native compilation of nonempty Frame/CSV-loader
controls, the enum payload binder blocks original logic, and original
stiffODE has independent callback/effect/legacy-bridge errors. These are
not hidden by the successful subset receipt. Original empty Frame now
runs with typed array-value length helpers; no raw array-header or fixed
small replacement was introduced. Frame/CSV source ports remain separate
until their native controls qualify.

## ByteBuffer owner budget accessor

Slurm **18229** completed with status 0 for the pure `byte_buffer_budget`
accessor and its check/compile/native control. The getter returns the existing
owner budget before and after reserve, including a rejected over-budget
reserve; it changes neither the layout nor the allocation policy.
Marker: `BYTE_BUFFER_BUDGET_OK`.
Receipt: `/scratch/sounio-stdlib-wave3-20261027/services/byte-buffer-budget-01/results`.
Raw compiler SHA-256: `9919d03afb60e418dc111797895860e27e93a22ca52a9cc51cd0e0edac1571b2`.
Source, compiler/runtime, and executable manifests pass final rechecks.

## Public service entry points and recorded prerequisites

Slurm **18210** completed with status 0 for the service-root fixture through
the explicit `io::lib` entry point. All other root imports and all assertions
are unchanged. The derivation changes one import path only, checks reverse
substitution for byte equality, and records both hashes in
`IMPORT_VARIANT.json`. Check, native compilation and execution pass on raw
9919d03a with platform ABI112; the executable prints
`SERVICE_ROOT_EXPORTS_OK`. It exercises actual readiness/UDP sender data,
executor reap and monotonic IDs, owned file/CSV/JSON operations, and separate
GFF graph/VCF document/FASTA REF validation. It also asserts that the legacy
socket stub remains unavailable.

Receipt: `/scratch/sounio-stdlib-wave3-20261027/services/root-exports-io-lib-01/results`.
Source/compiler/runtime/executable manifests have final successful rechecks.

Canonical `io` root acceptance is still open. Slurm **18204** preserves four
type errors where legacy `io::binary::read_i64` shadows the pointer-reading
builtin used by IntegerBuffer. No legacy decoder or owner implementation was
removed to conceal that compiler namespace defect. Slurm18197 preserves the
earlier duplicate-DataFrame facade refusal. The approved canonical data root
now exports the existing fixed DataFrame/df_*; heterogeneous DataFrame and
dataframe_* remain explicit at data::frame and the data::lib compatibility
facade. Their differing nominal layouts must not be co-loaded.

Slurm **18230** records successful checks for the nonempty Series and
argument-parser controls, followed by native compilation refusals (rc12)
from readonly-array concatenation on raw9919. No native execution took place.
This is a prerequisite replay, not a passing native gate. Frame/CSV-loader
nonempty qualification likewise remains open in18189; compiler support must
land before claiming these persistent-array source ports fully qualified.

The final facade-commit fixture additionally verifies the canonical fixed
DataFrame through df_new/df_add_row/df_get with two distinct nonzero cells
and an out-of-range zero result. Slurm **18232** repeats the explicit
io::lib variant with this added control: all check/compile/native phases
pass, marker SERVICE_ROOT_EXPORTS_OK, final manifests verified.
Receipt: `/scratch/sounio-stdlib-wave3-20261027/services/root-exports-io-lib-02/results`.
The canonical io namespace refusal remains open;18232 does not close18204.

## Reusable mandatory services gate

`scripts/stdlib/wave3_services_gate.sh FRESH_ABSOLUTE_OUTPUT` requires a
scheduled Slurm job, absolute `MADAROS_RAW_BIN` and
`STDLIB_PLATFORM_LIBRARY` paths, and Node (`NODE` may specify its path).
It uses the strict raw-compiler wrapper with separate bounded check,
compile and native phases. Check/compile allow300 seconds and4 MiB each;
native allows120 seconds and1 MiB. All27 fixtures are mandatory, including
the canonical service roots and unchanged original Frame/CSV tests.
There is no subset or explicit-io-lib acceptance mode.

Every shared /tmp string literal is relocated into a unique short private
directory. The gate refuses ambiguous paths/collisions, preserves original
fixtures, records original/derived hashes and substitution multiplicities,
and requires reverse substitution to reproduce the original bytes. Relative
file outputs run in a separate directory per fixture. Produced private files
are copied into the output for custody. Source, raw compiler, platform,
tools, derived inputs, executables and output files have final hash guards.

Slurm **18247** exercises the complete gate on raw9919 and ABI112.
Exactly21 fixture native markers pass, plus independent C++23 ABI112 checks
against the supplied library and against a fully ASan/UBSan-instrumented
runtime/reference build with leak detection. All final manifest guards pass.
The mandatory gate correctly exits1: original CSV-loader and four new
nonempty Frame/CSV/Series/argparse controls retain native concatenation
refusals, and the canonical root retains the read_i64 namespace check
failure. These six failures are explicit in RESULTS.tsv and remain required
for final acceptance.

Receipt: `/scratch/sounio-stdlib-wave3-20261027/services/reusable-gate-before-prereqs-03/results/gate`.
Earlier18243 caught an overbroad private-path guard;18245 caught the raw
compiler's non-executable output mode. The final gate validates ELF magic
and sets executable permission only after strict compilation succeeds.
Those earlier harness failures are preserved and are not passing receipts.

The gate also accepts an explicitly identified official GitHub Actions
context: GITHUB_ACTIONS=true, positive numeric GITHUB_RUN_ID and nonempty
safe GITHUB_JOB. It records EXECUTION_AUTHORITY.json. Scheduled Slurm
remains required for manual cluster execution. The --execution-context
mode reports only authority metadata and cannot emit the acceptance marker
or execute/accept fixtures.

Slurm **18252** qualifies the authority guard itself: ordinary workspace,
missing run/job, false context and malformed run/job are rejected with2;
identified official context and actual Slurm context produce the expected
JSON metadata. No compilation or native test was performed under the
simulated GitHub context. Source final hash recheck passes.
Receipt: `/scratch/sounio-stdlib-wave3-20261027/services/reusable-gate-authority-01/results`.

## Active metrics use the shared logarithm

Slurm **18255** preserves the failing numerical before witness: native
execution passes its original internal checks, but the independent C++23
oracle rejects active EFE at input4 (got0.6931471740887124,
expected0.69314718055994529) with the unchanged absolute threshold1e-11.
Receipt: `/scratch/sounio-stdlib-wave3-20261027/services/active-log-before-02/results`.

Slurm **18258** checks, compiles and executes the same input fixture against
the canonical-log adapter. All30 finite rows and4 domain rows agree with
C++23 std::log under1e-11. Inputs are emitted as exact IEEE bit words and
verified in order against the independent reference: subnormal extrema,
minimum normal, values next to range-reduction boundaries, powers of two,
wide magnitudes through maximum finite, and zero/negative/NaN/+infinity
public EFE behavior. Marker: ACTIVE_LOG_CPP_CONCORDANCE_OK.
Receipt: `/scratch/sounio-stdlib-wave3-20261027/services/active-log-after-03/results`.
Source/compiler/reference/executable final hash guards pass. Raw compiler
SHA-256 remains9919d03afb60e418dc111797895860e27e93a22ca52a9cc51cd0e0edac1571b2.

The private adapter preserves the nonpositive -999999 sentinel, propagates
NaN and handles positive infinity before calling shared math::pure::ln_checked.
There is no independent logarithm approximation in active. Canonical checked
log requires Mut/Div, so those effects are now explicit on its eight public
active callers; existing Panic requirements remain. This is an intentional
effect-contract migration, not a weakening of effects. Public names and
metric formulas are unchanged. Before18253 and after18256/18257 preserve
harness output-routing and incomplete effect-annotation refusals, respectively.
The original bridge/test workloads and numerical thresholds are untouched.

The mandatory gate now also includes the active logarithm fixture and its
independent C++23 oracle, bringing the required plan to28 fixtures.
Slurm **18259** records22 successful fixture native markers, both ABI112
reference modes and ACTIVE_LOG_CPP_CONCORDANCE_OK. Actual Slurm execution
authority is recorded as run18259. All source/tool/derived/executable/output
final guards pass. Overall status remains1 for the same six explicit
compiler prerequisites; no canonical-root or nonempty-array check is
omitted from final acceptance.
Receipt: `/scratch/sounio-stdlib-wave3-20261027/services/reusable-gate-log-04/results/gate`.
