# Dynamic incremental count TSV

`genomics::io::dynamic_count_tsv` is an integer-only incremental count-table
decoder. It accepts caller-supplied byte chunks in IntegerBuffer owners and
publishes one completed count row at a time in caller scratch. It does not read
files, recognize sample/gene identifiers, parse headers, validate a biological
dataset or implement a whole-file genomics pipeline. The bounded legacy
`count_tsv.sio` is unchanged.

The generated current-source reference program completed natively with exit0
at central Slurm job16124. Its scope includes all small split/byte cases, the
512-sample ×20000-row stream and1,000,000 single-byte feed calls. The reported
fixed-program resource observation is16.26 seconds and MaxRSS1668KiB. This is
focused execution evidence, not a whole-file I/O pipeline, full genomic-format
coverage, comparative performance benchmark, CI or merge acceptance. Exact
source/compiler/generated-artifact identities belong to the central receipt.

## API and storage contract

- `dynamic_count_tsv_new(samples,max_count)->DynamicCountTsv`.
- `dynamic_count_tsv_info(&state)->DynamicCountTsvInfo`.
- `dynamic_count_tsv_feed_slot()->DynamicCountTsvFeed` initializes reusable metadata.
- `dynamic_count_tsv_feed_into(&!state,&!row,&bytes,start,length,&!slot)->i64`.
- `dynamic_count_tsv_ack_row(&!state,&!row)->bool`.
- `dynamic_count_tsv_finish_into(&!state,&!row,&!slot)->i64`.
- `dynamic_count_tsv_append_row(&state,&row,&!table,rows)->DynamicCountMatrixAppend`.

The parser state contains private scalar fields. Clients obtain counters,
readiness and error metadata through a public snapshot, rather than modifying
the parser's intermediate decimal or CRLF state. No public field permits
fabricating a different sample count or injecting an accumulated decimal value.

The state does not own a row allocation. The caller provides a successful
IntegerBuffer of EXACT length samples and keeps that SAME owner and its
contents unchanged across a partial row, through readiness and until ack.
Ordinary linear borrows prevent overlapping mutable calls, but the API does
not authenticate owner identity across separate calls. Replacing or modifying
row scratch between calls violates this caller contract. Private parser fields
are not an authenticated capability tying arbitrary row data to the byte stream.

The caller also owns input chunks. Feed borrows them and never modifies them.
The start/length interval is checked by subtraction before computing an indexed
offset, including negative and out-of-range inputs. A chunk's unconsumed suffix
can be supplied again with an advanced start; it need not be copied or sliced.

The canonical feed_into/finish_into APIs update caller-owned reusable metadata
and return scalar status. Their byte loop and return paths construct no new
aggregate record. All hot parser/append APIs have no Alloc effect. Feed and
append have Mut,Div,Panic; ack and finish require Mut. No fixed sample8, chunk256 or row256 limit
is retained. Convenience feed/finish adapters preserve the same parsing rules
but return a record by value; on the current managed compiler that creates a
long-lived handle per call. Long-running or tiny-chunk consumers must use the
into APIs with one reusable slot. The by-value info snapshot is for checkpoints,
not a per-byte telemetry loop. Practical sizes remain bounded by the caller's IntegerBuffer
lengths/budgets and checked i64 counters. There is no hidden growing matrix.

## Grammar, limits and row boundaries

Each field is a nonempty sequence of ASCII digits0 through9. Leading zeros are
accepted. A TAB separates fields. LF or CRLF terminates a row; CRLF can be split
between chunks. A final unterminated row is accepted only when finish confirms
EOF. A CR without its following LF is not accepted as a row terminator.

Negative/positive signs, spaces, empty fields, blank lines, NUL, non-ASCII bytes
and IntegerBuffer elements outside0..255 are invalid. Exactly samples fields
are required per completed row. Empty input can finish successfully with zero
rows; a consumer requiring genes must make that separate decision.

max_count is an explicit inclusive nonnegative i64 limit. Passing i64MAX
supports the full nonnegative i64 parsing domain. Passing2^53 explicitly selects
the inclusive lossless-binary64 count boundary required by some model kernels.
The parser performs no f64 conversion or inference about downstream models.

Before decimal multiplication/addition, the accumulated value and new digit
are compared with max_count/10 and max_count%10. A rejected overflowing digit
is never multiplied into the accumulator. The byte position is likewise
checked before increment; row count is checked before a completion increment.
No wraparound is used as overflow detection.

Feed stops immediately after one completed row and returns consumed relative
to its supplied start. It can return less than length. While a row is ready,
another valid feed interval consumes0 and leaves it ready. The caller reads
or appends the row, acknowledges it, then resumes the untouched suffix.
Ack zeroes active row scratch and resets its field parser. Ack without a ready
row returns false and does not invent a new row.

Finish pauses if a newline-terminated ready row has not been acknowledged;
it does not mark finished and silently discard an unconsumed caller suffix.
After that row is acknowledged, the caller either resumes its suffix or calls
finish again if EOF is known. For an unterminated final row, finish can return
both row_ready=true and finished=true; the caller still acknowledges the row.
A nonempty feed after finished is a sticky protocol error. Empty feed/finish
after completed EOF remains a no-op.

## Results and exact positions

The reused DynamicCountTsvFeed slot exposes status,consumed,row_ready,finished,position,
error_position. DynamicCountTsvInfo additionally exposes samples,max_count,rows.
Rows counts completed records, including the currently ready record before ack.
It is not an appended-table row count.

position is the number of accepted stream bytes, starting at0. On a syntax or
count-limit failure, error_position is the absolute zero-based position of the
first rejected byte and that byte is not consumed. On incomplete EOF it is the
position just after all accepted bytes. A pending CR has already been consumed;
an incorrect following byte or missing LF fails at that next byte/EOF position.

Configuration, interval, scratch and protocol errors use the current position
because no new stream byte was accepted. The API does not inspect a filesystem:
these are offsets in the byte sequence actually supplied by the caller. Feeding
a duplicate or altered suffix cannot be detected as a different physical file.

| Status | Meaning |
| --- | --- |
| 0 | Valid progress, row-ready pause or completed EOF |
| 1 | Invalid samples/max_count, failed owner or wrong row length |
| 2 | Invalid chunk start/length interval |
| 3 | Invalid byte/ASCII grammar, empty field or incomplete CRLF/EOF |
| 4 | Too few or too many columns |
| 5 | Count exceeds explicit limit; append also rejects invalid row values |
| 6 | Invalid protocol or append without an available completed row |
| 7 | Stream-position, completed-row or append-row counter overflow |
| 8 | Matrix append capacity exhausted |

Parser errors are sticky. Subsequent calls return the same failure without
consuming more input. Construct a new state for a new stream. Row scratch can
contain partial fields on a parser failure; only a ready row from status0 is
available to a conforming consumer. The parser does not promise row-scratch
rollback or replace an invalid field with zero.

## Transactional packed matrix append

Append requires parser status0 and row_ready. It validates the supplied row
length, all values0..max_count, the destination owner, nonnegative rows, row-count
addition and whole-row capacity before modifying any destination entry.

The table uses packed row-major stride=samples. Its length, not an allocation
hidden in the parser, determines capacity. The quotient table_length/samples is
checked before computing rows*samples or rows+1; trailing cells insufficient
for a complete row are unused. Every append failure leaves the table unchanged
and returns the supplied rows unchanged. Success returns rows+1.

The append rows argument is caller bookkeeping and can differ from decoded rows
because a streaming consumer may discard rows instead of storing every one.
The function is not an authenticated proof that a caller-modified valid row
matches its earlier byte stream. No QC totals, sample IDs or model assumptions
are certified here.

## Verification inputs and limits

`stdlib_deep_dynamic_count_tsv_status.sio` checks CRLF split, ready-row pause,
unconsumed suffix resumption, EOF acknowledgement, packed append, capacity and
invalid-row transactional failure, chunk/row-shape rejection and12 samples.

`tests/reference/deep_dynamic_count_tsv_reference.cpp` uses a separate
line-oriented C++23 parser and std::from_chars. On a count overflow it locates
the first failing prefix; it does not duplicate the production multiply/add
recurrence. The emitter provides23 small cases, each exercised over EVERY
two-chunk split plus individual-byte chunks. Cases include LF/CRLF, final EOF,
empty fields, signs, wrong column counts, i64MAX and i64MAX+1, the explicit2^53
boundary and bytes outside the allowed domain.

The same emitter generates a procedural512-sample ×20000-row stream test.
Sounio stores only its reusable4096-element byte chunk,512-element row and
scalar producer/parser/checksum state; it does not allocate a whole file or
the complete count table. LF/CRLF alternate and the last record is unterminated.
Both the large stream and small split/reference paths use the canonical into
APIs. An additional1,000,000 single-byte feed sequence (alternating0 and LF)
uses one row cell and one reusable result slot, acknowledging500,000 rows.
It checks every consumed count and completed zero value, then checks exact
final byte/row counts from the independent C++ parser, without constructing
per-byte result or info snapshots. The independent C++ route materializes
one row at a time, parses its decimal
fields with from_chars, and supplies the expected consumed-byte count and
ordered modular checksum.

The checksum recurrence is (h*131+value%1000000007)%1000000007. Its arithmetic is
bounded within i64, but it is a regression checksum, not a cryptographic digest
or proof against adversarial collisions. File/source custody uses separate
cryptographic manifests in the central execution record.

Build the emitter on the authorized worker with C++23 -O2 -Wall -Wextra
-pedantic. Generate
`tests/run-pass/stdlib_deep_dynamic_count_tsv_reference.sio` and coordinate
current-source checks, compilation and native execution through Slurm.
No code was executed locally by the implementation agent.

## Executed resource boundary

The accepted large cases use the canonical into APIs and a reusable metadata
slot. The1,000,000-call path acknowledges500,000 zero-count rows and obtains
its info snapshot only after the stream. This demonstrates that this fixed
workload completed without the per-call by-value handle exhaustion that the
canonical API was designed to avoid. It does not prove arbitrary allocation
behavior for every caller, compiler or chunk schedule. The convenience
by-value wrappers remain unsuitable as a claimed unlimited managed hot path.
No test or compiler was executed locally by the implementation agent.
