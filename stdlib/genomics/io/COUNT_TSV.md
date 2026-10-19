# Incremental integer count TSV

Module: `genomics::io::count_tsv`.

This module parses caller-supplied byte chunks. It does not open files, decompress
streams, infer a format, or establish sample/feature identities. The caller owns
transport and supplies the declared sample order.

## Format

Each row is a feature and each column a sample. Declare 1..8 samples when creating
the state. Values contain one or more ASCII decimal digits and are nonnegative
i64 integers, including 9223372036854775807. Leading zeros are accepted.
Tabs separate fields. LF or CRLF terminate rows. CRLF may cross chunk boundaries;
a lone CR, including at EOF, is rejected. The final row may omit a terminator.

There are no headers, comments, feature-name columns, blank rows, spaces, signs,
decimal points, exponents, missing values, BOM, or trailing tabs. Reorder or
transform named tables explicitly before supplying this numeric format. Empty
input is a valid empty stream; downstream algorithms may require nonempty data.

Counts stay integer during parsing, QC, and matrix assembly. This does not alter
the documented binary64 conversion limits of downstream numerical algorithms.

## Public API and state

- `count_tsv_new(samples:i64)->CountTsvState`.
- `count_tsv_feed(state:&!CountTsvState,bytes:&[i64;256],length:i64)->CountTsvFeed`,
  effects Mut,Div,Panic. Each active buffer cell must hold a byte 0..255.
  Inactive buffer cells are ignored.
- `count_tsv_finish(state:&!CountTsvState)->CountTsvFeed`, effects Mut,Panic.
- `count_tsv_ack_row(state:&!CountTsvState)->bool`, effects Mut,Panic.

State fields are public for inspection: status,samples,row[8],ready,finished,rows,
position,error_position,columns,value,has_digits,pending_cr.
The last five parsing fields are implementation state, not a mutation API.
Construct with new; do not forge or mutate parser fields. Basic consistency
guards reject detected malformed states but are not a serialization validator.

Feed result fields: status,consumed,row_ready,finished,position,error_position.
Status zero does not mean a row is ready: inspect row_ready separately.

Feed stops immediately after a complete row terminator and leaves row[0..samples)
available. Remaining row cells are zero. It consumes no further input while
ready is true. Drain the row through QC or matrix assembly, acknowledge it, then
resubmit the unconsumed suffix. Acknowledgement clears all eight row cells and
row-local metadata; it preserves stream counters and EOF state.

Rows counts completed rows, whether or not already acknowledged. Position is a
zero-based absolute byte count: it is the offset of the next unconsumed byte.
Consumed is the number of bytes accepted from THIS feed call. A failing byte is
excluded from both counters. Error_position records that byte's offset; at EOF
it records position. A CR already accepted before a bad/missing LF remains counted.
Do not resubmit successfully consumed bytes.

Finish consumes no bytes. If a newline-terminated row is still ready, it returns
that event without committing EOF; acknowledge and drain any unconsumed chunk
suffix before calling finish again. With an unfinished valid final row it emits
one ready row and sets finished=true. Acknowledge that row; finished remains true.
At a clean row boundary it sets finished=true without a row event.
Repeated finish after completion is idempotent. An empty feed after EOF succeeds;
nonempty feed after EOF is a state error. A ready row pauses feed even after EOF.

All parser errors are sticky. Subsequent calls retain the error, consume nothing,
and expose no ready row. Start a fresh state for a new stream. Previously
acknowledged rows remain committed; a malformed current row is never emitted.
Partial current-row cells can remain in an errored state and are not valid data.

## Parser statuses

| Status | Meaning |
|---|---|
| 0 | operation available; inspect row_ready and finished |
| 1 | invalid declared sample count |
| 2 | invalid active chunk length, outside 0..256 |
| 3 | invalid byte/token syntax, empty field, or incomplete CRLF |
| 4 | wrong column count, including a separator after the last allowed field |
| 5 | integer count overflow |
| 6 | detected invalid state or input after completed EOF |
| 7 | absolute position or completed-row counter cannot advance |

Validation precedence is existing sticky error, detected state inconsistency,
chunk length, row pause, EOF, then the next byte. An excessive tab is classified
as a column error before checking whether its field is empty. For other
separators, an empty field is a syntax error. Finishing a short nonempty row is
a column error; finishing after a trailing tab is a syntax error.

Decimal overflow is checked against the maximum quotient and final digit BEFORE
multiplication or addition. Byte and row positions are also guarded before
advancing; no wraparound is used as an overflow detector.

## Incremental QC

`count_qc_new(samples:i64)->CountQcState`.

`count_qc_add_row(state:&!CountQcState,row:&[i64;8])->i64`, effects Mut,Panic.

Public state: status,samples,rows,totals[8],zeros[8]. Active columns accumulate
exact i64 counts and zero frequencies. Inactive row cells are ignored; inactive
QC cells remain zero when state is constructed normally. There is no pooled
total whose overflow could reject individually representable sample totals.

The row is validated and every prospective total checked before numerical fields
are changed. Failure changes only status; rows, totals, and zeros remain intact.
Statuses are sticky: 1 invalid samples at construction, 2 negative active count,
5 total/row-counter overflow, 6 detected malformed QC metadata. Zero is success.
Within each active column, detected-state checks precede negative-count validation;
all these checks precede overflow validation.

A zero fraction is zeros[s]/rows when rows>0. An empty stream has no defined zero
fraction. QC does not assess sequencing quality, compositional assumptions,
sample independence, normalization suitability, or biological expression changes.

## Bounded matrix adapter

`count_matrix_append_checked(matrix:&![i64;2048],rows:i64,
row:&[i64;8],samples:i64)->CountMatrixAppend`, effects Mut,Panic.

The result contains status and rows. On success, one full physical stride8 row
is written: declared samples followed by zero padding, and returned rows=rows+1.
The caller updates its row count from the result. The existing matrix prefix is
not copied, validated, or modified. All input-row checks precede all writes.

Statuses: 1 invalid samples, 6 rows outside 0..256, 8 capacity already256,
2 negative active row count, 0 success. This is also their validation precedence.
Any failure leaves the entire matrix unchanged and returns the original rows.

The adapter supplies the existing 256x8 normalization layout. A parser/QC stream
can contain more rows; the matrix adapter explicitly rejects capacity excess.
Never silently discard excess features. Never normalize independent blocks with
median-ratio normalization and describe the result as the global estimator:
the global median does not decompose that way.

## Resource and scientific boundaries

Parser state holds one row and scalar token/position metadata; QC holds eight
sample totals and eight zero counters. State does not grow with input length.
Caller buffers and the optional matrix are separate. This is a representation
bound, not a measured RSS, stack, throughput, or artifact-size claim.

This is numeric count interchange, not FASTA, Matrix Market, 10x, CSV, a feature
identity table, or a filesystem reader. No uncertainty model or inferential
interval is generated. A plot consuming QC must label totals as counts and
zero fractions with the correct row denominator.

## Native acceptance — 2026-10-19

Slurm15625 passes105/105 including the count parser/QC fixture and its native
Canvas/SVG consumer. Independent C++23 from_chars reference generates24 input
cases walked at chunk sizes1,2,3,7,256, plus dedicated transaction/capacity checks.
An injected overflow-to-zero control compiles and fails unchanged assertions
in15629. A weaker threshold mutation in15627 remains passing and is recorded as
non-discriminating. See docs/plans/HC3_COUNT_TSV_ACCEPTANCE_2026_10_19.md.
