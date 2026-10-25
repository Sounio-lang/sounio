<!-- docs:meta
topic_id: repo.stdlib.genomics.count-identities-csr
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.stdlib.genomics.count-identities-csr
-->

# Named integer counts, identity joins and canonical CSR

These modules extend the existing incremental integer parser and exact-median normalization. They do not infer biological identifiers, merge spelling variants, impute missing counts, choose a normalization method, or calibrate a differential-expression model.

## Dynamic named TSV adapter

`genomics::io::named_count_tsv` exposes `named_count_tsv_new(samples,max_id_bytes)`, `named_count_tsv_feed(state,bytes,length,id_scratch,count_row)`, `named_count_tsv_ack(state)` and `named_count_tsv_finish(state,id_scratch,count_row)`.

The first line is a corner label followed by exactly `samples` sample IDs. An empty corner label is permitted; its text is ignored. Every subsequent line is a nonempty feature ID followed by exactly `samples` nonnegative i64 decimal counts. The sample count is caller-declared and is dimension-dynamic. A tested 512-sample table is not a maximum. The caller must allocate an IntegerBuffer of at least max_id_bytes for ID scratch and another of at least samples for the row. There is no transport, file opening or decompression in this API.

IDs are byte-exact sequences: values 1..255 except TAB/LF/CR. Spaces, case, composed/decomposed UTF-8 and other opaque bytes remain distinct. No Unicode validation or normalization is implied. The pools and ID scratch store one byte value in each i64 cell: the payload is eight times the byte count, not compact u8 storage. Lengths and offsets refer to byte counts. A corner label also obeys the explicit ID length limit even though its text is ignored.

Each feed accepts up to 256 caller bytes. LF and CRLF are accepted, including a CR/LF split across calls; a lone CR refuses. Counts use the original `count_tsv` numeric state machine, including its pre-multiply i64 overflow check. No independent decimal parser is copied into the adapter. A retained `CountTsvFeed` slot avoids constructing a parser-result record for each byte. `count_tsv_feed_into` and `count_tsv_finish_into` are reusable-result forms of the original parser API; the original snapshot-returning functions remain available.

The returned `NamedCountFeed` contains status, consumed, kind, index, id_len, rows, header_complete, finished, position and error_position. Kind 1 is one sample ID in id_scratch, with its original zero-based sample index. Kind 2 is one complete feature row, with feature ID in id_scratch and all sample counts in count_row; index is its zero-based row number. Kind 0 emits no event. Acknowledge each event after copying/using its data. An unacknowledged event pauses feed and finish without consuming more bytes. Supply only the unconsumed input suffix next. A row/ID scratch prefix is not data while parsing is partial or status is nonzero.

Position counts accepted original bytes, not the internal replacement of numeric TAB separators by LF in the one-sample parser. The offending byte is not counted. EOF errors use the accepted position, including any accepted CR. Previously acknowledged events stay committed; a malformed row emits no event. Errors are sticky. Finishing a valid final row without a newline emits it once and marks finished; acknowledge it before subsequent empty operations. Empty feed/finish after EOF are idempotent; nonempty feed after EOF refuses. An empty input is invalid because this named format requires its header.

Statuses are 0 available, 1 invalid declared shape/scratch, 2 invalid active chunk length, 3 invalid syntax/byte/empty ID, 4 wrong field count, 5 integer count overflow, 6 invalid state/input after EOF, 7 position/row counter overflow, 8 ID length budget. Count/ID duplication is deliberately handled by the identity layer after framing, not silently resolved during tokenization.

## Validated identity owners and joins

`genomics::count_identities::count_identities_payload(byte_count,n)` reports the required IntegerBuffer payload for concatenated byte values, n+1 offsets, n sort indices and n group indices. It validates sums/products before computing bytes. Allocator metadata and process RSS are separate. Construct all buffers explicitly with their budgets.

`count_identities_build(bytes,offsets,order,groups,duplicate_policy,max_byte_comparisons)` consumes four linear buffers and returns a private `CountIdentities` owner. Offsets start at zero, end at the exact byte-pool length and increase strictly for every ID. Zero IDs with an empty pool and offsets[0]=0 are valid. Build uses deterministic indexed heapsort and exact byte comparisons; ties retain original index order. It does not match hash digests.

Duplicate policy 0 refuses repeated byte-exact IDs. Policy 1 builds explicit groups with lexicographic group numbers. This groups identities only: counts are not added by this constructor. Getters report status, original count, unique count, comparison count, group for an original row, original row at a sorted position, and ID length/bytes. No mutable payload escapes. Close the owner exactly once, including after refusal.

Build statuses: 0 available, 1 invalid layout/options, 2 invalid offsets/ID bytes, 3 duplicate refusal, 4 comparison budget. The budget counts actual compared byte pairs; length-only decisions and validation scans are separate. Failed owners remain unavailable; sort/group scratch may have changed.

`count_identities_join_into(left,right,missing_policy,max_byte_comparisons,scratch,out)` performs a sorted merge and maps each original left position to an original right position. Both owners must be valid and unique; ambiguous duplicate groups refuse. Missing policy 0 refuses any absent left ID, preserving the complete output. Policy 1 explicitly emits -1 for absence. Both integer buffers must hold at least left.count elements. Only after the whole join succeeds is scratch copied to out. Join statuses: 0 committed, 1 invalid layout/options, 2 invalid owner, 4 comparison budget, 5 ambiguous duplicates, 6 missing under strict policy. The receipt reports actual matched/missing counts and compared byte pairs, including work before refusal.

## Canonical count CSR

`genomics::count_csr::count_csr_payload(rows,samples,capacity)` reports `(rows+1+2*capacity)*8` bytes for offsets, sample indices and exact i64 values. The input matrix is rows by samples, with samples positive and rows possibly zero. The query is not an allocation guarantee.

`count_csr_build(rows,samples,nnz,offsets,columns,values,duplicate_policy)` consumes the three owners. Offsets must have exactly rows+1 elements, start at zero, end at nnz and be monotone. Column/value buffers must hold at least nnz active entries. Each row's columns must be nondecreasing and within [0,samples). Counts must be nonnegative i64. Policy 0 refuses repeated columns; policy 1 explicitly adds adjacent repeated columns with checked exact i64 sums. Unsorted columns refuse. Explicit zeros are removed. A valid returned owner has strictly increasing row columns and positive stored values, so absence has the explicit structural meaning count zero.

CSR statuses are 0 available, 1 invalid shape/options, 2 invalid offsets/column/count, 3 duplicate refusal, 4 unrepresentable coalesced sum. Compaction happens within moved buffers. A failed result is closeable but is never an available partial view. Unused capacity after the active nnz is not data. The raw buffers cannot be mutated after construction. Getters provide shape, active nnz, duplicate count, row offsets, columns and stored values. `count_csr_at(row,sample)` uses binary search; it returns a nonnegative count on success and -1 for an unavailable accessor. Close each owner once.

## Explicit feature-count coalescence

`genomics::count_identity_coalesce::count_identity_coalesce_build` takes validated feature identities, a CSR with the same number of original rows, caller sample-sum scratch, three moved output buffers and an explicit point-work budget. This call explicitly declares that the supplied identity row order belongs to the source CSR; matching row counts cannot independently prove that association.

It aggregates all original rows of each byte-exact feature group using checked integer sums and emits groups in lexicographic order. The first pass determines exact required nnz and checks overflow. Capacity and the remaining budget for the same second pass are checked before writing output buffers. Work charges each accumulator reset, original-row visit, stored-count accumulation and sample scan, in both passes. The receipt reports work actually executed, not reserved future work. Metadata/factory validation is separate.

The private linear `CountCoalesced` wrapper reports status, whether required_nnz is known, its value and point_work. Statuses: 0 available, 1 layout/options, 2 invalid input/association, 3 point-work budget, 4 integer sum overflow, 5 output capacity, 6 final CSR validation. On failure the enclosed CSR remains unavailable. Read detailed status before consuming `count_coalesced_take` to obtain the CSR; alternatively close the wrapper. Taking an invalid result does not manufacture a usable prefix. Scratch may change on refusal.

## Normalization without dense materialization

`count_csr_size_factors_into` forwards the immutable payload to `dynamic_median_ratio_size_factors_csr_into` in the existing `dynamic_normalization` module. The raw-buffer adapter validates CSR offsets and sorted columns each call. Both dense and CSR representations share the exact same log-ratio, compensated sum, median selection, centering and exponentiation code. Dense public signatures remain unchanged.

Mode 0 still requires genes positive in every sample. Mode 1 still uses positive-count geometric means and per-sample positive ratios; it is an explicit alternative, never an automatic response to missing eligibility. Counts above 2^53 refuse conversion exactly as the dense API does. A missing eligible ratio or all-zero input refuses, preserving output factors/logs. There is no zero imputation or silent positive-count policy change.

Use `dynamic_median_ratio_size_factors_workspace` for scratch: 2*genes+2*samples numerical elements and genes integer anchors, in addition to the two sample-sized output vectors. The same reusable `DynamicSizeFactorsSlot` and statuses apply: 0 success, 1 shape/mode, 2 negative count, 3 no eligible ratios, 4 arithmetic, 5 count not exactly convertible to binary64. The CSR path validates/counts and computes row means by visiting stored entries. Sample medians still scan gene eligibility and look up only eligible rows. It does not allocate a genes-by-samples dense matrix.

Qualification and measured limits are recorded separately in `docs/evidence/stdlib-SIO-170/README.md`. In particular, a row-sparse benchmark is not evidence of speed or memory savings for every sparsity pattern, and CSR can use more storage than dense at high density.

## Compiler scope of encapsulation

Private-field enforcement is qualified on the canonical source-built Madaros. Actual compilation with the committed bootstrap seed rejects repeated linear consumption, but accepts the reachable private-read probes for both this CSR and the unchanged existing IntegerBuffer.size. Adding an explicit module declaration did not change that result in the retained control. The gate records these accepted reads as `PRIVATE_READ_NOT_ENFORCED_BY_BOOTSTRAP` in SEED_LIMITATIONS.tsv; they are not seed privacy passes. Numerical agreement on the seed does not establish its enforcement of the private-owner contract. Use the canonical compiler for that contract. No compiler fix is claimed by this library increment.

The retained token-slot benchmark measures repeated canonical token operations. Named-table framing additionally constructs its outer feed result per feed call and resets its scalar number state per feature row. Constant RSS in the token benchmark is not a universal memory bound for arbitrarily long named streams.
