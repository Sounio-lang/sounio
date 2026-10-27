# Separate genomic document validation

These APIs add document/graph/reference checks to the owned lexical readers.
They preserve exact identifiers and do not perform Unicode normalization,
variant normalization, ontology inference or liftover.

## GFF3 ID/Parent graph

`genomics::io::gff_graph::gff_validate_graph(d, max_attribute_bytes,
max_index_bytes, max_graph_bytes, max_byte_comparisons)` borrows a
lexically validated `DynamicGff` and returns `GffGraphReport`.

Attribute keys and ID/Parent values are percent-decoded. Parent lists split
at raw commas before decoding, so `Parent=a%2Cb` names the single ID
`a,b`. Duplicate reserved attributes and raw comma lists in scalar ID
attributes refuse. Parent references may precede their declarations.

The exact-byte `CountIdentities` index groups names. Private reversible
ASCII hex encoding permits opaque decoded bytes without changing the
existing count-index byte policy; hashes never stand in for identity.
Canonically equivalent Unicode spellings remain distinct. Repeated IDs
represent one named node; this validation profile requires equal decoded
seqid, type and strand across its rows. It combines Parent occurrences from
those rows. It does not resolve ontology aliases or cross-sequence multipart
interpretations.

The graph resolves every Parent, including those on anonymous leaf features.
Reverse CSR and Kahn elimination detect cycles. On a cycle, row and
related_row identify representative rows for an actual edge of a cycle.
Nodes counts distinct declared IDs. Edges counts Parent occurrences from
named features, including repeated occurrences in multipart rows. Anonymous
leaves have no name by which another row could refer to them.

The [GFF3 specification](https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md)
allows discontinuous features to repeat an ID and does not require child
coordinates to lie within parent coordinates. This graph validator therefore
applies no containment rule. Sequence Ontology Part-Of membership needs
separately supplied ontology data and is outside this API's success claim.

Memory budgets are explicit and separate:

- Each of two scratch buffers holds at most max_attribute_bytes.
- max_index_bytes covers the four exact-key identity payloads. Each decoded
  key byte becomes two encoded byte values stored in the existing integer
  byte pool. `count_identities_payload` supplies the checked payload query.
- max_graph_bytes covers16 bytes/token metadata,8 bytes/physical row,
  four8-byte arrays per distinct key plus one offset sentinel, and
 8 bytes/Parent occurrence for adjacency.
- max_byte_comparisons counts encoded byte-pair comparisons in the existing
  identity sort. Validation scans and the failure-only cycle witness walk
  are separate work.

Status0 means the checks above passed. Status22 means malformed attributes,
incompatible multipart profile or a cycle;95 is unresolved Parent;105 is
payload/comparison budget;12 is allocation;75 is arithmetic overflow.
Reasons:1 lexical input,2 attributes,3 multipart compatibility,4 missing
Parent,5 cycle,6 payload/configuration,7 comparison budget. Row indices are
zero-based; -1 means unavailable. Comparisons reports consumed identity work.

## VCF4.3 document context

`genomics::io::vcf_document::vcf_validate_document(input,
max_header_bytes, max_record_bytes, max_index_bytes, info_ploidy,
max_byte_comparisons)` applies the existing owned VCF4.3 header/schema and
record validators, then checks document ordering and identities.

Every CHROM occupies one contiguous block. POS is nondecreasing within its
block, and ties are allowed. Each nonmissing record ID is globally unique,
including IDs within semicolon lists. A final data line requires LF (CRLF
is accepted). These rules follow the targeted
[VCF4.3 specification](https://samtools.github.io/hts-specs/VCFv4.3.pdf);
this module does not claim the later VCF4.4/4.5 profiles.

The existing exact-byte identity index uses separate private namespaces for
CHROM block names and record IDs. Equal hashes are never accepted as equal
identities. No case folding or Unicode normalization occurs on identifiers.
The index budget includes identity payloads plus24 bytes/source-token
metadata; header and reusable record byte budgets are separate.

`VcfDocumentReport` returns status, reason, zero-based record and byte
error_offset, related_record, records and comparisons. Records counts rows
validated lexically before a failure; identity failures are discovered after
the complete lexical scan. Reasons:1 header/record syntax,2 decreasing POS,
3 reopened CHROM,4 duplicate ID,6 payload/configuration,7 comparison budget.
Statuses retain lexical/header errors and use22 for ordering/identity,
105 for budget and12 for allocation.

## Explicit FASTA REF concordance

`vcf_record_validate_reference(record, fasta, max_scratch_bytes)` is
separate from document validation. It borrows an owned `DynamicFasta` and
one VCF record. Each of its two scratch buffers has max_scratch_bytes payload.
The exact first FASTA header token identifies the contig. Missing contigs
return95; multiple matching FASTA IDs refuse with22.

`dynamic_fasta_copy_range` appends a checked zero-based sequence window
without copying the entire contig. Bounds and reservation failures preserve
the output prefix. It scans from the record's sequence start to the requested
window; it is not a random-access genomic index. REF validation checks the
complete concrete one-based window and folds ASCII base case explicitly.
N matches literal N, not any base. A mismatch returns22 and its zero-based
offset inside REF; range failures use mismatch_offset -1.

Structural alleles, telomere coordinates outside the current lexical core,
breakend resolution, variant normalization, circular-reference rules and
ontology membership require separate APIs/data. Success here asserts only
the declared document and concrete REF checks.
