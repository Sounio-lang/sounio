# VCF core record validation

`genomics::io::vcf` now implements checked record validation over a borrowed `ByteBuffer`. It replaces the disabled placeholder. The caller frames one record and controls the input buffer's byte budget. The result contains summary values, with no borrowed pointers or views.

`vcf_record_summary(&buffer)` accepts an optional LF or CRLF terminator and checks:

- Valid UTF-8, no leading BOM, no embedded line break/NUL/control or ASCII space.
- Eight nonempty core columns, or FORMAT plus at least one sample.
- Nonnegative decimal POS, including 0 and the maximum i64; overflow is detected before arithmetic.
- A nonmissing CHROM and a REF made from case-insensitive A/C/G/T/N.
- Nonempty comma-delimited ALT entries, with '.' representing no alternate allele.
- Lexical nonnegative decimal/scientific QUAL, or '.'. The summary reports presence; it does not decode QUAL or certify an f64 value.
- If GT occurs in FORMAT, it must be first. Alleles are nonnegative indices bounded by the ALT count, or missing '.'. Variable ploidy and mixed '/' and '|' separators are supported; empty allele tokens are rejected.

The returned `VcfRecordSummary` provides status, position, column/sample counts, alternate-allele count, and quality/genotype presence. Status 0 is success, 22 is invalid input, 75 is decimal overflow, and 84 is invalid UTF-8. `vcf_parse_position` and `vcf_genotype_status` expose the same checked helpers for callers that already know field offsets.

ID, FILTER, INFO and non-GT sample values remain opaque nonempty column payloads. This API does not validate a complete VCF file: header declarations, sample/header count concordance, INFO/FORMAT schemas, reference-contig length, symbolic/breakend allele semantics, compression and indexing remain separate work.

Fixtures cover telomeric POS=0, i64 limits/overflow, missing alleles, mixed-phase polyploid GT, out-of-range indices, misplaced GT, empty fields/alleles, malformed QUAL, invalid UTF-8, and a record over 10,000 bytes. An independent C++23 parser checks the same fixture matrix using standard-library tokenization and numeric parsing. Both ran successfully in Slurm job 17874.

Normative format reference: [VCF 4.3 specification source](https://github.com/samtools/hts-specs/blob/master/VCFv4.3.tex). This record-level contract covers the checks listed above.
