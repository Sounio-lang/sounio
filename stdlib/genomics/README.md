# stdlib/genomics

## Checked between-sample reference-panel normalization

median_ratio_size_factors estimates median-log-ratio size factors from a
supplied raw-integer reference panel, with physical stride8,1..256 features
and1..8 samples. Zero-containing features are excluded; invalid negative
counts are detected before exclusion. The factors have explicit geometric-
mean-one scaling. A scalar helper applies them without a second dense matrix.
Close integer log contrasts are anchored before conversion; shared sort256
and two256-element buffers are reused. Composition assumptions remain
unassessed; this is not count-likelihood or differential-expression inference.
Final main-based Forge15457:95/95, retaining90 prior fixtures; independent
C++23 quad33cases and actual pre-anchor control15458:94/95.
See [contract](NORMALIZATION.md) and
[receipt](../../docs/plans/COUNTS_COVARIANCE_EXECUTION_2026_10_19.md).


Mathematical genomics: DNA as GF(4) field elements, sequence operators, and format parsers.

## Modules

| Path | Status | Description |
|------|--------|-------------|
| `genomics::core` | **real** | Complement, GC, Hamming, alignment score, codon translation, k-mers |
| `genomics::io::fasta` | **real** | In-memory FASTA parser → `Seq256` |
| `genomics::gpu::gf4` | **real** | GF(4) CPU fallback kernels + self-test |

## Usage

```sounio
let parsed = genomics::io::fasta::fasta_parse_string(">gene1\nACGT\n")
if parsed.code == genomics::io::fasta::fasta_err_ok() {
    let gc = genomics::core::dna_gc_content(&parsed.record.seq.data, parsed.record.seq.len)
}
```

Sequences use fixed `[i32; 256]` buffers with GF(4) encoding: A=0, C=1, G=2, T=3.

## Tests

- `tests/stdlib/genomics/test_genomics.sio` — core operators (check-only)
- `tests/stdlib/genomics/test_fasta_parse.sio` — FASTA parser (check-only)
- `tests/stdlib/genomics/test_gf4_gpu_e2e.sio` — GF(4) self-test (check-only)
## Expression normalization

`genomics::expression` now provides a two-pass streaming TPM normalizer:
`tpm_new`, `tpm_accumulate`, and `tpm_value`. It consumes supplied counts and
positive effective lengths in a common unit; it does not estimate read assignments
or effective lengths from sequence data. Transcript-level TPM is a typical use.
All supplied records must belong to the same sample in both passes.

The accumulator uses compensated summation and has no fixed array capacity.
Any invalid row invalidates that normalizer. Empty/all-zero totals, negative
counts, invalid lengths, non-finite inputs, and detected numeric failures are
observable through status fields. Normalization is not differential-expression
inference or a substitute for a count likelihood model.

The native fixture exercises three unequal lengths, scaling invariance,
5,000 streamed records, and a compensated-sum stress case. Execution and limits:
[Multi-domain receipt](../../docs/plans/STDLIB_MULTIDOMAIN_EXECUTION_2026_10_17.md).
