# stdlib/genomics

Mathematical genomics: DNA as GF(4) field elements, sequence operators, and format parsers.

## Modules

| Path | Status | Description |
|------|--------|-------------|
| `genomics::core` | **real** | Complement, GC, Hamming, alignment score, codon translation, k-mers |
| `genomics::io::fasta` | **real** | In-memory FASTA parser → `Seq256` |
| `genomics::gpu::gf4` | **real** | GF(4) CPU fallback kernels + self-test |
| `alignment`, `variant`, `types`, … | stub | Pending parser / design support |

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