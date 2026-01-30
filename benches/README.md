# Sounio Benchmarks

Performance benchmarks for the Sounio compiler and runtime.

## Organization

- **compiler/** - Compiler performance (parser, typechecker, codegen)
- **misc/** - Additional benchmarks

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench --workspace

# Run specific benchmark
cargo bench compiler_bench

# With baseline comparison
cargo bench --bench compiler_bench -- --save-baseline main

# Compare against baseline
cargo bench --bench compiler_bench -- --baseline main
```

## Benchmark Categories

### Compiler (compiler/)
- `compiler_bench.rs` - Overall compiler performance
- `layout_bench.rs` - Memory layout optimization
- `locality_bench.rs` - Cache locality analysis
- `gpu_bench.rs` - GPU code generation
- `sir_gpu_bench.rs` - GPU IR performance
- `qnn_performance_bench.rs` - Quantized NN performance
- `octonion_benchmark.rs` - Octonion algebra
- `quat_bench.rs` - Quaternion operations
- `ontology_bench.rs` - Ontology queries
- `glm_performance_bench.rs` - ML-guided optimization

## Adding Benchmarks

1. Create benchmark in appropriate directory
2. Use Criterion for statistical analysis
3. Set `harness = false` in Cargo.toml
4. Document what is being measured

Example:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_feature(c: &mut Criterion) {
    c.bench_function("feature_name", |b| {
        b.iter(|| {
            // Benchmark code
        })
    });
}

criterion_group!(benches, bench_feature);
criterion_main!(benches);
```

## CI Integration

Benchmarks run on:
- Push to main
- Pull requests (comparison mode)
- Weekly scheduled runs

See `.github/workflows/` for CI configuration.
