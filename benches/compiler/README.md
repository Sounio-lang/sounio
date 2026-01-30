# Sounio Compiler Benchmarks

This directory contains performance benchmarks for the Sounio compiler using the [Criterion](https://docs.rs/criterion) benchmarking framework.

## Available Benchmarks

### `compiler_bench.rs`
Comprehensive benchmarks for the core compiler pipeline:

- **Lexer Benchmarks**: Tokens/sec, lines/sec, throughput
- **Parser Benchmarks**: AST construction speed, various program sizes
- **Type Checker Benchmarks**: Type inference and checking performance
- **Interpreter Benchmarks**: Execution speed, loop iterations, recursion
- **Full Pipeline Benchmarks**: End-to-end compilation time
- **Memory Benchmarks**: AST and token stream memory usage

### `layout_bench.rs`
Benchmarks for semantic clustering and cache optimization:
- Cache simulation for different workloads
- Hierarchical clustering algorithms
- Layout effectiveness validation

### `locality_bench.rs`
Benchmarks for memory locality and access patterns.

### `ontology_bench.rs`
Benchmarks for the ontology and semantic type system:
- Hierarchy building and indexing
- Semantic distance calculations
- ANN index operations

### `gpu_bench.rs`
GPU codegen benchmarks (requires `gpu` feature).

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench compiler_bench

# Run benchmarks matching a pattern
cargo bench -- lexer
cargo bench -- "parser/simple"

# Run with release optimizations (default)
cargo bench --release
```

### Benchmark Options

```bash
# Save baseline for comparison
cargo bench -- --save-baseline main

# Compare against baseline
cargo bench -- --baseline main

# Generate HTML report
cargo bench -- --plotting-backend gnuplot

# Quick run (fewer samples)
cargo bench -- --quick

# Specific number of samples
cargo bench -- --sample-size 50
```

### Filtering

```bash
# Run only lexer benchmarks
cargo bench --bench compiler_bench -- lexer

# Run only parser benchmarks for small programs
cargo bench --bench compiler_bench -- "parser/simple"

# Run typecheck benchmarks
cargo bench --bench compiler_bench -- typecheck
```

## Benchmark Output

Results are saved to `target/criterion/`:
- `target/criterion/<group>/<name>/` - Individual benchmark data
- `target/criterion/report/` - HTML reports

Open `target/criterion/report/index.html` in a browser to view results.

## Writing New Benchmarks

### Basic Benchmark

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn my_benchmark(c: &mut Criterion) {
    c.bench_function("my_function", |b| {
        b.iter(|| my_function(black_box(input)))
    });
}

criterion_group!(benches, my_benchmark);
criterion_main!(benches);
```

### Parameterized Benchmark

```rust
fn parameterized_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_group");

    for size in [10, 100, 1000] {
        group.bench_with_input(
            BenchmarkId::new("operation", size),
            &size,
            |b, &s| b.iter(|| operation(s)),
        );
    }

    group.finish();
}
```

### Throughput Measurement

```rust
fn throughput_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    let input = generate_input();

    group.throughput(Throughput::Bytes(input.len() as u64));
    group.bench_function("process", |b| {
        b.iter(|| process(black_box(&input)))
    });

    group.finish();
}
```

## Performance Tips

1. **Use `black_box`**: Prevents compiler from optimizing away the computation
2. **Prepare input outside `iter()`**: Setup should not be measured
3. **Use `Throughput`**: For meaningful comparisons across different sizes
4. **Use benchmark groups**: For related benchmarks with shared setup
5. **Run multiple times**: For stable results on a quiet system

## CI Integration

Add to your CI pipeline:

```yaml
benchmark:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v2
    - name: Run benchmarks
      run: cargo bench --no-run && cargo bench -- --noplot

    # Optional: Compare against main branch
    - name: Compare benchmarks
      run: |
        git fetch origin main
        git checkout origin/main
        cargo bench -- --save-baseline main
        git checkout -
        cargo bench -- --baseline main
```

## Benchmark Results

### Expected Performance

Based on internal testing, approximate performance on a modern workstation:

| Metric | Expected Range |
|--------|---------------|
| Lexer throughput | 50-100 MB/s |
| Parser throughput | 10-50k lines/sec |
| Type check | 1-10k items/sec |
| Interpreter | 100k-1M ops/sec |

Actual numbers depend on:
- CPU speed and cache size
- Memory bandwidth
- Program complexity
- Enabled features

## Troubleshooting

### Noisy Results
- Close other applications
- Disable CPU frequency scaling
- Use `--sample-size` to increase samples
- Run on a quiet system

### Benchmark Too Slow
- Use `--quick` for faster iteration
- Reduce input sizes during development
- Profile with `cargo flamegraph`

### Compiler Optimizations
- Always use `black_box` for inputs
- Check that work isn't being eliminated
- Verify output is used

## Related

- [Criterion User Guide](https://bheisler.github.io/criterion.rs/book/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
- [Flamegraph profiling](https://github.com/flamegraph-rs/flamegraph)
