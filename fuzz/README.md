# Sounio Compiler Fuzzing

This directory contains fuzzing targets for the Sounio compiler using `cargo-fuzz` and `libfuzzer`.

## Prerequisites

Install `cargo-fuzz`:

```bash
cargo install cargo-fuzz
```

Note: Fuzzing requires the nightly Rust toolchain.

## Available Fuzz Targets

### `fuzz_lexer`
Tests the lexer with arbitrary byte sequences. The lexer should handle any valid UTF-8 input without panicking.

```bash
cargo +nightly fuzz run fuzz_lexer
```

### `fuzz_parser`
Tests the parser with structured fuzzing that generates syntactically plausible Sounio code. Uses the `arbitrary` crate to generate token-like fragments.

```bash
cargo +nightly fuzz run fuzz_parser
```

### `fuzz_typecheck`
Tests the type checker with complete, syntactically valid programs. Uses structured fuzzing to generate well-formed ASTs that exercise the type system.

```bash
cargo +nightly fuzz run fuzz_typecheck
```

## Running Fuzzing

### Basic Usage

Run a fuzz target for a specific duration:

```bash
# Run indefinitely (Ctrl+C to stop)
cargo +nightly fuzz run fuzz_lexer

# Run for 60 seconds
cargo +nightly fuzz run fuzz_lexer -- -max_total_time=60

# Run with multiple jobs (parallel fuzzing)
cargo +nightly fuzz run fuzz_lexer -- -jobs=4 -workers=4
```

### Useful Options

```bash
# Limit input size
cargo +nightly fuzz run fuzz_lexer -- -max_len=1024

# Use a corpus directory
cargo +nightly fuzz run fuzz_lexer corpus/lexer/

# Run with address sanitizer (default)
cargo +nightly fuzz run fuzz_lexer

# Run with memory sanitizer (requires special setup)
cargo +nightly fuzz run fuzz_lexer --sanitizer=memory

# Print statistics every 10 seconds
cargo +nightly fuzz run fuzz_lexer -- -print_pcs=1 -print_coverage=1
```

### Reproducing Crashes

When a crash is found, it is saved to `fuzz/artifacts/<target>/`:

```bash
# Reproduce a crash
cargo +nightly fuzz run fuzz_lexer fuzz/artifacts/fuzz_lexer/crash-<hash>

# Minimize the crash input
cargo +nightly fuzz tmin fuzz_lexer fuzz/artifacts/fuzz_lexer/crash-<hash>
```

### Coverage

Generate coverage reports:

```bash
cargo +nightly fuzz coverage fuzz_lexer
```

## Corpus Management

The fuzzing corpus is stored in `fuzz/corpus/<target>/`. You can seed the corpus with interesting inputs:

```bash
# Add example Sounio files to corpus
cp examples/*.d fuzz/corpus/fuzz_parser/
```

## Structured Fuzzing

The parser and type checker fuzz targets use structured fuzzing via the `arbitrary` crate. This generates syntactically valid code fragments rather than random bytes, which helps explore deeper parts of the compiler.

Key structures:
- `FuzzInput` - Generates token-like fragments for the parser
- `FuzzProgram` - Generates complete programs for the type checker
- `FuzzType` - Generates valid type expressions
- `FuzzExpr` - Generates valid expressions

## Continuous Fuzzing

For CI/CD integration, consider running fuzzing as part of your test suite:

```bash
# Run each target for 30 seconds
for target in fuzz_lexer fuzz_parser fuzz_typecheck; do
    cargo +nightly fuzz run $target -- -max_total_time=30
done
```

## Troubleshooting

### "error: could not compile"
Make sure you're using the nightly toolchain:
```bash
rustup default nightly
# or
cargo +nightly fuzz run fuzz_lexer
```

### Out of memory
Limit the input size:
```bash
cargo +nightly fuzz run fuzz_lexer -- -max_len=4096 -rss_limit_mb=2048
```

### Slow fuzzing
- Use multiple workers: `-jobs=N -workers=N`
- Reduce input size: `-max_len=1024`
- Disable coverage instrumentation for dependencies

## Integration with OSS-Fuzz

This fuzzing setup is compatible with [OSS-Fuzz](https://github.com/google/oss-fuzz). To integrate:

1. Create a `project.yaml` in the OSS-Fuzz repo
2. Create a Dockerfile that builds the fuzz targets
3. Submit a PR to OSS-Fuzz

See the OSS-Fuzz documentation for details.
