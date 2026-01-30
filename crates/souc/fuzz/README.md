# Sounio Fuzz Testing

Coverage-guided fuzzing for the Sounio compiler frontend.

## Invariant

**The lexer and parser must NEVER panic on any input.** They may return errors, but they must handle all byte sequences gracefully. This is a crash-proof requirement: adversarial input must be tolerated forever.

## Prerequisites

```bash
# Install cargo-fuzz
cargo install cargo-fuzz

# Requires nightly Rust
rustup install nightly
```

## Fuzz Targets

### `fuzz_lexer`
Tests: bytes → tokens

The lexer must handle any UTF-8 (or invalid UTF-8) input without panicking.

```bash
cargo +nightly fuzz run fuzz_lexer
```

### `fuzz_parser`
Tests: lexer output → AST

The parser must handle any token stream without panicking.

```bash
cargo +nightly fuzz run fuzz_parser
```

### `fuzz_full_pipeline`
Tests: lexer → parser → type checker

The entire compilation pipeline must handle any input without panicking.

```bash
cargo +nightly fuzz run fuzz_full_pipeline
```

## Running Fuzzing

```bash
cd compiler/fuzz

# Run a specific target (runs indefinitely until Ctrl+C)
cargo +nightly fuzz run fuzz_lexer

# Run with a timeout (e.g., 60 seconds)
cargo +nightly fuzz run fuzz_lexer -- -max_total_time=60

# Run with specific corpus directory
cargo +nightly fuzz run fuzz_lexer corpus/lexer/
```

## Reproducing Crashes

When a crash is found, libFuzzer saves the input to `artifacts/`. To reproduce:

```bash
cargo +nightly fuzz run fuzz_lexer artifacts/fuzz_lexer/crash-<hash>
```

## Corpus Management

The fuzzer builds a corpus of interesting inputs in `corpus/<target>/`. You can:

- Seed it with existing test files
- Share corpus across machines
- Minimize corpus: `cargo +nightly fuzz cmin fuzz_lexer`

## CI Integration

For CI, run fuzzing with a time limit:

```bash
cargo +nightly fuzz run fuzz_lexer -- -max_total_time=300
cargo +nightly fuzz run fuzz_parser -- -max_total_time=300
cargo +nightly fuzz run fuzz_full_pipeline -- -max_total_time=300
```
