# HumanEval-Sounio Benchmark

Classic coding interview problems translated to Sounio syntax with test harnesses.
Inspired by OpenAI's HumanEval benchmark for measuring code generation capabilities.

## Problems

| # | Problem | Algorithm | Complexity |
|---|---------|-----------|------------|
| 001 | Two Sum | Brute-force pair search | O(n^2) |
| 002 | FizzBuzz | Modular arithmetic | O(n) |
| 003 | Reverse String | Two-pointer swap | O(n) |
| 004 | Palindrome Check | Two-pointer comparison | O(n) |
| 005 | Fibonacci | Iterative DP | O(n) |
| 006 | Max Subarray | Kadane's algorithm | O(n) |
| 007 | Binary Search | Divide and conquer | O(log n) |
| 008 | GCD | Euclidean algorithm | O(log min(a,b)) |
| 009 | Fast Power | Binary exponentiation | O(log n) |
| 010 | Count Primes | Sieve of Eratosthenes | O(n log log n) |

## Running

```bash
# Run all benchmarks
bash benchmarks/humaneval_sounio/run_bench.sh

# Run a single problem
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
$SOUC run benchmarks/humaneval_sounio/005_fibonacci.sio
```

## Sounio-Specific Patterns

These benchmarks demonstrate idiomatic Sounio patterns:

- **No semicolons** -- statements are newline-terminated
- **`var` for mutable bindings** -- not `let mut`
- **`&!` for exclusive references** -- not `&mut`
- **Struct wrappers for mutable arrays** -- workaround for `&![T;N]` JIT bug
- **`0 - x` for negation** -- no unary minus operator
- **Effects system** -- `with IO, Mut, Panic, Div` declares side effects
- **Fixed-size arrays** -- `[i64; 256]` with explicit length parameter
- **`assert()` not `assert!()`** -- no Rust-style macros
- **Bit shifts use u8** -- `x >> 4u8`
