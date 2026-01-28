# Integration Tests for CLI, REPL, and LSP

This document describes the comprehensive integration test suites for user-facing components of the Sounio compiler.

## Overview

Three integration test suites have been implemented to address issue 3.1.1 (P0 severity) from the audit:

1. **CLI Integration Tests** (`integration_cli.rs`)
2. **REPL Integration Tests** (`integration_repl.rs`)
3. **LSP Integration Tests** (`integration_lsp.rs`)

These tests verify actual end-to-end user workflows rather than just unit-level functionality.

## CLI Integration Tests

**File**: `tests/integration_cli.rs`

**Coverage**: Tests for the `souc` command-line interface.

### Test Categories

1. **Check Command Tests**
   - Valid file type checking
   - Syntax error detection
   - Type error detection
   - File not found handling
   - Flag behavior (--show-ast, --show-types, --show-effects)

2. **Run Command Tests**
   - Simple program execution
   - Arithmetic evaluation
   - Variable binding
   - Runtime error handling

3. **JIT Command Tests** (requires `--features jit`)
   - JIT compilation and execution
   - Optimization flag handling
   - Feature detection when JIT not available

4. **Build Command Tests**
   - Native backend compilation
   - Optimization levels
   - Output file generation

5. **Help and Version Tests**
   - --help output
   - --version output
   - Command-specific help

6. **Complex Program Tests**
   - Function definitions
   - Function calls
   - Multi-file handling

### Running CLI Tests

```bash
# Run all CLI tests
cargo test --test integration_cli

# Run specific test
cargo test --test integration_cli test_cli_check_valid_file

# With JIT feature
cargo test --test integration_cli --features jit
```

## REPL Integration Tests

**File**: `tests/integration_repl.rs`

**Coverage**: Tests for the interactive REPL (Read-Eval-Print Loop).

### Test Categories

1. **Basic REPL Tests**
   - Startup and banner display
   - Simple expression evaluation
   - Arithmetic operations
   - Negative numbers

2. **Variable Binding Tests**
   - let bindings
   - var bindings
   - Variable persistence across lines
   - Variable shadowing

3. **REPL Command Tests**
   - :help command
   - :env command (show environment)
   - :clear command
   - :ast, :types, :jit toggles
   - :funcs command

4. **Function Definition Tests**
   - Multi-line function definitions
   - Function listing

5. **Error Recovery Tests**
   - Syntax error recovery
   - Type error recovery
   - Undefined variable handling
   - Continued execution after errors

6. **Complex Expression Tests**
   - Nested arithmetic
   - Variable expressions
   - Multiple operations

7. **Epistemic Features Tests**
   - :confidence command
   - :info command
   - Epistemic metadata display

8. **Multi-line Input Tests**
   - Function definitions spanning multiple lines
   - Brace counting

9. **Exit Tests**
   - :quit command
   - :q shorthand
   - Goodbye message

### Running REPL Tests

```bash
# Run all REPL tests
cargo test --test integration_repl

# Run specific test
cargo test --test integration_repl test_repl_simple_expression

# With output
cargo test --test integration_repl -- --nocapture
```

### Test Implementation Notes

REPL tests use a subprocess-based approach:
- Spawn REPL process with stdio pipes
- Send input via stdin
- Capture output via stdout
- Verify expected behavior

## LSP Integration Tests

**File**: `tests/integration_lsp.rs`

**Coverage**: Tests for the Language Server Protocol implementation.

**Requirements**: Requires `--features lsp` to compile and run.

### Test Categories

1. **Initialization Tests**
   - Server initialization handshake
   - Capabilities negotiation
   - Shutdown sequence

2. **Document Lifecycle Tests**
   - textDocument/didOpen
   - textDocument/didChange
   - textDocument/didClose
   - Multiple documents

3. **Diagnostics Tests**
   - Syntax error reporting
   - Type error reporting
   - Diagnostic publication

4. **Hover Tests**
   - Hover information requests
   - Type information display

5. **Completion Tests**
   - Code completion requests
   - Completion items

6. **Go to Definition Tests**
   - Definition location requests
   - Function definitions

7. **Error Recovery Tests**
   - Continued operation after invalid documents
   - Recovery from errors

8. **Stress Tests**
   - Rapid document changes
   - Multiple concurrent documents

### Running LSP Tests

```bash
# Run all LSP tests (requires lsp feature)
cargo test --test integration_lsp --features lsp

# Run specific test
cargo test --test integration_lsp test_lsp_initialize --features lsp
```

### Test Implementation Notes

LSP tests use JSON-RPC over stdio:
- Spawn `sounio-lsp --stdio` process
- Send JSON-RPC messages with Content-Length headers
- Parse JSON-RPC responses
- Verify protocol compliance

## Test Infrastructure

All integration tests share common patterns:

### Binary Discovery

Tests locate the compiled binaries in `target/debug/` or `target/release/`:
- `souc` (CLI)
- `sounio-lsp` (LSP server)

If binaries don't exist, tests will attempt to build them automatically.

### Temporary Files

Tests use `tempfile` crate for safe temporary file handling:
```rust
let temp_dir = TempDir::new().unwrap();
let test_file = create_test_file(&temp_dir, "test.sio", "...");
```

### Process Management

Tests spawn subprocesses and manage their lifecycle:
- Proper cleanup in Drop implementations
- Timeout handling for long-running operations
- Signal handling for graceful shutdown

## Running All Integration Tests

```bash
# Run all integration tests for CLI, REPL, and LSP
cargo test --tests

# Run with specific features
cargo test --tests --features "jit,lsp"

# Run with output
cargo test --tests -- --nocapture

# Run tests sequentially (useful for debugging)
cargo test --tests -- --test-threads=1
```

## Test Coverage Summary

| Component | Test File | Tests | Features Required |
|-----------|-----------|-------|-------------------|
| CLI | `integration_cli.rs` | 22+ | None (some require `jit`) |
| REPL | `integration_repl.rs` | 27+ | None |
| LSP | `integration_lsp.rs` | 14+ | `lsp` |

## Known Limitations

1. **LSP Tests**: Currently require a working LSP implementation. The LSP server has compilation errors that need to be fixed before these tests can run successfully.

2. **Timing Sensitivity**: Some tests use sleep/timeouts which may be sensitive to system load. Tests include reasonable timeouts but may need adjustment on slower systems.

3. **Process Communication**: REPL and LSP tests use subprocess communication which can be platform-dependent. Tests are designed to be robust but may behave differently on different operating systems.

## Quick Verification

A set of quick verification scripts are available in `/tmp/`:

```bash
# Quick CLI test
/tmp/test_cli_quick.sh

# Quick REPL test
/tmp/test_repl_quick.sh
```

These scripts test basic functionality without full cargo test overhead.

## Continuous Integration

For CI environments, run tests with:

```bash
# Basic tests (no special features)
cargo test --test integration_cli
cargo test --test integration_repl

# With features
cargo test --test integration_cli --features jit
cargo test --test integration_lsp --features lsp

# All together
cargo test --tests --features "jit,lsp"
```

## Debugging Failed Tests

When tests fail:

1. **Run with output**: `cargo test --test integration_cli -- --nocapture`
2. **Run single test**: `cargo test --test integration_cli test_name`
3. **Check binary exists**: `ls -la target/debug/souc`
4. **Test binary manually**: `./target/debug/souc --version`
5. **Check test output**: Look for error messages and stack traces

## Future Improvements

Potential enhancements to the test suites:

1. **Property-based testing**: Use proptest for fuzzing inputs
2. **Performance benchmarks**: Add timing assertions
3. **Error message validation**: More precise error message checking
4. **LSP protocol compliance**: Full LSP spec conformance testing
5. **Cross-platform testing**: Ensure tests work on Windows, macOS, Linux
6. **Integration with external tools**: Test editor integration workflows

## Contributing

When adding new CLI/REPL/LSP features:

1. Add corresponding integration tests
2. Follow existing test patterns
3. Use descriptive test names: `test_<component>_<feature>_<scenario>`
4. Document any special requirements or setup
5. Ensure tests are deterministic and reproducible
