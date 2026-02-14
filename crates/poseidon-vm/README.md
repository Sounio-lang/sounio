# Poseidon VM - Safe Rust Wrapper

Safe, idiomatic Rust wrapper around the Poseidon C VM for executing SOIR (Sounio Intermediate Representation) bytecode.

## Overview

The Poseidon VM is a lightweight bytecode interpreter written in C99 that executes SOIR bytecode. This crate provides a safe Rust API with:

- **Zero unsafe in public API** - All FFI calls wrapped in safe abstractions
- **Automatic memory management** - No leaks, proper cleanup on drop
- **Comprehensive error handling** - No panics, detailed error messages
- **Type safety** - Register bounds checking, validated bytecode loading
- **Integration with SOIR** - Works seamlessly with the `soir` crate

## Features

- Execute SOIR bytecode with configurable step limits
- Single-step debugging support
- Register and stack introspection
- VM state reset for multiple executions
- Cross-platform (Linux, macOS, Windows)

## Usage

```rust
use poseidon_vm::PoseidonVm;
use soir::serialize;

// Create VM instance
let mut vm = PoseidonVm::new()?;

// Load SOIR bytecode
let bytecode = serialize(&ir_module)?;
vm.load(&bytecode)?;

// Execute program
let exit_code = vm.execute()?;
println!("Program exited with code: {}", exit_code);
```

### Custom Step Limit

```rust
// Create VM with custom timeout
let mut vm = PoseidonVm::with_max_steps(5000)?;
vm.load(&bytecode)?;

match vm.execute() {
    Ok(code) => println!("Exited with: {}", code),
    Err(VmError::Timeout(steps)) => println!("Timeout after {} steps", steps),
    Err(e) => println!("Error: {}", e),
}
```

### Single-Step Debugging

```rust
let mut vm = PoseidonVm::new()?;
vm.load(&bytecode)?;

while vm.step()? {
    println!("PC: {}, R0: {}", vm.program_counter(), vm.register(0)?);
}

println!("Exit code: {:?}", vm.exit_code());
```

### Reset and Re-run

```rust
let mut vm = PoseidonVm::new()?;
vm.load(&bytecode)?;

// First execution
let code1 = vm.execute()?;

// Reset and run again
vm.reset()?;
let code2 = vm.execute()?;

assert_eq!(code1, code2);
```

## API Overview

### PoseidonVm

- `new()` - Create new VM instance
- `with_max_steps(i64)` - Create VM with custom step limit
- `load(&[u8])` - Load SOIR bytecode
- `execute()` - Run until halt or timeout
- `step()` - Execute one instruction
- `reset()` - Reset VM to initial state
- `register(usize)` - Get register value
- `stack_depth()` - Get call stack depth
- `program_counter()` - Get current PC
- `is_halted()` - Check if halted
- `exit_code()` - Get exit code (if halted)

### Error Handling

All operations return `Result<T, VmError>`:

- `LoadError` - Bytecode loading failed
- `ExecutionError` - Execution error occurred
- `InvalidRegister` - Register index out of bounds
- `Timeout` - Max steps exceeded
- `AlreadyHalted` - VM already halted
- `NoMainFunction` - No main function in module
- `InvalidBytecode` - Malformed bytecode

## Safety

This crate ensures memory safety through:

- **Proper lifetime management** - Module freed before VMState
- **Bounds checking** - All register accesses validated
- **Null pointer checks** - All C pointers validated before use
- **No use-after-free** - Drop implementation ensures proper cleanup
- **No double-free** - Module pointer nulled after free

All unsafe code is encapsulated in the FFI layer and wrapped in safe abstractions.

## Architecture

```
┌─────────────────────────────────────────┐
│           Rust Application              │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      PoseidonVm (Safe Wrapper)          │
│  - Load/Execute API                     │
│  - Register access                      │
│  - Error handling                       │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          FFI Layer (ffi.rs)             │
│  - C struct definitions                 │
│  - extern "C" bindings                  │
│  - Unsafe wrapping                      │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│      Poseidon C VM (bootstrap/)         │
│  - vm.c: Execution engine               │
│  - loader.c: SOIR deserializer          │
│  - runtime.c: Builtin functions         │
└─────────────────────────────────────────┘
```

## Building

The C VM is compiled automatically via `build.rs` using the `cc` crate. No manual build steps required.

```bash
cargo build
cargo test
```

## Testing

Comprehensive integration tests cover:

- Simple execution (load immediate, return)
- Arithmetic operations (add, sub, mul, etc.)
- Function calls with parameters
- Conditional branches
- Register access and bounds checking
- VM reset and re-execution
- Error handling (invalid bytecode, timeouts)
- Stack depth tracking
- Single-step debugging

Run tests:

```bash
cargo test --package poseidon-vm
```

## Performance

The Poseidon VM is optimized for:

- **Fast startup** - No JIT compilation overhead
- **Predictable execution** - No GC pauses
- **Small footprint** - ~50KB compiled size
- **Deterministic** - Same input always produces same output

Suitable for:

- Testing and validation
- Bootstrap compilation
- Embedded environments
- Deterministic replay

Not suitable for:

- Production high-performance execution (use LLVM backend)
- Long-running computations (use step limit)

## License

Same as Sounio compiler (see repository root).
