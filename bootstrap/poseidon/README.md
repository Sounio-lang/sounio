# Poseidon - SOIR v1 Bytecode VM

**Stage 0 VM for Sounio self-hosted bootstrap**

## Overview

Poseidon is a minimal, correct C99-based virtual machine for executing SOIR (Sounio Intermediate Representation) bytecode. It serves as the bootstrap VM that enables the self-hosted Sounio compiler to compile itself.

## Architecture

- **Register-based VM**: 1024 general-purpose registers
- **Stack for calls**: 1024-deep call stack
- **No GC**: Manual memory management only (for bootstrap phase)
- **No JIT**: Pure interpreter
- **Instruction set**: Matches self-hosted IR exactly

## Components

| File | Purpose |
|------|---------|
| `opcodes.h` | Opcode definitions (matches SOIR v1 spec) |
| `vm.h/c` | Core execution engine |
| `loader.h/c` | SOIR bytecode deserializer |
| `runtime.h/c` | Minimal runtime (print, panic) |
| `main.c` | Entry point |

## Building

```bash
make
```

Produces `poseidon` executable.

## Usage

```bash
./poseidon program.soir
```

Exit code is the return value of `main()`.

## SOIR Binary Format

See `self-hosted/ir/serialize.sio` for full specification.

Header (8 bytes):
- Magic: "SOIR" (0x534F4952 LE)
- Version: 1
- Reserved: 3 bytes padding

Body:
- fn_count: i64
- functions[]: serialized IrFunction
- string_count: i64
- strings[]: serialized Name

## Supported Opcodes

- `LOAD_IMM`, `LOAD_BOOL` - Load immediate values
- `COPY` - Register copy
- `BINOP` - Binary operations (add, sub, mul, div, cmp, etc.)
- `UNARYOP` - Unary operations (neg, not)
- `CALL`, `RETURN` - Function calls
- `JUMP`, `BRANCH_TRUE`, `BRANCH_FALSE` - Control flow
- `LABEL` - Jump target marker
- `NOP` - No operation

## Builtins

- `print_int(i64)` - Print integer to stdout

## Limitations

- No heap allocation (bootstrap only needs stack)
- No float support (only i64)
- Max 1024 registers
- Max 1024 call depth
- Max 10000 execution steps
- No string operations (except literals)

## Testing

```bash
make test
```

Runs all test SOIR files in `tests/`.

## License

Part of the Sounio project.
