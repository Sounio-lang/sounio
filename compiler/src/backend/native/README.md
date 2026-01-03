# Native Backend

The native backend for Sounio generates optimized machine code directly from SIR (Sounio Intermediate Representation) without relying on external code generators like LLVM.

## Architecture

The native backend consists of several components:

1. **Metrics Estimation** (`metrics.rs`): Cycle-accurate and power-aware estimation based on microarchitecture literature
2. **Thermal Modeling** (`thermal.rs`): Arrhenius-based thermal degradation models
3. **Register Allocation** (`alloc.rs`): Epistemic-aware register allocator using modified Linear Scan
4. **ELF Generation** (`elf.rs`): Direct ELF64 object file generation
5. **Linking** (`linker.rs`): Integration with system linkers
6. **Runtime** (`runtime.rs`): Minimal runtime for standalone executables

## Pipeline

```
Source → AST → HIR → HLIR → SIR → NativeBackend → ELF → Linker → Executable
```

## Usage

### Basic Compilation

```bash
# Compile to object file
souc build --backend=native -o output.o input.sio

# Compile to shared library
souc build --backend=native -o output.so input.sio

# Compile to executable
souc build --backend=native -o output input.sio
```

### Options

- `--thermal=<model>`: Thermal model (7nm, 5nm, conservative)
- `--alloc=<strategy>`: Allocation strategy (epistemic, greedy)
- `--arch=<arch>`: Target architecture (skylake, zen3)

## Features

### Epistemic-Aware Register Allocation

The register allocator considers epistemic metadata when making spilling decisions:

- Values with lower confidence (ε) are preferentially spilled
- High-confidence values are preserved in registers
- Provenance degradation is tracked through spill/reload cycles

### Hardware Metrics

The backend estimates:

- **Cycles**: Instruction latencies based on Agner Fog's tables
- **Power**: Energy consumption in picojoules (7nm/5nm FinFET models)
- **Thermal**: Temperature rise and degradation using Arrhenius model

### Direct ELF Generation

The backend generates ELF64 object files directly, including:

- `.text` section with machine code
- Symbol table with function/data symbols
- Relocation entries for linking
- Custom `.demetrios.epistemic` section for metadata

## Troubleshooting

### Linker Not Found

If you get "linker not found" errors:

```bash
# Install system linker
sudo apt-get install binutils  # Debian/Ubuntu
sudo yum install binutils      # RHEL/CentOS
```

### Invalid ELF Format

If generated ELF files are invalid:

1. Check that the ELF writer is generating correct headers
2. Verify section alignment and sizes
3. Use `readelf -a output.o` to inspect the file

### Register Allocation Fails

If allocation produces errors:

1. Check that epistemic metadata is being extracted correctly
2. Verify live interval computation
3. Increase spill slot size if needed

## Examples

See `compiler/examples/native/` for example programs.

## References

- Agner Fog, "Instruction Tables" (2023)
- Intel® 64 and IA-32 Architectures Optimization Reference Manual
- "Energy-Efficient Neural Networks using Data-Flow ISAs" (IEEE MICRO 2023)
- Poletto & Sarkar, "Linear Scan Register Allocation" (PLDI 1999)
