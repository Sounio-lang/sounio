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
- Spill/reload operations are automatically inserted during code generation

#### Integration Pipeline

1. **Metadata Extraction**: Epistemic metadata is extracted from the SIR `MetadataStore` during lowering
2. **Interval Building**: Live intervals are constructed with confidence information
3. **Allocation**: `EpistemicAllocator` runs modified Linear Scan with confidence-based priority
4. **Code Generation**: `AllocResult` is passed to the emitter, which applies register assignments and inserts spill/reload operations

#### Configuration

The allocator can be configured via `AllocConfig`:
- `confidence_weight`: Weight for confidence in priority (default: 0.5)
- `spill_confidence_factor`: Confidence degradation per spill (default: 0.95)
- `min_confidence_threshold`: Below this, always spill first (default: 0.1)

Use `AllocConfig::scientific()` for scientific computing (preserves confidence) or `AllocConfig::performance()` for performance (minimizes spills).

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
   - Use `--verbose` to see allocation statistics
   - Verify that values have epistemic annotations in source code
2. Verify live interval computation
   - Check that `build_intervals_from_sir` is receiving correct metadata
3. Increase spill slot size if needed
   - Modify `AllocConfig::max_spill_attempts` if allocation times out
4. Fallback behavior: If epistemic allocation fails, the compiler falls back to internal allocator

#### Debugging Allocation

To debug allocation issues:

```bash
# Verbose output shows allocation metrics
souc build --backend=native --verbose input.sio

# Output includes:
# - Number of intervals allocated vs spilled
# - Average confidence of allocated/spilled values
# - Critical spills (high-confidence values that had to be spilled)
```

#### Common Issues

- **No metadata extracted**: Ensure source code has epistemic annotations (e.g., `Knowledge<T>` types)
- **All intervals spilled**: May indicate too few registers or incorrect register class assignment
- **Critical spills**: High-confidence values being spilled - consider increasing `confidence_weight`

## Examples

See `compiler/examples/native/` for example programs.

## References

- Agner Fog, "Instruction Tables" (2023)
- Intel® 64 and IA-32 Architectures Optimization Reference Manual
- "Energy-Efficient Neural Networks using Data-Flow ISAs" (IEEE MICRO 2023)
- Poletto & Sarkar, "Linear Scan Register Allocation" (PLDI 1999)
