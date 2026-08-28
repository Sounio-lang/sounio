<!-- docs:meta
topic_id: repo.docs.architecture.llvm-codegen
authority: historical
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.llvm-codegen
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio LLVM Code Generation

This document describes the LLVM code generation backend for Sounio, including type mapping, supported operations, and build configuration.

## Table of Contents

- [Overview](#overview)
- [Build Configuration](#build-configuration)
- [Type System](#type-system)
  - [Primitive Types](#primitive-types)
  - [SIMD Types](#simd-types)
  - [Epistemic Types](#epistemic-types)
  - [QNN Types](#qnn-types)
- [Architecture](#architecture)
- [Debug Information](#debug-information)
- [Performance](#performance)

---

## Overview

The LLVM backend generates optimized native code using the [inkwell](https://github.com/TheDan64/inkwell) Rust bindings for LLVM. Sounio supports LLVM 15, 16, and 17.

**Key capabilities**:
- Full HLIR-to-LLVM type conversion
- SIMD vector operations (Vec2, Vec3, Vec4, Mat2-4, Quat)
- Double-precision vectors (Vec2d, Vec3d, Vec4d)
- Epistemic types with configurable overhead
- DWARF debug information generation
- Opaque pointer support (LLVM 15+)

---

## Build Configuration

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt install llvm-17 llvm-17-dev libllvm17 clang-17 libzstd-dev

# macOS
brew install llvm@17
```

### Feature Flags

| Feature | LLVM Version | Environment Variable |
|---------|--------------|----------------------|
| `llvm15` | 15.x | `LLVM_SYS_150_PREFIX` |
| `llvm16` | 16.x | `LLVM_SYS_160_PREFIX` |
| `llvm17` | 17.x | `LLVM_SYS_170_PREFIX` |

**Note**: Features are mutually exclusive. Only enable one LLVM version.

### Building

```bash
# Set LLVM path
export LLVM_SYS_170_PREFIX=/usr/lib/llvm-17

# Build with LLVM 17
cargo build --release --features llvm17

# Verify LLVM integration
cargo test --features llvm17 llvm_
```

---

## Type System

The `TypeConverter` (`compiler/src/codegen/llvm/types.rs`) handles conversion from HLIR types to LLVM types.

### Primitive Types

| HLIR Type | LLVM Type | Size (bits) | Alignment |
|-----------|-----------|-------------|-----------|
| `Bool` | `i1` | 1 | 1 |
| `I8`/`U8` | `i8` | 8 | 1 |
| `I16`/`U16` | `i16` | 16 | 2 |
| `I32`/`U32` | `i32` | 32 | 4 |
| `I64`/`U64` | `i64` | 64 | 8 |
| `I128`/`U128` | `i128` | 128 | 8 |
| `F32` | `float` | 32 | 4 |
| `F64` | `double` | 64 | 8 |

### Pointer Types

LLVM 15+ uses **opaque pointers**. All pointers are the same type regardless of pointee:

```rust
// All pointers become opaque
HlirType::Ptr(_) => context.ptr_type(AddressSpace::default())
```

### Compound Types

| HLIR Type | LLVM Representation |
|-----------|---------------------|
| `Array(T, N)` | `[N x T]` |
| `Tuple(T1, T2, ...)` | `{ T1, T2, ... }` (anonymous struct) |
| `Struct(name)` | Named struct type |
| `Function` | Opaque pointer (function type for calls) |

### SIMD Types

Single-precision (f32) vectors:

| Type | LLVM | Size | Alignment | Notes |
|------|------|------|-----------|-------|
| `Vec2` | `<2 x float>` | 64 bits | 8 bytes | 2D vector |
| `Vec3` | `<4 x float>` | 128 bits | 16 bytes | Padded to 4 elements |
| `Vec4` | `<4 x float>` | 128 bits | 16 bytes | 4D vector |
| `Quat` | `<4 x float>` | 128 bits | 16 bytes | Quaternion (w,x,y,z) |

Double-precision (f64) vectors:

| Type | LLVM | Size | Alignment | Notes |
|------|------|------|-----------|-------|
| `Vec2d` | `<2 x double>` | 128 bits | 16 bytes | 2D double vector |
| `Vec3d` | `<4 x double>` | 256 bits | 32 bytes | Padded to 4 elements |
| `Vec4d` | `<4 x double>` | 256 bits | 32 bytes | 4D double vector |

Matrices:

| Type | LLVM | Size | Notes |
|------|------|------|-------|
| `Mat2` | `[4 x float]` | 128 bits | 2x2 matrix |
| `Mat3` | `[12 x float]` | 384 bits | 3x3 with row padding |
| `Mat4` | `[16 x float]` | 512 bits | 4x4 matrix |

Hypercomplex:

| Type | LLVM | Size | Notes |
|------|------|------|-------|
| `Dual` | `{ double, double }` | 128 bits | Dual number (value, derivative) |
| `Octonion` | `<8 x float>` | 256 bits | 8D hypercomplex |

### Epistemic Types

`Knowledge<T>` wraps values with uncertainty metadata. Three modes control the overhead:

#### Full Mode (default)

```rust
struct KnowledgeFull<T> {
    value: T,           // Inner value
    confidence: f64,    // 0.0-1.0 confidence level
    lower: f64,         // Lower bound of interval
    upper: f64,         // Upper bound of interval
    provenance: [u8; 32], // Data origin hash
    timestamp: u64,     // When measured
}
// Overhead: +64 bytes (512 bits)
```

#### Compact Mode

```rust
struct KnowledgeCompact<T> {
    value: T,           // Inner value
    confidence: u16,    // 0-65535 (scaled confidence)
    _padding: [u8; 14], // Alignment padding
}
// Overhead: +16 bytes (128 bits)
```

#### Erased Mode

```rust
// Zero overhead - just the inner value
// Epistemic operations become no-ops
```

**Type representation**:

```rust
HlirType::Knowledge { inner, mode, .. } => {
    match mode {
        EpistemicMode::Full => /* struct with all fields */,
        EpistemicMode::Compact => /* struct with u16 confidence */,
        EpistemicMode::Erased => self.convert(inner), // Just inner type
    }
}
```

### QNN Types (Quaternionic Neural Networks)

Specialized types for quaternion-valued neural networks:

| Type | LLVM | Description |
|------|------|-------------|
| `QuatLinear` | `ptr` | Quaternion linear layer |
| `QuatConv2d` | `ptr` | Quaternion 2D convolution |
| `QuatRnnState` | `ptr` | Quaternion RNN hidden state |
| `QuatGate` | `ptr` | Quaternion gate (LSTM/GRU) |

All QNN types are represented as opaque pointers to runtime-managed structures.

---

## Architecture

### Pipeline

```
HLIR (High-Level IR)
       │
       ▼
┌──────────────────┐
│  TypeConverter   │  Convert types
├──────────────────┤
│  CodeGen         │  Emit instructions
├──────────────────┤
│  DebugInfo       │  DWARF generation
└──────────────────┘
       │
       ▼
   LLVM Module
       │
       ▼
   Native Code
```

### Key Components

| File | Purpose |
|------|---------|
| `codegen.rs` | Main code generation |
| `types.rs` | HLIR → LLVM type conversion |
| `debug.rs` | DWARF debug info generation |

### Constants

The codegen handles various constant types:

```rust
match constant {
    HlirConstant::Int(v) => /* i64 constant */,
    HlirConstant::Float(v) => /* f64 constant */,
    HlirConstant::Bool(v) => /* i1 constant */,
    HlirConstant::CString(s) => /* global string pointer */,
}
```

String constants are interned as global string pointers.

---

## Debug Information

The LLVM backend generates DWARF debug information for source-level debugging.

### Type Information

Each type generates appropriate DWARF entries:

```rust
fn create_type_info(&self, ty: &HlirType) -> u64 {
    let bits = type_size_bits(ty);
    // Returns size for DWARF DIType
}
```

Size calculations account for epistemic modes:

```rust
HlirType::Knowledge { inner, mode, .. } => {
    let inner_bits = type_size_bits(inner);
    match mode {
        EpistemicMode::Full => inner_bits + 512,    // +64 bytes
        EpistemicMode::Compact => inner_bits + 128, // +16 bytes
        EpistemicMode::Erased => inner_bits,        // Zero overhead
    }
}
```

### Using Debug Info

```bash
# Compile with debug info
souc build --debug program.sio -o program

# Debug with LLDB
lldb ./program

# Debug with GDB
gdb ./program
```

---

## Performance

### Alignment

Types are aligned for optimal SIMD performance:

- `Vec2`: 8-byte alignment
- `Vec3`, `Vec4`, `Quat`, `Mat*`: 16-byte alignment
- `Vec3d`, `Vec4d`, `Octonion`: 32-byte alignment

### Optimization Levels

```bash
# Debug build (O0)
cargo build

# Release build (O3)
cargo build --release

# Custom opt level
souc build -O2 program.sio
```

### Epistemic Mode Selection

Choose the appropriate mode based on requirements:

| Mode | Overhead | Use Case |
|------|----------|----------|
| `Full` | +64 bytes | Research, auditing, full provenance |
| `Compact` | +16 bytes | Production, confidence-only |
| `Erased` | 0 bytes | Performance-critical, confidence unnecessary |

---

## API Reference

### TypeConverter

```rust
impl TypeConverter<'ctx> {
    // Create converter
    fn new(context: &'ctx Context) -> Self;

    // Convert HLIR type to LLVM type
    fn convert(&mut self, ty: &HlirType) -> BasicTypeEnum<'ctx>;

    // Get specific types
    fn i32_type(&self) -> IntType<'ctx>;
    fn f64_type(&self) -> FloatType<'ctx>;
    fn ptr_type(&self, element: BasicTypeEnum<'ctx>) -> PointerType<'ctx>;
    fn string_type(&self) -> StructType<'ctx>;
    fn slice_type(&mut self, elem: &HlirType) -> StructType<'ctx>;

    // Create compound types
    fn create_struct_type(&mut self, name: &str, fields: &[HlirType]) -> StructType<'ctx>;
    fn function_type(&mut self, params: &[HlirType], ret: &HlirType) -> FunctionType<'ctx>;

    // Size calculations
    fn size_bits(&self, ty: &HlirType) -> u64;
    fn size_bytes(&self, ty: &HlirType) -> u64;
    fn align_bytes(&self, ty: &HlirType) -> u64;

    // Type queries
    fn is_integer_type(&self, ty: &HlirType) -> bool;
    fn is_float_type(&self, ty: &HlirType) -> bool;
    fn is_signed(&self, ty: &HlirType) -> bool;
}
```

---

## Troubleshooting

### Error: `unable to find library -lzstd`

```bash
# Install zstd library
sudo apt install libzstd-dev  # Debian/Ubuntu
brew install zstd             # macOS
```

### Error: `LLVM_SYS_170_PREFIX not set`

```bash
# Find LLVM installation
llvm-config-17 --prefix

# Set environment variable
export LLVM_SYS_170_PREFIX=/usr/lib/llvm-17
```

### Error: `Unknown type variant`

Ensure all HLIR type variants are handled in:
- `codegen/llvm/types.rs` - Type conversion
- `codegen/llvm/debug.rs` - Debug info generation

---

## Related Documentation

- [Installation Guide](INSTALLATION.md) - LLVM setup
- [Feature Flags](FEATURE_FLAGS.md) - Build configuration
- [Epistemic API](api/EPISTEMIC_API.md) - Knowledge<T> usage
- [GPU Codegen](codegen/gpu/numerical_README.md) - GPU backend

---

*Last updated: January 2026 (v1.0.0)*
