# IR Verification Pipeline

## Overview

The IR verification pipeline enables correctness validation by comparing the semantic equivalence of different code generation strategies. This is critical for the self-hosted compiler bootstrap where Stage 1 (Rust-compiled) and Stage 2 (self-hosted) must produce equivalent IR for identical source programs.

## Architecture

### Phase 0: Serialization and Normalization (Complete)

**Status:** ✅ Implemented and tested

**Modules:**
- `serialize.sio` - SOIR v1 binary format serialization/deserialization
- `normalize.sio` - Deterministic IR canonicalization
- `verify.sio` - High-level verification API

**Binary Format:** SOIR v1
- Magic: "SOIR" (0x534F4952 little-endian)
- Version: 1
- Fixed-size instruction encoding (128 bytes)
- Max module size: 128KB

**Normalization Strategy:**
1. Sort functions alphabetically by name
2. Renumber labels by first definition order (L0, L1, L2...)
3. Renumber registers by first use order (R0, R1, R2...)
4. Sort string table alphabetically

**Why this matters:** Different compilation strategies may allocate registers and labels in different orders. Normalization ensures that semantically equivalent IR produces byte-identical serialized modules despite syntactic differences in register/label numbering.

### Phase 1: Verification Pipeline Integration (Current)

**Status:** 🚧 In Progress

**Goal:** Wire the Phase 0 modules into the Rust test harness for automated verification.

**Components:**

1. **Rust Test Harness** (`crates/souc/tests/ir_verification.rs`)
   - Loads pairs of semantically equivalent .sio programs
   - Type-checks both programs
   - TODO: Lower to IR and invoke self-hosted serialization/normalization

2. **Test Fixtures** (`tests/verify-ir/`)
   - `math_a.sio` / `math_b.sio` - Commutative arithmetic (a+b vs b+a)
   - `control_a.sio` / `control_b.sio` - Equivalent control flow (negated conditions)
   - `call_a.sio` / `call_b.sio` - Commutative function calls

3. **Self-Hosted Tests** (`self-hosted/test_ir.sio`)
   - T01-T16: IR lowering tests (int/float literals, binops, calls, control flow)
   - T17-T21: Phase 0 serialization/normalization tests
   - Run via: `cargo test native_phase1_selfhost_tests_pass` (directory compilation mode)

## Usage

### Running Verification Tests

```bash
# Basic test infrastructure (type-checking only)
cargo test --test ir_verification

# Full self-hosted IR test suite (includes Phase 0 tests)
# Note: Requires directory compilation mode
cargo test native_phase1_selfhost_tests_pass
```

### Verification API (Sounio)

```sio
// High-level verification (byte-for-byte comparison)
fn verify_ir_equivalence(module_a: IrModule, module_b: IrModule) -> i64

// Structural comparison (semantic)
fn compare_ir_modules(module_a: IrModule, module_b: IrModule) -> i64

// Low-level primitives
fn serialize_ir_module(module: IrModule) -> ([i8; 131072], i64)
fn deserialize_ir_module(buf: [i8; 131072], len: i64) -> IrModule
fn normalize_ir_module(module: IrModule) -> IrModule
```

Return codes:
- `0` = Equivalent
- `1` = Length mismatch
- `2` = Byte mismatch
- `100+` = Function-level differences

## Test Coverage

### Phase 0 Tests (T17-T21)

| Test | Description | Status |
|------|-------------|--------|
| T17  | Serialize/deserialize roundtrip | ✅ Pass |
| T18  | Deterministic register renumbering | ✅ Pass |
| T19  | Label ordering normalization | ✅ Pass |
| T20  | Function name sorting | ✅ Pass |
| T21  | Normalized IR comparison | ✅ Pass |

### Phase 1 Tests

| Test | Description | Status |
|------|-------------|--------|
| Math commutative | `a+b` ≡ `b+a` | ✅ Type-check |
| Control equivalent | `if x>0` ≡ `if x<=0 else` | ✅ Type-check |
| Call commutative | `f(a)+f(b)` ≡ `f(b)+f(a)` | ✅ Type-check |

## Known Limitations

1. **Module System:** test_ir.sio requires directory compilation mode and cannot be run in isolation
2. **IR Lowering:** Full verification requires integration between Rust HIR and self-hosted IR lowering
3. **Float Representation:** SOIR serialization needs proper IEEE 754 bit reinterpretation (currently casts)

## Future Work

### Phase 2: Full Pipeline Integration

- [ ] Bridge Rust HIR → Self-hosted IR lowering
- [ ] Integrate serialize/normalize into Rust test harness
- [ ] Add property-based testing for commutative/associative operations
- [ ] Extend test fixtures (loops, nested functions, closures)

### Phase 3: Cross-Stage Verification

- [ ] Verify Stage 1 (Rust-compiled) ≡ Stage 2 (self-hosted) for identical inputs
- [ ] Automated regression testing for IR changes
- [ ] Performance benchmarks for serialization/normalization

## References

- **SOIR Spec:** See comments in `serialize.sio`
- **Normalization Algorithm:** See `normalize.sio` header comments
- **IR Definition:** `ir.sio` - IrModule, IrFunction, IrInstr structures
