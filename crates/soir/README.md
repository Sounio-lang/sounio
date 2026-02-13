# SOIR - Sounio Intermediate Representation Binary Format

SOIR (Sounio Intermediate Representation) is a versioned binary serialization format for the Sounio compiler's intermediate representation. It provides deterministic serialization, normalization for semantic equivalence checking, and efficient encoding.

## Features

- **Deterministic serialization** - Byte-identical output for equivalent IR
- **Normalization** - Canonical form for comparing compilation outputs
- **Versioned format** - Magic bytes and version number for compatibility
- **Compact encoding** - Fixed-size 128-byte instruction format
- **No dependencies** (except `thiserror` for error handling)

## Format Specification

```text
SOIR v1 Binary Format:
──────────────────────
Header (8 bytes):
  - Magic: "SOIR" (0x534F4952)
  - Version: 1 (1 byte)
  - Reserved: 0x00 (3 bytes)

Body:
  - fn_count: i64
  - functions[]: IrFunction[]
  - string_count: i64
  - strings[]: Name[]

IrFunction encoding:
  - name: Name (128 bytes + 8 byte length)
  - instr_count: i64
  - reg_count: i64
  - label_count: i64
  - param_count: i64
  - param_regs: [i64; 64]
  - instrs: IrInstr[]

IrInstr encoding (fixed 128 bytes):
  - op: IrOpcode (1 byte + 7 padding)
  - dst, src1, src2: i64
  - imm_i64: i64, imm_f64: f64
  - label_id, fn_id, field_idx: i64
  - bin_op: BinaryOp (1 byte + 7 padding)
  - un_op: UnaryOp (1 byte + 7 padding)
  - name: Name (128 bytes + 8 byte length)
  - arg_count: i64
```

## Usage

```rust
use soir::{SoirModule, serialize, deserialize, normalize, compare};

// Serialize IR module to bytes
let bytes = serialize(&ir_module)?;

// Deserialize from bytes
let module = deserialize(&bytes)?;

// Normalize for deterministic comparison
let normalized = normalize(&module);

// Compare two normalized modules
if compare(&normalized1, &normalized2) {
    println!("Modules are semantically equivalent");
}
```

## Normalization

SOIR normalization transforms IR into canonical form:

1. **Sort functions by name** (alphabetically)
2. **Renumber labels** by first definition order (L0, L1, L2...)
3. **Renumber registers** by first use order (R0, R1, R2...)
4. **Sort string table** alphabetically

This ensures that semantically equivalent IR from different compilation strategies produces byte-identical SOIR artifacts after normalization.

## Use Case: Rustless Cutover

SOIR is a key component of Sounio's "rustless cutover" - the ability to bootstrap the compiler using only self-hosted code. By serializing IR to SOIR format, we can:

1. Compile Stage 1 using the Rust-based compiler
2. Compile Stage 2 using the self-hosted compiler
3. Normalize both outputs
4. Compare byte-for-byte to verify correctness

This enables reproducible builds and eliminates dependency on Rust for the bootstrap chain.

## Testing

```bash
cargo test
```

## License

MIT OR Apache-2.0
