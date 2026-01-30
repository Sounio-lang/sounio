# Epistemic Runtime Backend Integration Status

## Objective
Integrate epistemic runtime (Knowledge<T>) support across all 5 backends: Interpreter, Native, LLVM, Cranelift, and GPU.

## Implementation Status

### ✅ 1. Interpreter Backend (COMPLETE)

**Files Modified:**
- `compiler/src/interp/value.rs` - Added Knowledge variant to Value enum
- `compiler/src/interp/eval.rs` - Full Knowledge construction and propagation

**Implementation:**

1. **Value Representation:**
   ```rust
   Knowledge {
       value: Box<Value>,
       confidence: f64,        // 0.0 to 1.0
       provenance_id: u32,     // Hashed provenance
   }
   ```

2. **Knowledge Construction:**
   - Evaluate inner value
   - Extract confidence from epsilon parameter (confidence = 1 - epsilon)
   - Generate provenance ID by hashing provenance expression
   - Wrap result in Knowledge variant

3. **Epistemic Propagation in Binary Operations:**
   - `unwrap_knowledge()` - Extract value and metadata
   - `combine_confidence()` - Multiply confidences (degradation)
   - `combine_provenance()` - XOR provenance IDs
   - `wrap_knowledge()` - Wrap results when inputs have epistemic data
   - All arithmetic/comparison ops automatically propagate

4. **Testing:**
   - ✅ All 36 interpreter tests pass
   - ✅ Knowledge values preserve metadata
   - ✅ Confidence degrades properly through operations
   - ✅ Provenance combines correctly

### 🔨 2. Native Backend (IN PROGRESS)

**Current State:**
- `compiler/src/backend/native/epistemic_runtime.rs` exists with C API stubs
- Fixed compilation errors in epistemic_runtime (ptr::copy issues)
- Native backend has infrastructure for function calls (BL instruction on ARM64)

**What's Needed:**

1. **Define External Functions:**
   ```rust
   // In runtime linking
   extern "C" {
       fn sounio_epistemic_add_f64(
           a_val: f64, a_conf: f64, a_prov: u32,
           b_val: f64, b_conf: f64, b_prov: u32,
           result: *mut EpistemicF64
       );
   }
   ```

2. **Emit Function Calls in Code Generation:**
   - Detect Knowledge types in HLIR operations
   - For Add/Sub/Mul on Knowledge<f64>:
     - Load operands into registers (x0-x5 for ARM64, RDI-RDX for x86-64)
     - Emit BL instruction to epistemic runtime function
     - Store result from output pointer

3. **Register Allocation:**
   - Knowledge<T> needs space for (value, confidence, provenance)
   - Allocate struct on stack or in registers
   - Pass pointer to runtime functions

4. **Files to Modify:**
   - `src/backend/native/mod.rs` - Add Knowledge op detection
   - `src/backend/native/aarch64.rs` - Emit BL calls
   - `src/backend/native/runtime.rs` - Link epistemic functions

### ⏳ 3. LLVM Backend (NOT STARTED)

**What's Needed:**

1. **Declare External Functions:**
   ```llvm
   declare void @sounio_epistemic_add_f64(
       double, double, i32,  ; a: value, confidence, provenance
       double, double, i32,  ; b: value, confidence, provenance
       %EpistemicF64* sret   ; result pointer
   )
   ```

2. **Lower Knowledge Operations:**
   - In `src/codegen/llvm/codegen.rs`
   - Detect Knowledge types in HLIR
   - Generate call instructions to epistemic runtime
   - Handle struct types in LLVM IR

3. **Type Lowering:**
   - Map `HlirType::Knowledge<T>` to LLVM struct type
   - Handle different epistemic modes (Full/Compact/Erased)

**Files to Modify:**
- `src/codegen/llvm/codegen.rs` - Main codegen
- `src/codegen/llvm/types.rs` - Type mapping
- `src/codegen/llvm/mod.rs` - Runtime function declarations

### ⏳ 4. Cranelift Backend (NOT STARTED)

**Current State:**
- JIT runtime uses handler IDs for effects
- Handler ID range 40-49 reserved for Epistemic operations

**What's Needed:**

1. **Register Handler Functions:**
   ```rust
   runtime.register_handler(40, epistemic_add_handler);
   runtime.register_handler(41, epistemic_mul_handler);
   // etc.
   ```

2. **Implement Handler Functions:**
   ```rust
   fn epistemic_add_handler(args: &[Value]) -> Value {
       let (a_val, a_conf, a_prov) = extract_knowledge(&args[0]);
       let (b_val, b_conf, b_prov) = extract_knowledge(&args[1]);
       // Call epistemic runtime
       wrap_knowledge(a_val + b_val, a_conf * b_conf, a_prov ^ b_prov)
   }
   ```

3. **Codegen for Knowledge Ops:**
   - Allocate stack slots for Knowledge structs
   - Generate calls to handler functions
   - Pass handler ID + arguments

**Files to Modify:**
- `src/codegen/cranelift.rs` - Add Knowledge op lowering
- `src/codegen/mir_cranelift.rs` - MIR to Cranelift with epistemic

### ⏳ 5. GPU Backend (NOT STARTED)

**Current State:**
- `src/codegen/gpu/epistemic_ptx.rs` exists (stub)
- PTX and SPIR-V codegen infrastructure present

**What's Needed:**

1. **PTX Codegen for Knowledge Operations:**
   ```ptx
   // Knowledge addition
   add.f64 %result_val, %a_val, %b_val
   mul.f64 %result_conf, %a_conf, %b_conf
   xor.b32 %result_prov, %a_prov, %b_prov
   ```

2. **SPIR-V Equivalent:**
   - OpFAdd for value addition
   - OpFMul for confidence combination
   - OpBitwiseXor for provenance

3. **Device Functions:**
   - Define epistemic operations as device functions
   - Inline in kernels for performance

**Files to Modify:**
- `src/codegen/gpu/epistemic_ptx.rs` - Complete implementation
- `src/codegen/gpu/hlir_to_gpu.rs` - Lower Knowledge ops to PTX/SPIR-V

## Epistemic Runtime API

The core runtime functions that all backends need to call:

```c
// Full mode (64 bytes overhead)
void sounio_epistemic_add_full(
    const KnowledgeFull* a,
    const KnowledgeFull* b,
    KnowledgeFull* result
);

// Compact mode (16 bytes overhead)
void sounio_epistemic_add_compact(
    const KnowledgeCompact* a,
    const KnowledgeCompact* b,
    KnowledgeCompact* result
);

// Erased mode (0 bytes overhead) - just regular ops
```

**Struct Layouts:**

```c
// Full mode (64 bytes)
struct KnowledgeFull {
    double value;
    double confidence;
    double lower_bound;
    double upper_bound;
    uint64_t provenance_chain;
    uint64_t timestamp;
    uint64_t flags;
    double variance;
};

// Compact mode (16 bytes)
struct KnowledgeCompact {
    double value;
    uint16_t confidence_q;  // Quantized
    uint32_t provenance_hash;
    uint16_t timestamp_delta;
};
```

## Testing Strategy

### Interpreter (✅ Done)
```rust
cargo test --lib interpreter
// All 36 tests pass
```

### Native Backend (🔨 To Do)
```rust
cargo test --test native_backend
```

### LLVM Backend (⏳ To Do)
```rust
cargo test --features llvm llvm_epistemic
```

### Cranelift Backend (⏳ To Do)
```rust
cargo test --features jit jit_epistemic
```

### GPU Backend (⏳ To Do)
```rust
cargo test --features gpu gpu_epistemic
```

### Cross-Backend Integration Test (⏳ To Do)
```rust
// Test same computation across all backends
// Verify results are consistent within floating-point tolerance
```

## Next Steps

### Priority 1: Complete Native Backend
1. Implement Knowledge op detection in native codegen
2. Emit BL calls to epistemic runtime functions
3. Test with native_backend integration tests

### Priority 2: LLVM Backend
1. Add external function declarations
2. Lower Knowledge ops to LLVM IR calls
3. Test with LLVM integration tests

### Priority 3: Cranelift Backend
1. Register epistemic handler functions
2. Implement handlers using Rust epistemic runtime
3. Test with JIT integration tests

### Priority 4: GPU Backend
1. Complete epistemic_ptx.rs with PTX codegen
2. Add SPIR-V equivalent
3. Test with GPU integration tests

### Priority 5: Cross-Backend Validation
1. Create integration test that runs same code on all backends
2. Verify confidence propagation is consistent
3. Benchmark performance across backends

## Acceptance Criteria

- ✅ Interpreter: Knowledge operations work with confidence tracking
- ⏳ Native: At least addition and multiplication emit runtime calls
- ⏳ LLVM: Knowledge ops lower to external function calls
- ⏳ Cranelift: JIT handlers for epistemic operations registered
- ⏳ GPU: PTX codegen for Knowledge operations complete
- ⏳ At least 3 backends working (Interpreter + 2 others minimum)
- ⏳ Cross-backend test shows consistent results

## Time Estimate

- Native backend: 2-3 hours (function call emission + testing)
- LLVM backend: 1-2 hours (external declarations + lowering)
- Cranelift backend: 2-3 hours (handler registration + implementation)
- GPU backend: 3-4 hours (PTX codegen + SPIR-V)
- Testing & integration: 1-2 hours

Total: ~10-14 hours for full implementation

## Current Achievement

- ✅ Interpreter backend: Fully working (1/5 backends complete)
- ✅ All interpreter tests pass
- ✅ Knowledge construction and propagation working
- ✅ Confidence degrades correctly through operations
- ✅ Provenance combines across operations

**Status: 20% complete** (1 of 5 backends working, with solid foundation for others)
