# Auto-Vectorization Pass for Sounio

## Overview

This document describes the auto-vectorization optimization pass implemented for the Sounio compiler's MIR (Mid-level Intermediate Representation) optimization pipeline.

## Implementation Status

**Status**: Core infrastructure implemented, transformation logic pending
**Location**: `crates/souc/src/mir/optimization/performance_tuning.rs`
**Tests**: 4 comprehensive tests in `performance_tuning::tests`

## What Was Implemented

### 1. Loop Vectorizer Structure (`LoopVectorizer`)

The vectorizer maintains vectorization decisions for each loop header block:

```rust
pub struct LoopVectorizer {
    decisions: HashMap<BlockId, VectorizationDecision>,
}
```

### 2. Main Entry Point (`run()` method)

The `run()` method orchestrates the vectorization pass:

1. **Loop Detection**: Uses `LoopAnalysis::analyze_function()` to identify natural loops
2. **Decision Making**: Evaluates each loop for vectorization profitability
3. **Transformation**: Applies vectorization when beneficial (currently stubbed)

```rust
pub fn run(&mut self, func: &mut MirFunction) -> Result<bool, String>
```

### 3. Vectorization Decision Logic

#### Decision Types

```rust
pub enum DecisionType {
    Vectorize { factor: usize },
    NoVectorize { reason: String },
    PartiallyVectorize { factor: usize, reason: String },
}
```

#### Decision Criteria

The `should_vectorize()` method evaluates:

- **Trip Count**: Minimum 8 iterations required
- **Loop Complexity**: Must be Simple or Medium
- **Vectorization Blockers**: No function calls, memory stores checked

### 4. Loop Complexity Analysis

Classifies loops into complexity levels:

```rust
pub enum LoopComplexity {
    Simple,       // < 10 instructions, no calls/memory
    Medium,       // 10-20 instructions, memory ops allowed
    Complex,      // 20-50 instructions
    VeryComplex,  // > 50 instructions or contains calls
}
```

**Analysis factors**:
- Instruction count
- Presence of function calls
- Memory operation patterns

### 5. Vectorization Blocker Detection

Identifies patterns that prevent safe vectorization:

- **Function calls**: Cannot vectorize across call boundaries
- **Memory stores**: Require dependency analysis (not yet implemented)
- **Complex control flow**: Multiple branches within loop body

### 6. Vector Factor Determination

Chooses optimal SIMD width based on data types:

| Type | Vector Factor | SIMD Width |
|------|---------------|------------|
| F32  | 4             | 128-bit    |
| I32  | 4             | 128-bit    |
| F64  | 2             | 128-bit    |
| Default | 4          | 128-bit    |

Future work will support:
- AVX-512: 8×F32 or 4×F64 (512-bit)
- NEON: 4×F32 or 2×F64 (128-bit)
- SVE: Variable width

## Algorithmic Foundation

### Loop Detection

Uses standard compiler algorithms (see `crates/souc/src/mir/analysis/loops.rs`):

1. **Dominator Tree Construction**: Identifies block dominance relationships
2. **Back Edge Detection**: Finds edges where source dominates target
3. **Natural Loop Formation**: Constructs loop body from back edges

Reference: Wolfe (1996) "High-Performance Compilers", Muchnick (1997) "Advanced Compiler Design"

### Cost Model

Simple threshold-based model:

```rust
let min_trip_count = 8;  // Amortize SIMD setup overhead
let max_complexity = LoopComplexity::Medium;
```

Future improvements (from literature review):
- Machine learning-based cost model (autograph from Intelligent Computing 2025)
- Detailed instruction latency modeling
- Memory bandwidth considerations

## Test Coverage

### 1. `test_loop_vectorizer_creation`
Validates basic vectorizer initialization.

### 2. `test_loop_complexity_analysis`
Tests complexity classification:
- Simple loop with 1 arithmetic operation → `LoopComplexity::Simple`
- Verifies instruction counting logic

### 3. `test_vectorization_blocker_detection`
Ensures function calls are detected as blockers:
```rust
assert!(blocker.unwrap().contains("function calls"));
```

### 4. `test_vector_factor_determination`
Validates factor selection:
- F32 operations → factor 4 (F32X4)
- F64 operations → factor 2 (F64X2)

All tests pass (verified 2026-01-30).

## Integration with Existing Infrastructure

### Available SIMD Support

The vectorizer builds on existing Sounio SIMD infrastructure:

1. **SIR Vector Operations** (`crates/souc/src/sir/ops.rs`)
   ```rust
   pub enum VectorOp {
       Splat(ValueId),
       Extract { vec, idx },
       Insert { vec, val, idx },
       Shuffle { v1, v2, mask },
       HAdd(ValueId),
       HMul(ValueId),
       Fma { a, b, c },
       Reduce { op, vec },
   }
   ```

2. **Cranelift SIMD Types** (`crates/souc/src/codegen/simd.rs`)
   - F32X4: 4×f32 vector
   - SimdVec, SimdQuat: Geometric types

3. **LLVM Vectorization Passes** (`crates/souc/src/codegen/llvm/passes.rs`)
   - `-slp-vectorizer`: Superword-level parallelism at O3
   - `-loop-vectorize`: LLVM's built-in loop vectorizer at O3

### Architecture Feature Detection

Target-specific features available (`crates/souc/src/target/spec.rs`):
- **x86_64**: SSE, SSE2, AVX, AVX2, AVX-512
- **ARM64**: NEON
- **RISC-V**: Vector extensions

## What's NOT Yet Implemented

### 1. Actual Loop Transformation

The `vectorize_loop()` method is stubbed (returns `Ok(false)`).

Full implementation requires:

```rust
fn vectorize_loop(
    &mut self,
    func: &mut MirFunction,
    natural_loop: &NaturalLoop,
    factor: usize,
) -> Result<bool, String> {
    // 1. Identify induction variable (i = 0; i < N; i++)
    // 2. Create vectorized loop: i += factor
    // 3. Transform operations:
    //    - Scalar load → vector load
    //    - Scalar add  → vector add
    //    - Scalar store → vector store
    // 4. Generate epilogue for remaining iterations
    // 5. Insert runtime checks (alignment, trip count)
    todo!()
}
```

### 2. Dependency Analysis

Currently blocks all loops with memory stores. Need:

- **Data dependence analysis**: Identify read-after-write (RAW), write-after-read (WAR), write-after-write (WAW) hazards
- **Pointer aliasing**: Determine if different pointers can overlap
- **Runtime disambiguation**: Insert checks when compile-time analysis inconclusive

Reference: AliasAnalysis in `crates/souc/src/mir/analysis/alias.rs` (partially implemented)

### 3. Induction Variable Recognition

Need to identify patterns like:
```sio
var i = 0
while i < N {
    a[i] = b[i] + c[i]
    i = i + 1
}
```

Extract: initial value (0), bound (N), stride (1)

### 4. Advanced Transformations

From literature review (VecTrans, autograph papers):

- **Loop interchange**: Reorder nested loops for better vectorization
- **Loop distribution**: Split loops to isolate vectorizable parts
- **Loop fusion**: Combine adjacent loops
- **Strip mining**: Break large loops into vectorizable chunks

### 5. Polyhedral Analysis

Partial infrastructure exists (`crates/souc/src/hlir/polyhedral/`) with TODOs.

Polyhedral model enables:
- Precise dependency analysis
- Complex loop transformations
- Locality optimization

Reference: Bastoul (2004) "Code Generation in the Polyhedral Model"

## Example: What Would Be Generated

### Original Sounio Code
```sio
fn vector_add(a: &[f32], b: &[f32], c: &![f32], n: i32) with Mut {
    var i = 0
    while i < n {
        c[i] = a[i] + b[i]
        i = i + 1
    }
}
```

### After Vectorization (Conceptual MIR)

```rust
// Vectorized loop: process 4 elements at a time
var i = 0
while i < (n - 3) {
    let va = vector_load(a + i, align=4)  // Load 4×f32
    let vb = vector_load(b + i, align=4)
    let vc = vector_add(va, vb)           // SIMD add
    vector_store(c + i, vc, align=4)
    i = i + 4
}

// Epilogue: process remaining elements
while i < n {
    c[i] = a[i] + b[i]
    i = i + 1
}
```

### Expected Assembly (x86_64 with SSE)

```asm
.vectorized_loop:
    movups  xmm0, [rsi + rcx*4]  ; Load 4 floats from a
    movups  xmm1, [rdx + rcx*4]  ; Load 4 floats from b
    addps   xmm0, xmm1            ; SIMD add
    movups  [rdi + rcx*4], xmm0  ; Store 4 floats to c
    add     rcx, 4
    cmp     rcx, r8
    jl      .vectorized_loop

.epilogue:
    ; Scalar loop for remainder
```

## Performance Expectations

### Theoretical Speedup

For perfectly vectorizable code:
- **4×F32 (SSE/NEON)**: Up to 4× speedup
- **8×F32 (AVX-256)**: Up to 8× speedup
- **16×F32 (AVX-512)**: Up to 16× speedup

### Realistic Speedup

Real-world factors reduce gains:
- Memory bandwidth saturation: 2-3× typical
- Cache effects: Vectorization increases memory traffic
- Control overhead: Branch mispredictions
- Epilogue: Scalar cleanup reduces average speedup

### Benchmarks (Future Work)

Target test cases:
1. **SAXPY**: `a[i] = a[i] + b[i] * s` (BLAS level-1)
2. **DGEMV**: Matrix-vector multiply (BLAS level-2)
3. **Stencil computations**: ODE/PDE solvers
4. **Reduction**: Sum, max, min operations

## Literature Review Integration

This implementation aligns with Q1 Literature Review priorities:

### Papers Referenced

1. **VecTrans (2025)**: "Enhancing Compiler Auto-Vectorization"
   - Standard transformation pipeline
   - Scalar-to-vector pattern matching

2. **Autograph (Intelligent Computing 2025)**: "Graph-Based Learning Framework"
   - ML-guided vectorization decisions
   - Future: Train model on Sounio scientific workloads

3. **Parsimony (CGO 2023)**: "SIMD Programming in Standard Compiler Flows"
   - Integration with existing MIR pipeline
   - Pass-based architecture

### Future Research Integration

From literature review, next steps:

1. **LLM-Vectorizer (CGO 2025)**: Explore LLM-assisted vectorization for complex patterns
2. **Partial SIMD Parallelism (ACM TACO)**: Handle loops with non-vectorizable parts
3. **Polyhedral Loop Optimization**: Complete `hlir/polyhedral/` implementation

## Known Limitations

1. **No transformation logic**: Decision-making only, no code generation
2. **Conservative blocker detection**: Rejects all loops with stores
3. **No multi-level vectorization**: Doesn't handle nested loops
4. **Fixed thresholds**: No adaptive cost model
5. **Architecture-agnostic**: Doesn't target AVX-512/NEON specifically yet

## Usage

### Programmatic API

```rust
use souc::mir::optimization::LoopVectorizer;

let mut vectorizer = LoopVectorizer::new();
let modified = vectorizer.run(&mut function)?;

if modified {
    println!("Function was vectorized");
}

// Inspect decisions
for (block_id, decision) in &vectorizer.decisions {
    println!("Block {:?}: {:?}", block_id, decision.decision);
}
```

### Integration with Optimization Pipeline

Currently the vectorizer is standalone. To integrate into the full pipeline:

```rust
// In crates/souc/src/mir/optimization/mod.rs
pub fn run_optimization_pipeline(module: &mut MirModule) {
    // ... other passes ...

    for func in &mut module.functions {
        let mut vectorizer = LoopVectorizer::new();
        vectorizer.run(func)?;
    }

    // ... more passes ...
}
```

## Next Steps (Priority Order)

### Immediate (0-3 months)
1. **Implement induction variable recognition**
   - Pattern match increment/decrement
   - Extract bounds and strides
   - Validate simple linear patterns

2. **Basic dependency analysis**
   - Integrate with `mir/analysis/alias.rs`
   - Conservative: reject ambiguous cases
   - Progressive: enable known-safe patterns

### Short-Term (3-6 months)
3. **Implement loop transformation**
   - Generate vectorized body
   - Create epilogue
   - Insert runtime guards

4. **Add vector IR operations**
   - Extend MirInstruction with VectorLoad/Store
   - Codegen to Cranelift SIMD
   - Map to LLVM vector intrinsics

### Medium-Term (6-12 months)
5. **Polyhedral analysis integration**
   - Complete dependency computation
   - Enable loop interchange
   - Support non-unit strides

6. **Architecture-specific tuning**
   - AVX-512 targeting (factor 8-16)
   - NEON intrinsics
   - Cost model per architecture

### Long-Term (12+ months)
7. **ML-guided optimization**
   - Collect vectorization success/failure data
   - Train autograph-style model
   - Adaptive threshold tuning

8. **Advanced transformations**
   - Loop distribution
   - Loop fusion
   - Strip mining

## References

- Wolfe, M. (1996). *High-Performance Compilers for Parallel Computing*
- Muchnick, S. (1997). *Advanced Compiler Design and Implementation*
- Allen, R. & Kennedy, K. (2001). *Optimizing Compilers for Modern Architectures*
- [VecTrans Paper (2025)](https://arxiv.org/pdf/2503.19449)
- [Autograph Paper (Intelligent Computing 2025)](https://spj.science.org/doi/10.34133/icomputing.0113)
- [Parsimony (CGO 2023)](https://dl.acm.org/doi/10.1145/3579990.3580019)

## Related Files

- **Core implementation**: `crates/souc/src/mir/optimization/performance_tuning.rs:720-1010`
- **Loop analysis**: `crates/souc/src/mir/analysis/loops.rs`
- **SIMD operations**: `crates/souc/src/sir/ops.rs:142-165`
- **Cranelift SIMD**: `crates/souc/src/codegen/simd.rs`
- **LLVM passes**: `crates/souc/src/codegen/llvm/passes.rs`
- **Target features**: `crates/souc/src/target/spec.rs:580-655`
- **Tests**: `crates/souc/src/mir/optimization/performance_tuning.rs:1144-1284`

---

**Last Updated**: 2026-01-30
**Implementation**: Core decision-making complete, transformation pending
**Tests**: 4/4 passing
**Integration**: Standalone, ready for pipeline inclusion
