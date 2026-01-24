# Phase 2 GPU Optimization - Implementation Strategy

## Completed: Framework Layer (2,610+ LOC)

### ✅ Training Framework (`examples/phase2_training_mnist.rs`)
- `Phase2Trainer` orchestrates all 4 features
- Mixed-precision forward/backward with dynamic loss scaling
- Quaternion-aware 2:4 sparsity pruning
- Kernel fusion simulation (Linear+BN+ReLU)
- QAT with warmup-based activation
- **Test coverage**: 8+ unit tests

### ✅ Profiling Utilities (`compiler/src/codegen/gpu/profiling.rs`)
- Feature-aware measurement: latency, throughput, memory BW, energy
- Statistical analysis: mean, stddev, percentiles (p50/p95/p99)
- Baseline vs optimized comparison
- CSV export for external analysis
- **Test coverage**: 4+ tests

### ✅ Model Serialization (`compiler/src/codegen/gpu/serialization.rs`)
- Binary checkpoint format with version control (v2)
- QuantTensor with scales, zero-points, bit-widths (4/8 bit)
- Sparsity state (2:4, N:M, CSR patterns)
- Mixed-precision state (loss scale, overflow count)
- QAT state (EMA min/max, warmup tracking)
- Full validation pipeline (120+ lines of checks)
- **Test coverage**: 2+ tests

### ✅ Validation Framework (`compiler/src/codegen/gpu/validation.rs`)
- 8 specialized validators with distinct responsibilities:
  1. NumericalCorrectnessValidator (<0.5% threshold)
  2. FeatureIsolationValidator (individual feature impact)
  3. IntegrationValidator (<2% combined features)
  4. AccuracyPreservationValidator (per-feature thresholds)
  5. SparsityValidator (pattern verification)
  6. LossScalingValidator (scale bounds enforcement)
  7. QuantizationValidator (quantization error bounds)
  8. FusionValidator (fused vs unfused equivalence)
- ValidationSuite for aggregation and reporting
- **Test coverage**: 128 integration tests (123 passing)

### ✅ Documentation (`compiler/docs/PHASE2_OPTIMIZATIONS.md`)
- 4-layer architecture overview
- Feature-specific guides (mixed-precision, fusion, sparsity, QAT)
- API reference with method signatures
- Performance benchmarks and expectations
- 20+ troubleshooting scenarios
- 9 detailed usage examples
- Best practices and optimization guidelines

---

## Next: Compiler Integration Layer (Remaining Work)

### Phase 2A: IR Operations & Autodiff Integration (2-3 parallel tasks)

#### Task 1: Extend ir.rs with Phase 2 GpuOp Variants

**Current State**: 100+ operations, mostly quantization + quaternion support

**Required Additions**:
```rust
// Mixed-Precision Operations (3 ops)
MixedPrecisionCast {
  from: MixedPrecisionType,  // Fp32, Fp16, Bf16, Fp8
  to: MixedPrecisionType,
  loss_scale: Option<f32>,
  overflow_detection: bool,
}

// Kernel Fusion Markers (2 ops for semantic tracking)
SemanticFusionBegin { pattern: SemanticFusionPattern }
SemanticFusionEnd { pattern: SemanticFusionPattern }

// Sparse Quaternion Load/Store Format Support (4 ops)
SparseQuatLoad {
  format: SparseQuatFormat,
  layout: MemoryLayout,
}
SparseQuatStore { ... }
```

**Implementation Steps**:
1. Add `MixedPrecisionType` enum (Fp32, Fp16, Bf16, Fp8E4M3, Fp8E5M2)
2. Add `SemanticFusionPattern` enum variants for Phase 2 (LinearBnRelu, etc.)
3. Add `SparseQuatFormat` comprehensive support to ir.rs
4. Add helper methods: `is_low_precision()`, `requires_conversion()`
5. Update codegen to handle new variants in PTX/Metal emission

**Files to Modify**:
- `compiler/src/codegen/gpu/ir.rs` - Add 9+ new GpuOp variants and types
- `compiler/src/codegen/gpu/ptx.rs` - Update PTX emission for new ops
- `compiler/src/codegen/gpu/metal.rs` - Update Metal codegen for new ops

**Expected Outcome**: ~200 LOC, clean compilation, all ops codegen-ready

---

#### Task 2: Create mixed_precision Module (`compiler/src/codegen/gpu/mixed_precision/`)

**New Files**:
```
compiler/src/codegen/gpu/mixed_precision/
├── mod.rs                 # Module exports
├── config.rs              # MixedPrecisionConfig, LowPrecision enum
├── loss_scaler.rs         # Dynamic loss scaling state machine
└── transform.rs           # MixedPrecisionTransform (IR pass)
```

**Features**:

**config.rs** (~80 LOC):
```rust
pub enum LowPrecision { Fp16, Bf16, Fp8E4M3, Fp8E5M2 }

pub struct MixedPrecisionConfig {
    pub compute_dtype: LowPrecision,
    pub accumulate_dtype: Fp32,  // Always FP32
    pub apply_loss_scale: bool,
    pub initial_loss_scale: f32,
    pub max_loss_scale: f32,
    pub min_loss_scale: f32,
}

pub struct LossScaleState {
    pub current_scale: f32,
    pub scale_history: Vec<f32>,
    pub overflow_count: usize,
    pub step_count: usize,
}

impl LossScaleState {
    pub fn update(&mut self, has_overflow: bool, config: &LossScaleConfig);
    pub fn scale(&self, loss: f32) -> f32;
    pub fn unscale(&self, gradient: f32) -> f32;
    pub fn should_skip_step(&self) -> bool;
}
```

**loss_scaler.rs** (~120 LOC):
```rust
pub struct DynamicLossScaler {
    state: LossScaleState,
    config: LossScaleConfig,
}

pub struct LossScaleConfig {
    pub growth_interval: usize,  // Steps before scaling up
    pub growth_factor: f32,       // Usually 2.0
    pub backoff_factor: f32,      // Usually 0.5
    pub max_scale_size: f32,      // 16M default
}

impl DynamicLossScaler {
    pub fn step(&mut self, has_overflow: bool);
    pub fn apply(&self, loss: f32) -> f32;
    pub fn apply_to_gradient(&self, gradient: &mut [f32]);
}
```

**transform.rs** (~150 LOC):
```rust
pub struct MixedPrecisionTransform {
    config: MixedPrecisionConfig,
    precision_map: FxHashMap<ValueId, MixedPrecisionType>,
}

impl MixedPrecisionTransform {
    pub fn classify_operations(&mut self, kernel: &GpuKernel);
    pub fn insert_conversions(&mut self, kernel: &mut GpuKernel);

    // Ops that stay FP32: exp, log, sqrt, reductions, softmax, layer_norm
    // Ops that go FP16: matmul, conv, activations (relu, sigmoid)

    pub fn transform_kernel(&self, kernel: &GpuKernel) -> GpuKernel;
}
```

**Modifications**:
- `compiler/src/codegen/gpu/mod.rs` - Add `mod mixed_precision`
- `compiler/src/codegen/gpu/hlir_to_gpu.rs` - Insert MP transform pass
- `compiler/src/autodiff/transform.rs` - Add MP configuration option

**Expected Outcome**: ~350 LOC, integrated into kernel lowering pipeline

---

#### Task 3: Extend autodiff/primitives.rs with QAT Straight-Through Estimator

**Current State**: autodiff has full AD infrastructure but no FakeQuantize rule

**Required Additions**:
```rust
// In autodiff/primitives.rs

pub struct FakeQuantizeBackward {
    scale: f32,
    zero_point: i32,
    bit_width: u8,
    // STE: gradient passes through unchanged
    // d(output)/d(input) = 1.0
}

impl Primitive for FakeQuantizeBackward {
    fn vjp(&self, output_grad: &Value) -> Vec<Value> {
        // Straight-through estimator: just return output_grad
        vec![output_grad.clone()]
    }
}

// Register in primitive registry
registry.register("fake_quantize_backward", Box::new(FakeQuantizeBackward::new));
```

**Files to Modify**:
- `compiler/src/autodiff/primitives.rs` - Add FakeQuantizeBackward
- `compiler/src/autodiff/tape.rs` - Add TapeOp::FakeQuantize variant
- `compiler/src/autodiff/mod.rs` - Export new primitive

**Expected Outcome**: ~100 LOC, STE gradient rule fully integrated

---

### Phase 2B: Semantic Fusion Patterns (1-2 parallel tasks)

#### Task 4: Extend fusion.rs with Phase 2 Semantic Patterns

**Current State**: fusion.rs has framework but limited semantic patterns

**Required Additions**:
```rust
pub enum SemanticFusionPattern {
    // Phase 2 patterns
    LinearBnRelu,           // 10% speedup
    QuatLinearBnRelu,       // Quaternion-aware
    LinearBn,
    LinearRelu,
    BnRelu,

    // Phase 3 (future)
    ConvBnRelu,
    DwConvBnRelu,
    AttentionLayerNorm,
    // ...
}

pub struct SemanticPattern {
    ops: Vec<OpCategory>,  // Sequence to match
    fused_kernel_name: String,
    benefit_multiplier: f32,  // 1.5 for LinearBnRelu
    register_impact: f32,
    shared_mem_impact: f32,
}
```

**Modifications to fusion.rs** (~300 LOC additions):
1. Extend `SemanticFusionPattern` enum with 5+ Phase 2 variants
2. Add pattern matchers for each variant
3. Compute benefit multipliers based on operation characteristics
4. Integrate with existing cost model
5. Generate fused kernel signatures

**Expected Outcome**: ~300 LOC, patterns discoverable and fusible

---

#### Task 5: Implement sparse_quat.rs

**New File**: `compiler/src/codegen/gpu/sparse_quat.rs`

**Features** (~400 LOC):
```rust
pub enum SparseQuatFormat {
    Dense,
    QuatCSR,           // CSR at quaternion granularity
    QuatBCSR { block_size: usize },
    QuatStructured2x4, // NVIDIA Ampere+
    QuatStructuredNxM { n: usize, m: usize },
}

pub struct QuatPruningEngine {
    format: SparseQuatFormat,
    target_sparsity: f32,  // 0.5 for 2:4
}

impl QuatPruningEngine {
    pub fn prune(&self, weights: &[Quaternion]) -> (Vec<Quaternion>, SparseMetadata);

    // Key insight: Prune at quaternion granularity (4 floats together)
    // Never individual floats—preserves algebraic structure

    pub fn compute_norms(&self, quats: &[Quaternion]) -> Vec<f32> {
        // |q| = sqrt(w² + x² + y² + z²)
        // Per-quaternion norm, not per-float
    }

    pub fn apply_2x4_pattern(&self, norms: &[f32]) -> Vec<(usize, usize)> {
        // Keep top 2 of every 4 quaternions by norm
        // Return (index, kept_count) for metadata
    }
}

pub struct SparseMetadata {
    format: SparseQuatFormat,
    sparsity_ratio: f32,
    nnz_quats: usize,  // Non-zero quaternions
    metadata_size: usize,
}
```

**Modifications**:
- `compiler/src/codegen/gpu/ir.rs` - Extend sparse format support
- `compiler/src/codegen/gpu/qnn_tensor_core.rs` - Add SparseQuatWMMAFragment

**Expected Outcome**: ~400 LOC, format-aware pruning fully implemented

---

### Phase 2C: Integration & Validation (1-2 tasks)

#### Task 6: Create Integration Tests for Compiler Passes

**Location**: `compiler/tests/phase2_compiler_integration/`

**Test Structure**:
```
compiler/tests/phase2_compiler_integration/
├── mixed_precision_pass.rs     # Test MP transform
├── semantic_fusion.rs           # Test pattern detection
├── sparse_quat_lowering.rs      # Test sparse format selection
├── qat_autodiff.rs              # Test STE gradient rules
└── full_pipeline.rs             # End-to-end compilation
```

**Example Test** (mixed_precision_pass.rs):
```rust
#[test]
fn test_mp_transform_classifies_ops_correctly() {
    let kernel = create_test_kernel_with_mixed_ops();
    let config = MixedPrecisionConfig::default();
    let transformer = MixedPrecisionTransform::new(config);

    let transformed = transformer.transform_kernel(&kernel);

    // Verify: exp/log → FP32, matmul → FP16
    assert_has_fp32_exp_ops(&transformed);
    assert_has_fp16_matmul_ops(&transformed);
}
```

**Expected Outcome**: 100+ lines, validates compiler pass correctness

---

## Implementation Order & Parallelization

### Round 1 (Parallel - Day 1):
- [ ] Task 1: ir.rs extensions (IR operations)
- [ ] Task 2: mixed_precision module (MP training pass)
- [ ] Task 3: autodiff QAT integration (STE gradients)

### Round 2 (Parallel - Day 2):
- [ ] Task 4: fusion.rs semantic patterns
- [ ] Task 5: sparse_quat.rs implementation
- [ ] Task 6: Integration tests

### Validation:
- Build verification: `cargo build --lib`
- All tests pass: `cargo test --lib`
- No new warnings/clippy issues
- Codegen backends (PTX/Metal) compile without errors

---

## Success Criteria

✅ **Phase 2A Complete**:
- ir.rs has all Phase 2 GpuOp variants
- mixed_precision module compiles and integrates
- QAT STE gradients working
- No compilation errors

✅ **Phase 2B Complete**:
- Semantic fusion patterns discoverable
- Sparse quaternion formats fully supported
- PTX/Metal emit correct instructions

✅ **Phase 2C Complete**:
- Integration tests: 100%+ passing (same as framework tests)
- End-to-end compilation: small model → PTX/Metal → executable

✅ **Final Verification**:
- All Phase 2 features co-function without conflicts
- Benchmark: training 100 iterations with all features enabled
- Performance targets met or characterized

---

## File Summary

**New Files** (~1,450 LOC):
- `compiler/src/codegen/gpu/mixed_precision/mod.rs` (~50 LOC)
- `compiler/src/codegen/gpu/mixed_precision/config.rs` (~80 LOC)
- `compiler/src/codegen/gpu/mixed_precision/loss_scaler.rs` (~120 LOC)
- `compiler/src/codegen/gpu/mixed_precision/transform.rs` (~150 LOC)
- `compiler/src/codegen/gpu/sparse_quat.rs` (~400 LOC)
- `compiler/tests/phase2_compiler_integration/*.rs` (~150+ LOC)

**Modified Files** (~600 LOC additions):
- `compiler/src/codegen/gpu/ir.rs` (+200 LOC)
- `compiler/src/codegen/gpu/fusion.rs` (+300 LOC)
- `compiler/src/codegen/gpu/ptx.rs` (+50 LOC)
- `compiler/src/codegen/gpu/metal.rs` (+50 LOC)
- `compiler/src/autodiff/primitives.rs` (+100 LOC)
- Various supporting files (+300 LOC)

**Total**: ~2,050 LOC new compiler integration code

---

## Dependencies & Risks

**Hard Dependencies**:
- SparseQuatFormat must be in ir.rs before sparse_quat.rs implementation
- MixedPrecisionType must be in ir.rs before mixed_precision module
- All ir.rs changes must compile before fusion/sparse extensions

**Risks**:
- PTX/Metal codegen may need updates for new GpuOp variants (mitigated by existing infrastructure)
- Semantic patterns require careful cost model tuning (mitigated by profiling utilities from Phase 1)
- QAT STE gradients must match TensorFlow behavior (validated against examples)

---

## Related Documentation

- `/home/demetrios/sounio-1/PHASE2_DELIVERABLES.md` - Framework layer summary
- `/home/demetrios/sounio-1/compiler/docs/PHASE2_OPTIMIZATIONS.md` - User guide
- `/home/demetrios/.claude/plans/robust-crafting-kettle.md` - Original implementation plan
- `/home/demetrios/sounio-1/CLAUDE.md` - Project guidelines

---

## Next Steps

1. Verify current test status (tests running in background)
2. Launch Phase 2A tasks in parallel (ir.rs, mixed_precision, autodiff)
3. Verify each produces clean compilation
4. Launch Phase 2B tasks (fusion patterns, sparse_quat)
5. Create Phase 2C integration tests
6. Full validation and benchmarking
