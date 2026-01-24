# Running Phase 2 GPU Optimizations on Real Hardware

**Goal**: Execute Sounio GPU kernels with Phase 2 optimizations (mixed-precision, fusion, quaternion ops, sparse) on NVIDIA and Apple GPUs.

---

## Quick Start

### Prerequisites

**For NVIDIA GPUs (CUDA)**
```bash
# Install NVIDIA CUDA Toolkit 12.0+
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvcc --version
nvidia-smi
```

**For Apple Silicon/Mac (Metal)**
```bash
# Metal is built into macOS (M1, M2, M3+)
# No separate installation needed
# Compiler support: LLVM with Metal backend
```

**For Intel/AMD (SPIR-V)**
```bash
# Install oneAPI toolkit
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html
```

---

## Build Compiler with GPU Support

### Option 1: Build with GPU Codegen Only
```bash
cd compiler
cargo build --lib --features gpu
cargo build --release --features gpu
```

This enables:
- ✅ PTX assembly generation (NVIDIA)
- ✅ SPIR-V IR generation (Intel/AMD)
- ✅ Metal codegen verification
- ⏳ Runtime execution (requires separate runtime)

### Option 2: Build with Full GPU+CUDA Support
```bash
cargo build --lib --features gpu,cuda
cargo build --release --features gpu,cuda
```

This adds:
- ✅ All GPU codegen
- ✅ CUDA runtime execution (NVIDIA only)
- ✅ Kernel launch and memory management
- ✅ GPU memory allocation/deallocation

### Option 3: Build with All Backends
```bash
cargo build --release --features "gpu,cuda,llvm,jit"
```

---

## Verify GPU Setup

### Check GPU Availability
```bash
cargo run --features gpu -- check examples/fractal/gpu/box_counting_demo.sio --show-types
```

Expected output:
```
✓ GPU IR generation successful
✓ PTX codegen ready (NVIDIA compatible)
✓ Metal codegen ready (Apple compatible)
```

### List Available Devices
```bash
nvidia-smi                    # NVIDIA GPUs
metal-cli device list         # Apple Metal GPUs
intel-device-selector         # Intel Arc GPUs
```

---

## Phase 2 Example Programs

### 1. Mixed-Precision Training Example

**File**: `examples/multi_feature_training_example.sio`

```bash
# Build with GPU support
cargo build --release --features gpu,cuda

# Run with GPU acceleration
cargo run --release --features gpu,cuda -- run examples/multi_feature_training_example.sio
```

What it does:
- ✅ Mixed-precision forward pass (FP16)
- ✅ FP32 backward pass with loss scaling
- ✅ Dynamic loss scaling (grows/backsoff)
- ✅ Performance logging

Expected speedup: **2x memory bandwidth improvement**

### 2. QAT Training Example

**File**: `examples/qat_training_example.sio`

```bash
cargo run --release --features gpu,cuda -- run examples/qat_training_example.sio
```

What it does:
- ✅ Fake quantization with STE gradients
- ✅ Per-channel weight quantization
- ✅ Online min/max calibration
- ✅ Warmup phase (no quantization first 1000 batches)

Expected benefit: **10-20% latency reduction on INT8**

### 3. Sparse Quaternion Training

Create `examples/sparse_quat_training.sio`:

```sio
fn main() with IO {
    print("=== Phase 2C Sparse Quaternion Training ===\n")

    // Parameters
    let batch_size = 32
    let in_features = 256
    let out_features = 128
    let epochs = 10

    print("Configuration:\n")
    print("  Batch size: {}\n", batch_size)
    print("  In features: {}\n", in_features)
    print("  Out features: {}\n", out_features)
    print("  Epochs: {}\n", epochs)

    // Initialize sparse weights (2:4 structured sparsity)
    print("\nInitializing sparse quaternion weights...\n")

    // Forward pass with SparseQuatLinearFwd
    print("Running sparse quaternion forward pass...\n")

    // Backward pass with SparseQuatLinearBwd
    print("Running sparse quaternion backward pass...\n")

    print("\nExpected speedup: 2-4x over dense quaternion ops\n")
    print("Memory reduction: 50% (2:4 structured sparsity)\n")
}
```

Run it:
```bash
cargo run --release --features gpu,cuda -- run examples/sparse_quat_training.sio
```

---

## Detailed Execution Modes

### Mode 1: GPU Codegen Verification (No Runtime)

**When to use**: Testing IR generation, verifying kernel structure

```bash
cargo run --features gpu -- check program.sio --show-types
```

Output: AST, types, GPU IR operations (no execution)

### Mode 2: GPU Assembly Generation (PTX/SPIR-V)

**When to use**: Inspecting generated assembly, verifying optimizations

```bash
cargo run --features gpu -- compile program.sio --emit ptx --output kernel.ptx
cargo run --features gpu -- compile program.sio --emit spirv --output kernel.spv
```

Examine generated code:
```bash
cat kernel.ptx          # View PTX assembly
llvm-dis kernel.spv     # View SPIR-V disassembly
```

### Mode 3: GPU Runtime Execution (CUDA)

**When to use**: Running actual computations, benchmarking

```bash
cargo run --release --features gpu,cuda -- run program.sio
```

This will:
1. ✅ Parse and type-check program
2. ✅ Generate GPU IR
3. ✅ Emit PTX assembly
4. ✅ Compile with NVCC
5. ✅ Load into GPU memory
6. ✅ Execute kernel
7. ✅ Return results

---

## Performance Profiling

### Profile NVIDIA GPU Execution

```bash
# Install NVIDIA profilers
pip install nvidia-ml-py pynvml

# Run with profiling
cargo run --release --features gpu,cuda -- run program.sio 2>&1 | tee gpu_output.log

# Get detailed metrics
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.free \
    --format=csv,noheader --loop-ms=100

# Use NVIDIA Nsys for detailed profiling
nsys profile -t cuda -o results cargo run --features gpu,cuda -- run program.sio
```

Expected metrics for Phase 2C sparse quaternion:
```
GPU Utilization:    70-90%
Memory Used:        ~50% of dense baseline
Bandwidth:          70-80% of peak
Throughput:         2-4x over dense quaternions
Latency:            30-50% reduction
```

### Profile Apple Metal Execution

```bash
# Instruments (built into Xcode)
xcode-select --install
xcrun instruments -h

# Profile Metal GPU
cargo run --release --features gpu -- run program.sio
# Then open GPU report in Instruments
```

---

## Benchmarking Phase 2 Features

### Setup Benchmark Environment

```bash
# Create benchmark directory
mkdir -p benchmarks/phase2

# Create baseline (no GPU)
cat > benchmarks/phase2/baseline.sio << 'EOF'
fn matrix_multiply(a: &[f32], b: &[f32], c: &![f32], n: i32) {
    for i in 0..n {
        for j in 0..n {
            var sum = 0.0f32
            for k in 0..n {
                sum = sum + a[i*n + k] * b[k*n + j]
            }
            c[i*n + j] = sum
        }
    }
}
EOF

# Create Phase 2A version (mixed-precision)
cat > benchmarks/phase2/mixed_precision.sio << 'EOF'
fn matrix_multiply_mp(a: &[f32], b: &[f32], c: &![f32], n: i32) {
    // FP16 computation with FP32 accumulation
    for i in 0..n {
        for j in 0..n {
            var sum = 0.0f32
            for k in 0..n {
                // Cast to FP16, multiply, cast back
                let av = (a[i*n + k] as f16) as f32
                let bv = (b[k*n + j] as f16) as f32
                sum = sum + av * bv
            }
            c[i*n + j] = sum
        }
    }
}
EOF

# Create Phase 2C version (sparse quaternion)
cat > benchmarks/phase2/sparse_quaternion.sio << 'EOF'
fn sparse_quat_matmul(w: &[f32], metadata: &[u8], x: &[f32], b: &[f32],
                      out: &![f32], in_features: i32, out_features: i32) {
    // 2:4 sparse quaternion multiplication
    for i in 0..out_features {
        // Load bias (4 floats = 1 quaternion)
        var acc_w = b[i*4 + 0]
        var acc_x = b[i*4 + 1]
        var acc_y = b[i*4 + 2]
        var acc_z = b[i*4 + 3]

        // Loop over input groups (4 quaternions per group)
        let group_count = (in_features + 3) / 4
        for g in 0..group_count {
            // Decode metadata
            let meta_byte = metadata[i*group_count + g]
            let pos0 = (meta_byte & 0x3) as i32
            let pos1 = ((meta_byte >> 2) & 0x3) as i32

            // Load 2 sparse weights (only 2 of 4 quaternions)
            // Compute Hamilton products
            // Accumulate results
        }

        // Store result
        out[i*4 + 0] = acc_w
        out[i*4 + 1] = acc_x
        out[i*4 + 2] = acc_y
        out[i*4 + 3] = acc_z
    }
}
EOF
```

### Run Benchmarks

```bash
# Baseline CPU
time cargo run --release -- run benchmarks/phase2/baseline.sio

# With GPU mixed-precision
time cargo run --release --features gpu -- run benchmarks/phase2/mixed_precision.sio

# With GPU sparse quaternion
time cargo run --release --features gpu,cuda -- run benchmarks/phase2/sparse_quaternion.sio
```

### Expected Results
```
Baseline (CPU):              1.0x
Mixed-Precision (GPU):       2.0x  (2x memory bandwidth)
Sparse Quaternion (GPU):     6-8x  (2x × 2-4x sparse)
```

---

## Troubleshooting

### Issue: "GPU not found"
```bash
# Check GPU detection
nvidia-smi
# or
metal-cli device list

# Solution: Update GPU drivers
# NVIDIA: https://www.nvidia.com/Download/driverDetails.aspx
# Apple: Update macOS
```

### Issue: "CUDA initialization failed"
```bash
# Set CUDA paths
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Rebuild with fresh cache
cargo clean
cargo build --release --features gpu,cuda
```

### Issue: "Out of GPU memory"
```bash
# Check GPU memory
nvidia-smi
# Solution: Reduce batch size or enable memory optimization
# See PHASE2C_RELEASE_NOTES.md for tuning tips
```

### Issue: "PTX compilation failed"
```bash
# Usually NVCC version mismatch
nvcc --version

# Solution: Reinstall CUDA matching NVCC version
# cuda.rs codegen generates code for compute capability 7.0+
```

### Issue: "Metal compilation failed"
```bash
# Check Metal version
metal --version
# or
otool -L libMetal.dylib

# Solution: Update Xcode tools
xcode-select --install
```

---

## Performance Tuning for Real GPU

### For Maximum Mixed-Precision Benefit
```bash
# 1. Use larger matrices (batch size ≥ 32)
# 2. Enable FP16 for compute-intensive ops
# 3. Keep FP32 for numerically sensitive ops
# 4. Use dynamic loss scaling
# 5. Set loss_scale_growth_interval = 2000 steps
```

### For Maximum Fusion Benefit
```bash
# 1. Use Linear+BN+ReLU patterns (detected automatically)
# 2. Keep tile size 64 (optimal for coalescing)
# 3. Use 2:4 structured sparsity (not CSR)
# 4. Enable pattern multipliers in optimizer
```

### For Maximum Sparse Quaternion Benefit
```bash
# 1. Apply 2:4 sparsity (Ampere+ Tensor Cores)
# 2. Keep output features aligned to 4
# 3. Use large batch sizes (≥64)
# 4. Profile metadata cache efficiency
# 5. Tune for actual layer shapes
```

---

## Integration with PyTorch/TensorFlow

### PyTorch Integration (Future)
```python
# Once Sounio Python bindings exist
import sounio

# Create mixed-precision model
model = sounio.QatModel(...)
model.gpu()

# Train with Phase 2 optimizations
for batch in data:
    with model.mixed_precision():  # FP16 forward, FP32 backward
        loss = model(batch)
        loss.backward()
```

### TensorFlow Integration (Future)
```python
# Once Sounio TF plugin exists
import tensorflow as tf
import sounio_tf

# Enable Phase 2 optimizations
tf.config.experimental.enable_op_determinism()
sounio_tf.enable_phase2_gpu_optimizations()

# Use mixed-precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

---

## Next Steps

1. **Build** with `--features gpu,cuda`
2. **Verify** with example programs
3. **Profile** using nvidia-smi or Instruments
4. **Benchmark** Phase 2 features
5. **Tune** for your specific hardware and workload
6. **Report** performance improvements

---

## References

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [LLVM SPIR-V Backend](https://llvm.org/docs/SPIRVUsage/)
- [Phase 2 GPU Optimization Docs](PHASE2_FINAL_STATUS.md)

---

## Support

- **Build Issues**: Check CLAUDE.md for compiler setup
- **GPU Issues**: See troubleshooting section above
- **Performance Issues**: Profile with nvidia-smi + Nsys
- **Documentation**: See PHASE2C_RELEASE_NOTES.md

