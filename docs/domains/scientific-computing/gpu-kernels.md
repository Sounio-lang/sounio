# GPU Kernels

Sounio provides native GPU kernel syntax for high-performance parallel computing. The syntax is designed to be familiar to CUDA programmers while integrating with Sounio's type system.

## Overview

GPU computing in Sounio follows the SIMT (Single Instruction, Multiple Threads) model:

- **Threads** execute the same kernel code with different indices
- **Blocks** group threads that can share memory and synchronize
- **Grids** organize blocks for large-scale parallelism

## Kernel Syntax

### Basic Kernel Declaration

```sio
kernel fn vector_add(
    a: &[f64],      // Input array (read-only)
    b: &[f64],      // Input array (read-only)
    c: &![f64],     // Output array (mutable)
    n: i32          // Array length
) {
    let tid = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    if tid < n {
        c[tid] = a[tid] + b[tid]
    }
}
```

**Key syntax elements:**
- `kernel fn` declares a GPU kernel (not a regular function)
- `&[T]` for read-only input arrays
- `&![T]` for mutable output arrays (note: `&!` not `&mut`)
- `gpu.thread_id.x` etc. for thread indexing

### Thread Indexing

```sio
kernel fn index_demo(output: &![i32]) {
    // Thread index within block (0 to block_dim-1)
    let tid_x = gpu.thread_id.x
    let tid_y = gpu.thread_id.y
    let tid_z = gpu.thread_id.z

    // Block index within grid
    let bid_x = gpu.block_id.x
    let bid_y = gpu.block_id.y
    let bid_z = gpu.block_id.z

    // Block dimensions
    let bdim_x = gpu.block_dim.x
    let bdim_y = gpu.block_dim.y
    let bdim_z = gpu.block_dim.z

    // Grid dimensions
    let gdim_x = gpu.grid_dim.x
    let gdim_y = gpu.grid_dim.y
    let gdim_z = gpu.grid_dim.z

    // Global linear index (1D)
    let global_id = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    // Global linear index (2D)
    let global_x = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x
    let global_y = gpu.block_id.y * gpu.block_dim.y + gpu.thread_id.y
    let global_id_2d = global_y * (gpu.grid_dim.x * gpu.block_dim.x) + global_x
}
```

### Shared Memory

Shared memory is fast on-chip memory accessible to all threads in a block.

```sio
kernel fn reduction_sum(
    input: &[f64],
    output: &![f64],
    n: i32
) {
    // Declare shared memory (visible to all threads in block)
    shared sdata: [f64; 256]

    let tid = gpu.thread_id.x
    let gid = gpu.block_id.x * gpu.block_dim.x + tid

    // Load from global memory to shared memory
    if gid < n {
        sdata[tid] = input[gid]
    } else {
        sdata[tid] = 0.0
    }

    // Synchronize: all threads must reach this point
    gpu.sync()

    // Parallel reduction in shared memory
    var s = gpu.block_dim.x / 2
    while s > 0 {
        if tid < s {
            sdata[tid] = sdata[tid] + sdata[tid + s]
        }
        gpu.sync()
        s = s / 2
    }

    // Thread 0 writes result
    if tid == 0 {
        output[gpu.block_id.x] = sdata[0]
    }
}
```

**Shared memory guidelines:**
- Declare with `shared name: [Type; SIZE]`
- Size must be compile-time constant
- Typical sizes: 256, 512, 1024 elements
- Limited per block (usually 48KB)

### Synchronization

```sio
kernel fn sync_example(data: &![f64]) {
    shared temp: [f64; 256]

    let tid = gpu.thread_id.x

    // Load phase
    temp[tid] = data[tid]

    // Barrier: wait for all threads in block
    gpu.sync()

    // Process phase (all data now available)
    if tid > 0 {
        data[tid] = temp[tid] + temp[tid - 1]
    }
}
```

**Synchronization rules:**
- `gpu.sync()` synchronizes all threads in the **same block**
- Threads in different blocks cannot synchronize directly
- All threads must reach the same `gpu.sync()` call (no divergent paths)

## Common Patterns

### Coalesced Memory Access

For optimal memory bandwidth, adjacent threads should access adjacent memory:

```sio
// GOOD: Coalesced access (threads read consecutive addresses)
kernel fn coalesced(input: &[f64], output: &![f64]) {
    let tid = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x
    output[tid] = input[tid] * 2.0  // Thread i reads address i
}

// BAD: Strided access (threads read non-consecutive addresses)
kernel fn strided(input: &[f64], output: &![f64], stride: i32) {
    let tid = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x
    output[tid] = input[tid * stride] * 2.0  // Thread i reads address i*stride
}
```

### Warp-Level Operations

A warp is 32 threads that execute in lockstep. Within a warp, no explicit synchronization is needed.

```sio
kernel fn warp_reduce_sum(
    input: &[f64],
    output: &![f64],
    n: i32
) {
    shared sdata: [f64; 256]

    let tid = gpu.thread_id.x
    let gid = gpu.block_id.x * gpu.block_dim.x * 2 + tid

    // Load two elements per thread
    var val = 0.0
    if gid < n {
        val = input[gid]
    }
    if gid + gpu.block_dim.x < n {
        val = val + input[gid + gpu.block_dim.x]
    }
    sdata[tid] = val
    gpu.sync()

    // Reduce within block (non-warp portion)
    var s = gpu.block_dim.x / 2
    while s > 32 {
        if tid < s {
            sdata[tid] = sdata[tid] + sdata[tid + s]
        }
        gpu.sync()
        s = s / 2
    }

    // Warp-level reduction (no sync needed within warp)
    if tid < 32 {
        if gpu.block_dim.x >= 64 { sdata[tid] = sdata[tid] + sdata[tid + 32] }
        if gpu.block_dim.x >= 32 { sdata[tid] = sdata[tid] + sdata[tid + 16] }
        if gpu.block_dim.x >= 16 { sdata[tid] = sdata[tid] + sdata[tid + 8] }
        if gpu.block_dim.x >= 8 { sdata[tid] = sdata[tid] + sdata[tid + 4] }
        if gpu.block_dim.x >= 4 { sdata[tid] = sdata[tid] + sdata[tid + 2] }
        if gpu.block_dim.x >= 2 { sdata[tid] = sdata[tid] + sdata[tid + 1] }
    }

    if tid == 0 {
        output[gpu.block_id.x] = sdata[0]
    }
}
```

### Grid-Stride Loop

For processing arrays larger than the grid:

```sio
kernel fn grid_stride_process(
    input: &[f64],
    output: &![f64],
    n: i64
) {
    let grid_size = gpu.grid_dim.x * gpu.block_dim.x
    var i = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    while i < n {
        output[i] = process(input[i])
        i = i + grid_size
    }
}
```

**Benefits:**
- Single kernel handles any array size
- Better occupancy for small arrays
- Reuses threads (amortizes launch overhead)

### Tiled Matrix Operations

For matrix operations, load tiles into shared memory:

```sio
kernel fn matrix_multiply_tiled(
    A: &[f64],      // [M x K]
    B: &[f64],      // [K x N]
    C: &![f64],     // [M x N]
    M: i32,
    N: i32,
    K: i32
) {
    let TILE_SIZE: i32 = 16

    shared As: [f64; 256]  // 16x16 tile of A
    shared Bs: [f64; 256]  // 16x16 tile of B

    let row = gpu.block_id.y * TILE_SIZE + gpu.thread_id.y
    let col = gpu.block_id.x * TILE_SIZE + gpu.thread_id.x

    var sum = 0.0

    // Loop over tiles
    var t: i32 = 0
    while t < (K + TILE_SIZE - 1) / TILE_SIZE {
        // Load tile of A into shared memory
        let a_col = t * TILE_SIZE + gpu.thread_id.x
        if row < M && a_col < K {
            As[gpu.thread_id.y * TILE_SIZE + gpu.thread_id.x] = A[row * K + a_col]
        } else {
            As[gpu.thread_id.y * TILE_SIZE + gpu.thread_id.x] = 0.0
        }

        // Load tile of B into shared memory
        let b_row = t * TILE_SIZE + gpu.thread_id.y
        if b_row < K && col < N {
            Bs[gpu.thread_id.y * TILE_SIZE + gpu.thread_id.x] = B[b_row * N + col]
        } else {
            Bs[gpu.thread_id.y * TILE_SIZE + gpu.thread_id.x] = 0.0
        }

        gpu.sync()

        // Compute partial dot product
        var k: i32 = 0
        while k < TILE_SIZE {
            sum = sum + As[gpu.thread_id.y * TILE_SIZE + k] * Bs[k * TILE_SIZE + gpu.thread_id.x]
            k = k + 1
        }

        gpu.sync()
        t = t + 1
    }

    // Write result
    if row < M && col < N {
        C[row * N + col] = sum
    }
}
```

## GPU Statistics Module

The `gpu::stats` module provides optimized statistical kernels:

### Mean and Variance

```sio
use gpu::stats::*

// Welford's algorithm for online mean/variance
struct WelfordState {
    count: i64,
    mean: f64,
    m2: f64
}

fn welford_new() -> WelfordState {
    WelfordState { count: 0, mean: 0.0, m2: 0.0 }
}

fn welford_update(state: &!WelfordState, x: f64) {
    state.count = state.count + 1
    let delta = x - state.mean
    state.mean = state.mean + delta / (state.count as f64)
    let delta2 = x - state.mean
    state.m2 = state.m2 + delta * delta2
}

fn welford_variance(state: &WelfordState) -> f64 {
    if state.count < 2 { 0.0 }
    else { state.m2 / ((state.count - 1) as f64) }
}
```

**GPU kernel for voxel-wise mean:**

```sio
kernel fn compute_mean_kernel(
    data: &[f64],           // [n_voxels * n_times]
    means: &![f64],         // [n_voxels]
    n_voxels: i32,
    n_times: i32
) {
    let voxel_id = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    if voxel_id < n_voxels {
        var sum = 0.0
        var t: i32 = 0
        while t < n_times {
            let idx = voxel_id * n_times + t
            sum = sum + data[idx]
            t = t + 1
        }
        means[voxel_id] = sum / n_times as f64
    }
}
```

### Z-Score Normalization

```sio
kernel fn zscore_kernel(
    data: &![f64],
    means: &[f64],
    stds: &[f64],
    n_voxels: i32,
    n_times: i32
) {
    let voxel_id = gpu.block_id.x
    let time_id = gpu.thread_id.x

    if voxel_id < n_voxels && time_id < n_times {
        let idx = voxel_id * n_times + time_id
        let mean = means[voxel_id]
        let std = stds[voxel_id]

        if std > 1e-10 {
            data[idx] = (data[idx] - mean) / std
        } else {
            data[idx] = 0.0
        }
    }
}
```

### Correlation Matrix

```sio
kernel fn correlation_matrix_kernel(
    data: &[f64],           // [n_regions * n_times], z-scored
    corr_matrix: &![f64],   // [n_regions * n_regions]
    n_regions: i32,
    n_times: i32
) {
    shared ts_i: [f64; 512]  // Cache one timeseries

    let region_i = gpu.block_id.x
    let region_j = gpu.block_id.y * gpu.block_dim.x + gpu.thread_id.x

    // Load region_i timeseries into shared memory
    if gpu.thread_id.x < n_times {
        ts_i[gpu.thread_id.x] = data[region_i * n_times + gpu.thread_id.x]
    }
    gpu.sync()

    if region_j < n_regions {
        // Compute correlation with region_j
        var dot_sum = 0.0
        var t: i32 = 0
        while t < n_times {
            let val_j = data[region_j * n_times + t]
            dot_sum = dot_sum + ts_i[t] * val_j
            t = t + 1
        }

        let corr = dot_sum / (n_times - 1) as f64
        corr_matrix[region_i * n_regions + region_j] = corr
    }
}
```

### Fisher Z-Transform

```sio
kernel fn fisher_z_kernel(
    corr: &![f64],
    n: i32
) {
    let i = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    if i < n * n {
        let r = corr[i]

        // Clamp to (-1, 1) to avoid infinity
        var r_clamped = r
        if r_clamped > 0.9999 { r_clamped = 0.9999 }
        if r_clamped < -0.9999 { r_clamped = -0.9999 }

        // z = 0.5 * ln((1+r)/(1-r)) = arctanh(r)
        let ratio = (1.0 + r_clamped) / (1.0 - r_clamped)

        // Log approximation for GPU
        var log_val = 0.0
        if ratio > 0.0 {
            var x = ratio
            var n_iter: i32 = 0
            while x > 2.0 && n_iter < 20 {
                x = x / 2.718281828
                log_val = log_val + 1.0
                n_iter = n_iter + 1
            }
            let y = x - 1.0
            log_val = log_val + y - y*y/2.0 + y*y*y/3.0 - y*y*y*y/4.0
        }

        corr[i] = 0.5 * log_val
    }
}
```

## Motion Regression

```sio
kernel fn regress_out_kernel(
    data: &![f64],          // [n_voxels * n_times]
    design: &[f64],         // [n_times * n_regressors]
    betas: &[f64],          // [n_voxels * n_regressors] - precomputed
    n_voxels: i32,
    n_times: i32,
    n_regressors: i32
) {
    let voxel_id = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    if voxel_id < n_voxels {
        var t: i32 = 0
        while t < n_times {
            // Compute predicted value from regressors
            var predicted = 0.0
            var r: i32 = 0
            while r < n_regressors {
                let beta = betas[voxel_id * n_regressors + r]
                let design_val = design[t * n_regressors + r]
                predicted = predicted + beta * design_val
                r = r + 1
            }

            // Subtract prediction (residualize)
            let idx = voxel_id * n_times + t
            data[idx] = data[idx] - predicted

            t = t + 1
        }
    }
}
```

## GPU Context and Launch Configuration

```sio
use gpu::*

// GPU execution context
struct GPUContext {
    device_id: i32,
    compute_capability: i32,
    max_threads_per_block: i32,
    shared_memory_size: i64,
    warp_size: i32
}

fn gpu_context_default() -> GPUContext {
    GPUContext {
        device_id: 0,
        compute_capability: 75,      // SM 7.5 (Turing)
        max_threads_per_block: 1024,
        shared_memory_size: 49152,   // 48 KB
        warp_size: 32
    }
}

// Check if GPU is available
fn gpu_available() -> bool {
    // Would check for actual GPU at runtime
    true
}

// Get optimal block size for a kernel
fn optimal_block_size(n_elements: i64) -> i32 {
    if n_elements <= 64 { 64 }
    else if n_elements <= 128 { 128 }
    else if n_elements <= 256 { 256 }
    else { 256 }  // Default
}

// Get grid size for given problem
fn grid_size(n_elements: i64, block_size: i32) -> i32 {
    ((n_elements + block_size as i64 - 1) / block_size as i64) as i32
}
```

## Performance Guidelines

### 1. Maximize Occupancy

```sio
// Use enough threads to hide memory latency
// Typical: 256-512 threads per block
let BLOCK_SIZE: i32 = 256

// Calculate grid size to cover all elements
let grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE
```

### 2. Minimize Divergence

```sio
// BAD: Divergent branches within warp
kernel fn divergent(data: &![f64], n: i32) {
    let tid = gpu.thread_id.x
    if tid % 2 == 0 {
        data[tid] = expensive_path_a(data[tid])
    } else {
        data[tid] = expensive_path_b(data[tid])
    }
}

// BETTER: Process in two passes or restructure
kernel fn non_divergent(data: &![f64], n: i32, phase: i32) {
    let tid = gpu.thread_id.x
    if phase == 0 && tid % 2 == 0 {
        data[tid] = expensive_path_a(data[tid])
    }
    if phase == 1 && tid % 2 == 1 {
        data[tid] = expensive_path_b(data[tid])
    }
}
```

### 3. Use Shared Memory Wisely

```sio
// Cache frequently accessed data
kernel fn cache_friendly(input: &[f64], output: &![f64]) {
    shared cache: [f64; 256]

    let tid = gpu.thread_id.x
    let gid = gpu.block_id.x * gpu.block_dim.x + tid

    // One global load
    cache[tid] = input[gid]
    gpu.sync()

    // Multiple reads from shared memory (fast)
    var sum = 0.0
    sum = sum + cache[tid]
    if tid > 0 { sum = sum + cache[tid - 1] }
    if tid < 255 { sum = sum + cache[tid + 1] }

    output[gid] = sum
}
```

### 4. Avoid Bank Conflicts

Shared memory is organized in banks (typically 32). Concurrent access to the same bank serializes.

```sio
// BAD: All threads access same bank
shared data: [f64; 256]
let val = data[gpu.thread_id.x * 32]  // Stride of 32 = bank conflict

// GOOD: Consecutive access
let val = data[gpu.thread_id.x]  // Each thread different bank
```

### 5. Prefer Compute Over Memory

```sio
// BAD: Multiple loads for same computation
kernel fn multiple_loads(a: &[f64], b: &[f64], c: &![f64]) {
    let tid = gpu.thread_id.x
    let x = a[tid]
    let y = b[tid]
    c[tid] = x + y
    c[tid] = c[tid] * x  // Reloads c[tid]
}

// GOOD: Compute in registers
kernel fn register_compute(a: &[f64], b: &[f64], c: &![f64]) {
    let tid = gpu.thread_id.x
    let x = a[tid]
    let y = b[tid]
    var result = x + y
    result = result * x
    c[tid] = result  // Single store
}
```

## Debugging Tips

### 1. Bounds Checking

```sio
kernel fn safe_kernel(data: &![f64], n: i32) {
    let tid = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    // Always check bounds
    if tid >= n {
        return
    }

    data[tid] = process(data[tid])
}
```

### 2. Verify with CPU Reference

```sio
fn verify_gpu_result(gpu_result: &[f64], n: i64) -> bool {
    // Compute reference on CPU
    var cpu_result: [f64; 10000] = [0.0; 10000]
    compute_cpu_reference(&!cpu_result, n)

    // Compare
    var max_err: f64 = 0.0
    var i: i64 = 0
    while i < n {
        let err = abs_f64(gpu_result[i as usize] - cpu_result[i as usize])
        if err > max_err { max_err = err }
        i = i + 1
    }

    if max_err > 1e-6 {
        println("GPU/CPU mismatch: max error = ", max_err)
        return false
    }
    return true
}
```

### 3. Print from Single Thread

```sio
kernel fn debug_kernel(data: &[f64]) {
    if gpu.block_id.x == 0 && gpu.thread_id.x == 0 {
        // Only first thread prints (avoid flood)
        // Note: GPU printing may not be available on all backends
    }
}
```
