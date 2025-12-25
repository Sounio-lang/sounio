// gpu::stats — GPU-Accelerated Statistics for Neuroimaging
//
// High-performance statistical computations essential for fMRI analysis.
// Optimized for computing correlations, means, and variances across voxels.
//
// Features:
// - Parallel mean/variance computation (Welford's algorithm)
// - Batch Pearson correlation for connectivity matrices
// - Z-score normalization across time
// - Motion regression (nuisance removal)
//
// Key optimizations:
// - Warp-level reduction for sum operations
// - Shared memory for correlation computation
// - Coalesced memory access patterns

// ============================================================================
// CONSTANTS
// ============================================================================

fn WARP_SIZE() -> i32 { 32 }
fn BLOCK_SIZE() -> i32 { 256 }

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// GPU KERNEL: PARALLEL REDUCTION (SUM)
// ============================================================================

/// Warp-level sum reduction (no sync needed within warp)
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

    // Reduce within block
    var s = gpu.block_dim.x / 2
    while s > 32 {
        if tid < s {
            sdata[tid] = sdata[tid] + sdata[tid + s]
        }
        gpu.sync()
        s = s / 2
    }

    // Warp-level reduction (no sync needed)
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

// ============================================================================
// GPU KERNEL: MEAN AND VARIANCE (Welford's Algorithm)
// ============================================================================

/// Online mean/variance for streaming data
struct WelfordState {
    count: i64,
    mean: f64,
    m2: f64,
}

fn welford_new() -> WelfordState {
    WelfordState {
        count: 0,
        mean: 0.0,
        m2: 0.0,
    }
}

fn welford_update(state: &!WelfordState, x: f64) {
    state.count = state.count + 1
    let delta = x - state.mean
    state.mean = state.mean + delta / (state.count as f64)
    let delta2 = x - state.mean
    state.m2 = state.m2 + delta * delta2
}

fn welford_variance(state: &WelfordState) -> f64 {
    if state.count < 2 {
        0.0
    } else {
        state.m2 / ((state.count - 1) as f64)
    }
}

fn welford_std(state: &WelfordState) -> f64 {
    sqrt(welford_variance(state))
}

/// GPU kernel: compute mean for each voxel across time
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

/// GPU kernel: compute variance for each voxel across time
kernel fn compute_variance_kernel(
    data: &[f64],
    means: &[f64],
    variances: &![f64],
    n_voxels: i32,
    n_times: i32
) {
    let voxel_id = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    if voxel_id < n_voxels {
        let mean = means[voxel_id]
        var sum_sq = 0.0
        var t: i32 = 0
        while t < n_times {
            let idx = voxel_id * n_times + t
            let diff = data[idx] - mean
            sum_sq = sum_sq + diff * diff
            t = t + 1
        }
        variances[voxel_id] = sum_sq / (n_times - 1) as f64
    }
}

/// GPU kernel: Z-score normalization (demean and divide by std)
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

// ============================================================================
// GPU KERNEL: PEARSON CORRELATION
// ============================================================================

/// GPU kernel: compute correlation between two timeseries
/// Uses shared memory for efficient dot product
kernel fn correlation_pair_kernel(
    ts1: &[f64],            // [n_times] - already z-scored
    ts2: &[f64],            // [n_times]
    result: &![f64],        // [1]
    n_times: i32
) {
    shared sdata: [f64; 256]

    let tid = gpu.thread_id.x
    let gid = tid

    // Each thread handles one or more time points
    var local_sum = 0.0
    var t = tid
    while t < n_times {
        local_sum = local_sum + ts1[t] * ts2[t]
        t = t + gpu.block_dim.x
    }
    sdata[tid] = local_sum
    gpu.sync()

    // Reduce
    var s = gpu.block_dim.x / 2
    while s > 0 {
        if tid < s {
            sdata[tid] = sdata[tid] + sdata[tid + s]
        }
        gpu.sync()
        s = s / 2
    }

    if tid == 0 {
        // For z-scored data, correlation = dot(x,y) / (n-1)
        result[0] = sdata[0] / (n_times - 1) as f64
    }
}

/// GPU kernel: compute full correlation matrix
/// Each block computes one row of correlations
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

/// GPU kernel: batch correlation for large number of region pairs
/// Optimized for computing upper triangle only
kernel fn correlation_batch_kernel(
    data: &[f64],           // [n_regions * n_times], z-scored
    pairs_i: &[i32],        // Region i indices
    pairs_j: &[i32],        // Region j indices
    correlations: &![f64],  // Output correlations
    n_pairs: i32,
    n_times: i32
) {
    let pair_id = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    if pair_id < n_pairs {
        let i = pairs_i[pair_id]
        let j = pairs_j[pair_id]

        var dot_sum = 0.0
        var t: i32 = 0
        while t < n_times {
            let val_i = data[i * n_times + t]
            let val_j = data[j * n_times + t]
            dot_sum = dot_sum + val_i * val_j
            t = t + 1
        }

        correlations[pair_id] = dot_sum / (n_times - 1) as f64
    }
}

// ============================================================================
// GPU KERNEL: MOTION REGRESSION
// ============================================================================

/// Regress out nuisance variables (motion, WM, CSF signals)
/// Uses ordinary least squares: data = design @ beta + residuals
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

// ============================================================================
// HOST-SIDE STATISTICS INTERFACE
// ============================================================================

/// Compute mean for each voxel (CPU reference)
fn compute_means_cpu(
    data: &[f64; 10000000],
    means: &![f64; 100000],
    n_voxels: i64,
    n_times: i64
) {
    var v: i64 = 0
    while v < n_voxels {
        var sum = 0.0
        var t: i64 = 0
        while t < n_times {
            let idx = v * n_times + t
            sum = sum + data[idx as usize]
            t = t + 1
        }
        means[v as usize] = sum / (n_times as f64)
        v = v + 1
    }
}

/// Compute variance for each voxel (CPU reference)
fn compute_variances_cpu(
    data: &[f64; 10000000],
    means: &[f64; 100000],
    variances: &![f64; 100000],
    n_voxels: i64,
    n_times: i64
) {
    var v: i64 = 0
    while v < n_voxels {
        let mean = means[v as usize]
        var sum_sq = 0.0
        var t: i64 = 0
        while t < n_times {
            let idx = v * n_times + t
            let diff = data[idx as usize] - mean
            sum_sq = sum_sq + diff * diff
            t = t + 1
        }
        variances[v as usize] = sum_sq / ((n_times - 1) as f64)
        v = v + 1
    }
}

/// Z-score normalization (CPU reference)
fn zscore_cpu(
    data: &![f64; 10000000],
    n_voxels: i64,
    n_times: i64
) {
    var v: i64 = 0
    while v < n_voxels {
        // Compute mean
        var sum = 0.0
        var t: i64 = 0
        while t < n_times {
            let idx = v * n_times + t
            sum = sum + data[idx as usize]
            t = t + 1
        }
        let mean = sum / (n_times as f64)

        // Compute std
        var sum_sq = 0.0
        t = 0
        while t < n_times {
            let idx = v * n_times + t
            let diff = data[idx as usize] - mean
            sum_sq = sum_sq + diff * diff
            t = t + 1
        }
        let std = sqrt(sum_sq / ((n_times - 1) as f64))

        // Normalize
        if std > 1e-10 {
            t = 0
            while t < n_times {
                let idx = v * n_times + t
                data[idx as usize] = (data[idx as usize] - mean) / std
                t = t + 1
            }
        } else {
            t = 0
            while t < n_times {
                let idx = v * n_times + t
                data[idx as usize] = 0.0
                t = t + 1
            }
        }

        v = v + 1
    }
}

/// Pearson correlation between two timeseries (CPU reference)
fn pearson_correlation_cpu(
    ts1: &[f64; 1000],
    ts2: &[f64; 1000],
    n: i64
) -> f64 {
    // Compute means
    var sum1 = 0.0
    var sum2 = 0.0
    var i: i64 = 0
    while i < n {
        sum1 = sum1 + ts1[i as usize]
        sum2 = sum2 + ts2[i as usize]
        i = i + 1
    }
    let mean1 = sum1 / (n as f64)
    let mean2 = sum2 / (n as f64)

    // Compute correlation
    var num = 0.0
    var denom1 = 0.0
    var denom2 = 0.0
    i = 0
    while i < n {
        let d1 = ts1[i as usize] - mean1
        let d2 = ts2[i as usize] - mean2
        num = num + d1 * d2
        denom1 = denom1 + d1 * d1
        denom2 = denom2 + d2 * d2
        i = i + 1
    }

    let denom = sqrt(denom1 * denom2)
    if denom > 1e-10 {
        num / denom
    } else {
        0.0
    }
}

/// Compute correlation matrix (CPU reference)
fn correlation_matrix_cpu(
    data: &[f64; 1000000],     // [n_regions * n_times], z-scored
    corr: &![f64; 10000],      // [n_regions * n_regions]
    n_regions: i64,
    n_times: i64
) {
    var i: i64 = 0
    while i < n_regions {
        var j: i64 = 0
        while j < n_regions {
            if i == j {
                corr[(i * n_regions + j) as usize] = 1.0
            } else {
                // Compute correlation
                var dot_sum = 0.0
                var t: i64 = 0
                while t < n_times {
                    let val_i = data[(i * n_times + t) as usize]
                    let val_j = data[(j * n_times + t) as usize]
                    dot_sum = dot_sum + val_i * val_j
                    t = t + 1
                }
                corr[(i * n_regions + j) as usize] = dot_sum / ((n_times - 1) as f64)
            }
            j = j + 1
        }
        i = i + 1
    }
}

// ============================================================================
// FISHER Z-TRANSFORM
// ============================================================================

extern "C" {
    fn log(x: f64) -> f64;
    fn tanh(x: f64) -> f64;
}

/// Fisher Z-transform: z = 0.5 * ln((1+r)/(1-r)) = arctanh(r)
fn fisher_z(r: f64) -> f64 {
    // Clamp to (-1, 1) to avoid infinity
    var r_clamped = r
    if r_clamped > 0.9999 {
        r_clamped = 0.9999
    }
    if r_clamped < -0.9999 {
        r_clamped = -0.9999
    }
    0.5 * log((1.0 + r_clamped) / (1.0 - r_clamped))
}

/// Inverse Fisher Z-transform: r = tanh(z)
fn fisher_z_inverse(z: f64) -> f64 {
    tanh(z)
}

/// GPU kernel: Fisher Z-transform correlation matrix
kernel fn fisher_z_kernel(
    corr: &![f64],
    n: i32
) {
    let i = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x

    if i < n * n {
        let r = corr[i]

        // Clamp
        var r_clamped = r
        if r_clamped > 0.9999 { r_clamped = 0.9999 }
        if r_clamped < -0.9999 { r_clamped = -0.9999 }

        // Transform (inline log approximation for GPU)
        let ratio = (1.0 + r_clamped) / (1.0 - r_clamped)

        // Newton-Raphson log approximation
        var log_val = 0.0
        if ratio > 0.0 {
            var x = ratio
            var n_iter: i32 = 0
            while x > 2.0 && n_iter < 20 {
                x = x / 2.718281828
                log_val = log_val + 1.0
                n_iter = n_iter + 1
            }
            // Taylor series for ln(1+y) where y = x-1
            let y = x - 1.0
            log_val = log_val + y - y*y/2.0 + y*y*y/3.0 - y*y*y*y/4.0
        }

        corr[i] = 0.5 * log_val
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_welford() -> bool {
    // Test Welford's algorithm against known values
    var state = welford_new()

    welford_update(&!state, 1.0)
    welford_update(&!state, 2.0)
    welford_update(&!state, 3.0)
    welford_update(&!state, 4.0)
    welford_update(&!state, 5.0)

    // Mean should be 3.0, variance should be 2.5
    fabs(state.mean - 3.0) < 1e-10 && fabs(welford_variance(&state) - 2.5) < 1e-10
}

fn test_correlation_identity() -> bool {
    // Correlation of signal with itself should be 1.0
    var ts: [f64; 1000] = [0.0; 1000]
    let n: i64 = 100

    var i: i64 = 0
    while i < n {
        ts[i as usize] = (i as f64) * 0.1
        i = i + 1
    }

    let corr = pearson_correlation_cpu(&ts, &ts, n)
    fabs(corr - 1.0) < 1e-10
}

fn test_correlation_negative() -> bool {
    // Perfectly anti-correlated signals should have r = -1
    var ts1: [f64; 1000] = [0.0; 1000]
    var ts2: [f64; 1000] = [0.0; 1000]
    let n: i64 = 100

    var i: i64 = 0
    while i < n {
        ts1[i as usize] = (i as f64)
        ts2[i as usize] = -(i as f64)
        i = i + 1
    }

    let corr = pearson_correlation_cpu(&ts1, &ts2, n)
    fabs(corr - (-1.0)) < 1e-10
}

fn test_fisher_z() -> bool {
    // Test Fisher transform properties
    // z(0) = 0
    // z(0.5) ≈ 0.549
    // z(-0.5) ≈ -0.549

    fabs(fisher_z(0.0)) < 1e-10 &&
    fabs(fisher_z(0.5) - 0.5493) < 0.001 &&
    fabs(fisher_z(-0.5) - (-0.5493)) < 0.001
}

fn test_fisher_z_roundtrip() -> bool {
    // z^-1(z(r)) = r
    let r = 0.75
    let z = fisher_z(r)
    let r_back = fisher_z_inverse(z)
    fabs(r - r_back) < 1e-10
}

fn main() -> i32 {
    print("Testing gpu::stats module...\n")

    if !test_welford() {
        print("FAIL: welford\n")
        return 1
    }
    print("PASS: welford\n")

    if !test_correlation_identity() {
        print("FAIL: correlation_identity\n")
        return 2
    }
    print("PASS: correlation_identity\n")

    if !test_correlation_negative() {
        print("FAIL: correlation_negative\n")
        return 3
    }
    print("PASS: correlation_negative\n")

    if !test_fisher_z() {
        print("FAIL: fisher_z\n")
        return 4
    }
    print("PASS: fisher_z\n")

    if !test_fisher_z_roundtrip() {
        print("FAIL: fisher_z_roundtrip\n")
        return 5
    }
    print("PASS: fisher_z_roundtrip\n")

    print("All gpu::stats tests PASSED\n")
    0
}
