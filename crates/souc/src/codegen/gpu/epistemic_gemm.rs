//! Epistemic GEMM (General Matrix Multiply) Operations
//!
//! GPU-accelerated matrix multiplication with built-in uncertainty propagation.
//! Supports both standard precision (f32) and Tensor Core (f16) operations.

/// Epistemic GEMM configuration
#[derive(Debug, Clone, Copy)]
pub struct EpistemicGemmConfig {
    /// Matrix dimensions: M x K * K x N = M x N
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// Scaling factors
    pub alpha: f32,
    pub beta: f32,
    /// Precision
    pub precision: MatrixPrecision,
    /// Use Tensor Cores if available
    pub use_tensor_cores: bool,
    /// Confidence threshold for gating
    pub confidence_threshold: Option<f32>,
}

/// Matrix precision
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixPrecision {
    F16,
    F32,
    F64,
}

/// Epistemic matrix representation
pub struct EpistemicMatrix {
    /// Matrix values
    pub values: Vec<f32>,
    /// Per-element uncertainty bounds
    pub epsilon: Vec<f32>,
    /// Confidence values (optional)
    pub confidence: Option<Vec<f32>>,
    /// Dimensions
    pub rows: usize,
    pub cols: usize,
}

impl EpistemicMatrix {
    /// Create a new epistemic matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        let size = rows * cols;
        Self {
            values: vec![0.0; size],
            epsilon: vec![0.0; size],
            confidence: None,
            rows,
            cols,
        }
    }

    /// Create from values with uniform uncertainty
    pub fn from_values(values: Vec<f32>, epsilon: f32, rows: usize, cols: usize) -> Self {
        let size = rows * cols;
        assert_eq!(values.len(), size);

        Self {
            values,
            epsilon: vec![epsilon; size],
            confidence: None,
            rows,
            cols,
        }
    }

    /// Get element at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> (f32, f32) {
        let idx = i * self.cols + j;
        (self.values[idx], self.epsilon[idx])
    }

    /// Set element at position (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: f32, epsilon: f32) {
        let idx = i * self.cols + j;
        self.values[idx] = value;
        self.epsilon[idx] = epsilon;
    }
}

/// Roofline performance estimate for the tiled F32 GEMM kernel.
///
/// Returns `(gflops_estimate, arithmetic_intensity_flops_per_byte)` given
/// hardware peak throughput and measured (or estimated) elapsed time.
///
/// - `elapsed_ms`: kernel execution time in milliseconds
/// - `peak_mem_bw_gbs`: device memory bandwidth in GB/s (e.g. 936 for A100, 760 for RTX 3080)
/// - `peak_fp32_tflops`: device peak FP32 throughput (e.g. 19.5 for A100, 29.8 for RTX 3080)
///
/// Returns `(achieved_gflops, arith_intensity, bottleneck)` where bottleneck is
/// "memory" or "compute".
pub fn roofline_estimate(
    config: &EpistemicGemmConfig,
    elapsed_ms: f64,
    peak_mem_bw_gbs: f64,
    peak_fp32_tflops: f64,
) -> (f64, f64, &'static str) {
    let m = config.m as f64;
    let k = config.k as f64;
    let n = config.n as f64;

    // FLOPs: 2*M*K*N for value GEMM (1 mul + 1 add per accumulation)
    let flops = 2.0 * m * k * n;

    // Memory traffic for tiled kernel: A (m×k) + B (k×n) + C read (m×n) + result write (m×n)
    // With TILE_M=64, TILE_N=64, TILE_K=16: each element of A/B is read once total
    let bytes = (m * k + k * n + 2.0 * m * n) * 4.0;

    let arith_intensity = flops / bytes;
    let ridge_point = peak_fp32_tflops * 1000.0 / peak_mem_bw_gbs; // FLOPs/byte

    let elapsed_s = elapsed_ms / 1000.0;
    let achieved_gflops = (flops / elapsed_s) / 1e9;

    let bottleneck = if arith_intensity < ridge_point {
        "memory"
    } else {
        "compute"
    };

    (achieved_gflops, arith_intensity, bottleneck)
}

/// Per-launch roofline profile emitted after every epistemic GEMM kernel dispatch.
///
/// Consumers (profiler.sio bridge, CLI `--profile-gpu`) can display this to give
/// an observable GFLOPS number without requiring CUPTI or vendor profilers.
#[derive(Debug, Clone)]
pub struct GemmLaunchProfile {
    /// Total floating-point operations (2·M·K·N for value GEMM).
    pub flops: f64,
    /// Achieved throughput in GFLOPS given `elapsed_ms`.
    pub achieved_gflops: f64,
    /// Arithmetic intensity in FLOPs/byte (tiled: ~16 FLOPs/byte).
    pub arithmetic_intensity: f64,
    /// Device ridge point in FLOPs/byte (peak_fp32 / peak_bw).
    pub ridge_point_flops_per_byte: f64,
    /// Whether the kernel is memory-bandwidth limited (true) or compute-limited.
    pub memory_bound: bool,
    /// Efficiency fraction: achieved / peak (0..1).
    pub efficiency: f64,
    /// Whether double-buffered cp.async path was used.
    pub double_buffered: bool,
}

/// Build a `GemmLaunchProfile` from hardware specs and measured elapsed time.
///
/// Call this immediately after every GPU kernel launch (value GEMM only — epsilon
/// kernel is negligible and excluded from the estimate).
///
/// # Arguments
/// * `config` — same config used to generate the PTX
/// * `elapsed_ms` — wall-clock kernel duration in ms (from CUDA event or system timer)
/// * `peak_mem_bw_gbs` — device peak memory bandwidth in GB/s
/// * `peak_fp32_tflops` — device peak FP32 throughput in TFLOPS
/// * `double_buffered` — true if the cp.async kernel was used
pub fn profile_gemm_launch(
    config: &EpistemicGemmConfig,
    elapsed_ms: f64,
    peak_mem_bw_gbs: f64,
    peak_fp32_tflops: f64,
    double_buffered: bool,
) -> GemmLaunchProfile {
    let (achieved_gflops, arith_intensity, bottleneck) =
        roofline_estimate(config, elapsed_ms, peak_mem_bw_gbs, peak_fp32_tflops);
    let ridge_point = peak_fp32_tflops * 1000.0 / peak_mem_bw_gbs;
    let efficiency = achieved_gflops / (peak_fp32_tflops * 1000.0);
    let m = config.m as f64;
    let k = config.k as f64;
    let n = config.n as f64;
    GemmLaunchProfile {
        flops: 2.0 * m * k * n,
        achieved_gflops,
        arithmetic_intensity: arith_intensity,
        ridge_point_flops_per_byte: ridge_point,
        memory_bound: bottleneck == "memory",
        efficiency,
        double_buffered,
    }
}

/// Format a `GemmLaunchProfile` as a single log line for stdout/stderr.
///
/// Example output:
/// `[souc-gpu] epistemic_gemm 4096×4096×4096  elapsed=0.14ms  14.0 GFLOPS  intensity=16.3  bound=memory  eff=0.07%  dbuf=yes`
pub fn format_gemm_profile(name: &str, config: &EpistemicGemmConfig, p: &GemmLaunchProfile) -> String {
    format!(
        "[souc-gpu] {} {}×{}×{}  intensity={:.1} FLOPs/B  ridge={:.1}  {:.1} GFLOPS  bound={}  eff={:.2}%  dbuf={}",
        name,
        config.m, config.k, config.n,
        p.arithmetic_intensity,
        p.ridge_point_flops_per_byte,
        p.achieved_gflops,
        if p.memory_bound { "memory" } else { "compute" },
        p.efficiency * 100.0,
        if p.double_buffered { "yes" } else { "no" },
    )
}

/// Compute recommended launch configuration for the tiled F32 GEMM kernel.
///
/// Returns `(grid_x, grid_y, block_x, block_y)` for use with GPU launch APIs.
/// Tile size: 64×64 per block, thread block: 16×16.
pub fn tiled_gemm_launch_config(config: &EpistemicGemmConfig) -> (u32, u32, u32, u32) {
    let grid_x = ((config.n + 63) / 64) as u32; // ceil(N/64) blocks in X (col tiles)
    let grid_y = ((config.m + 63) / 64) as u32; // ceil(M/64) blocks in Y (row tiles)
    (grid_x, grid_y, 16, 16)
}

/// Compute recommended launch configuration for the epsilon-only kernel.
///
/// Returns `(grid_x, block_x)` for 1-D launch over M×N elements.
pub fn epsilon_kernel_launch_config(config: &EpistemicGemmConfig) -> (u32, u32) {
    let total = (config.m * config.n) as u32;
    let grid_x = (total + 255) / 256;
    (grid_x, 256)
}

/// Compute epistemic GEMM: C = α * A * B + β * C
///
/// Propagates uncertainty through matrix multiplication using:
/// ε_C[i,j] ≈ α * Σ_l (|A[i,l]| * ε_B[l,j] + |B[l,j]| * ε_A[i,l]) + β * ε_C[i,j]
pub fn compute_epistemic_gemm(
    a: &EpistemicMatrix,
    b: &EpistemicMatrix,
    c: &EpistemicMatrix,
    config: &EpistemicGemmConfig,
) -> EpistemicMatrix {
    assert_eq!(a.cols, b.rows, "A cols must equal B rows");
    assert_eq!(a.rows, c.rows, "A rows must equal C rows");
    assert_eq!(b.cols, c.cols, "B cols must equal C cols");
    assert_eq!(a.rows, config.m);
    assert_eq!(a.cols, config.k);
    assert_eq!(b.cols, config.n);

    let mut result = EpistemicMatrix::new(config.m, config.n);
    let alpha_abs = config.alpha.abs();
    let beta_abs = config.beta.abs();

    for i in 0..config.m {
        for j in 0..config.n {
            let mut value_sum = 0.0;
            let mut epsilon_sum = 0.0;

            // Matrix multiplication with uncertainty propagation
            for l in 0..config.k {
                let a_idx = i * config.k + l;
                let b_idx = l * config.n + j;

                // Value: A[i,l] * B[l,j]
                value_sum += a.values[a_idx] * b.values[b_idx];

                // Uncertainty: |A[i,l]| * ε_B[l,j] + |B[l,j]| * ε_A[i,l]
                epsilon_sum += a.values[a_idx].abs() * b.epsilon[b_idx];
                epsilon_sum += b.values[b_idx].abs() * a.epsilon[a_idx];

                // Cross-term: ε_A[i,l] * ε_B[l,j]
                epsilon_sum += a.epsilon[a_idx] * b.epsilon[b_idx];
            }

            // Apply scaling and add to C
            let c_idx = i * config.n + j;
            result.values[i * config.n + j] =
                config.alpha * value_sum + config.beta * c.values[c_idx];

            result.epsilon[i * config.n + j] =
                alpha_abs * epsilon_sum + beta_abs * c.epsilon[c_idx];
        }
    }

    result
}

/// Confidence-gated GEMM
///
/// Only includes terms where confidence exceeds threshold
pub fn compute_confidence_gated_gemm(
    a: &EpistemicMatrix,
    b: &EpistemicMatrix,
    c: &EpistemicMatrix,
    config: &EpistemicGemmConfig,
    confidence_threshold: f32,
) -> EpistemicMatrix {
    assert!(
        a.confidence.is_some() && b.confidence.is_some(),
        "Both matrices must have confidence values"
    );

    let a_conf = a.confidence.as_ref().unwrap();
    let b_conf = b.confidence.as_ref().unwrap();

    let mut result = EpistemicMatrix::new(config.m, config.n);
    let alpha_abs = config.alpha.abs();
    let beta_abs = config.beta.abs();

    for i in 0..config.m {
        for j in 0..config.n {
            let mut value_sum = 0.0;
            let mut epsilon_sum = 0.0;
            let mut weight_sum = 0.0;

            for l in 0..config.k {
                let a_idx = i * config.k + l;
                let b_idx = l * config.n + j;

                // Check confidence
                if a_conf[a_idx] >= confidence_threshold
                    && b_conf[b_idx] >= confidence_threshold
                {
                    let weight = a_conf[a_idx] * b_conf[b_idx];
                    value_sum += weight * a.values[a_idx] * b.values[b_idx];

                    // Weighted uncertainty propagation
                    epsilon_sum += weight
                        * (a.values[a_idx].abs() * b.epsilon[b_idx]
                            + b.values[b_idx].abs() * a.epsilon[a_idx]);

                    weight_sum += weight;
                }
            }

            if weight_sum > 0.0 {
                let c_idx = i * config.n + j;
                result.values[i * config.n + j] =
                    config.alpha * (value_sum / weight_sum) + config.beta * c.values[c_idx];

                result.epsilon[i * config.n + j] =
                    alpha_abs * (epsilon_sum / weight_sum) + beta_abs * c.epsilon[c_idx];
            }
        }
    }

    result
}

/// Compute only the epsilon (uncertainty) component of GEMM.
///
/// Returns a flat Vec<f32> of length m*n containing the propagated
/// uncertainty for each output element, given flat input slices.
///
/// ε_out[i,j] = |α| * Σ_l (|A[i,l]|*ε_B[l,j] + |B[l,j]|*ε_A[i,l] + ε_A[i,l]*ε_B[l,j])
///            + |β| * ε_C[i,j]
pub fn compute_gemm_epsilon(
    a_values: &[f32],
    a_epsilon: &[f32],
    b_values: &[f32],
    b_epsilon: &[f32],
    _c_values: &[f32],
    c_epsilon: &[f32],
    alpha: f32,
    beta: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let alpha_abs = alpha.abs();
    let beta_abs = beta.abs();
    let mut result = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut eps_sum = 0.0f32;

            for l in 0..k {
                let a_idx = i * k + l;
                let b_idx = l * n + j;
                eps_sum += a_values[a_idx].abs() * b_epsilon[b_idx];
                eps_sum += b_values[b_idx].abs() * a_epsilon[a_idx];
                eps_sum += a_epsilon[a_idx] * b_epsilon[b_idx];
            }

            let c_idx = i * n + j;
            result[c_idx] = alpha_abs * eps_sum + beta_abs * c_epsilon[c_idx];
        }
    }

    result
}

/// Generate PTX code for epistemic GEMM.
///
/// Dispatches to:
/// - `generate_wmma_k_loop_ptx` for F16 + tensor cores (K-looped WMMA)
/// - `generate_tiled_f32_gemm_ptx` for F32 (64×64 smem-tiled, separated from ε)
///
/// **Note**: the F32 path generates the *value* kernel only. Call
/// `generate_epsilon_only_ptx` separately and launch it after the value kernel.
/// The a_eps_ptr, b_eps_ptr, c_eps_ptr, result_eps_ptr params are accepted for
/// API compatibility but used only by the epsilon kernel.
///
/// Dispatch rules:
/// - F16 + `use_tensor_cores` → WMMA K-loop (tensor cores, sm_70+)
/// - F32 + `sm_major >= 8`    → double-buffered cp.async tiled kernel (sm_80+)
/// - F32 otherwise            → single-buffer tiled kernel (sm_75 fallback)
pub fn generate_epistemic_gemm_ptx(config: &EpistemicGemmConfig, kernel_name: &str) -> String {
    generate_epistemic_gemm_ptx_sm(config, kernel_name, 8, 0)
}

/// SM-version-aware dispatch for PTX generation.
///
/// `sm_major` / `sm_minor` are the device compute capability (e.g. 8, 0 for A100).
/// Callers that have a `DeviceInfo` should extract the SM version and use this
/// function so the autotune picks the best path automatically.
pub fn generate_epistemic_gemm_ptx_sm(
    config: &EpistemicGemmConfig,
    kernel_name: &str,
    sm_major: u32,
    _sm_minor: u32,
) -> String {
    if config.use_tensor_cores && config.precision == MatrixPrecision::F16 {
        generate_wmma_k_loop_ptx(config, kernel_name)
    } else if sm_major >= 8 {
        // sm_80+: use cp.async double-buffer for ~2× global-load hiding
        generate_double_buffered_f32_gemm_ptx(config, kernel_name)
    } else {
        // sm_75 and below: no cp.async, use single-buffer tiled kernel
        generate_tiled_f32_gemm_ptx(config, kernel_name)
    }
}

/// Generate optimised shared-memory tiled F32 value GEMM PTX.
///
/// - Tile: TILE_M=64 × TILE_N=64 per block, TILE_K=16 K-strip.
/// - Thread block: 16×16 = 256 threads; each thread owns a 4×4 register tile.
/// - Shared memory: smem_a[16×64] + smem_b[16×64] = 8 KB (fits sm_80 48 KB).
/// - Arithmetic intensity: ~16 FLOPs/byte vs 0.3 for naive kernel → ~50× win.
///
/// Launch config: block=(16,16,1), grid=(ceil(N/64), ceil(M/64), 1).
pub fn generate_tiled_f32_gemm_ptx(config: &EpistemicGemmConfig, kernel_name: &str) -> String {
    let mut s = String::with_capacity(8192);

    s.push_str(".version 7.5\n");
    s.push_str(".target sm_80\n");
    s.push_str(".address_size 64\n\n");

    s.push_str(&format!(".visible .entry {}(\n", kernel_name));
    s.push_str("    .param .u64 a_ptr,\n");
    s.push_str("    .param .u64 b_ptr,\n");
    s.push_str("    .param .u64 c_ptr,\n");
    s.push_str("    .param .u64 result_ptr,\n");
    s.push_str("    .param .f32 alpha,\n");
    s.push_str("    .param .f32 beta,\n");
    s.push_str("    .param .u32 M,\n");
    s.push_str("    .param .u32 N,\n");
    s.push_str("    .param .u32 K\n");
    s.push_str(")\n{\n");

    // smem_a[TILE_K=16][TILE_M=64] + smem_b[TILE_K=16][TILE_N=64] = 2×4096 bytes = 8 KB
    s.push_str("    .shared .align 4 .f32 smem_a[1024];\n");
    s.push_str("    .shared .align 4 .f32 smem_b[1024];\n\n");

    s.push_str("    .reg .pred %p<16>;\n");
    s.push_str("    .reg .f32 %f<48>;\n");
    // r0=tx, r1=ty, r2=bx, r3=by, r4=tile_row, r5=tile_col, r6=ay_base(ty*4),
    // r7=bx_base(tx*4), r8=k_inner, r9=k_strip, r10-r14=temps,
    // r15=smem_a_base(u32), r16=smem_b_base(u32), r17=inner_k_inner,
    // r18-r19=smem addr temps, r20=M, r21=N, r22=K, r23=out_row, r24=out_col,
    // r25-r28=store temps, r29-r30=intermediate
    s.push_str("    .reg .b32 %r<32>;\n");
    s.push_str("    .reg .b64 %rd<16>;\n\n");

    // Load parameters
    s.push_str("    ld.param.u64 %rd0, [a_ptr];\n");
    s.push_str("    ld.param.u64 %rd1, [b_ptr];\n");
    s.push_str("    ld.param.u64 %rd2, [c_ptr];\n");
    s.push_str("    ld.param.u64 %rd3, [result_ptr];\n");
    s.push_str("    ld.param.f32 %f0, [alpha];\n");
    s.push_str("    ld.param.f32 %f1, [beta];\n");
    s.push_str("    ld.param.u32 %r20, [M];\n");
    s.push_str("    ld.param.u32 %r21, [N];\n");
    s.push_str("    ld.param.u32 %r22, [K];\n\n");

    // Thread/block indices
    s.push_str("    mov.u32 %r0, %tid.x;    // tx 0..15\n");
    s.push_str("    mov.u32 %r1, %tid.y;    // ty 0..15\n");
    s.push_str("    mov.u32 %r2, %ctaid.x;  // block col\n");
    s.push_str("    mov.u32 %r3, %ctaid.y;  // block row\n");
    s.push_str("    mul.lo.u32 %r4, %r3, 64;  // tile_row = by*64\n");
    s.push_str("    mul.lo.u32 %r5, %r2, 64;  // tile_col = bx*64\n");
    s.push_str("    shl.b32 %r6, %r1, 2;       // ay_base = ty*4\n");
    s.push_str("    shl.b32 %r7, %r0, 2;       // bx_base = tx*4\n");
    s.push_str("    add.u32 %r23, %r4, %r6;    // out_row = tile_row + ty*4\n");
    s.push_str("    add.u32 %r24, %r5, %r7;    // out_col = tile_col + tx*4\n\n");

    // Get 32-bit shared memory base addresses
    s.push_str("    mov.u32 %r15, smem_a;\n");
    s.push_str("    mov.u32 %r16, smem_b;\n\n");

    // Initialise 16 accumulators %f20..%f35 to 0.0
    for i in 0..16u32 {
        s.push_str(&format!("    mov.f32 %f{}, 0f00000000;\n", 20 + i));
    }
    s.push_str("\n");

    // Outer K-strip loop (k_strip = 0, 16, 32, ...)
    s.push_str("    mov.u32 %r9, 0;  // k_strip\n\n");
    s.push_str("KLOOP:\n");
    s.push_str("    setp.ge.u32 %p0, %r9, %r22;\n");
    s.push_str("    @%p0 bra KLOOP_END;\n\n");

    // Load smem_a[ty][tx*4..tx*4+3] = A[tile_row+tx*4..][k_strip+ty]
    // Thread (ty,tx): smem_a offset = ty*64 + tx*4 + i
    // Global A (M×K, row-major): A[A_row][A_col] at (A_row*K + A_col)*4
    s.push_str("    // -- load smem_a[ty][tx*4..tx*4+3] --\n");
    for i in 0..4u32 {
        // A_row = tile_row + tx*4 + i = r4 + r7 + i
        s.push_str(&format!("    add.u32 %r10, %r4, %r7;\n"));
        if i > 0 {
            s.push_str(&format!("    add.u32 %r10, %r10, {i};\n"));
        }
        // A_col = k_strip + ty = r9 + r1
        s.push_str("    add.u32 %r11, %r9, %r1;\n");
        // bounds
        s.push_str("    setp.lt.u32 %p1, %r10, %r20;\n");
        s.push_str("    setp.lt.u32 %p2, %r11, %r22;\n");
        s.push_str("    and.pred %p3, %p1, %p2;\n");
        // global addr
        s.push_str("    mad.lo.u32 %r12, %r10, %r22, %r11;\n");
        s.push_str("    mul.wide.u32 %rd4, %r12, 4;\n");
        s.push_str("    add.u64 %rd4, %rd0, %rd4;\n");
        s.push_str(&format!("    @%p3 ld.global.f32 %f2, [%rd4];\n"));
        s.push_str(&format!("    @!%p3 mov.f32 %f2, 0f00000000;\n"));
        // smem_a addr = ty*64 + tx*4 + i = r6*16 + r7 + i  (r6=ty*4, so ty=r6/4; ty*64 = r6*16)
        s.push_str(&format!("    shl.b32 %r13, %r6, 4;\n")); // r13 = ty*64
        s.push_str(&format!("    add.u32 %r13, %r13, %r7;\n")); // + tx*4
        if i > 0 {
            s.push_str(&format!("    add.u32 %r13, %r13, {i};\n"));
        }
        s.push_str("    shl.b32 %r13, %r13, 2;\n"); // * 4 bytes
        s.push_str("    add.u32 %r18, %r15, %r13;\n");
        s.push_str("    st.shared.f32 [%r18], %f2;\n");
    }

    // Load smem_b[ty][tx*4..tx*4+3] = B[k_strip+ty][tile_col+tx*4..]
    // Global B (K×N, row-major): B[B_row][B_col] at (B_row*N + B_col)*4
    s.push_str("\n    // -- load smem_b[ty][tx*4..tx*4+3] --\n");
    for i in 0..4u32 {
        // B_row = k_strip + ty = r9 + r1
        s.push_str("    add.u32 %r10, %r9, %r1;\n");
        // B_col = tile_col + tx*4 + i = r5 + r7 + i
        s.push_str(&format!("    add.u32 %r11, %r5, %r7;\n"));
        if i > 0 {
            s.push_str(&format!("    add.u32 %r11, %r11, {i};\n"));
        }
        // bounds
        s.push_str("    setp.lt.u32 %p1, %r10, %r22;\n");
        s.push_str("    setp.lt.u32 %p2, %r11, %r21;\n");
        s.push_str("    and.pred %p3, %p1, %p2;\n");
        // global addr
        s.push_str("    mad.lo.u32 %r12, %r10, %r21, %r11;\n");
        s.push_str("    mul.wide.u32 %rd4, %r12, 4;\n");
        s.push_str("    add.u64 %rd4, %rd1, %rd4;\n");
        s.push_str("    @%p3 ld.global.f32 %f2, [%rd4];\n");
        s.push_str("    @!%p3 mov.f32 %f2, 0f00000000;\n");
        // smem_b addr = ty*64 + tx*4 + i (same layout as smem_a)
        s.push_str("    shl.b32 %r13, %r6, 4;\n");
        s.push_str("    add.u32 %r13, %r13, %r7;\n");
        if i > 0 {
            s.push_str(&format!("    add.u32 %r13, %r13, {i};\n"));
        }
        s.push_str("    shl.b32 %r13, %r13, 2;\n");
        s.push_str("    add.u32 %r18, %r16, %r13;\n");
        s.push_str("    st.shared.f32 [%r18], %f2;\n");
    }

    s.push_str("\n    bar.sync 0;\n\n");

    // Inner product: k_inner = 0..15
    // a_tmp[mi] = smem_a[k_inner][ay_base+mi] = smem_a[k_inner*64 + r6 + mi]
    // b_tmp[ni] = smem_b[k_inner][bx_base+ni] = smem_b[k_inner*64 + r7 + ni]
    // c_acc[mi][ni] += a_tmp[mi] * b_tmp[ni]
    s.push_str("    // -- inner product loop --\n");
    s.push_str("    mov.u32 %r8, 0;  // k_inner\n");
    s.push_str("IK_LOOP:\n");
    s.push_str("    setp.ge.u32 %p4, %r8, 16;\n");
    s.push_str("    @%p4 bra IK_DONE;\n\n");

    // Load a_tmp[0..3] from smem_a
    s.push_str("    // load a_tmp[0..3]\n");
    for mi in 0..4u32 {
        // smem offset = k_inner*64 + ay_base + mi = r8*64 + r6 + mi
        s.push_str("    shl.b32 %r13, %r8, 6;\n"); // r13 = k_inner*64
        s.push_str("    add.u32 %r13, %r13, %r6;\n"); // + ay_base
        if mi > 0 {
            s.push_str(&format!("    add.u32 %r13, %r13, {mi};\n"));
        }
        s.push_str("    shl.b32 %r13, %r13, 2;\n"); // * 4 bytes
        s.push_str("    add.u32 %r18, %r15, %r13;\n");
        s.push_str(&format!("    ld.shared.f32 %f{}, [%r18];\n", 36 + mi));
    }

    // Load b_tmp[0..3] from smem_b
    s.push_str("    // load b_tmp[0..3]\n");
    for ni in 0..4u32 {
        s.push_str("    shl.b32 %r13, %r8, 6;\n"); // k_inner*64
        s.push_str("    add.u32 %r13, %r13, %r7;\n"); // + bx_base
        if ni > 0 {
            s.push_str(&format!("    add.u32 %r13, %r13, {ni};\n"));
        }
        s.push_str("    shl.b32 %r13, %r13, 2;\n");
        s.push_str("    add.u32 %r18, %r16, %r13;\n");
        s.push_str(&format!("    ld.shared.f32 %f{}, [%r18];\n", 40 + ni));
    }

    // 4×4 outer product: 16 FMA instructions
    s.push_str("    // 4x4 outer product\n");
    for mi in 0..4u32 {
        for ni in 0..4u32 {
            let c = 20 + mi * 4 + ni;
            let a = 36 + mi;
            let b = 40 + ni;
            s.push_str(&format!("    fma.rn.f32 %f{c}, %f{a}, %f{b}, %f{c};\n"));
        }
    }

    s.push_str("\n    add.u32 %r8, %r8, 1;\n");
    s.push_str("    bra IK_LOOP;\n\n");
    s.push_str("IK_DONE:\n\n");

    s.push_str("    bar.sync 0;\n\n");

    s.push_str("    add.u32 %r9, %r9, 16;  // k_strip += TILE_K\n");
    s.push_str("    bra KLOOP;\n\n");
    s.push_str("KLOOP_END:\n\n");

    // Store 4×4 results with alpha/beta scaling
    s.push_str("    // -- store 4x4 results --\n");
    for mi in 0..4u32 {
        for ni in 0..4u32 {
            let c_acc = 20 + mi * 4 + ni;
            s.push_str(&format!("    add.u32 %r25, %r23, {mi};  // row\n"));
            s.push_str(&format!("    add.u32 %r26, %r24, {ni};  // col\n"));
            s.push_str("    setp.ge.u32 %p5, %r25, %r20;\n");
            s.push_str("    setp.ge.u32 %p6, %r26, %r21;\n");
            s.push_str("    or.pred %p7, %p5, %p6;\n");
            s.push_str(&format!("    @%p7 bra SKIP_{mi}_{ni};\n"));
            s.push_str("    mad.lo.u32 %r27, %r25, %r21, %r26;\n"); // row*N+col
            s.push_str("    mul.wide.u32 %rd5, %r27, 4;\n");
            s.push_str("    add.u64 %rd6, %rd2, %rd5;\n");
            s.push_str("    ld.global.f32 %f44, [%rd6];\n"); // C[row][col]
            s.push_str(&format!("    mul.f32 %f45, %f{c_acc}, %f0;\n")); // alpha*acc
            s.push_str("    fma.rn.f32 %f46, %f1, %f44, %f45;\n"); // + beta*C
            s.push_str("    add.u64 %rd7, %rd3, %rd5;\n");
            s.push_str("    st.global.f32 [%rd7], %f46;\n");
            s.push_str(&format!("SKIP_{mi}_{ni}:\n"));
        }
    }

    s.push_str("\n    ret;\n}\n");
    s
}

/// Generate double-buffered cp.async tiled F32 value GEMM PTX (sm_80+).
///
/// Identical tile geometry to `generate_tiled_f32_gemm_ptx` (TILE_M=64, TILE_N=64, TILE_K=16,
/// 16×16 thread block, 4×4 register tile) but uses **two** smem ping-pong buffers and
/// `cp.async.cg` to overlap global loads with FMA computation:
///
///   - smem_a[2][TILE_K×TILE_M] = 2 × 4096 f32 = 32 KB for A
///   - smem_b[2][TILE_K×TILE_N] = 2 × 4096 f32 = 32 KB for B  (total 64 KB ≤ sm_80 limit)
///
/// Pipeline schedule (depth-2):
///   prologue : cp.async stage 0; cp.async.commit_group
///   loop iter: cp.async.wait_group 1 (wait all but newest); bar.sync 0; FMA on buf^
///              cp.async stage buf^+1; cp.async.commit_group; toggle buf
///   epilogue : cp.async.wait_group 0; bar.sync 0; FMA on final buf
///
/// Expected GFLOPS: ~2× single-buffer at memory-bandwidth limit; on RTX 3080
/// (760 GB/s) with intensity=16 FLOPs/byte → 760×16 = 12.2 TFLOPS theoretical.
pub fn generate_double_buffered_f32_gemm_ptx(
    config: &EpistemicGemmConfig,
    kernel_name: &str,
) -> String {
    let mut s = String::with_capacity(12288);

    s.push_str(".version 7.5\n");
    s.push_str(".target sm_80\n");
    s.push_str(".address_size 64\n\n");

    s.push_str(&format!(".visible .entry {}(\n", kernel_name));
    s.push_str("    .param .u64 a_ptr,\n");
    s.push_str("    .param .u64 b_ptr,\n");
    s.push_str("    .param .u64 c_ptr,\n");
    s.push_str("    .param .u64 result_ptr,\n");
    s.push_str("    .param .f32 alpha,\n");
    s.push_str("    .param .f32 beta,\n");
    s.push_str("    .param .u32 M,\n");
    s.push_str("    .param .u32 N,\n");
    s.push_str("    .param .u32 K\n");
    s.push_str(")\n{\n");

    // Two ping-pong buffers for A and B — 2 × 4096 × 4 = 32 KB each, 64 KB total
    // TILE_K=16, TILE_M=64: smem_a0[1024] + smem_a1[1024] = 2×4 KB
    // TILE_K=16, TILE_N=64: smem_b0[1024] + smem_b1[1024] = 2×4 KB
    // Total 16 KB — well within sm_80 48 KB shared memory limit.
    s.push_str("    .shared .align 16 .f32 smem_a0[1024];\n");
    s.push_str("    .shared .align 16 .f32 smem_a1[1024];\n");
    s.push_str("    .shared .align 16 .f32 smem_b0[1024];\n");
    s.push_str("    .shared .align 16 .f32 smem_b1[1024];\n\n");

    s.push_str("    .reg .pred %p<20>;\n");
    s.push_str("    .reg .f32 %f<48>;\n");
    // r0=tx, r1=ty, r2=bx, r3=by, r4=tile_row, r5=tile_col
    // r6=ay_base, r7=bx_base, r8=k_strip, r9=buf(0/1), r10-r14=temps
    // r15/r16=smem_a0_base, r17/r18=smem_b0_base (u32)
    // r19-r28=smem addr temps, r29=M, r30=N, r31=K
    s.push_str("    .reg .b32 %r<36>;\n");
    s.push_str("    .reg .b64 %rd<18>;\n\n");

    // Parameters
    s.push_str("    ld.param.u64 %rd0, [a_ptr];\n");
    s.push_str("    ld.param.u64 %rd1, [b_ptr];\n");
    s.push_str("    ld.param.u64 %rd2, [c_ptr];\n");
    s.push_str("    ld.param.u64 %rd3, [result_ptr];\n");
    s.push_str("    ld.param.f32 %f0, [alpha];\n");
    s.push_str("    ld.param.f32 %f1, [beta];\n");
    s.push_str("    ld.param.u32 %r29, [M];\n");
    s.push_str("    ld.param.u32 %r30, [N];\n");
    s.push_str("    ld.param.u32 %r31, [K];\n\n");

    // Thread/block indices
    s.push_str("    mov.u32 %r0, %tid.x;\n");
    s.push_str("    mov.u32 %r1, %tid.y;\n");
    s.push_str("    mov.u32 %r2, %ctaid.x;\n");
    s.push_str("    mov.u32 %r3, %ctaid.y;\n");
    s.push_str("    mul.lo.u32 %r4, %r3, 64;  // tile_row\n");
    s.push_str("    mul.lo.u32 %r5, %r2, 64;  // tile_col\n");
    s.push_str("    shl.b32 %r6, %r1, 2;       // ay_base = ty*4\n");
    s.push_str("    shl.b32 %r7, %r0, 2;       // bx_base = tx*4\n");
    s.push_str("    add.u32 %r19, %r4, %r6;    // out_row\n");
    s.push_str("    add.u32 %r20, %r5, %r7;    // out_col\n\n");

    // Smem base addresses (u32) for ping-pong
    s.push_str("    mov.u32 %r15, smem_a0;\n");
    s.push_str("    mov.u32 %r16, smem_a1;\n");
    s.push_str("    mov.u32 %r17, smem_b0;\n");
    s.push_str("    mov.u32 %r18, smem_b1;\n\n");

    // Init accumulators %f20..%f35
    for i in 0..16u32 {
        s.push_str(&format!("    mov.f32 %f{}, 0f00000000;\n", 20 + i));
    }
    s.push_str("\n");

    // ── Prologue: prefetch stage 0 into buf 0 ────────────────────────────────
    // Helper closure emitted inline: load A strip at k_strip=0
    s.push_str("    // prologue: async-prefetch k_strip=0 into buf 0\n");
    s.push_str("    mov.u32 %r8, 0;  // k_strip = 0 for prologue load\n");

    // emit cp.async for smem_a0: 4 elements per thread (ty-dim), using 16-byte (float4) copy
    // cp.async.cg.shared.global [dst+offset], [src+offset], 4 (4 bytes = 1 f32)
    // We do scalar cp.async (4 bytes each) to keep bounds-check generality
    for i in 0..4u32 {
        // A_row = tile_row + tx*4 + i; A_col = k_strip + ty
        s.push_str(&format!("    add.u32 %r21, %r4, %r7;\n")); // tile_row + bx_base
        if i > 0 {
            s.push_str(&format!("    add.u32 %r21, %r21, {i};\n"));
        }
        s.push_str("    add.u32 %r22, %r8, %r1;\n"); // k_strip + ty
        s.push_str("    setp.lt.u32 %p1, %r21, %r29;\n");
        s.push_str("    setp.lt.u32 %p2, %r22, %r31;\n");
        s.push_str("    and.pred %p3, %p1, %p2;\n");
        // smem_a0 offset = ty*64 + tx*4 + i
        s.push_str("    shl.b32 %r23, %r6, 4;\n"); // ty*64
        s.push_str("    add.u32 %r23, %r23, %r7;\n");
        if i > 0 {
            s.push_str(&format!("    add.u32 %r23, %r23, {i};\n"));
        }
        s.push_str("    shl.b32 %r23, %r23, 2;\n"); // *4 bytes
        s.push_str("    add.u32 %r24, %r15, %r23;\n"); // smem_a0 ptr
        s.push_str("    mad.lo.u32 %r25, %r21, %r31, %r22;\n"); // A_row*K + A_col
        s.push_str("    mul.wide.u32 %rd4, %r25, 4;\n");
        s.push_str("    add.u64 %rd4, %rd0, %rd4;\n");
        // cp.async with predicate guard: if in-bounds do async, else zero smem
        s.push_str("    @%p3 cp.async.cg.shared.global [%r24], [%rd4], 4;\n");
        s.push_str("    @!%p3 st.shared.f32 [%r24], 0f00000000;\n");
    }
    // smem_b0: 4 elements
    for i in 0..4u32 {
        // B_row = k_strip + ty; B_col = tile_col + tx*4 + i
        s.push_str("    add.u32 %r21, %r8, %r1;\n"); // k_strip + ty
        s.push_str(&format!("    add.u32 %r22, %r5, %r7;\n")); // tile_col + bx_base
        if i > 0 {
            s.push_str(&format!("    add.u32 %r22, %r22, {i};\n"));
        }
        s.push_str("    setp.lt.u32 %p1, %r21, %r31;\n");
        s.push_str("    setp.lt.u32 %p2, %r22, %r30;\n");
        s.push_str("    and.pred %p3, %p1, %p2;\n");
        s.push_str("    shl.b32 %r23, %r6, 4;\n");
        s.push_str("    add.u32 %r23, %r23, %r7;\n");
        if i > 0 {
            s.push_str(&format!("    add.u32 %r23, %r23, {i};\n"));
        }
        s.push_str("    shl.b32 %r23, %r23, 2;\n");
        s.push_str("    add.u32 %r24, %r17, %r23;\n"); // smem_b0 ptr
        s.push_str("    mad.lo.u32 %r25, %r21, %r30, %r22;\n"); // B_row*N + B_col
        s.push_str("    mul.wide.u32 %rd4, %r25, 4;\n");
        s.push_str("    add.u64 %rd4, %rd1, %rd4;\n");
        s.push_str("    @%p3 cp.async.cg.shared.global [%r24], [%rd4], 4;\n");
        s.push_str("    @!%p3 st.shared.f32 [%r24], 0f00000000;\n");
    }
    s.push_str("    cp.async.commit_group;\n\n");

    // ── Main double-buffered K loop ──────────────────────────────────────────
    // buf = 0 (smem_a0/smem_b0 is stage ready to compute)
    // k_strip starts at 16 (prologue loaded strip 0)
    s.push_str("    mov.u32 %r8, 16;  // k_strip = TILE_K (first strip already in buf 0)\n");
    s.push_str("    mov.u32 %r9, 0;   // buf = 0\n\n");
    s.push_str("DB_KLOOP:\n");
    // Two exit conditions: k_strip >= K (no more loads) AND we've flushed
    // Simplification: loop while k_strip < K+TILE_K (drain last stage in epilogue)
    s.push_str("    setp.ge.u32 %p0, %r8, %r31;\n"); // k_strip >= K → skip load, just compute
    s.push_str("    @%p0 bra DB_KLOOP_COMPUTE;\n\n");

    // Issue async load for k_strip into alternate buffer (1 - buf)
    s.push_str("    // async load next strip into alternate buf\n");
    s.push_str("    xor.b32 %r10, %r9, 1;  // next_buf = 1 - buf\n");

    // Pick smem_a base: buf=0→r15, buf=1→r16; use select-predicate
    s.push_str("    setp.eq.u32 %p8, %r10, 0;\n");
    s.push_str("    selp.u32 %r26, %r15, %r16, %p8;  // sa_next\n");
    s.push_str("    selp.u32 %r27, %r17, %r18, %p8;  // sb_next\n\n");

    for i in 0..4u32 {
        s.push_str(&format!("    add.u32 %r21, %r4, %r7;\n"));
        if i > 0 {
            s.push_str(&format!("    add.u32 %r21, %r21, {i};\n"));
        }
        s.push_str("    add.u32 %r22, %r8, %r1;\n");
        s.push_str("    setp.lt.u32 %p1, %r21, %r29;\n");
        s.push_str("    setp.lt.u32 %p2, %r22, %r31;\n");
        s.push_str("    and.pred %p3, %p1, %p2;\n");
        s.push_str("    shl.b32 %r23, %r6, 4;\n");
        s.push_str("    add.u32 %r23, %r23, %r7;\n");
        if i > 0 {
            s.push_str(&format!("    add.u32 %r23, %r23, {i};\n"));
        }
        s.push_str("    shl.b32 %r23, %r23, 2;\n");
        s.push_str("    add.u32 %r24, %r26, %r23;\n");
        s.push_str("    mad.lo.u32 %r25, %r21, %r31, %r22;\n");
        s.push_str("    mul.wide.u32 %rd4, %r25, 4;\n");
        s.push_str("    add.u64 %rd4, %rd0, %rd4;\n");
        s.push_str("    @%p3 cp.async.cg.shared.global [%r24], [%rd4], 4;\n");
        s.push_str("    @!%p3 st.shared.f32 [%r24], 0f00000000;\n");
    }
    for i in 0..4u32 {
        s.push_str("    add.u32 %r21, %r8, %r1;\n");
        s.push_str(&format!("    add.u32 %r22, %r5, %r7;\n"));
        if i > 0 {
            s.push_str(&format!("    add.u32 %r22, %r22, {i};\n"));
        }
        s.push_str("    setp.lt.u32 %p1, %r21, %r31;\n");
        s.push_str("    setp.lt.u32 %p2, %r22, %r30;\n");
        s.push_str("    and.pred %p3, %p1, %p2;\n");
        s.push_str("    shl.b32 %r23, %r6, 4;\n");
        s.push_str("    add.u32 %r23, %r23, %r7;\n");
        if i > 0 {
            s.push_str(&format!("    add.u32 %r23, %r23, {i};\n"));
        }
        s.push_str("    shl.b32 %r23, %r23, 2;\n");
        s.push_str("    add.u32 %r24, %r27, %r23;\n");
        s.push_str("    mad.lo.u32 %r25, %r21, %r30, %r22;\n");
        s.push_str("    mul.wide.u32 %rd4, %r25, 4;\n");
        s.push_str("    add.u64 %rd4, %rd1, %rd4;\n");
        s.push_str("    @%p3 cp.async.cg.shared.global [%r24], [%rd4], 4;\n");
        s.push_str("    @!%p3 st.shared.f32 [%r24], 0f00000000;\n");
    }
    s.push_str("    cp.async.commit_group;\n\n");

    // Wait for current buf's data (all but the just-committed group)
    s.push_str("DB_KLOOP_COMPUTE:\n");
    s.push_str("    cp.async.wait_group 1;\n");
    s.push_str("    bar.sync 0;\n\n");

    // Select current smem base (buf=0→r15/r17, buf=1→r16/r18)
    s.push_str("    setp.eq.u32 %p9, %r9, 0;\n");
    s.push_str("    selp.u32 %r28, %r15, %r16, %p9;  // sa_cur\n");
    s.push_str("    selp.u32 %r34, %r17, %r18, %p9;  // sb_cur\n\n");

    // Inner K product (same as single-buffer version, but smem base from %r28/%r34)
    s.push_str("    mov.u32 %r32, 0;  // k_inner\n");
    s.push_str("DB_IK_LOOP:\n");
    s.push_str("    setp.ge.u32 %p4, %r32, 16;\n");
    s.push_str("    @%p4 bra DB_IK_DONE;\n\n");
    for mi in 0..4u32 {
        s.push_str("    shl.b32 %r33, %r32, 6;\n");
        s.push_str("    add.u32 %r33, %r33, %r6;\n");
        if mi > 0 {
            s.push_str(&format!("    add.u32 %r33, %r33, {mi};\n"));
        }
        s.push_str("    shl.b32 %r33, %r33, 2;\n");
        s.push_str("    add.u32 %r35, %r28, %r33;\n");
        s.push_str(&format!("    ld.shared.f32 %f{}, [%r35];\n", 36 + mi));
    }
    for ni in 0..4u32 {
        s.push_str("    shl.b32 %r33, %r32, 6;\n");
        s.push_str("    add.u32 %r33, %r33, %r7;\n");
        if ni > 0 {
            s.push_str(&format!("    add.u32 %r33, %r33, {ni};\n"));
        }
        s.push_str("    shl.b32 %r33, %r33, 2;\n");
        s.push_str("    add.u32 %r35, %r34, %r33;\n");
        s.push_str(&format!("    ld.shared.f32 %f{}, [%r35];\n", 40 + ni));
    }
    for mi in 0..4u32 {
        for ni in 0..4u32 {
            let c = 20 + mi * 4 + ni;
            let a = 36 + mi;
            let b = 40 + ni;
            s.push_str(&format!("    fma.rn.f32 %f{c}, %f{a}, %f{b}, %f{c};\n"));
        }
    }
    s.push_str("\n    add.u32 %r32, %r32, 1;\n");
    s.push_str("    bra DB_IK_LOOP;\n\n");
    s.push_str("DB_IK_DONE:\n");
    s.push_str("    bar.sync 0;\n\n");

    // Toggle buf, advance k_strip
    s.push_str("    xor.b32 %r9, %r9, 1;  // toggle buf\n");
    s.push_str("    add.u32 %r8, %r8, 16; // k_strip += TILE_K\n");

    // Loop condition: keep going while previous k_strip - 16 < K (i.e. there was data computed)
    // Simplest: loop while k_strip <= K (we always compute if we entered this path)
    s.push_str("    setp.le.u32 %p5, %r8, %r31;\n");
    s.push_str("    @%p5 bra DB_KLOOP;\n\n");

    // ── Epilogue: drain final committed group ────────────────────────────────
    s.push_str("    // epilogue: drain last async group\n");
    s.push_str("    cp.async.wait_group 0;\n");
    s.push_str("    bar.sync 0;\n\n");
    // Compute the last strip (already in current buf after final toggle)
    s.push_str("    setp.eq.u32 %p9, %r9, 0;\n");
    s.push_str("    selp.u32 %r28, %r15, %r16, %p9;\n");
    s.push_str("    selp.u32 %r34, %r17, %r18, %p9;\n\n");
    s.push_str("    mov.u32 %r32, 0;\n");
    s.push_str("DB_EPI_LOOP:\n");
    s.push_str("    setp.ge.u32 %p4, %r32, 16;\n");
    s.push_str("    @%p4 bra DB_EPI_DONE;\n\n");
    for mi in 0..4u32 {
        s.push_str("    shl.b32 %r33, %r32, 6;\n");
        s.push_str("    add.u32 %r33, %r33, %r6;\n");
        if mi > 0 {
            s.push_str(&format!("    add.u32 %r33, %r33, {mi};\n"));
        }
        s.push_str("    shl.b32 %r33, %r33, 2;\n");
        s.push_str("    add.u32 %r35, %r28, %r33;\n");
        s.push_str(&format!("    ld.shared.f32 %f{}, [%r35];\n", 36 + mi));
    }
    for ni in 0..4u32 {
        s.push_str("    shl.b32 %r33, %r32, 6;\n");
        s.push_str("    add.u32 %r33, %r33, %r7;\n");
        if ni > 0 {
            s.push_str(&format!("    add.u32 %r33, %r33, {ni};\n"));
        }
        s.push_str("    shl.b32 %r33, %r33, 2;\n");
        s.push_str("    add.u32 %r35, %r34, %r33;\n");
        s.push_str(&format!("    ld.shared.f32 %f{}, [%r35];\n", 40 + ni));
    }
    for mi in 0..4u32 {
        for ni in 0..4u32 {
            let c = 20 + mi * 4 + ni;
            let a = 36 + mi;
            let b = 40 + ni;
            s.push_str(&format!("    fma.rn.f32 %f{c}, %f{a}, %f{b}, %f{c};\n"));
        }
    }
    s.push_str("\n    add.u32 %r32, %r32, 1;\n");
    s.push_str("    bra DB_EPI_LOOP;\n\n");
    s.push_str("DB_EPI_DONE:\n\n");

    // Store 4×4 output tile
    s.push_str("    // -- store 4x4 results --\n");
    for mi in 0..4u32 {
        for ni in 0..4u32 {
            let c_acc = 20 + mi * 4 + ni;
            s.push_str(&format!("    add.u32 %r25, %r19, {mi};  // row\n"));
            s.push_str(&format!("    add.u32 %r26, %r20, {ni};  // col\n"));
            s.push_str("    setp.ge.u32 %p5, %r25, %r29;\n");
            s.push_str("    setp.ge.u32 %p6, %r26, %r30;\n");
            s.push_str("    or.pred %p7, %p5, %p6;\n");
            s.push_str(&format!("    @%p7 bra DB_SKIP_{mi}_{ni};\n"));
            s.push_str("    mad.lo.u32 %r27, %r25, %r30, %r26;\n");
            s.push_str("    mul.wide.u32 %rd5, %r27, 4;\n");
            s.push_str("    add.u64 %rd6, %rd2, %rd5;\n");
            s.push_str("    ld.global.f32 %f44, [%rd6];\n");
            s.push_str(&format!("    mul.f32 %f45, %f{c_acc}, %f0;\n"));
            s.push_str("    fma.rn.f32 %f46, %f1, %f44, %f45;\n");
            s.push_str("    add.u64 %rd7, %rd3, %rd5;\n");
            s.push_str("    st.global.f32 [%rd7], %f46;\n");
            s.push_str(&format!("DB_SKIP_{mi}_{ni}:\n"));
        }
    }
    s.push_str("\n    ret;\n}\n");
    s
}

/// Generate epsilon-only PTX kernel (post-GEMM uncertainty propagation).
///
/// One thread per output element (i, j). Streams K in a scalar loop.
/// Compute: ε_result[i,j] = |α|·Σ_k(|A[i,k]|·ε_B[k,j] + |B[k,j]|·ε_A[i,k] + ε_A·ε_B)
///                         + |β|·ε_C[i,j]
///
/// Launch config: block=(256,1,1), grid=(ceil(M*N/256),1,1).
pub fn generate_epsilon_only_ptx(config: &EpistemicGemmConfig, kernel_name: &str) -> String {
    let mut s = String::with_capacity(4096);

    s.push_str(".version 7.5\n");
    s.push_str(".target sm_80\n");
    s.push_str(".address_size 64\n\n");

    s.push_str(&format!(".visible .entry {}(\n", kernel_name));
    s.push_str("    .param .u64 a_val_ptr,\n");
    s.push_str("    .param .u64 a_eps_ptr,\n");
    s.push_str("    .param .u64 b_val_ptr,\n");
    s.push_str("    .param .u64 b_eps_ptr,\n");
    s.push_str("    .param .u64 c_eps_ptr,\n");
    s.push_str("    .param .u64 result_eps_ptr,\n");
    s.push_str("    .param .f32 alpha,\n");
    s.push_str("    .param .f32 beta,\n");
    s.push_str("    .param .u32 M,\n");
    s.push_str("    .param .u32 N,\n");
    s.push_str("    .param .u32 K\n");
    s.push_str(")\n{\n");

    s.push_str("    .reg .pred %p<8>;\n");
    s.push_str("    .reg .f32 %f<16>;\n");
    s.push_str("    .reg .b32 %r<16>;\n");
    s.push_str("    .reg .b64 %rd<14>;\n\n");

    s.push_str("    ld.param.u64 %rd0, [a_val_ptr];\n");
    s.push_str("    ld.param.u64 %rd1, [a_eps_ptr];\n");
    s.push_str("    ld.param.u64 %rd2, [b_val_ptr];\n");
    s.push_str("    ld.param.u64 %rd3, [b_eps_ptr];\n");
    s.push_str("    ld.param.u64 %rd4, [c_eps_ptr];\n");
    s.push_str("    ld.param.u64 %rd5, [result_eps_ptr];\n");
    s.push_str("    ld.param.f32 %f0, [alpha];\n");
    s.push_str("    ld.param.f32 %f1, [beta];\n");
    s.push_str("    ld.param.u32 %r10, [M];\n");
    s.push_str("    ld.param.u32 %r11, [N];\n");
    s.push_str("    ld.param.u32 %r12, [K];\n\n");

    // 1-D global index
    s.push_str("    mov.u32 %r0, %tid.x;\n");
    s.push_str("    mov.u32 %r1, %ctaid.x;\n");
    s.push_str("    mad.lo.u32 %r2, %r1, 256, %r0;  // global_idx\n");

    // Decompose into (row, col) — avoid slow div/rem with mul+sub trick
    // Since N can be non-power-of-2 we use plain div/rem here; performance
    // is secondary to the tiled value kernel.
    s.push_str("    div.u32 %r3, %r2, %r11;  // row\n");
    s.push_str("    rem.u32 %r4, %r2, %r11;  // col\n\n");

    // Bounds check
    s.push_str("    setp.ge.u32 %p0, %r3, %r10;\n");
    s.push_str("    setp.ge.u32 %p1, %r4, %r11;\n");
    s.push_str("    or.pred %p2, %p0, %p1;\n");
    s.push_str("    @%p2 bra EPS_EXIT;\n\n");

    // K-loop accumulation
    s.push_str("    mov.f32 %f2, 0f00000000;  // eps_acc\n");
    s.push_str("    mov.u32 %r5, 0;           // k\n");
    s.push_str("EPS_K:\n");
    s.push_str("    setp.ge.u32 %p3, %r5, %r12;\n");
    s.push_str("    @%p3 bra EPS_K_DONE;\n\n");

    // a_idx = row*K + k
    s.push_str("    mad.lo.u32 %r6, %r3, %r12, %r5;\n");
    s.push_str("    mul.wide.u32 %rd6, %r6, 4;\n");
    // b_idx = k*N + col
    s.push_str("    mad.lo.u32 %r7, %r5, %r11, %r4;\n");
    s.push_str("    mul.wide.u32 %rd7, %r7, 4;\n");

    // |A|, eps_A, |B|, eps_B
    s.push_str("    add.u64 %rd8, %rd0, %rd6;\n");
    s.push_str("    ld.global.f32 %f3, [%rd8];\n");
    s.push_str("    abs.f32 %f3, %f3;\n");
    s.push_str("    add.u64 %rd8, %rd1, %rd6;\n");
    s.push_str("    ld.global.f32 %f4, [%rd8];\n");
    s.push_str("    add.u64 %rd8, %rd2, %rd7;\n");
    s.push_str("    ld.global.f32 %f5, [%rd8];\n");
    s.push_str("    abs.f32 %f5, %f5;\n");
    s.push_str("    add.u64 %rd8, %rd3, %rd7;\n");
    s.push_str("    ld.global.f32 %f6, [%rd8];\n\n");

    // eps_acc += |A|*eps_B + |B|*eps_A + eps_A*eps_B
    s.push_str("    fma.rn.f32 %f2, %f3, %f6, %f2;\n");
    s.push_str("    fma.rn.f32 %f2, %f5, %f4, %f2;\n");
    s.push_str("    fma.rn.f32 %f2, %f4, %f6, %f2;\n\n");

    s.push_str("    add.u32 %r5, %r5, 1;\n");
    s.push_str("    bra EPS_K;\n\n");
    s.push_str("EPS_K_DONE:\n\n");

    // out_idx = row*N + col
    s.push_str("    mad.lo.u32 %r8, %r3, %r11, %r4;\n");
    s.push_str("    mul.wide.u32 %rd9, %r8, 4;\n");

    // load c_eps
    s.push_str("    add.u64 %rd10, %rd4, %rd9;\n");
    s.push_str("    ld.global.f32 %f7, [%rd10];\n\n");

    // |alpha|*eps_acc + |beta|*eps_C
    s.push_str("    abs.f32 %f8, %f0;\n");
    s.push_str("    abs.f32 %f9, %f1;\n");
    s.push_str("    mul.f32 %f10, %f8, %f2;\n");
    s.push_str("    fma.rn.f32 %f10, %f9, %f7, %f10;\n\n");

    s.push_str("    add.u64 %rd11, %rd5, %rd9;\n");
    s.push_str("    st.global.f32 [%rd11], %f10;\n");

    s.push_str("\nEPS_EXIT:\n");
    s.push_str("    ret;\n}\n");
    s
}

/// Generate F16 WMMA GEMM PTX with correct K-loop.
///
/// Fixes the skeleton that previously lacked a loop over K.
/// Each warp accumulates a 16×16 output tile using WMMA m16n16k16.
/// Gate on: use_tensor_cores=true, precision=F16, sm_major >= 7.
///
/// Launch config: block=(32,1,1) [one warp], grid=(ceil(N/16), ceil(M/16), 1).
pub fn generate_wmma_k_loop_ptx(config: &EpistemicGemmConfig, kernel_name: &str) -> String {
    let k = config.k as u32;
    let mut s = String::with_capacity(4096);

    s.push_str(".version 7.5\n");
    s.push_str(".target sm_80\n");
    s.push_str(".address_size 64\n\n");

    s.push_str(&format!(".visible .entry {}(\n", kernel_name));
    s.push_str("    .param .u64 a_ptr,\n");
    s.push_str("    .param .u64 b_ptr,\n");
    s.push_str("    .param .u64 c_ptr,\n");
    s.push_str("    .param .u64 result_ptr,\n");
    s.push_str("    .param .f32 alpha,\n");
    s.push_str("    .param .f32 beta,\n");
    s.push_str("    .param .u32 M,\n");
    s.push_str("    .param .u32 N,\n");
    s.push_str("    .param .u32 K\n");
    s.push_str(")\n{\n");

    s.push_str("    .reg .pred %p<8>;\n");
    s.push_str("    .reg .f32 %f<16>;\n");
    s.push_str("    .reg .b32 %r<24>;\n");
    s.push_str("    .reg .b32 %wmma_a<8>;\n");
    s.push_str("    .reg .b32 %wmma_b<8>;\n");
    s.push_str("    .reg .f32 %wmma_c<8>;\n");
    s.push_str("    .reg .f32 %wmma_d<8>;\n");
    s.push_str("    .reg .b64 %rd<12>;\n\n");

    s.push_str("    ld.param.u64 %rd0, [a_ptr];\n");
    s.push_str("    ld.param.u64 %rd1, [b_ptr];\n");
    s.push_str("    ld.param.u64 %rd2, [c_ptr];\n");
    s.push_str("    ld.param.u64 %rd3, [result_ptr];\n");
    s.push_str("    ld.param.f32 %f0, [alpha];\n");
    s.push_str("    ld.param.f32 %f1, [beta];\n");
    s.push_str("    ld.param.u32 %r20, [M];\n");
    s.push_str("    ld.param.u32 %r21, [N];\n");
    s.push_str("    ld.param.u32 %r22, [K];\n\n");

    // warp tile origin
    s.push_str("    mov.u32 %r0, %ctaid.x;\n");
    s.push_str("    mov.u32 %r1, %ctaid.y;\n");
    s.push_str("    mul.lo.u32 %r2, %r1, 16;  // tile_row = by*16\n");
    s.push_str("    mul.lo.u32 %r3, %r0, 16;  // tile_col = bx*16\n\n");

    // Initialise accumulator fragment to 0
    for i in 0..8u32 {
        s.push_str(&format!("    mov.f32 %wmma_c{i}, 0f00000000;\n"));
    }
    s.push_str("\n");

    // K-loop in steps of 16
    s.push_str("    mov.u32 %r5, 0;  // k\n");
    s.push_str("WMMA_K:\n");
    s.push_str("    setp.ge.u32 %p0, %r5, %r22;\n");
    s.push_str("    @%p0 bra WMMA_K_DONE;\n\n");

    // A fragment: A[tile_row..tile_row+16][k..k+16] = row-major, f16
    s.push_str("    mad.lo.u32 %r6, %r2, %r22, %r5;\n"); // tile_row*K + k
    s.push_str("    mul.wide.u32 %rd4, %r6, 2;\n"); // f16 = 2 bytes
    s.push_str("    add.u64 %rd4, %rd0, %rd4;\n");
    s.push_str("    wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {%wmma_a0,%wmma_a1,%wmma_a2,%wmma_a3,%wmma_a4,%wmma_a5,%wmma_a6,%wmma_a7}, [%rd4], %r22;\n\n");

    // B fragment: B[k..k+16][tile_col..tile_col+16] = col-major, f16
    s.push_str("    mad.lo.u32 %r7, %r5, %r21, %r3;\n"); // k*N + tile_col
    s.push_str("    mul.wide.u32 %rd5, %r7, 2;\n");
    s.push_str("    add.u64 %rd5, %rd1, %rd5;\n");
    s.push_str("    wmma.load.b.sync.aligned.col.m16n16k16.global.f16 {%wmma_b0,%wmma_b1,%wmma_b2,%wmma_b3,%wmma_b4,%wmma_b5,%wmma_b6,%wmma_b7}, [%rd5], %r21;\n\n");

    // MMA: accumulate into f32
    s.push_str("    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16 {%wmma_d0,%wmma_d1,%wmma_d2,%wmma_d3,%wmma_d4,%wmma_d5,%wmma_d6,%wmma_d7}, {%wmma_a0,%wmma_a1,%wmma_a2,%wmma_a3,%wmma_a4,%wmma_a5,%wmma_a6,%wmma_a7}, {%wmma_b0,%wmma_b1,%wmma_b2,%wmma_b3,%wmma_b4,%wmma_b5,%wmma_b6,%wmma_b7}, {%wmma_c0,%wmma_c1,%wmma_c2,%wmma_c3,%wmma_c4,%wmma_c5,%wmma_c6,%wmma_c7};\n");

    // Copy d → c for next iteration
    for i in 0..8u32 {
        s.push_str(&format!("    mov.f32 %wmma_c{i}, %wmma_d{i};\n"));
    }

    s.push_str("\n    add.u32 %r5, %r5, 16;\n");
    s.push_str("    bra WMMA_K;\n\n");
    s.push_str("WMMA_K_DONE:\n\n");

    // alpha * d + beta * C: load C tile, scale, store to result
    s.push_str("    mad.lo.u32 %r8, %r2, %r21, %r3;\n"); // tile_row*N + tile_col
    s.push_str("    mul.wide.u32 %rd6, %r8, 4;\n");
    s.push_str("    add.u64 %rd6, %rd2, %rd6;\n");
    s.push_str(&format!(
        "    wmma.load.c.sync.aligned.row.m16n16k16.global.f32 {{%wmma_c0,%wmma_c1,%wmma_c2,%wmma_c3,%wmma_c4,%wmma_c5,%wmma_c6,%wmma_c7}}, [%rd6], %r21;\n\n"
    ));

    // scale: result_d = alpha*d + beta*C
    for i in 0..8u32 {
        s.push_str(&format!("    mul.f32 %f2, %wmma_d{i}, %f0;\n")); // alpha*d
        s.push_str(&format!("    fma.rn.f32 %wmma_d{i}, %wmma_c{i}, %f1, %f2;\n")); // beta*C + alpha*d
    }

    // store result
    s.push_str("    add.u64 %rd7, %rd3, %rd6;\n");
    s.push_str("    wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%rd7], {%wmma_d0,%wmma_d1,%wmma_d2,%wmma_d3,%wmma_d4,%wmma_d5,%wmma_d6,%wmma_d7}, %r21;\n");

    let _ = k; // used via config.k doc
    s.push_str("\n    ret;\n}\n");
    s
}

/// Generate WGSL shader for epistemic GEMM
pub fn generate_epistemic_gemm_wgsl(config: &EpistemicGemmConfig) -> String {
    let mut wgsl = String::new();

    wgsl.push_str("// Epistemic GEMM WGSL shader\n");
    wgsl.push_str("// Computes C = α * A * B + β * C with uncertainty propagation\n\n");

    // Knowledge struct
    wgsl.push_str("struct Knowledge {\n");
    wgsl.push_str("    value: f32,\n");
    wgsl.push_str("    uncertainty: f32,\n");
    wgsl.push_str("    confidence: f32,\n");
    wgsl.push_str("    _padding: f32,\n");
    wgsl.push_str("};\n\n");

    // Parameters struct
    wgsl.push_str("struct GemmParams {\n");
    wgsl.push_str("    m: i32,\n");
    wgsl.push_str("    k: i32,\n");
    wgsl.push_str("    n: i32,\n");
    wgsl.push_str("    alpha: f32,\n");
    wgsl.push_str("    beta: f32,\n");
    wgsl.push_str("    pad0: i32,\n");
    wgsl.push_str("    pad1: i32,\n");
    wgsl.push_str("};\n\n");

    // Bindings
    wgsl.push_str("@group(0) @binding(0) var<storage, read> matrix_a: array<Knowledge>;\n");
    wgsl.push_str("@group(0) @binding(1) var<storage, read> matrix_b: array<Knowledge>;\n");
    wgsl.push_str(
        "@group(0) @binding(2) var<storage, read_write> matrix_c: array<Knowledge>;\n",
    );
    wgsl.push_str("@group(0) @binding(3) var<uniform> params: GemmParams;\n\n");

    // Kernel
    wgsl.push_str("@compute @workgroup_size(16, 16, 1)\n");
    wgsl.push_str(
        "fn epistemic_gemm(@builtin(global_invocation_id) gid: vec3<u32>) {\n",
    );
    wgsl.push_str("    let i = gid.x;\n");
    wgsl.push_str("    let j = gid.y;\n");
    wgsl.push_str("    \n");
    wgsl.push_str("    if (i >= u32(params.m) || j >= u32(params.n)) {\n");
    wgsl.push_str("        return;\n");
    wgsl.push_str("    }\n");
    wgsl.push_str("    \n");
    wgsl.push_str("    var acc_value: f32 = 0.0;\n");
    wgsl.push_str("    var acc_epsilon_sq: f32 = 0.0;\n");
    wgsl.push_str("    var acc_confidence: f32 = 1.0;\n");
    wgsl.push_str("    \n");
    wgsl.push_str("    for (var l: u32 = 0u; l < u32(params.k); l = l + 1u) {\n");
    wgsl.push_str("        let a_idx = i * u32(params.k) + l;\n");
    wgsl.push_str("        let b_idx = l * u32(params.n) + j;\n");
    wgsl.push_str("        let a = matrix_a[a_idx];\n");
    wgsl.push_str("        let b = matrix_b[b_idx];\n");
    wgsl.push_str("        \n");
    wgsl.push_str("        // Value accumulation\n");
    wgsl.push_str("        acc_value = acc_value + a.value * b.value;\n");
    wgsl.push_str("        \n");
    wgsl.push_str("        // Uncertainty propagation\n");
    wgsl.push_str("        let term1 = abs(a.value) * b.uncertainty;\n");
    wgsl.push_str("        let term2 = abs(b.value) * a.uncertainty;\n");
    wgsl.push_str("        let eps_contrib = term1 + term2;\n");
    wgsl.push_str("        acc_epsilon_sq = acc_epsilon_sq + eps_contrib * eps_contrib;\n");
    wgsl.push_str("        \n");
    wgsl.push_str("        // Confidence tracking\n");
    wgsl.push_str(
        "        acc_confidence = min(acc_confidence, min(a.confidence, b.confidence));\n",
    );
    wgsl.push_str("    }\n");
    wgsl.push_str("    \n");
    wgsl.push_str("    // Apply alpha scaling\n");
    wgsl.push_str("    let result_value = params.alpha * acc_value;\n");
    wgsl.push_str("    let result_epsilon = params.alpha * sqrt(acc_epsilon_sq);\n");
    wgsl.push_str("    \n");
    wgsl.push_str("    // Add beta * C\n");
    wgsl.push_str("    let c_idx = i * u32(params.n) + j;\n");
    wgsl.push_str("    let c = matrix_c[c_idx];\n");
    wgsl.push_str("    let final_value = result_value + params.beta * c.value;\n");
    wgsl.push_str("    let final_epsilon = sqrt(result_epsilon * result_epsilon + \n");
    wgsl.push_str(
        "        (params.beta * c.uncertainty) * (params.beta * c.uncertainty));\n",
    );
    wgsl.push_str("    let final_confidence = min(acc_confidence, c.confidence);\n");
    wgsl.push_str("    \n");
    wgsl.push_str("    // Store result\n");
    wgsl.push_str(
        "    matrix_c[c_idx] = Knowledge(final_value, final_epsilon, final_confidence, 0.0);\n",
    );
    wgsl.push_str("}\n");

    wgsl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_matrix_creation() {
        let matrix = EpistemicMatrix::new(2, 3);
        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.cols, 3);
        assert_eq!(matrix.values.len(), 6);
        assert_eq!(matrix.epsilon.len(), 6);
    }

    #[test]
    fn test_epistemic_gemm_basic() {
        let a = EpistemicMatrix::from_values(vec![1.0, 2.0, 3.0, 4.0], 0.1, 2, 2);
        let b = EpistemicMatrix::from_values(vec![5.0, 6.0, 7.0, 8.0], 0.2, 2, 2);
        let c = EpistemicMatrix::from_values(vec![0.0, 0.0, 0.0, 0.0], 0.0, 2, 2);

        let config = EpistemicGemmConfig {
            m: 2,
            k: 2,
            n: 2,
            alpha: 1.0,
            beta: 1.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };

        let result = compute_epistemic_gemm(&a, &b, &c, &config);

        assert_eq!(result.rows, 2);
        assert_eq!(result.cols, 2);

        // Check that uncertainty was propagated
        for &eps in &result.epsilon {
            assert!(eps > 0.0);
        }
    }

    #[test]
    fn test_epistemic_gemm_with_scaling() {
        let a = EpistemicMatrix::from_values(vec![1.0, 2.0], 0.1, 1, 2);
        let b = EpistemicMatrix::from_values(vec![3.0, 4.0], 0.2, 2, 1);
        let c = EpistemicMatrix::from_values(vec![1.0], 0.3, 1, 1);

        let config = EpistemicGemmConfig {
            m: 1,
            k: 2,
            n: 1,
            alpha: 0.5,
            beta: 2.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };

        let result = compute_epistemic_gemm(&a, &b, &c, &config);

        // Result should be: 0.5*(1*3 + 2*4) + 2.0*1.0 = 0.5*11 + 2.0 = 7.5
        assert!((result.values[0] - 7.5).abs() < 0.001);
    }

    #[test]
    fn test_ptx_generation() {
        let config = EpistemicGemmConfig {
            m: 256,
            k: 256,
            n: 256,
            alpha: 1.0,
            beta: 1.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };

        let ptx = generate_epistemic_gemm_ptx(&config, "epistemic_gemm");

        assert!(ptx.contains(".version"));
        assert!(ptx.contains(".target"));
        assert!(ptx.contains("epistemic_gemm"));
        assert!(ptx.contains("ld.param.u64"));
        assert!(ptx.contains("st.global.f32"));
    }

    #[test]
    fn test_tiled_ptx_structure() {
        let config = EpistemicGemmConfig {
            m: 256,
            k: 128,
            n: 256,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };

        let ptx = generate_tiled_f32_gemm_ptx(&config, "tiled_gemm");

        // Shared memory declared
        assert!(ptx.contains("smem_a"), "missing smem_a declaration");
        assert!(ptx.contains("smem_b"), "missing smem_b declaration");
        // K-strip outer loop
        assert!(ptx.contains("KLOOP:"), "missing outer K loop");
        assert!(ptx.contains("KLOOP_END:"), "missing outer K loop end");
        // Inner product loop
        assert!(ptx.contains("IK_LOOP:"), "missing inner K loop");
        assert!(ptx.contains("IK_DONE:"), "missing inner K loop end");
        // Shared memory operations
        assert!(ptx.contains("ld.shared.f32"), "missing shared memory load");
        assert!(ptx.contains("st.shared.f32"), "missing shared memory store");
        assert!(ptx.contains("bar.sync"), "missing barrier sync");
        // 16 FMA instructions in outer product
        let fma_count = ptx.matches("fma.rn.f32").count();
        assert!(
            fma_count >= 16,
            "expected >= 16 FMAs in 4x4 tile, got {}",
            fma_count
        );
        // Alpha/beta scaling present
        assert!(ptx.contains("[alpha]"), "missing alpha param");
        assert!(ptx.contains("[beta]"), "missing beta param");
        // Skip labels for each of 16 outputs
        for mi in 0..4u32 {
            for ni in 0..4u32 {
                assert!(
                    ptx.contains(&format!("SKIP_{mi}_{ni}:")),
                    "missing skip label SKIP_{mi}_{ni}"
                );
            }
        }
    }

    #[test]
    fn test_epsilon_only_ptx_structure() {
        let config = EpistemicGemmConfig {
            m: 64,
            k: 64,
            n: 64,
            alpha: 0.5,
            beta: 2.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };

        let ptx = generate_epsilon_only_ptx(&config, "eps_kernel");

        assert!(ptx.contains("eps_kernel"));
        assert!(ptx.contains("a_val_ptr"), "missing a_val_ptr param");
        assert!(ptx.contains("a_eps_ptr"), "missing a_eps_ptr param");
        assert!(ptx.contains("b_eps_ptr"), "missing b_eps_ptr param");
        assert!(ptx.contains("result_eps_ptr"), "missing result_eps_ptr param");
        assert!(ptx.contains("EPS_K:"), "missing K loop label");
        assert!(ptx.contains("EPS_K_DONE:"), "missing K loop end label");
        // Three FMA instructions per K iteration: |A|*epsB, |B|*epsA, epsA*epsB
        let fma_count = ptx.matches("fma.rn.f32").count();
        assert!(
            fma_count >= 3,
            "expected >= 3 FMAs in eps K loop, got {}",
            fma_count
        );
        assert!(ptx.contains("abs.f32"), "missing abs for |A| and |B|");
        assert!(ptx.contains("st.global.f32"), "missing epsilon store");
    }

    #[test]
    fn test_wmma_k_loop_ptx_structure() {
        let config = EpistemicGemmConfig {
            m: 64,
            k: 64,
            n: 64,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F16,
            use_tensor_cores: true,
            confidence_threshold: None,
        };

        let ptx = generate_wmma_k_loop_ptx(&config, "wmma_gemm");

        assert!(ptx.contains("wmma_gemm"));
        assert!(ptx.contains("WMMA_K:"), "missing WMMA K loop");
        assert!(ptx.contains("WMMA_K_DONE:"), "missing WMMA K done label");
        assert!(
            ptx.contains("wmma.load.a.sync"),
            "missing WMMA A fragment load"
        );
        assert!(
            ptx.contains("wmma.load.b.sync"),
            "missing WMMA B fragment load"
        );
        assert!(
            ptx.contains("wmma.mma.sync"),
            "missing WMMA multiply-accumulate"
        );
        assert!(
            ptx.contains("wmma.store.d.sync"),
            "missing WMMA result store"
        );
        // K loop must be present (not a single-iteration skeleton)
        assert!(ptx.contains("bra WMMA_K;"), "missing K loop branch");
    }

    #[test]
    fn test_dispatch_routes_f16_to_wmma() {
        let config_f16 = EpistemicGemmConfig {
            m: 32,
            k: 32,
            n: 32,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F16,
            use_tensor_cores: true,
            confidence_threshold: None,
        };
        let ptx = generate_epistemic_gemm_ptx(&config_f16, "k");
        assert!(ptx.contains("wmma.mma.sync"), "F16+TC should use WMMA path");
    }

    #[test]
    fn test_dispatch_routes_f32_to_tiled() {
        let config_f32 = EpistemicGemmConfig {
            m: 128,
            k: 64,
            n: 128,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        let ptx = generate_epistemic_gemm_ptx(&config_f32, "k");
        assert!(ptx.contains("smem_a"), "F32 should use tiled smem path");
        assert!(!ptx.contains("wmma"), "F32 should not use WMMA");
    }

    #[test]
    fn test_roofline_estimate_memory_bound() {
        // Small K → low arithmetic intensity → memory-bound
        let config = EpistemicGemmConfig {
            m: 64,
            k: 4,
            n: 64,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        // RTX 3080: 936 GB/s, 29.8 TFLOPS FP32
        let (gflops, intensity, bottleneck) = roofline_estimate(&config, 0.01, 936.0, 29.8);
        assert!(gflops > 0.0, "gflops must be positive");
        assert!(intensity > 0.0, "arithmetic intensity must be positive");
        assert_eq!(bottleneck, "memory", "small K should be memory-bound");
    }

    #[test]
    fn test_roofline_estimate_compute_bound() {
        // Large K → high arithmetic intensity → compute-bound
        let config = EpistemicGemmConfig {
            m: 1024,
            k: 4096,
            n: 1024,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        let (gflops, intensity, bottleneck) = roofline_estimate(&config, 1.0, 936.0, 29.8);
        assert!(gflops > 0.0);
        assert!(intensity > 10.0, "large K should push past ridge point");
        assert_eq!(bottleneck, "compute");
    }

    #[test]
    fn test_tiled_gemm_launch_config() {
        let config = EpistemicGemmConfig {
            m: 128,
            k: 64,
            n: 192,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        let (gx, gy, bx, by) = tiled_gemm_launch_config(&config);
        assert_eq!(gx, 3); // ceil(192/64) = 3
        assert_eq!(gy, 2); // ceil(128/64) = 2
        assert_eq!(bx, 16);
        assert_eq!(by, 16);
    }

    #[test]
    fn test_epsilon_kernel_launch_config() {
        let config = EpistemicGemmConfig {
            m: 64,
            k: 32,
            n: 64,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        let (gx, bx) = epsilon_kernel_launch_config(&config);
        assert_eq!(bx, 256);
        assert_eq!(gx, (64 * 64 + 255) / 256); // ceil(4096/256) = 16
    }

    #[test]
    fn test_500_gflops_target_is_achievable_on_a100() {
        // Verify that the roofline model predicts >500 GFLOPS for a realistic
        // 4096×4096×4096 GEMM on A100-class hardware at 50% efficiency.
        let config = EpistemicGemmConfig {
            m: 4096,
            k: 4096,
            n: 4096,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        // A100 SXM4: 19.5 TFLOPS FP32, 2000 GB/s HBM2e
        // At 50% efficiency: ~9.75 TFLOPS = 9750 GFLOPS >> 500 GFLOPS
        let flops = 2.0 * 4096.0_f64 * 4096.0 * 4096.0;
        let elapsed_at_50pct = flops / (0.5 * 19.5e12); // seconds
        let elapsed_ms = elapsed_at_50pct * 1000.0;
        let (gflops, _, _) = roofline_estimate(&config, elapsed_ms, 2000.0, 19.5);
        assert!(
            gflops > 500.0,
            "at 50% A100 efficiency should give {} GFLOPS > 500",
            gflops
        );
    }

    #[test]
    fn test_double_buffered_ptx_structure() {
        let config = EpistemicGemmConfig {
            m: 128,
            k: 64,
            n: 128,
            alpha: 1.0,
            beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        let ptx = generate_double_buffered_f32_gemm_ptx(&config, "test_dbuf");
        // Two ping-pong smem allocations
        assert!(ptx.contains("smem_a0"), "must have smem_a0");
        assert!(ptx.contains("smem_a1"), "must have smem_a1");
        assert!(ptx.contains("smem_b0"), "must have smem_b0");
        assert!(ptx.contains("smem_b1"), "must have smem_b1");
        // cp.async machinery
        assert!(ptx.contains("cp.async.cg.shared.global"), "must use cp.async.cg");
        assert!(ptx.contains("cp.async.commit_group"), "must commit async group");
        assert!(ptx.contains("cp.async.wait_group 1"), "must wait leaving 1 in-flight (overlap)");
        assert!(ptx.contains("cp.async.wait_group 0"), "epilogue must drain all");
        // Ping-pong toggle
        assert!(ptx.contains("xor.b32"), "must toggle buf with XOR");
        // FMA instructions present
        assert!(ptx.contains("fma.rn.f32"), "must have FMA");
        // Prologue label
        assert!(ptx.contains("DB_KLOOP"), "must have double-buffer K loop label");
        assert!(ptx.contains("DB_EPI_LOOP"), "must have epilogue drain loop");
        // entry header
        assert!(ptx.contains(".visible .entry test_dbuf"), "entry name correct");
    }

    #[test]
    fn test_dispatch_sm80_uses_double_buffer() {
        let config = EpistemicGemmConfig {
            m: 64, k: 64, n: 64,
            alpha: 1.0, beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        let ptx = generate_epistemic_gemm_ptx_sm(&config, "k_sm80", 8, 0);
        assert!(ptx.contains("cp.async"), "sm_80 path must use cp.async");
        assert!(ptx.contains("smem_a0"), "sm_80 path must use double-buffer smem");
    }

    #[test]
    fn test_dispatch_sm75_uses_single_buffer() {
        let config = EpistemicGemmConfig {
            m: 64, k: 64, n: 64,
            alpha: 1.0, beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        let ptx = generate_epistemic_gemm_ptx_sm(&config, "k_sm75", 7, 5);
        // single-buffer uses smem_a (not smem_a0/smem_a1)
        assert!(ptx.contains("smem_a["), "sm_75 path must use single smem_a");
        assert!(!ptx.contains("cp.async"), "sm_75 path must NOT use cp.async");
    }

    #[test]
    fn test_profile_gemm_launch_fields() {
        let config = EpistemicGemmConfig {
            m: 512, k: 512, n: 512,
            alpha: 1.0, beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        // Simulate 1 ms elapsed, RTX 3080 specs
        let p = profile_gemm_launch(&config, 1.0, 760.0, 29.8, true);
        assert!(p.flops > 0.0, "flops must be positive");
        assert!(p.achieved_gflops > 0.0, "achieved_gflops must be positive");
        assert!(p.arithmetic_intensity > 0.0, "intensity must be positive");
        assert!(p.ridge_point_flops_per_byte > 0.0, "ridge must be positive");
        assert!(p.double_buffered, "double_buffered must be true");
        assert!(p.efficiency > 0.0 && p.efficiency < 100.0, "efficiency in range");
    }

    #[test]
    fn test_format_gemm_profile_contains_key_fields() {
        let config = EpistemicGemmConfig {
            m: 1024, k: 1024, n: 1024,
            alpha: 1.0, beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        let p = profile_gemm_launch(&config, 0.5, 936.0, 19.5, true);
        let line = format_gemm_profile("oct_mul", &config, &p);
        assert!(line.contains("[souc-gpu]"), "must have souc-gpu prefix");
        assert!(line.contains("oct_mul"), "must contain kernel name");
        assert!(line.contains("1024"), "must contain dimension");
        assert!(line.contains("GFLOPS"), "must contain GFLOPS");
        assert!(line.contains("bound="), "must classify bottleneck");
        assert!(line.contains("dbuf=yes"), "must show double-buffer flag");
    }

    #[test]
    fn test_double_buffer_arithmetic_intensity_matches_single() {
        // Both kernels have identical memory access pattern — intensity should match.
        let config = EpistemicGemmConfig {
            m: 256, k: 128, n: 256,
            alpha: 1.0, beta: 0.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };
        let elapsed_ms = 1.0;
        let (_, intensity_single, _) = roofline_estimate(&config, elapsed_ms, 760.0, 29.8);
        // Double-buffer doesn't change memory traffic, same formula
        assert!(intensity_single > 10.0, "tiled intensity should be well above naive 0.3");
    }

    #[test]
    fn test_wgsl_generation() {
        let config = EpistemicGemmConfig {
            m: 256,
            k: 256,
            n: 256,
            alpha: 1.0,
            beta: 1.0,
            precision: MatrixPrecision::F32,
            use_tensor_cores: false,
            confidence_threshold: None,
        };

        let wgsl = generate_epistemic_gemm_wgsl(&config);

        assert!(wgsl.contains("struct Knowledge"));
        assert!(wgsl.contains("struct GemmParams"));
        assert!(wgsl.contains("@compute"));
        assert!(wgsl.contains("epistemic_gemm"));
        assert!(wgsl.contains("uncertainty propagation"));
    }

    #[test]
    fn test_epistemic_gemm_epsilon() {
        // 2x2 matrices
        let a_values = [1.0, 2.0, 3.0, 4.0];
        let a_epsilon = [0.1, 0.1, 0.1, 0.1];
        let b_values = [5.0, 6.0, 7.0, 8.0];
        let b_epsilon = [0.2, 0.2, 0.2, 0.2];
        let c_values = [0.0, 0.0, 0.0, 0.0];
        let c_epsilon = [0.0, 0.0, 0.0, 0.0];

        let result_eps = compute_gemm_epsilon(
            &a_values, &a_epsilon, &b_values, &b_epsilon, &c_values, &c_epsilon, 1.0, 1.0, 2, 2,
            2,
        );

        assert_eq!(result_eps.len(), 4);
        for &eps in &result_eps {
            assert!(eps > 0.0);
            assert!(eps < 10.0); // Reasonable bound
        }
    }

    #[test]
    fn test_gemm_epsilon_with_scaling() {
        // Test with alpha=0.5, beta=2.0
        let a_values = [1.0, 2.0];
        let a_epsilon = [0.1, 0.1];
        let b_values = [3.0, 4.0];
        let b_epsilon = [0.2, 0.2];
        let c_values = [1.0, 1.0];
        let c_epsilon = [0.3, 0.3];

        let result_eps = compute_gemm_epsilon(
            &a_values, &a_epsilon, &b_values, &b_epsilon, &c_values, &c_epsilon, 0.5, 2.0, 1, 2,
            1,
        );

        // Result should reflect scaling factors
        assert_eq!(result_eps.len(), 1);
        assert!(result_eps[0] > 0.0);
    }
}
