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

/// Generate PTX code for epistemic GEMM
pub fn generate_epistemic_gemm_ptx(config: &EpistemicGemmConfig, kernel_name: &str) -> String {
    let mut ptx = String::new();

    // PTX header
    ptx.push_str(".version 7.0\n");
    ptx.push_str(&format!(
        ".target sm_{}{}\n",
        if config.use_tensor_cores { "80" } else { "50" },
        if config.precision == MatrixPrecision::F16 {
            ",ptx71"
        } else {
            ""
        }
    ));
    ptx.push_str(".address_size 64\n\n");

    // Kernel signature
    ptx.push_str(&format!(".visible .entry {}(", kernel_name));
    ptx.push_str("\n    .param .u64 a_ptr,");
    ptx.push_str("\n    .param .u64 a_eps_ptr,");
    ptx.push_str("\n    .param .u64 b_ptr,");
    ptx.push_str("\n    .param .u64 b_eps_ptr,");
    ptx.push_str("\n    .param .u64 c_ptr,");
    ptx.push_str("\n    .param .u64 c_eps_ptr,");
    ptx.push_str("\n    .param .u64 result_ptr,");
    ptx.push_str("\n    .param .u64 result_eps_ptr,");
    ptx.push_str("\n    .param .f32 alpha,");
    ptx.push_str("\n    .param .f32 beta");
    ptx.push_str("\n)\n");

    // Kernel body
    ptx.push_str("{\n");
    ptx.push_str("    .reg .pred %p<8>;\n");
    ptx.push_str("    .reg .f32 %f<32>;\n");
    ptx.push_str("    .reg .f16 %h<64>;\n");
    ptx.push_str("    .reg .b32 %r<32>;\n");
    ptx.push_str("    .reg .b64 %rd<20>;\n\n");

    // Load parameters
    ptx.push_str("    // Load matrix pointers\n");
    ptx.push_str("    ld.param.u64 %rd1, [a_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd2, [a_eps_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd3, [b_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd4, [b_eps_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd5, [c_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd6, [c_eps_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd7, [result_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd8, [result_eps_ptr];\n");
    ptx.push_str("    ld.param.f32 %f1, [alpha];\n");
    ptx.push_str("    ld.param.f32 %f2, [beta];\n\n");

    // Thread indices
    ptx.push_str("    // Thread indices\n");
    ptx.push_str("    mov.u32 %r1, %tid.x;\n");
    ptx.push_str("    mov.u32 %r2, %tid.y;\n");
    ptx.push_str("    mov.u32 %r3, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %r4, %ctaid.y;\n");
    ptx.push_str("    mad.lo.u32 %r5, %r3, 16, %r1;  // row = blockIdx.x * 16 + threadIdx.x\n");
    ptx.push_str("    mad.lo.u32 %r6, %r4, 16, %r2;  // col = blockIdx.y * 16 + threadIdx.y\n\n");

    // Bounds checking
    ptx.push_str("    // Bounds checking\n");
    ptx.push_str(&format!("    setp.ge.u32 %p0, %r5, {};\n", config.m));
    ptx.push_str(&format!("    setp.ge.u32 %p1, %r6, {};\n", config.n));
    ptx.push_str("    or.pred %p2, %p0, %p1;\n");
    ptx.push_str("    @%p2 bra EXIT;\n\n");

    if config.use_tensor_cores && config.precision == MatrixPrecision::F16 {
        // Tensor Core implementation
        ptx.push_str("    // Tensor Core WMMA implementation\n");
        ptx.push_str("    .reg .b32 %wmma_a<8>;\n");
        ptx.push_str("    .reg .b32 %wmma_b<8>;\n");
        ptx.push_str("    .reg .b32 %wmma_c<8>;\n");
        ptx.push_str("    .reg .b32 %wmma_d<8>;\n\n");

        ptx.push_str("    // Load matrix fragments\n");
        ptx.push_str(
            "    wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {%wmma_a}, [%rd1];\n",
        );
        ptx.push_str(
            "    wmma.load.b.sync.aligned.col.m16n16k16.global.f16 {%wmma_b}, [%rd3];\n",
        );
        ptx.push_str(
            "    wmma.load.c.sync.aligned.row.m16n16k16.global.f16 {%wmma_c}, [%rd5];\n\n",
        );

        ptx.push_str("    // Matrix multiply-accumulate\n");
        ptx.push_str("    wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 {%wmma_d}, {%wmma_a}, {%wmma_b}, {%wmma_c};\n\n");

        ptx.push_str("    // Store result\n");
        ptx.push_str(
            "    wmma.store.d.sync.aligned.row.m16n16k16.global.f16 [%rd7], {%wmma_d};\n",
        );
    } else {
        // Standard implementation
        ptx.push_str("    // Standard GEMM implementation\n");
        ptx.push_str("    mov.f32 %f10, 0.0;  // accumulator value\n");
        ptx.push_str("    mov.f32 %f11, 0.0;  // accumulator epsilon\n");
        ptx.push_str("    mov.u32 %r7, 0;     // k index\n\n");

        ptx.push_str("LOOP_K:\n");
        ptx.push_str(&format!("    setp.lt.u32 %p3, %r7, {};\n", config.k));
        ptx.push_str("    @!%p3 bra END_LOOP_K;\n\n");

        // Compute address offsets
        ptx.push_str("    // Compute address offsets\n");
        ptx.push_str("    mul.lo.u32 %r8, %r5, k;\n");
        ptx.push_str("    add.u32 %r8, %r8, %r7;  // A[i,k] offset\n");
        ptx.push_str("    mul.lo.u32 %r9, %r7, n;\n");
        ptx.push_str("    add.u32 %r9, %r9, %r6;  // B[k,j] offset\n");
        ptx.push_str("    mul.wide.u32 %rd9, %r8, 4;\n");
        ptx.push_str("    mul.wide.u32 %rd10, %r9, 4;\n\n");

        // Load values and uncertainties
        ptx.push_str("    // Load A[i,k] and B[k,j]\n");
        ptx.push_str("    add.u64 %rd11, %rd1, %rd9;\n");
        ptx.push_str("    add.u64 %rd12, %rd3, %rd10;\n");
        ptx.push_str("    ld.global.f32 %f20, [%rd11];  // A value\n");
        ptx.push_str("    ld.global.f32 %f21, [%rd12];  // B value\n\n");

        ptx.push_str("    // Load uncertainties\n");
        ptx.push_str("    add.u64 %rd13, %rd2, %rd9;\n");
        ptx.push_str("    add.u64 %rd14, %rd4, %rd10;\n");
        ptx.push_str("    ld.global.f32 %f22, [%rd13];  // A epsilon\n");
        ptx.push_str("    ld.global.f32 %f23, [%rd14];  // B epsilon\n\n");

        // Accumulate
        ptx.push_str("    // Accumulate value\n");
        ptx.push_str("    fma.rn.f32 %f10, %f20, %f21, %f10;\n\n");

        ptx.push_str("    // Accumulate uncertainty\n");
        ptx.push_str("    abs.f32 %f24, %f20;\n");
        ptx.push_str("    abs.f32 %f25, %f21;\n");
        ptx.push_str("    fma.rn.f32 %f11, %f24, %f23, %f11;  // |A| * ε_B\n");
        ptx.push_str("    fma.rn.f32 %f11, %f25, %f22, %f11;  // |B| * ε_A\n");
        ptx.push_str("    fma.rn.f32 %f11, %f22, %f23, %f11;  // ε_A * ε_B\n\n");

        ptx.push_str("    add.u32 %r7, %r7, 1;\n");
        ptx.push_str("    bra LOOP_K;\n\n");

        ptx.push_str("END_LOOP_K:\n");

        // Load C[i,j]
        ptx.push_str("    // Load C[i,j]\n");
        ptx.push_str("    mul.lo.u32 %r10, %r5, n;\n");
        ptx.push_str("    add.u32 %r10, %r10, %r6;\n");
        ptx.push_str("    mul.wide.u32 %rd15, %r10, 4;\n");
        ptx.push_str("    add.u64 %rd16, %rd5, %rd15;\n");
        ptx.push_str("    add.u64 %rd17, %rd6, %rd15;\n");
        ptx.push_str("    ld.global.f32 %f30, [%rd16];  // C value\n");
        ptx.push_str("    ld.global.f32 %f31, [%rd17];  // C epsilon\n\n");

        // Apply scaling
        ptx.push_str("    // Apply alpha and beta scaling\n");
        ptx.push_str("    mul.f32 %f10, %f10, %f1;  // alpha * (A*B)\n");
        ptx.push_str("    mul.f32 %f11, %f11, %f1;  // alpha * epsilon\n");
        ptx.push_str("    fma.rn.f32 %f10, %f2, %f30, %f10;  // + beta * C\n");
        ptx.push_str("    mul.f32 %f31, %f31, %f2;  // beta * ε_C\n");
        ptx.push_str("    add.f32 %f11, %f11, %f31;  // total epsilon\n\n");

        // Store result
        ptx.push_str("    // Store result\n");
        ptx.push_str("    mul.lo.u32 %r11, %r5, n;\n");
        ptx.push_str("    add.u32 %r11, %r11, %r6;\n");
        ptx.push_str("    mul.wide.u32 %rd18, %r11, 4;\n");
        ptx.push_str("    add.u64 %rd19, %rd7, %rd18;\n");
        ptx.push_str("    add.u64 %rd20, %rd8, %rd18;\n");
        ptx.push_str("    st.global.f32 [%rd19], %f10;\n");
        ptx.push_str("    st.global.f32 [%rd20], %f11;\n");
    }

    ptx.push_str("\nEXIT:\n");
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    ptx
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
