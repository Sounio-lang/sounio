//! GPU Kernels for Scientific Computing Primitives
//!
//! This module provides GPU-accelerated implementations of core scientific
//! computing operations for Sounio:
//!
//! - Matrix operations (GEMM, transpose, inverse)
//! - Tensor contractions (einsum-style)
//! - ODE solver parallelization
//! - PDE stencil operations
//! - Monte Carlo sampling
//! - Reduction operations
//!
//! Supports multiple GPU backends: CUDA, Metal, Vulkan, WebGPU

// ============================================================================
// GPU Device Abstraction
// ============================================================================

/// Supported GPU backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GPUBackend {
    CUDA,
    Metal,
    Vulkan,
    WebGPU,
    CPU, // Fallback
}

/// GPU device capabilities
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub backend: GPUBackend,
    pub name: String,
    pub compute_units: u32,
    pub max_workgroup_size: u32,
    pub max_shared_memory: usize,
    pub supports_f64: bool,
    pub supports_atomics: bool,
    pub total_memory: usize,
}

/// GPU memory buffer
#[derive(Debug, Clone)]
pub struct GPUBuffer {
    pub id: u64,
    pub size: usize,
    pub dtype: DataType,
    pub device: GPUBackend,
}

/// Supported data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataType {
    #[default]
    F32,
    F64,
    I32,
    I64,
    U32,
    Complex64,
    Complex128,
}

impl DataType {
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::F32 | DataType::I32 | DataType::U32 => 4,
            DataType::F64 | DataType::I64 | DataType::Complex64 => 8,
            DataType::Complex128 => 16,
        }
    }
}

/// GPU kernel launch configuration
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub grid_size: (u32, u32, u32),
    pub block_size: (u32, u32, u32),
    pub shared_memory: usize,
}

impl LaunchConfig {
    pub fn for_1d(n: usize, block_size: u32) -> Self {
        let grid_size = ((n as u32).div_ceil(block_size), 1, 1);
        Self {
            grid_size,
            block_size: (block_size, 1, 1),
            shared_memory: 0,
        }
    }

    pub fn for_2d(m: usize, n: usize, block_x: u32, block_y: u32) -> Self {
        Self {
            grid_size: (
                (n as u32).div_ceil(block_x),
                (m as u32).div_ceil(block_y),
                1,
            ),
            block_size: (block_x, block_y, 1),
            shared_memory: 0,
        }
    }

    pub fn with_shared_memory(mut self, bytes: usize) -> Self {
        self.shared_memory = bytes;
        self
    }
}

// ============================================================================
// GPU Kernel Registry
// ============================================================================

/// Scientific kernel type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    // Matrix operations
    MatMul,
    MatMulTransposed,
    MatrixAdd,
    MatrixScale,
    MatrixTranspose,

    // Tensor operations
    TensorContract,
    TensorPermute,
    TensorReduce,

    // ODE operations
    ODEEulerStep,
    ODERK4Step,
    ODEBatchEval,

    // PDE operations
    PDEStencil1D,
    PDEStencil2D,
    PDEStencil3D,
    CrankNicolsonStep,

    // Monte Carlo
    MonteCarloSample,
    ImportanceSample,
    MCMCStep,

    // Reductions
    Sum,
    Max,
    Min,
    Norm,
    Dot,

    // Element-wise
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    ReLU,

    // Autodiff
    DualMul,
    DualAdd,
    BackwardPass,
}

/// Compiled GPU kernel
pub struct CompiledKernel {
    pub kernel_type: KernelType,
    pub backend: GPUBackend,
    pub code: Vec<u8>, // Compiled shader/PTX
    pub entry_point: String,
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// GPU matrix multiplication kernel generator
pub struct MatMulKernel {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub tile_size: usize,
    pub dtype: DataType,
}

impl MatMulKernel {
    pub fn new(m: usize, n: usize, k: usize, dtype: DataType) -> Self {
        // Choose tile size based on problem size
        let tile_size = if m * n * k > 1_000_000 { 32 } else { 16 };
        Self {
            m,
            n,
            k,
            tile_size,
            dtype,
        }
    }

    /// Generate WGSL shader for WebGPU
    pub fn generate_wgsl(&self) -> String {
        let dtype_str = match self.dtype {
            DataType::F32 => "f32",
            DataType::F64 => "f64",
            _ => "f32",
        };

        format!(
            r#"
@group(0) @binding(0) var<storage, read> A: array<{dtype}>;
@group(0) @binding(1) var<storage, read> B: array<{dtype}>;
@group(0) @binding(2) var<storage, read_write> C: array<{dtype}>;

struct Dimensions {{
    M: u32,
    N: u32,
    K: u32,
}}
@group(0) @binding(3) var<uniform> dims: Dimensions;

const TILE_SIZE: u32 = {tile}u;

var<workgroup> tile_A: array<array<{dtype}, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_B: array<array<{dtype}, TILE_SIZE>, TILE_SIZE>;

@compute @workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {{
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum: {dtype} = 0.0;
    let num_tiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < num_tiles; t = t + 1u) {{
        // Load tiles cooperatively
        let a_col = t * TILE_SIZE + local_col;
        let b_row = t * TILE_SIZE + local_row;

        if (row < dims.M && a_col < dims.K) {{
            tile_A[local_row][local_col] = A[row * dims.K + a_col];
        }} else {{
            tile_A[local_row][local_col] = 0.0;
        }}

        if (b_row < dims.K && col < dims.N) {{
            tile_B[local_row][local_col] = B[b_row * dims.N + col];
        }} else {{
            tile_B[local_row][local_col] = 0.0;
        }}

        workgroupBarrier();

        // Compute partial dot product
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {{
            sum = sum + tile_A[local_row][k] * tile_B[k][local_col];
        }}

        workgroupBarrier();
    }}

    if (row < dims.M && col < dims.N) {{
        C[row * dims.N + col] = sum;
    }}
}}
"#,
            dtype = dtype_str,
            tile = self.tile_size
        )
    }

    /// Generate Metal Shading Language for Apple GPUs
    pub fn generate_metal(&self) -> String {
        format!(
            r#"
#include <metal_stdlib>
using namespace metal;

constant uint TILE_SIZE = {tile};

kernel void matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {{
    threadgroup float tile_A[TILE_SIZE][TILE_SIZE];
    threadgroup float tile_B[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0f;
    uint num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {{
        uint a_col = t * TILE_SIZE + lid.x;
        uint b_row = t * TILE_SIZE + lid.y;

        if (row < M && a_col < K) {{
            tile_A[lid.y][lid.x] = A[row * K + a_col];
        }} else {{
            tile_A[lid.y][lid.x] = 0.0f;
        }}

        if (b_row < K && col < N) {{
            tile_B[lid.y][lid.x] = B[b_row * N + col];
        }} else {{
            tile_B[lid.y][lid.x] = 0.0f;
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {{
            sum += tile_A[lid.y][k] * tile_B[k][lid.x];
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
"#,
            tile = self.tile_size
        )
    }

    /// Generate CUDA PTX-like pseudocode (actual PTX is more complex)
    pub fn generate_cuda(&self) -> String {
        format!(
            r#"
// CUDA kernel for matrix multiplication
// Compile with: nvcc -arch=sm_70

#define TILE_SIZE {tile}

__global__ void matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {{
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {{
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {{
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }}

        __syncthreads();
    }}

    if (row < M && col < N) {{
        C[row * N + col] = sum;
    }}
}}
"#,
            tile = self.tile_size
        )
    }
}

// ============================================================================
// PDE Stencil Operations
// ============================================================================

/// Stencil type for PDE operations
#[derive(Debug, Clone)]
pub enum StencilType {
    Laplacian1D,      // [-1, 2, -1] / dx^2
    Laplacian2D,      // 5-point stencil
    Laplacian3D,      // 7-point stencil
    Advection1D,      // Upwind or central
    Custom(Vec<f64>), // User-defined weights
}

/// GPU stencil kernel for PDEs
pub struct StencilKernel {
    pub stencil: StencilType,
    pub dims: Vec<usize>,
    pub dtype: DataType,
}

impl StencilKernel {
    pub fn laplacian_2d(nx: usize, ny: usize) -> Self {
        Self {
            stencil: StencilType::Laplacian2D,
            dims: vec![ny, nx],
            dtype: DataType::F64,
        }
    }

    /// Generate WGSL for 2D Laplacian
    pub fn generate_laplacian_2d_wgsl(&self) -> String {
        r#"
@group(0) @binding(0) var<storage, read> u_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> u_out: array<f32>;

struct Params {
    nx: u32,
    ny: u32,
    dx: f32,
    dy: f32,
    dt: f32,
    alpha: f32,  // Diffusion coefficient
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn laplacian_step(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let i = gid.x;
    let j = gid.y;
    let nx = params.nx;
    let ny = params.ny;

    if (i == 0u || i >= nx - 1u || j == 0u || j >= ny - 1u) {
        return;
    }

    let idx = j * nx + i;
    let idx_left = idx - 1u;
    let idx_right = idx + 1u;
    let idx_down = idx - nx;
    let idx_up = idx + nx;

    let dx2 = params.dx * params.dx;
    let dy2 = params.dy * params.dy;

    let laplacian =
        (u_in[idx_left] - 2.0 * u_in[idx] + u_in[idx_right]) / dx2 +
        (u_in[idx_down] - 2.0 * u_in[idx] + u_in[idx_up]) / dy2;

    u_out[idx] = u_in[idx] + params.dt * params.alpha * laplacian;
}
"#
        .to_string()
    }

    /// Generate Metal for 2D wave equation
    pub fn generate_wave_2d_metal(&self) -> String {
        r#"
#include <metal_stdlib>
using namespace metal;

struct WaveParams {
    uint nx;
    uint ny;
    float dx;
    float dy;
    float dt;
    float c;  // Wave speed
};

kernel void wave_step(
    device const float* u [[buffer(0)]],
    device const float* u_prev [[buffer(1)]],
    device float* u_next [[buffer(2)]],
    constant WaveParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    uint nx = params.nx;
    uint ny = params.ny;

    if (i == 0 || i >= nx - 1 || j == 0 || j >= ny - 1) {
        return;
    }

    uint idx = j * nx + i;

    float dx2 = params.dx * params.dx;
    float dy2 = params.dy * params.dy;
    float dt2 = params.dt * params.dt;
    float c2 = params.c * params.c;

    float laplacian =
        (u[idx - 1] - 2.0f * u[idx] + u[idx + 1]) / dx2 +
        (u[idx - nx] - 2.0f * u[idx] + u[idx + nx]) / dy2;

    // Wave equation: u_tt = c^2 * laplacian(u)
    u_next[idx] = 2.0f * u[idx] - u_prev[idx] + c2 * dt2 * laplacian;
}
"#
        .to_string()
    }
}

// ============================================================================
// Monte Carlo Operations
// ============================================================================

/// GPU Monte Carlo kernel for sampling
pub struct MonteCarloKernel {
    pub samples: usize,
    pub dimensions: usize,
    pub dtype: DataType,
}

impl MonteCarloKernel {
    /// Generate WGSL for parallel Monte Carlo integration
    pub fn generate_mc_integration_wgsl(&self) -> String {
        format!(
            r#"
// Monte Carlo integration kernel
// Uses PCG random number generator

struct PCGState {{
    state: u32,
    inc: u32,
}}

fn pcg_next(state: ptr<function, PCGState>) -> u32 {{
    let oldstate = (*state).state;
    (*state).state = oldstate * 747796405u + (*state).inc;
    let xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    let rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}}

fn pcg_uniform(state: ptr<function, PCGState>) -> f32 {{
    return f32(pcg_next(state)) / 4294967296.0;
}}

@group(0) @binding(0) var<storage, read_write> results: array<f32>;
@group(0) @binding(1) var<uniform> seed_base: u32;

const SAMPLES_PER_THREAD: u32 = {samples_per_thread}u;
const DIMS: u32 = {dims}u;

@compute @workgroup_size(256)
fn monte_carlo(
    @builtin(global_invocation_id) gid: vec3<u32>
) {{
    let thread_id = gid.x;

    // Initialize RNG with unique seed per thread
    var rng: PCGState;
    rng.state = seed_base + thread_id * 12345u;
    rng.inc = thread_id * 2u + 1u;

    var sum: f32 = 0.0;

    for (var i = 0u; i < SAMPLES_PER_THREAD; i = i + 1u) {{
        // Generate random point in [0,1]^d
        var x: array<f32, {dims}>;
        for (var d = 0u; d < DIMS; d = d + 1u) {{
            x[d] = pcg_uniform(&rng);
        }}

        // Evaluate integrand (user-defined function would be called here)
        sum = sum + eval_integrand(x);
    }}

    // Store partial sum
    results[thread_id] = sum / f32(SAMPLES_PER_THREAD);
}}

// Placeholder for user integrand
fn eval_integrand(x: array<f32, {dims}>) -> f32 {{
    return 1.0;  // Will be replaced with actual integrand
}}
"#,
            samples_per_thread = 1000,
            dims = self.dimensions
        )
    }

    /// Generate Metal for MCMC (Metropolis-Hastings)
    pub fn generate_mcmc_metal(&self) -> String {
        r#"
#include <metal_stdlib>
using namespace metal;

// PCG random number generator
struct PCGState {
    uint state;
    uint inc;
};

uint pcg_next(thread PCGState& rng) {
    uint oldstate = rng.state;
    rng.state = oldstate * 747796405u + rng.inc;
    uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

float pcg_uniform(thread PCGState& rng) {
    return float(pcg_next(rng)) / 4294967296.0f;
}

float pcg_normal(thread PCGState& rng) {
    // Box-Muller transform
    float u1 = pcg_uniform(rng);
    float u2 = pcg_uniform(rng);
    return sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2);
}

kernel void mcmc_step(
    device float* samples [[buffer(0)]],
    device const float* log_prob [[buffer(1)]],
    device float* new_samples [[buffer(2)]],
    device float* new_log_prob [[buffer(3)]],
    constant uint& n_chains [[buffer(4)]],
    constant uint& n_dims [[buffer(5)]],
    constant float& step_size [[buffer(6)]],
    constant uint& seed [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_chains) return;

    PCGState rng;
    rng.state = seed + gid * 12345u;
    rng.inc = gid * 2u + 1u;

    // Propose new state
    for (uint d = 0; d < n_dims; d++) {
        uint idx = gid * n_dims + d;
        new_samples[idx] = samples[idx] + step_size * pcg_normal(rng);
    }

    // Compute acceptance probability (log_prob is computed separately)
    // Accept/reject step happens after log_prob evaluation
    float accept_prob = exp(new_log_prob[gid] - log_prob[gid]);

    if (pcg_uniform(rng) < accept_prob) {
        // Accept: keep new samples
        for (uint d = 0; d < n_dims; d++) {
            samples[gid * n_dims + d] = new_samples[gid * n_dims + d];
        }
    } else {
        // Reject: copy old samples back
        for (uint d = 0; d < n_dims; d++) {
            new_samples[gid * n_dims + d] = samples[gid * n_dims + d];
        }
    }
}
"#
        .to_string()
    }
}

// ============================================================================
// ODE Solver Kernels
// ============================================================================

/// GPU ODE solver for parallel evaluation
pub struct ODEKernel {
    pub n_systems: usize, // Number of parallel ODEs
    pub state_dim: usize, // Dimension of each ODE
    pub method: ODEMethod,
    pub dtype: DataType,
}

#[derive(Debug, Clone, Copy)]
pub enum ODEMethod {
    Euler,
    RK4,
    DormandPrince,
}

impl ODEKernel {
    /// Generate WGSL for batch RK4 step
    pub fn generate_rk4_batch_wgsl(&self) -> String {
        format!(
            r#"
// Batch RK4 for {n_systems} parallel ODE systems, each with {state_dim} dimensions

@group(0) @binding(0) var<storage, read> y: array<f32>;
@group(0) @binding(1) var<storage, read_write> y_new: array<f32>;
@group(0) @binding(2) var<storage, read_write> k1: array<f32>;
@group(0) @binding(3) var<storage, read_write> k2: array<f32>;
@group(0) @binding(4) var<storage, read_write> k3: array<f32>;
@group(0) @binding(5) var<storage, read_write> k4: array<f32>;

struct Params {{
    t: f32,
    dt: f32,
    n_systems: u32,
    state_dim: u32,
}}
@group(0) @binding(6) var<uniform> params: Params;

// User-defined ODE function (will be replaced)
fn f(sys_id: u32, t: f32, y_idx: u32) -> f32 {{
    return 0.0;  // Placeholder
}}

@compute @workgroup_size(256)
fn rk4_k1(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.n_systems * params.state_dim) {{ return; }}

    let sys_id = idx / params.state_dim;
    k1[idx] = f(sys_id, params.t, idx);
}}

@compute @workgroup_size(256)
fn rk4_k2(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.n_systems * params.state_dim) {{ return; }}

    let sys_id = idx / params.state_dim;
    let y_temp = y[idx] + 0.5 * params.dt * k1[idx];
    k2[idx] = f(sys_id, params.t + 0.5 * params.dt, idx);
}}

@compute @workgroup_size(256)
fn rk4_k3(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.n_systems * params.state_dim) {{ return; }}

    let sys_id = idx / params.state_dim;
    let y_temp = y[idx] + 0.5 * params.dt * k2[idx];
    k3[idx] = f(sys_id, params.t + 0.5 * params.dt, idx);
}}

@compute @workgroup_size(256)
fn rk4_k4(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.n_systems * params.state_dim) {{ return; }}

    let sys_id = idx / params.state_dim;
    let y_temp = y[idx] + params.dt * k3[idx];
    k4[idx] = f(sys_id, params.t + params.dt, idx);
}}

@compute @workgroup_size(256)
fn rk4_update(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= params.n_systems * params.state_dim) {{ return; }}

    y_new[idx] = y[idx] + (params.dt / 6.0) * (
        k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]
    );
}}
"#,
            n_systems = self.n_systems,
            state_dim = self.state_dim
        )
    }
}

// ============================================================================
// Tensor Contraction Kernels
// ============================================================================

/// Einsum-style tensor contraction on GPU
pub struct TensorContractionKernel {
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub contraction_indices: Vec<char>,
    pub dtype: DataType,
}

impl TensorContractionKernel {
    /// Generate optimized einsum for common patterns
    pub fn generate_wgsl(&self) -> String {
        // For simplicity, generate a general contraction kernel
        // Real implementation would specialize for common patterns

        r#"
// General tensor contraction kernel
// Handles arbitrary einsum expressions through index computation

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

struct ContractionParams {
    // Shapes and strides encoded as flat arrays
    a_shape: array<u32, 8>,
    b_shape: array<u32, 8>,
    c_shape: array<u32, 8>,
    a_strides: array<u32, 8>,
    b_strides: array<u32, 8>,
    c_strides: array<u32, 8>,
    a_ndim: u32,
    b_ndim: u32,
    c_ndim: u32,
    contraction_size: u32,
}
@group(0) @binding(3) var<uniform> params: ContractionParams;

@compute @workgroup_size(256)
fn contract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c_idx = gid.x;
    if (c_idx >= arrayLength(&C)) { return; }

    // Compute multi-dimensional indices for output
    var c_indices: array<u32, 8>;
    var remaining = c_idx;
    for (var d = 0u; d < params.c_ndim; d = d + 1u) {
        c_indices[d] = remaining / params.c_strides[d];
        remaining = remaining % params.c_strides[d];
    }

    // Sum over contraction indices
    var sum: f32 = 0.0;
    for (var k = 0u; k < params.contraction_size; k = k + 1u) {
        // Compute A and B indices based on contraction pattern
        // (This is simplified - real implementation needs index mapping)
        let a_idx = compute_a_index(c_indices, k, params);
        let b_idx = compute_b_index(c_indices, k, params);

        sum = sum + A[a_idx] * B[b_idx];
    }

    C[c_idx] = sum;
}

fn compute_a_index(c_idx: array<u32, 8>, k: u32, p: ContractionParams) -> u32 {
    // Placeholder - actual implementation depends on einsum pattern
    return 0u;
}

fn compute_b_index(c_idx: array<u32, 8>, k: u32, p: ContractionParams) -> u32 {
    return 0u;
}
"#
        .to_string()
    }
}

// ============================================================================
// Reduction Kernels
// ============================================================================

/// Parallel reduction kernel
pub struct ReductionKernel {
    pub op: ReductionOp,
    pub size: usize,
    pub dtype: DataType,
}

#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Sum,
    Max,
    Min,
    Product,
    LogSumExp,
}

impl ReductionKernel {
    /// Generate WGSL for tree reduction
    pub fn generate_wgsl(&self) -> String {
        let op_code = match self.op {
            ReductionOp::Sum => "a + b",
            ReductionOp::Max => "max(a, b)",
            ReductionOp::Min => "min(a, b)",
            ReductionOp::Product => "a * b",
            ReductionOp::LogSumExp => "max(a, b) + log(1.0 + exp(-abs(a - b)))",
        };

        let identity = match self.op {
            ReductionOp::Sum => "0.0",
            ReductionOp::Max => "-3.4028235e+38", // -FLT_MAX
            ReductionOp::Min => "3.4028235e+38",  // FLT_MAX
            ReductionOp::Product => "1.0",
            ReductionOp::LogSumExp => "-3.4028235e+38",
        };

        format!(
            r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> n: u32;

const BLOCK_SIZE: u32 = 256u;
var<workgroup> shared: array<f32, BLOCK_SIZE>;

fn reduce_op(a: f32, b: f32) -> f32 {{
    return {op};
}}

@compute @workgroup_size(BLOCK_SIZE)
fn reduce(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {{
    let tid = lid.x;
    let idx = gid.x;

    // Load and reduce two elements per thread
    var val: f32 = {identity};
    if (idx < n) {{
        val = input[idx];
    }}
    if (idx + BLOCK_SIZE * gridDim.x < n) {{
        val = reduce_op(val, input[idx + BLOCK_SIZE * gridDim.x]);
    }}
    shared[tid] = val;

    workgroupBarrier();

    // Tree reduction in shared memory
    for (var s = BLOCK_SIZE / 2u; s > 0u; s = s / 2u) {{
        if (tid < s) {{
            shared[tid] = reduce_op(shared[tid], shared[tid + s]);
        }}
        workgroupBarrier();
    }}

    if (tid == 0u) {{
        output[wgid.x] = shared[0];
    }}
}}
"#,
            op = op_code,
            identity = identity
        )
    }
}

// ============================================================================
// Kernel Compiler/Manager
// ============================================================================

/// Manages GPU kernel compilation and caching
pub struct KernelManager {
    cached_kernels: std::collections::HashMap<(KernelType, GPUBackend), CompiledKernel>,
    capabilities: Option<DeviceCapabilities>,
}

impl KernelManager {
    pub fn new() -> Self {
        Self {
            cached_kernels: std::collections::HashMap::new(),
            capabilities: None,
        }
    }

    /// Detect available GPU
    pub fn detect_device(&mut self) -> DeviceCapabilities {
        // In real implementation, this would query actual hardware
        // For now, return CPU fallback
        let caps = DeviceCapabilities {
            backend: GPUBackend::CPU,
            name: "CPU Fallback".to_string(),
            compute_units: num_cpus(),
            max_workgroup_size: 1024,
            max_shared_memory: 48 * 1024,
            supports_f64: true,
            supports_atomics: true,
            total_memory: 16 * 1024 * 1024 * 1024,
        };
        self.capabilities = Some(caps.clone());
        caps
    }

    /// Get or compile a kernel
    pub fn get_kernel(
        &mut self,
        kernel_type: KernelType,
        params: &KernelParams,
    ) -> &CompiledKernel {
        let backend = self
            .capabilities
            .as_ref()
            .map(|c| c.backend)
            .unwrap_or(GPUBackend::CPU);

        let key = (kernel_type, backend);

        if !self.cached_kernels.contains_key(&key) {
            let kernel = self.compile_kernel(kernel_type, backend, params);
            self.cached_kernels.insert(key, kernel);
        }

        self.cached_kernels.get(&key).unwrap()
    }

    fn compile_kernel(
        &self,
        kernel_type: KernelType,
        backend: GPUBackend,
        params: &KernelParams,
    ) -> CompiledKernel {
        let code = match (kernel_type, backend) {
            (KernelType::MatMul, GPUBackend::WebGPU) => {
                let k = MatMulKernel::new(params.m, params.n, params.k, DataType::F32);
                k.generate_wgsl().into_bytes()
            }
            (KernelType::MatMul, GPUBackend::Metal) => {
                let k = MatMulKernel::new(params.m, params.n, params.k, DataType::F32);
                k.generate_metal().into_bytes()
            }
            (KernelType::MatMul, GPUBackend::CUDA) => {
                let k = MatMulKernel::new(params.m, params.n, params.k, DataType::F32);
                k.generate_cuda().into_bytes()
            }
            _ => Vec::new(),
        };

        CompiledKernel {
            kernel_type,
            backend,
            code,
            entry_point: "main".to_string(),
        }
    }
}

/// Kernel compilation parameters
#[derive(Debug, Clone, Default)]
pub struct KernelParams {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub tile_size: usize,
    pub dtype: DataType,
}

impl Default for KernelManager {
    fn default() -> Self {
        Self::new()
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(4)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_kernel_wgsl() {
        let kernel = MatMulKernel::new(256, 256, 256, DataType::F32);
        let wgsl = kernel.generate_wgsl();

        assert!(wgsl.contains("@compute"));
        assert!(wgsl.contains("TILE_SIZE"));
        assert!(wgsl.contains("workgroupBarrier"));
    }

    #[test]
    fn test_matmul_kernel_metal() {
        let kernel = MatMulKernel::new(128, 128, 128, DataType::F32);
        let metal = kernel.generate_metal();

        assert!(metal.contains("kernel void matmul"));
        assert!(metal.contains("threadgroup"));
    }

    #[test]
    fn test_stencil_kernel() {
        let kernel = StencilKernel::laplacian_2d(100, 100);
        let wgsl = kernel.generate_laplacian_2d_wgsl();

        assert!(wgsl.contains("laplacian"));
        assert!(wgsl.contains("@compute"));
    }

    #[test]
    fn test_monte_carlo_kernel() {
        let kernel = MonteCarloKernel {
            samples: 100000,
            dimensions: 3,
            dtype: DataType::F32,
        };
        let wgsl = kernel.generate_mc_integration_wgsl();

        assert!(wgsl.contains("pcg_uniform"));
        assert!(wgsl.contains("monte_carlo"));
    }

    #[test]
    fn test_reduction_kernel() {
        let kernel = ReductionKernel {
            op: ReductionOp::Sum,
            size: 1000000,
            dtype: DataType::F32,
        };
        let wgsl = kernel.generate_wgsl();

        assert!(wgsl.contains("reduce_op"));
        assert!(wgsl.contains("workgroupBarrier"));
    }

    #[test]
    fn test_ode_kernel() {
        let kernel = ODEKernel {
            n_systems: 1000,
            state_dim: 4,
            method: ODEMethod::RK4,
            dtype: DataType::F32,
        };
        let wgsl = kernel.generate_rk4_batch_wgsl();

        assert!(wgsl.contains("rk4_k1"));
        assert!(wgsl.contains("rk4_update"));
    }

    #[test]
    fn test_kernel_manager() {
        let mut manager = KernelManager::new();
        let caps = manager.detect_device();

        assert!(caps.compute_units > 0);
    }

    #[test]
    fn test_launch_config() {
        let config = LaunchConfig::for_2d(1024, 1024, 16, 16);

        assert_eq!(config.grid_size.0, 64);
        assert_eq!(config.grid_size.1, 64);
        assert_eq!(config.block_size.0, 16);
        assert_eq!(config.block_size.1, 16);
    }

    #[test]
    fn test_data_type_sizes() {
        assert_eq!(DataType::F32.size_bytes(), 4);
        assert_eq!(DataType::F64.size_bytes(), 8);
        assert_eq!(DataType::Complex128.size_bytes(), 16);
    }
}
