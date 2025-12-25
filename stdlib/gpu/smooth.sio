// gpu::smooth — GPU-Accelerated Spatial Smoothing
//
// High-performance Gaussian smoothing for neuroimaging.
// Implements separable 3D convolution for volumetric data.
//
// Features:
// - Separable 3D Gaussian convolution (O(n) per dimension)
// - FWHM-based kernel specification (standard in neuroimaging)
// - Shared memory optimization for data reuse
// - Batch processing for 4D fMRI data
//
// References:
// - Friston et al. (1995): "Spatial registration and normalization of images"
// - Pham et al. (2000): "Current methods in medical image segmentation"

// ============================================================================
// CONSTANTS
// ============================================================================

fn MAX_KERNEL_SIZE() -> i64 { 15 }
fn BLOCK_DIM_X() -> i32 { 16 }
fn BLOCK_DIM_Y() -> i32 { 16 }
fn BLOCK_DIM_Z() -> i32 { 4 }

fn pi() -> f64 { 3.14159265358979323846 }

extern "C" {
    fn exp(x: f64) -> f64;
    fn sqrt(x: f64) -> f64;
    fn fabs(x: f64) -> f64;
}

// ============================================================================
// GAUSSIAN KERNEL
// ============================================================================

/// 1D Gaussian kernel for separable convolution
struct GaussianKernel1D {
    weights: [f64; 15],
    size: i64,
    half_size: i64,
    sigma: f64,
    fwhm_mm: f64,
}

fn gaussian_kernel_new() -> GaussianKernel1D {
    GaussianKernel1D {
        weights: [0.0; 15],
        size: 1,
        half_size: 0,
        sigma: 1.0,
        fwhm_mm: 2.355,
    }
}

/// Create Gaussian kernel from FWHM (standard neuroimaging parameterization)
/// FWHM = 2.355 * sigma (Full Width at Half Maximum)
fn gaussian_kernel_from_fwhm(fwhm_mm: f64, voxel_size_mm: f64) -> GaussianKernel1D {
    var kernel = gaussian_kernel_new()
    kernel.fwhm_mm = fwhm_mm

    // FWHM to sigma conversion
    kernel.sigma = fwhm_mm / (2.355 * voxel_size_mm)

    // Kernel size: 3 sigma on each side, rounded to odd
    var size = (6.0 * kernel.sigma + 1.0) as i64
    if size % 2 == 0 {
        size = size + 1
    }
    if size > MAX_KERNEL_SIZE() {
        size = MAX_KERNEL_SIZE()
    }
    if size < 3 {
        size = 3
    }

    kernel.size = size
    kernel.half_size = size / 2

    // Generate weights
    var sum = 0.0
    var i: i64 = 0
    while i < size {
        let x = (i - kernel.half_size) as f64
        let w = exp(-0.5 * x * x / (kernel.sigma * kernel.sigma))
        kernel.weights[i as usize] = w
        sum = sum + w
        i = i + 1
    }

    // Normalize
    i = 0
    while i < size {
        kernel.weights[i as usize] = kernel.weights[i as usize] / sum
        i = i + 1
    }

    kernel
}

/// 3D Gaussian kernel (for non-separable reference or verification)
struct GaussianKernel3D {
    weights: [[[f64; 15]; 15]; 15],
    size_x: i64,
    size_y: i64,
    size_z: i64,
    sigma_x: f64,
    sigma_y: f64,
    sigma_z: f64,
}

fn gaussian_kernel_3d_new() -> GaussianKernel3D {
    GaussianKernel3D {
        weights: [[[0.0; 15]; 15]; 15],
        size_x: 1,
        size_y: 1,
        size_z: 1,
        sigma_x: 1.0,
        sigma_y: 1.0,
        sigma_z: 1.0,
    }
}

/// Create isotropic 3D kernel from FWHM
fn gaussian_kernel_3d_isotropic(fwhm_mm: f64, voxel_x: f64, voxel_y: f64, voxel_z: f64) -> GaussianKernel3D {
    var kernel = gaussian_kernel_3d_new()

    // Convert FWHM to sigma for each dimension
    kernel.sigma_x = fwhm_mm / (2.355 * voxel_x)
    kernel.sigma_y = fwhm_mm / (2.355 * voxel_y)
    kernel.sigma_z = fwhm_mm / (2.355 * voxel_z)

    // Size for each dimension
    var sx = (6.0 * kernel.sigma_x + 1.0) as i64
    if sx % 2 == 0 { sx = sx + 1 }
    if sx > 15 { sx = 15 }
    if sx < 3 { sx = 3 }
    kernel.size_x = sx

    var sy = (6.0 * kernel.sigma_y + 1.0) as i64
    if sy % 2 == 0 { sy = sy + 1 }
    if sy > 15 { sy = 15 }
    if sy < 3 { sy = 3 }
    kernel.size_y = sy

    var sz = (6.0 * kernel.sigma_z + 1.0) as i64
    if sz % 2 == 0 { sz = sz + 1 }
    if sz > 15 { sz = 15 }
    if sz < 3 { sz = 3 }
    kernel.size_z = sz

    // Generate 3D weights
    var sum = 0.0
    let hx = sx / 2
    let hy = sy / 2
    let hz = sz / 2

    var iz: i64 = 0
    while iz < sz {
        var iy: i64 = 0
        while iy < sy {
            var ix: i64 = 0
            while ix < sx {
                let dx = (ix - hx) as f64
                let dy = (iy - hy) as f64
                let dz = (iz - hz) as f64

                let w = exp(-0.5 * (
                    dx * dx / (kernel.sigma_x * kernel.sigma_x) +
                    dy * dy / (kernel.sigma_y * kernel.sigma_y) +
                    dz * dz / (kernel.sigma_z * kernel.sigma_z)
                ))

                kernel.weights[iz as usize][iy as usize][ix as usize] = w
                sum = sum + w

                ix = ix + 1
            }
            iy = iy + 1
        }
        iz = iz + 1
    }

    // Normalize
    iz = 0
    while iz < sz {
        var iy: i64 = 0
        while iy < sy {
            var ix: i64 = 0
            while ix < sx {
                kernel.weights[iz as usize][iy as usize][ix as usize] =
                    kernel.weights[iz as usize][iy as usize][ix as usize] / sum
                ix = ix + 1
            }
            iy = iy + 1
        }
        iz = iz + 1
    }

    kernel
}

// ============================================================================
// GPU KERNELS: SEPARABLE CONVOLUTION
// ============================================================================

/// GPU kernel for 1D convolution along X axis
/// Uses shared memory for coalesced access and kernel weight reuse
kernel fn convolve_x_kernel(
    input: &[f64],
    output: &![f64],
    kernel_weights: &[f64],
    dim_x: i32,
    dim_y: i32,
    dim_z: i32,
    kernel_size: i32,
    kernel_half: i32
) {
    // Shared memory for input tile + halo
    shared tile: [f64; 256]  // BLOCK_DIM_X + 2*kernel_half

    let tx = gpu.thread_id.x
    let ty = gpu.thread_id.y
    let tz = gpu.thread_id.z

    let gx = gpu.block_id.x * gpu.block_dim.x + tx
    let gy = gpu.block_id.y * gpu.block_dim.y + ty
    let gz = gpu.block_id.z * gpu.block_dim.z + tz

    // Linearized block thread ID
    let block_tid = tx + ty * gpu.block_dim.x

    // Load main tile
    let tile_width = gpu.block_dim.x + 2 * kernel_half
    if block_tid < tile_width {
        let load_x = gpu.block_id.x * gpu.block_dim.x + block_tid - kernel_half

        if load_x >= 0 && load_x < dim_x && gy < dim_y && gz < dim_z {
            let idx = gz * dim_y * dim_x + gy * dim_x + load_x
            tile[block_tid] = input[idx]
        } else {
            tile[block_tid] = 0.0  // Zero-padding at boundaries
        }
    }
    gpu.sync()

    // Compute convolution
    if gx < dim_x && gy < dim_y && gz < dim_z {
        var sum = 0.0
        var k: i32 = 0
        while k < kernel_size {
            let tile_idx = tx + k
            sum = sum + tile[tile_idx] * kernel_weights[k]
            k = k + 1
        }

        let out_idx = gz * dim_y * dim_x + gy * dim_x + gx
        output[out_idx] = sum
    }
}

/// GPU kernel for 1D convolution along Y axis
kernel fn convolve_y_kernel(
    input: &[f64],
    output: &![f64],
    kernel_weights: &[f64],
    dim_x: i32,
    dim_y: i32,
    dim_z: i32,
    kernel_size: i32,
    kernel_half: i32
) {
    shared tile: [f64; 256]

    let tx = gpu.thread_id.x
    let ty = gpu.thread_id.y
    let tz = gpu.thread_id.z

    let gx = gpu.block_id.x * gpu.block_dim.x + tx
    let gy = gpu.block_id.y * gpu.block_dim.y + ty
    let gz = gpu.block_id.z * gpu.block_dim.z + tz

    let block_tid = ty + tx * gpu.block_dim.y
    let tile_height = gpu.block_dim.y + 2 * kernel_half

    if block_tid < tile_height {
        let load_y = gpu.block_id.y * gpu.block_dim.y + block_tid - kernel_half

        if gx < dim_x && load_y >= 0 && load_y < dim_y && gz < dim_z {
            let idx = gz * dim_y * dim_x + load_y * dim_x + gx
            tile[block_tid] = input[idx]
        } else {
            tile[block_tid] = 0.0
        }
    }
    gpu.sync()

    if gx < dim_x && gy < dim_y && gz < dim_z {
        var sum = 0.0
        var k: i32 = 0
        while k < kernel_size {
            sum = sum + tile[ty + k] * kernel_weights[k]
            k = k + 1
        }

        let out_idx = gz * dim_y * dim_x + gy * dim_x + gx
        output[out_idx] = sum
    }
}

/// GPU kernel for 1D convolution along Z axis
kernel fn convolve_z_kernel(
    input: &[f64],
    output: &![f64],
    kernel_weights: &[f64],
    dim_x: i32,
    dim_y: i32,
    dim_z: i32,
    kernel_size: i32,
    kernel_half: i32
) {
    shared tile: [f64; 64]

    let tx = gpu.thread_id.x
    let ty = gpu.thread_id.y
    let tz = gpu.thread_id.z

    let gx = gpu.block_id.x * gpu.block_dim.x + tx
    let gy = gpu.block_id.y * gpu.block_dim.y + ty
    let gz = gpu.block_id.z * gpu.block_dim.z + tz

    let block_tid = tz
    let tile_depth = gpu.block_dim.z + 2 * kernel_half

    if block_tid < tile_depth {
        let load_z = gpu.block_id.z * gpu.block_dim.z + block_tid - kernel_half

        if gx < dim_x && gy < dim_y && load_z >= 0 && load_z < dim_z {
            let idx = load_z * dim_y * dim_x + gy * dim_x + gx
            tile[block_tid] = input[idx]
        } else {
            tile[block_tid] = 0.0
        }
    }
    gpu.sync()

    if gx < dim_x && gy < dim_y && gz < dim_z {
        var sum = 0.0
        var k: i32 = 0
        while k < kernel_size {
            sum = sum + tile[tz + k] * kernel_weights[k]
            k = k + 1
        }

        let out_idx = gz * dim_y * dim_x + gy * dim_x + gx
        output[out_idx] = sum
    }
}

// ============================================================================
// GPU KERNEL: BATCH SMOOTHING FOR 4D fMRI
// ============================================================================

/// Smooth single volume (launched per timepoint)
kernel fn smooth_volume_kernel(
    input: &[f64],          // Single 3D volume
    temp: &![f64],          // Temporary buffer
    output: &![f64],        // Output volume
    kernel_x: &[f64],
    kernel_y: &[f64],
    kernel_z: &[f64],
    dim_x: i32,
    dim_y: i32,
    dim_z: i32,
    k_size: i32,
    k_half: i32
) {
    let gx = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x
    let gy = gpu.block_id.y * gpu.block_dim.y + gpu.thread_id.y
    let gz = gpu.block_id.z * gpu.block_dim.z + gpu.thread_id.z

    if gx < dim_x && gy < dim_y && gz < dim_z {
        // X convolution
        var sum_x = 0.0
        var kx: i32 = 0
        while kx < k_size {
            let ix = gx + kx - k_half
            if ix >= 0 && ix < dim_x {
                let idx = gz * dim_y * dim_x + gy * dim_x + ix
                sum_x = sum_x + input[idx] * kernel_x[kx]
            }
            kx = kx + 1
        }

        let out_idx = gz * dim_y * dim_x + gy * dim_x + gx
        temp[out_idx] = sum_x
    }
    gpu.sync()

    if gx < dim_x && gy < dim_y && gz < dim_z {
        // Y convolution on temp
        var sum_y = 0.0
        var ky: i32 = 0
        while ky < k_size {
            let iy = gy + ky - k_half
            if iy >= 0 && iy < dim_y {
                let idx = gz * dim_y * dim_x + iy * dim_x + gx
                sum_y = sum_y + temp[idx] * kernel_y[ky]
            }
            ky = ky + 1
        }

        let out_idx = gz * dim_y * dim_x + gy * dim_x + gx
        output[out_idx] = sum_y
    }
    gpu.sync()

    if gx < dim_x && gy < dim_y && gz < dim_z {
        // Z convolution on output (in-place would need extra temp)
        var sum_z = 0.0
        var kz: i32 = 0
        while kz < k_size {
            let iz = gz + kz - k_half
            if iz >= 0 && iz < dim_z {
                let idx = iz * dim_y * dim_x + gy * dim_x + gx
                sum_z = sum_z + output[idx] * kernel_z[kz]
            }
            kz = kz + 1
        }

        let out_idx = gz * dim_y * dim_x + gy * dim_x + gx
        temp[out_idx] = sum_z
    }
    gpu.sync()

    // Copy back to output
    if gx < dim_x && gy < dim_y && gz < dim_z {
        let idx = gz * dim_y * dim_x + gy * dim_x + gx
        output[idx] = temp[idx]
    }
}

// ============================================================================
// HOST-SIDE SMOOTHING INTERFACE
// ============================================================================

/// Smoothing configuration
struct SmoothConfig {
    fwhm_mm: f64,
    voxel_x: f64,
    voxel_y: f64,
    voxel_z: f64,
    use_gpu: bool,
}

fn smooth_config_fmri(fwhm_mm: f64) -> SmoothConfig {
    SmoothConfig {
        fwhm_mm: fwhm_mm,
        voxel_x: 2.0,  // Typical fMRI voxel size
        voxel_y: 2.0,
        voxel_z: 2.0,
        use_gpu: true,
    }
}

fn smooth_config_custom(fwhm_mm: f64, vx: f64, vy: f64, vz: f64) -> SmoothConfig {
    SmoothConfig {
        fwhm_mm: fwhm_mm,
        voxel_x: vx,
        voxel_y: vy,
        voxel_z: vz,
        use_gpu: true,
    }
}

/// CPU reference: 3D Gaussian smoothing (separable)
fn smooth_3d_separable_cpu(
    input: &[f64; 1000000],
    output: &![f64; 1000000],
    dim_x: i64,
    dim_y: i64,
    dim_z: i64,
    kernel: &GaussianKernel1D
) {
    let n_voxels = dim_x * dim_y * dim_z
    var temp1: [f64; 1000000] = [0.0; 1000000]
    var temp2: [f64; 1000000] = [0.0; 1000000]

    let k_size = kernel.size
    let k_half = kernel.half_size

    // X convolution
    var z: i64 = 0
    while z < dim_z {
        var y: i64 = 0
        while y < dim_y {
            var x: i64 = 0
            while x < dim_x {
                var sum = 0.0
                var k: i64 = 0
                while k < k_size {
                    let ix = x + k - k_half
                    if ix >= 0 && ix < dim_x {
                        let idx = z * dim_y * dim_x + y * dim_x + ix
                        sum = sum + input[idx as usize] * kernel.weights[k as usize]
                    }
                    k = k + 1
                }
                let out_idx = z * dim_y * dim_x + y * dim_x + x
                temp1[out_idx as usize] = sum
                x = x + 1
            }
            y = y + 1
        }
        z = z + 1
    }

    // Y convolution
    z = 0
    while z < dim_z {
        var y: i64 = 0
        while y < dim_y {
            var x: i64 = 0
            while x < dim_x {
                var sum = 0.0
                var k: i64 = 0
                while k < k_size {
                    let iy = y + k - k_half
                    if iy >= 0 && iy < dim_y {
                        let idx = z * dim_y * dim_x + iy * dim_x + x
                        sum = sum + temp1[idx as usize] * kernel.weights[k as usize]
                    }
                    k = k + 1
                }
                let out_idx = z * dim_y * dim_x + y * dim_x + x
                temp2[out_idx as usize] = sum
                x = x + 1
            }
            y = y + 1
        }
        z = z + 1
    }

    // Z convolution
    z = 0
    while z < dim_z {
        var y: i64 = 0
        while y < dim_y {
            var x: i64 = 0
            while x < dim_x {
                var sum = 0.0
                var k: i64 = 0
                while k < k_size {
                    let iz = z + k - k_half
                    if iz >= 0 && iz < dim_z {
                        let idx = iz * dim_y * dim_x + y * dim_x + x
                        sum = sum + temp2[idx as usize] * kernel.weights[k as usize]
                    }
                    k = k + 1
                }
                let out_idx = z * dim_y * dim_x + y * dim_x + x
                output[out_idx as usize] = sum
                x = x + 1
            }
            y = y + 1
        }
        z = z + 1
    }
}

/// Smooth 4D fMRI data (CPU reference)
fn smooth_4d_cpu(
    input: &[f64; 10000000],
    output: &![f64; 10000000],
    dim_x: i64,
    dim_y: i64,
    dim_z: i64,
    n_volumes: i64,
    kernel: &GaussianKernel1D
) {
    let vol_size = dim_x * dim_y * dim_z

    var t: i64 = 0
    while t < n_volumes {
        let offset = t * vol_size

        // Extract volume (simplified - would use slicing in real impl)
        var vol_in: [f64; 1000000] = [0.0; 1000000]
        var vol_out: [f64; 1000000] = [0.0; 1000000]

        var i: i64 = 0
        while i < vol_size && i < 1000000 {
            vol_in[i as usize] = input[(offset + i) as usize]
            i = i + 1
        }

        // Smooth
        smooth_3d_separable_cpu(&vol_in, &!vol_out, dim_x, dim_y, dim_z, kernel)

        // Copy back
        i = 0
        while i < vol_size && i < 1000000 {
            output[(offset + i) as usize] = vol_out[i as usize]
            i = i + 1
        }

        t = t + 1
    }
}

// ============================================================================
// MASK-AWARE SMOOTHING
// ============================================================================

/// Smooth within brain mask only (preserves sharp mask edges)
fn smooth_masked_cpu(
    input: &[f64; 1000000],
    mask: &[bool; 1000000],
    output: &![f64; 1000000],
    dim_x: i64,
    dim_y: i64,
    dim_z: i64,
    kernel: &GaussianKernel1D
) {
    let k_size = kernel.size
    let k_half = kernel.half_size

    var z: i64 = 0
    while z < dim_z {
        var y: i64 = 0
        while y < dim_y {
            var x: i64 = 0
            while x < dim_x {
                let center_idx = z * dim_y * dim_x + y * dim_x + x

                if !mask[center_idx as usize] {
                    output[center_idx as usize] = 0.0
                    x = x + 1
                    continue
                }

                // Weighted average within mask
                var sum = 0.0
                var weight_sum = 0.0

                var kz: i64 = 0
                while kz < k_size {
                    let iz = z + kz - k_half
                    if iz < 0 || iz >= dim_z { kz = kz + 1; continue }

                    var ky: i64 = 0
                    while ky < k_size {
                        let iy = y + ky - k_half
                        if iy < 0 || iy >= dim_y { ky = ky + 1; continue }

                        var kx: i64 = 0
                        while kx < k_size {
                            let ix = x + kx - k_half
                            if ix < 0 || ix >= dim_x { kx = kx + 1; continue }

                            let idx = iz * dim_y * dim_x + iy * dim_x + ix
                            if mask[idx as usize] {
                                let w = kernel.weights[kx as usize] *
                                        kernel.weights[ky as usize] *
                                        kernel.weights[kz as usize]
                                sum = sum + input[idx as usize] * w
                                weight_sum = weight_sum + w
                            }

                            kx = kx + 1
                        }
                        ky = ky + 1
                    }
                    kz = kz + 1
                }

                if weight_sum > 0.0 {
                    output[center_idx as usize] = sum / weight_sum
                } else {
                    output[center_idx as usize] = input[center_idx as usize]
                }

                x = x + 1
            }
            y = y + 1
        }
        z = z + 1
    }
}

// ============================================================================
// GPU KERNEL: MASKED SMOOTHING
// ============================================================================

/// GPU kernel for mask-aware Gaussian smoothing
kernel fn smooth_masked_kernel(
    input: &[f64],
    mask: &[i32],           // 1 = brain, 0 = outside
    output: &![f64],
    kernel_weights: &[f64], // Flattened 3D kernel
    dim_x: i32,
    dim_y: i32,
    dim_z: i32,
    k_size: i32,
    k_half: i32
) {
    let gx = gpu.block_id.x * gpu.block_dim.x + gpu.thread_id.x
    let gy = gpu.block_id.y * gpu.block_dim.y + gpu.thread_id.y
    let gz = gpu.block_id.z * gpu.block_dim.z + gpu.thread_id.z

    if gx >= dim_x || gy >= dim_y || gz >= dim_z {
        return
    }

    let center_idx = gz * dim_y * dim_x + gy * dim_x + gx

    if mask[center_idx] == 0 {
        output[center_idx] = 0.0
        return
    }

    var sum = 0.0
    var weight_sum = 0.0

    var kz: i32 = 0
    while kz < k_size {
        let iz = gz + kz - k_half
        if iz < 0 || iz >= dim_z { kz = kz + 1; continue }

        var ky: i32 = 0
        while ky < k_size {
            let iy = gy + ky - k_half
            if iy < 0 || iy >= dim_y { ky = ky + 1; continue }

            var kx: i32 = 0
            while kx < k_size {
                let ix = gx + kx - k_half
                if ix < 0 || ix >= dim_x { kx = kx + 1; continue }

                let idx = iz * dim_y * dim_x + iy * dim_x + ix
                if mask[idx] != 0 {
                    let k_idx = kz * k_size * k_size + ky * k_size + kx
                    let w = kernel_weights[k_idx]
                    sum = sum + input[idx] * w
                    weight_sum = weight_sum + w
                }

                kx = kx + 1
            }
            ky = ky + 1
        }
        kz = kz + 1
    }

    if weight_sum > 0.0 {
        output[center_idx] = sum / weight_sum
    } else {
        output[center_idx] = input[center_idx]
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_kernel_normalization() -> bool {
    // Kernel weights should sum to 1.0
    let kernel = gaussian_kernel_from_fwhm(6.0, 2.0)  // 6mm FWHM, 2mm voxels

    var sum = 0.0
    var i: i64 = 0
    while i < kernel.size {
        sum = sum + kernel.weights[i as usize]
        i = i + 1
    }

    fabs(sum - 1.0) < 1e-10
}

fn test_kernel_symmetry() -> bool {
    // Kernel should be symmetric
    let kernel = gaussian_kernel_from_fwhm(8.0, 2.0)

    var symmetric = true
    var i: i64 = 0
    while i < kernel.size / 2 {
        let j = kernel.size - 1 - i
        if fabs(kernel.weights[i as usize] - kernel.weights[j as usize]) > 1e-10 {
            symmetric = false
        }
        i = i + 1
    }

    symmetric
}

fn test_smoothing_identity() -> bool {
    // Delta function should become Gaussian
    var input: [f64; 1000000] = [0.0; 1000000]
    var output: [f64; 1000000] = [0.0; 1000000]

    // 5x5x5 volume with central spike
    let dim: i64 = 5
    let center = dim * dim * (dim / 2) + dim * (dim / 2) + (dim / 2)
    input[center as usize] = 1.0

    let kernel = gaussian_kernel_from_fwhm(4.0, 2.0)
    smooth_3d_separable_cpu(&input, &!output, dim, dim, dim, &kernel)

    // Output should be non-zero at center
    output[center as usize] > 0.0
}

fn test_smoothing_preserves_sum() -> bool {
    // Total intensity should be preserved
    var input: [f64; 1000000] = [0.0; 1000000]
    var output: [f64; 1000000] = [0.0; 1000000]

    let dim: i64 = 10
    let n = dim * dim * dim

    // Fill with 1s
    var i: i64 = 0
    while i < n {
        input[i as usize] = 1.0
        i = i + 1
    }

    let kernel = gaussian_kernel_from_fwhm(4.0, 2.0)
    smooth_3d_separable_cpu(&input, &!output, dim, dim, dim, &kernel)

    // Sum should be preserved (within boundary effects)
    var sum_in = 0.0
    var sum_out = 0.0
    i = 0
    while i < n {
        sum_in = sum_in + input[i as usize]
        sum_out = sum_out + output[i as usize]
        i = i + 1
    }

    // Allow 10% difference due to boundary effects
    fabs(sum_in - sum_out) / sum_in < 0.1
}

fn main() -> i32 {
    print("Testing gpu::smooth module...\n")

    if !test_kernel_normalization() {
        print("FAIL: kernel_normalization\n")
        return 1
    }
    print("PASS: kernel_normalization\n")

    if !test_kernel_symmetry() {
        print("FAIL: kernel_symmetry\n")
        return 2
    }
    print("PASS: kernel_symmetry\n")

    if !test_smoothing_identity() {
        print("FAIL: smoothing_identity\n")
        return 3
    }
    print("PASS: smoothing_identity\n")

    if !test_smoothing_preserves_sum() {
        print("FAIL: smoothing_preserves_sum\n")
        return 4
    }
    print("PASS: smoothing_preserves_sum\n")

    print("All gpu::smooth tests PASSED\n")
    0
}
