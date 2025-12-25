// fmri::nifti — NIfTI File Format Support for Neuroimaging
//
// Native support for neuroimaging volumetric data.
// NIfTI stores 3D/4D brain volumes with spatial metadata.
//
// References:
// - NIfTI-1 spec: https://nifti.nimh.nih.gov/nifti-1

// ============================================================================
// MATH HELPERS (inline implementations)
// ============================================================================

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    var y = x
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y = (y + x / y) / 2.0
    y
}

// ============================================================================
// NIFTI IMAGE STRUCTURES
// ============================================================================

/// NIfTI image with data (16^3 max)
struct NiftiImage {
    nx: i64,
    ny: i64,
    nz: i64,
    nvox: i64,
    dx: f64,
    dy: f64,
    dz: f64,
    tr: f64,
    xform: [[f64; 4]; 4],
    data: [f64; 4096],
}

fn nifti_image_new() -> NiftiImage {
    NiftiImage {
        nx: 0,
        ny: 0,
        nz: 0,
        nvox: 0,
        dx: 1.0,
        dy: 1.0,
        dz: 1.0,
        tr: 2.0,
        xform: [[0.0; 4]; 4],
        data: [0.0; 4096],
    }
}

/// Create image with dimensions
fn nifti_create(nx: i64, ny: i64, nz: i64) -> NiftiImage {
    var img = nifti_image_new()
    img.nx = nx
    img.ny = ny
    img.nz = nz
    img.nvox = nx * ny * nz
    img.xform[0][0] = 1.0
    img.xform[1][1] = 1.0
    img.xform[2][2] = 1.0
    img.xform[3][3] = 1.0
    img
}

/// Get voxel index
fn voxel_idx(nx: i64, ny: i64, x: i64, y: i64, z: i64) -> i64 {
    x + y * nx + z * nx * ny
}

/// Get voxel value
fn get_voxel(img: &NiftiImage, x: i64, y: i64, z: i64) -> f64 {
    if x < 0 || x >= img.nx || y < 0 || y >= img.ny || z < 0 || z >= img.nz {
        return 0.0
    }
    let idx = voxel_idx(img.nx, img.ny, x, y, z)
    img.data[idx as usize]
}

/// Voxel to world coordinates
fn voxel_to_world(img: &NiftiImage, i: f64, j: f64, k: f64) -> (f64, f64, f64) {
    let x = img.xform[0][0] * i + img.xform[0][3]
    let y = img.xform[1][1] * j + img.xform[1][3]
    let z = img.xform[2][2] * k + img.xform[2][3]
    (x, y, z)
}

// ============================================================================
// STATISTICS
// ============================================================================

fn array_mean_5(data: [f64; 5]) -> f64 {
    var sum = 0.0
    sum = data[0] + data[1] + data[2] + data[3] + data[4]
    sum / 5.0
}

fn array_std_5(data: [f64; 5]) -> f64 {
    let mean = array_mean_5(data)
    var sum_sq = 0.0
    var i: i64 = 0
    while i < 5 {
        let diff = data[i as usize] - mean
        sum_sq = sum_sq + diff * diff
        i = i + 1
    }
    sqrt_f64(sum_sq / 4.0)
}

// ============================================================================
// TESTS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn test_array_stats() -> bool {
    var data: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    let mean = array_mean_5(data)
    if abs_f64(mean - 3.0) > 0.01 {
        return false
    }
    let std = array_std_5(data)
    if abs_f64(std - 1.58) > 0.1 {
        return false
    }
    true
}

fn test_voxel_idx() -> bool {
    let idx = voxel_idx(10, 10, 5, 5, 5)
    idx == 555
}

fn main() -> i32 {
    print("Testing fmri::nifti module...\n")

    if !test_array_stats() {
        print("FAIL: array_stats\n")
        return 1
    }
    print("PASS: array_stats\n")

    if !test_voxel_idx() {
        print("FAIL: voxel_idx\n")
        return 2
    }
    print("PASS: voxel_idx\n")

    print("All fmri::nifti tests PASSED\n")
    0
}
