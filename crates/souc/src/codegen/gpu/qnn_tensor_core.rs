//! GPU Tensor Core Quaternion Operations (WMMA/MFMA mapping)
//!
//! This module provides abstractions for mapping quaternion operations to GPU tensor cores:
//! - NVIDIA WMMA (Warp Matrix Multiply-Accumulate) for FP16/BF16
//! - AMD MFMA (Matrix Fused Multiply-Add) for RDNA/CDNA
//!
//! A 4×4 quaternion tile maps to 16×16 WMMA fragments with optimized Hamilton products.

use std::vec::Vec;

/// Quaternion tile representation (row-major storage)
/// For dim=4: 16 quaternions × 4 components = 64 f32
#[derive(Clone, Debug, Default)]
pub struct QuatTile {
    pub dim: i32,        // Tile dimension in quaternions
    pub quats: Vec<f32>, // Flattened data
}

impl QuatTile {
    pub fn new(dim: i32) -> Self {
        let size = (dim as usize).pow(2) * 4;
        Self {
            dim,
            quats: vec![0.0; size],
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        let expected = (self.dim as usize).pow(2) * 4;
        if self.quats.len() != expected {
            return Err(format!(
                "Invalid size: expected {}, got {}",
                expected,
                self.quats.len()
            ));
        }
        Ok(())
    }

    pub fn from_vec(dim: i32, data: Vec<f32>) -> Result<Self, String> {
        let tile = Self { dim, quats: data };
        tile.validate()?;
        Ok(tile)
    }
}

/// WMMA Fragment (16×16 matrix, row-major)
#[derive(Clone, Copy, Debug)]
pub struct WmmaFragment {
    data: [f32; 256], // 16 × 16
}

impl Default for WmmaFragment {
    fn default() -> Self {
        Self { data: [0.0; 256] }
    }
}

impl WmmaFragment {
    pub fn zero() -> Self {
        Self { data: [0.0; 256] }
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * 16 + col]
    }

    #[inline]
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * 16 + col] = val;
    }

    pub fn clear(&mut self) {
        self.data = [0.0; 256];
    }
}

/// Quaternion WMMA fragments (w, x, y, z components)
#[derive(Clone, Debug, Default)]
pub struct QuatWMMAFragment {
    pub w_frag: WmmaFragment,
    pub x_frag: WmmaFragment,
    pub y_frag: WmmaFragment,
    pub z_frag: WmmaFragment,
}

impl QuatWMMAFragment {
    pub fn zero() -> Self {
        Self {
            w_frag: WmmaFragment::zero(),
            x_frag: WmmaFragment::zero(),
            y_frag: WmmaFragment::zero(),
            z_frag: WmmaFragment::zero(),
        }
    }

    pub fn clear(&mut self) {
        self.w_frag.clear();
        self.x_frag.clear();
        self.y_frag.clear();
        self.z_frag.clear();
    }
}

/// Load quaternion matrix A into WMMA fragments
/// Extracts 4×4 components from row-major quaternion data
pub fn quat_wmma_load_a(mem: &[f32], stride: i32, row_offset: i32) -> QuatWMMAFragment {
    let mut frag = QuatWMMAFragment::zero();
    let dim = 4usize;
    let stride = stride as usize;

    for comp in 0..4 {
        let comp_frag = match comp {
            0 => &mut frag.w_frag,
            1 => &mut frag.x_frag,
            2 => &mut frag.y_frag,
            3 => &mut frag.z_frag,
            _ => unreachable!(),
        };

        for m in 0..dim {
            let global_m = row_offset as usize + m;
            let row_start = global_m * stride;

            for k in 0..dim {
                let idx = row_start + k * 4 + comp;
                let val = if idx < mem.len() { mem[idx] } else { 0.0 };
                comp_frag.set(m, k, val);
            }
        }
    }

    frag
}

/// Load quaternion matrix B into WMMA fragments
pub fn quat_wmma_load_b(mem: &[f32], stride: i32, col_offset: i32) -> QuatWMMAFragment {
    let mut frag = QuatWMMAFragment::zero();
    let dim = 4usize;
    let stride = stride as usize;

    for comp in 0..4 {
        let comp_frag = match comp {
            0 => &mut frag.w_frag,
            1 => &mut frag.x_frag,
            2 => &mut frag.y_frag,
            3 => &mut frag.z_frag,
            _ => unreachable!(),
        };

        for k in 0..dim {
            let global_k = col_offset as usize + k;
            let row_start = global_k * stride;

            for n in 0..dim {
                let idx = row_start + n * 4 + comp;
                let val = if idx < mem.len() { mem[idx] } else { 0.0 };
                comp_frag.set(k, n, val);
            }
        }
    }

    frag
}

/// Store WMMA result fragments back to quaternion memory
pub fn quat_wmma_store_c(frag: &QuatWMMAFragment, mem: &mut [f32], stride: i32) {
    let dim = 4usize;
    let stride = stride as usize;

    for m in 0..dim {
        let row_start = m * stride;
        for n in 0..dim {
            let base = row_start + n * 4;
            if base + 4 <= mem.len() {
                mem[base + 0] = frag.w_frag.get(m, n);
                mem[base + 1] = frag.x_frag.get(m, n);
                mem[base + 2] = frag.y_frag.get(m, n);
                mem[base + 3] = frag.z_frag.get(m, n);
            }
        }
    }
}

/// WMMA multiply-accumulate: C += A * B (with optional negation)
fn wmma_mma(c: &mut WmmaFragment, a: &WmmaFragment, b: &WmmaFragment, negate: bool) {
    let sign = if negate { -1.0 } else { 1.0 };

    // Simulates WMMA.MMA instruction on Tensor Core
    // In GPU codegen: Maps to mma.sync.aligned.m16n16k16.row.col.f32.f32
    for m in 0..16 {
        for n in 0..16 {
            let mut acc = 0.0f32;
            for k in 0..16 {
                acc += a.get(m, k) * (b.get(k, n) * sign);
            }
            c.set(m, n, c.get(m, n) + acc);
        }
    }
}

/// Compute quaternion matrix product using WMMA tiles
/// Implements: C = A ⊗ B (Hamilton product for quaternions)
///
/// Performance: Each tile multiply = 16 WMMA ops = ~8192 FLOPS/op
/// Total: ~131k hardware FLOPS per tile (4×4 quats)
pub fn quat_tile_multiply_wmma(a: &QuatTile, b: &QuatTile) -> QuatTile {
    a.validate().unwrap();
    b.validate().unwrap();

    let stride = a.dim * 4;

    // Load tiles into WMMA fragments
    let a_frags = quat_wmma_load_a(&a.quats, stride, 0);
    let b_frags = quat_wmma_load_b(&b.quats, stride, 0);

    // Initialize result accumulators
    let mut result_w = WmmaFragment::zero();
    let mut result_x = WmmaFragment::zero();
    let mut result_y = WmmaFragment::zero();
    let mut result_z = WmmaFragment::zero();

    // Hamilton product: 16 WMMA operations (4 per component)
    // w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    wmma_mma(&mut result_w, &a_frags.w_frag, &b_frags.w_frag, false);
    wmma_mma(&mut result_w, &a_frags.x_frag, &b_frags.x_frag, true);
    wmma_mma(&mut result_w, &a_frags.y_frag, &b_frags.y_frag, true);
    wmma_mma(&mut result_w, &a_frags.z_frag, &b_frags.z_frag, true);

    // x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    wmma_mma(&mut result_x, &a_frags.w_frag, &b_frags.x_frag, false);
    wmma_mma(&mut result_x, &a_frags.x_frag, &b_frags.w_frag, false);
    wmma_mma(&mut result_x, &a_frags.y_frag, &b_frags.z_frag, false);
    wmma_mma(&mut result_x, &a_frags.z_frag, &b_frags.y_frag, true);

    // y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    wmma_mma(&mut result_y, &a_frags.w_frag, &b_frags.y_frag, false);
    wmma_mma(&mut result_y, &a_frags.x_frag, &b_frags.z_frag, true);
    wmma_mma(&mut result_y, &a_frags.y_frag, &b_frags.w_frag, false);
    wmma_mma(&mut result_y, &a_frags.z_frag, &b_frags.x_frag, false);

    // z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    wmma_mma(&mut result_z, &a_frags.w_frag, &b_frags.z_frag, false);
    wmma_mma(&mut result_z, &a_frags.x_frag, &b_frags.y_frag, false);
    wmma_mma(&mut result_z, &a_frags.y_frag, &b_frags.x_frag, true);
    wmma_mma(&mut result_z, &a_frags.z_frag, &b_frags.w_frag, false);

    // Store result back to memory
    let mut result = QuatTile::new(a.dim);
    let result_frag = QuatWMMAFragment {
        w_frag: result_w,
        x_frag: result_x,
        y_frag: result_y,
        z_frag: result_z,
    };
    quat_wmma_store_c(&result_frag, &mut result.quats, stride);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wmma_fragment() {
        let mut frag = WmmaFragment::zero();
        frag.set(0, 0, 1.5);
        assert!((frag.get(0, 0) - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_quat_tile_creation() {
        let tile = QuatTile::new(4);
        assert_eq!(tile.dim, 4);
        assert_eq!(tile.quats.len(), 64);
    }

    #[test]
    fn test_quat_tile_multiply() {
        // Identity quaternion [1, 0, 0, 0] tiled
        let mut a = QuatTile::new(4);
        for i in 0..4 {
            a.quats[i * 4] = 1.0; // w = 1 for all quats
        }

        let mut b = QuatTile::new(4);
        for i in 0..4 {
            b.quats[i * 4] = 0.707; // w
            b.quats[i * 4 + 1] = 0.707; // x
        }

        let result = quat_tile_multiply_wmma(&a, &b);

        // Identity * q = q, so check first quaternion
        assert!((result.quats[0] - 0.707).abs() < 1e-4);
        assert!((result.quats[1] - 0.707).abs() < 1e-4);
    }
}
