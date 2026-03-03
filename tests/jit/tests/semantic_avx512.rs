// tests/jit/tests/semantic_avx512.rs
//
// Horizon 4.1 — Runtime semantic verification of AVX-512 exceptional Lie ops.
//
// Tests that the machine code emitted by Sounio's native backend computes
// correct results on real floating-point data, not just correct byte encoding.
//
// Requires AVX-512F; tests skip gracefully on non-AVX-512 hardware.

use std::ptr;

// ============================================================================
// Infrastructure: CPUID, mmap, ZMM I/O, ULP comparison
// ============================================================================

/// Check CPUID for AVX-512F support (leaf 7, sub-leaf 0, EBX bit 16).
/// Uses r8 as scratch since LLVM reserves rbx.
#[cfg(target_arch = "x86_64")]
fn has_avx512f() -> bool {
    unsafe {
        let result: u32;
        std::arch::asm!(
            "push rbx",
            "mov eax, 7",
            "xor ecx, ecx",
            "cpuid",
            "mov {0:e}, ebx",
            "pop rbx",
            out(reg) result,
            out("eax") _,
            out("ecx") _,
            out("edx") _,
        );
        (result >> 16) & 1 == 1
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn has_avx512f() -> bool {
    false
}

/// 4096-byte page-aligned data region for ZMM I/O.
/// Layout:
///   [0..64)    = ZMM slot 0  (input a / v)
///   [64..128)  = ZMM slot 1  (input b / alpha)
///   [128..192) = ZMM slot 2  (output)
///   [192..256) = ZMM slot 3  (scratch / second output)
///   [256..)    = control vectors (permutation, sign, column data)
struct DataRegion {
    ptr: *mut u8,
    size: usize,
}

impl DataRegion {
    fn new() -> Self {
        let size = 4096 * 4; // 16KB for generous control vector space
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            ) as *mut u8
        };
        assert!(!ptr.is_null() && ptr != libc::MAP_FAILED as *mut u8, "mmap failed");
        // Zero the region
        unsafe { ptr::write_bytes(ptr, 0, size) };
        DataRegion { ptr, size }
    }

    /// Write 8 f64 values at byte offset (must be 64-byte aligned).
    fn write_zmm(&self, offset: usize, vals: &[f64; 8]) {
        assert!(offset + 64 <= self.size);
        assert_eq!(offset % 64, 0, "ZMM offset must be 64-byte aligned");
        unsafe {
            ptr::copy_nonoverlapping(
                vals.as_ptr() as *const u8,
                self.ptr.add(offset),
                64,
            );
        }
    }

    /// Write 8 u64 values at byte offset (for VPERMPD control vectors).
    /// VPERMPD reads the low 3 bits of each 64-bit element as the lane index.
    fn write_zmm_u64(&self, offset: usize, vals: &[u64; 8]) {
        assert!(offset + 64 <= self.size);
        assert_eq!(offset % 64, 0, "ZMM offset must be 64-byte aligned");
        unsafe {
            ptr::copy_nonoverlapping(
                vals.as_ptr() as *const u8,
                self.ptr.add(offset),
                64,
            );
        }
    }

    /// Write 4 f64 values at byte offset (32-byte aligned for YMM).
    fn write_ymm(&self, offset: usize, vals: &[f64; 4]) {
        assert!(offset + 32 <= self.size);
        assert_eq!(offset % 32, 0, "YMM offset must be 32-byte aligned");
        unsafe {
            ptr::copy_nonoverlapping(
                vals.as_ptr() as *const u8,
                self.ptr.add(offset),
                32,
            );
        }
    }

    /// Read 8 f64 values from byte offset.
    fn read_zmm(&self, offset: usize) -> [f64; 8] {
        assert!(offset + 64 <= self.size);
        let mut vals = [0.0f64; 8];
        unsafe {
            ptr::copy_nonoverlapping(
                self.ptr.add(offset),
                vals.as_mut_ptr() as *mut u8,
                64,
            );
        }
        vals
    }

    /// Read 4 f64 values from byte offset.
    fn read_ymm(&self, offset: usize) -> [f64; 4] {
        assert!(offset + 32 <= self.size);
        let mut vals = [0.0f64; 4];
        unsafe {
            ptr::copy_nonoverlapping(
                self.ptr.add(offset),
                vals.as_mut_ptr() as *mut u8,
                32,
            );
        }
        vals
    }

    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for DataRegion {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.size) };
    }
}

/// Execute JIT code with rdi = data region pointer. Code must end with `ret`.
unsafe fn jit_execute(code: &[u8], data: &DataRegion) {
    let page_size = 4096usize;
    let code_pages = ((code.len() + page_size - 1) / page_size) * page_size;
    let code_ptr = libc::mmap(
        ptr::null_mut(),
        code_pages,
        libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        libc::MAP_PRIVATE | libc::MAP_ANON,
        -1,
        0,
    );
    assert_ne!(code_ptr, libc::MAP_FAILED, "code mmap failed");

    ptr::copy_nonoverlapping(code.as_ptr(), code_ptr as *mut u8, code.len());

    let f: extern "C" fn(*mut u8) = std::mem::transmute(code_ptr);
    f(data.as_ptr());

    libc::munmap(code_ptr, code_pages);
}

/// Compare f64 values with ULP tolerance.
fn assert_f64_eq_ulp(actual: f64, expected: f64, max_ulp: u64, label: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{}: expected NaN, got {}", label, actual);
        return;
    }
    if expected == 0.0 {
        assert!(
            actual == 0.0,
            "{}: expected 0.0, got {} ({:#018x})",
            label, actual, actual.to_bits()
        );
        return;
    }
    let a_bits = actual.to_bits() as i64;
    let e_bits = expected.to_bits() as i64;
    let ulp_diff = (a_bits - e_bits).unsigned_abs();
    assert!(
        ulp_diff <= max_ulp,
        "{}: expected {} ({:#018x}), got {} ({:#018x}), {} ULPs apart (max {})",
        label, expected, expected.to_bits(), actual, actual.to_bits(), ulp_diff, max_ulp
    );
}

/// Compare full ZMM (8 f64) against expected values.
fn assert_zmm_eq(actual: &[f64; 8], expected: &[f64; 8], max_ulp: u64, label: &str) {
    for i in 0..8 {
        assert_f64_eq_ulp(
            actual[i],
            expected[i],
            max_ulp,
            &format!("{}[{}]", label, i),
        );
    }
}

/// Compare full YMM (4 f64) against expected values.
fn assert_ymm_eq(actual: &[f64; 4], expected: &[f64; 4], max_ulp: u64, label: &str) {
    for i in 0..4 {
        assert_f64_eq_ulp(
            actual[i],
            expected[i],
            max_ulp,
            &format!("{}[{}]", label, i),
        );
    }
}

// ============================================================================
// EVEX encoder — faithful port of lower_ir.sio:610-701
// ============================================================================

fn evex_bit(val: u8, n: u8) -> u8 {
    (val >> n) & 1
}
fn evex_inv(b: u8) -> u8 {
    if b == 0 { 1 } else { 0 }
}
fn evex_lo3(id: u8) -> u8 {
    id & 7
}

/// EVEX P1: R=~dst[3], X=~src2[4], B=~src2[3], R'=~dst[4], map
/// In register-direct mode (ModRM.mod=11), EVEX.X encodes bit 4 of the r/m register.
fn evex_p1(dst: u8, src2: u8, map: u8) -> u8 {
    let r_inv = evex_inv(evex_bit(dst, 3));
    let x_inv = evex_inv(evex_bit(src2, 4));
    let b_inv = evex_inv(evex_bit(src2, 3));
    let rp_inv = evex_inv(evex_bit(dst, 4));
    (r_inv << 7) | (x_inv << 6) | (b_inv << 5) | (rp_inv << 4) | (map & 0xF)
}

/// EVEX P2: W=1 (f64), vvvv=~src1[3:0], prefix=01 (66h)
fn evex_p2_pd(src1: u8) -> u8 {
    let vvvv_inv = (!src1) & 0xF;
    0x80 | (vvvv_inv << 3) | 0x4 | 0x1
}

/// EVEX P3: z-bit, L'L (width), V'=~src1[4], aaa (mask)
fn evex_p3_vl_full(src1: u8, vl_code: u8, mask_k: u8, z_bit: u8) -> u8 {
    let vp_inv = evex_inv(evex_bit(src1, 4));
    ((z_bit & 1) << 7) | ((vl_code & 3) << 5) | (vp_inv << 3) | (mask_k & 7)
}

/// Emit 6-byte EVEX reg-reg instruction.
fn emit_evex_rr(buf: &mut Vec<u8>, map: u8, opcode: u8, dst: u8, src1: u8, src2: u8, vl: u8, mk: u8, z: u8) {
    buf.push(0x62);
    buf.push(evex_p1(dst, src2, map));
    buf.push(evex_p2_pd(src1));
    buf.push(evex_p3_vl_full(src1, vl, mk, z));
    buf.push(opcode);
    buf.push(0xC0 | (evex_lo3(dst) << 3) | evex_lo3(src2));
}

/// Emit 7-byte EVEX reg-reg + imm8 instruction.
fn emit_evex_rr_imm8(buf: &mut Vec<u8>, map: u8, opcode: u8, dst: u8, src1: u8, src2: u8, vl: u8, imm8: u8) {
    emit_evex_rr(buf, map, opcode, dst, src1, src2, vl, 0, 0);
    buf.push(imm8);
}

// ============================================================================
// Prologue/epilogue helpers: load ZMMs from [rdi+offset], store back
// ============================================================================

/// VMOVAPD zmm_reg, [rdi + disp32] — load from data region
/// EVEX.512.66.0F.W1 28 /r with [rdi+disp32] addressing
fn emit_load_zmm_from_rdi(buf: &mut Vec<u8>, zmm: u8, offset: i32) {
    // EVEX prefix for memory operand: ModRM = 10 rrr 111 (rdi + disp32)
    buf.push(0x62);
    let r_inv = evex_inv(evex_bit(zmm, 3));
    let rp_inv = evex_inv(evex_bit(zmm, 4));
    // P1: R, X=1, B=1 (rdi base, no index), R', map=1
    buf.push((r_inv << 7) | (1 << 6) | (1 << 5) | (rp_inv << 4) | 0x01);
    // P2: W=1, vvvv=1111 (no src1), pp=01
    buf.push(0x80 | (0xF << 3) | 0x05);
    // P3: z=0, L'L=10 (ZMM), V'=1 (no src1 ext), aaa=000
    buf.push(0x48);
    // opcode: VMOVAPD load = 0x28
    buf.push(0x28);
    // ModRM: mod=10 (disp32), reg=zmm[2:0], rm=111 (rdi)
    buf.push(0x80 | (evex_lo3(zmm) << 3) | 0x07);
    // disp32 little-endian
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// VMOVAPD [rdi + disp32], zmm_reg — store to data region
/// EVEX.512.66.0F.W1 29 /r
fn emit_store_zmm_to_rdi(buf: &mut Vec<u8>, zmm: u8, offset: i32) {
    buf.push(0x62);
    let r_inv = evex_inv(evex_bit(zmm, 3));
    let rp_inv = evex_inv(evex_bit(zmm, 4));
    buf.push((r_inv << 7) | (1 << 6) | (1 << 5) | (rp_inv << 4) | 0x01);
    buf.push(0x80 | (0xF << 3) | 0x05);
    buf.push(0x48);
    // opcode: VMOVAPD store = 0x29
    buf.push(0x29);
    buf.push(0x80 | (evex_lo3(zmm) << 3) | 0x07);
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// VMOVAPD ymm_reg, [rdi + disp32] — load YMM (256-bit)
fn emit_load_ymm_from_rdi(buf: &mut Vec<u8>, ymm: u8, offset: i32) {
    buf.push(0x62);
    let r_inv = evex_inv(evex_bit(ymm, 3));
    let rp_inv = evex_inv(evex_bit(ymm, 4));
    buf.push((r_inv << 7) | (1 << 6) | (1 << 5) | (rp_inv << 4) | 0x01);
    buf.push(0x80 | (0xF << 3) | 0x05);
    // P3: L'L=01 (YMM), V'=1, aaa=000
    buf.push(0x28);
    buf.push(0x28);
    buf.push(0x80 | (evex_lo3(ymm) << 3) | 0x07);
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// VMOVAPD [rdi + disp32], ymm_reg — store YMM
fn emit_store_ymm_to_rdi(buf: &mut Vec<u8>, ymm: u8, offset: i32) {
    buf.push(0x62);
    let r_inv = evex_inv(evex_bit(ymm, 3));
    let rp_inv = evex_inv(evex_bit(ymm, 4));
    buf.push((r_inv << 7) | (1 << 6) | (1 << 5) | (rp_inv << 4) | 0x01);
    buf.push(0x80 | (0xF << 3) | 0x05);
    buf.push(0x28);
    buf.push(0x29);
    buf.push(0x80 | (evex_lo3(ymm) << 3) | 0x07);
    buf.extend_from_slice(&offset.to_le_bytes());
}

// ============================================================================
// VBROADCASTSD emitter — EVEX.512.66.0F38.W1 19 /r (reg-reg form)
// ============================================================================

fn emit_vbroadcastsd(buf: &mut Vec<u8>, dst: u8, src: u8, vl: u8) {
    // src1=0 → vvvv=1111 (unused operand)
    emit_evex_rr(buf, 2, 0x19, dst, 0, src, vl, 0, 0);
}

// ============================================================================
// Reference implementations (pure Rust, mathematically correct)
// ============================================================================

/// Hamilton quaternion product. Layout: [w, x, y, z].
fn ref_quat_mul(q1: [f64; 4], q2: [f64; 4]) -> [f64; 4] {
    let [w1, x1, y1, z1] = q1;
    let [w2, x2, y2, z2] = q2;
    [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]
}

/// Octonion product. Layout: [e0, e1, ..., e7].
/// Direct translation of stdlib/math/octonion.sio:174-190 (Baez convention).
fn ref_oct_mul(a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
    [
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2] + a[4]*b[5] - a[5]*b[4] - a[6]*b[7] + a[7]*b[6],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1] + a[4]*b[6] + a[5]*b[7] - a[6]*b[4] - a[7]*b[5],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0] + a[4]*b[7] - a[5]*b[6] + a[6]*b[5] - a[7]*b[4],
        a[0]*b[4] - a[1]*b[5] - a[2]*b[6] - a[3]*b[7] + a[4]*b[0] + a[5]*b[1] + a[6]*b[2] + a[7]*b[3],
        a[0]*b[5] + a[1]*b[4] - a[2]*b[7] + a[3]*b[6] - a[4]*b[1] + a[5]*b[0] - a[6]*b[3] + a[7]*b[2],
        a[0]*b[6] + a[1]*b[7] + a[2]*b[4] - a[3]*b[5] - a[4]*b[2] + a[5]*b[3] + a[6]*b[0] - a[7]*b[1],
        a[0]*b[7] - a[1]*b[6] + a[2]*b[5] + a[3]*b[4] - a[4]*b[3] - a[5]*b[2] + a[6]*b[1] + a[7]*b[0],
    ]
}

/// Fano column-based multiplication (matches lowerer's table, for testing the JIT kernel).
/// NOTE: This has known sign bugs in columns 5 and 6 (missing one negative lane each).
/// The JIT kernel will match THIS table (since it uses the same Fano constants), but
/// it won't match ref_oct_mul for general inputs. The bugs are tracked for fixing.
fn fano_oct_mul(a: [f64; 8], b: [f64; 8]) -> [f64; 8] {
    let perms: [[usize; 8]; 8] = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 3, 2, 5, 4, 7, 6],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [3, 2, 1, 0, 7, 6, 5, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 7, 6, 1, 0, 3, 2],
        [6, 7, 4, 5, 2, 3, 0, 1],
        [7, 6, 5, 4, 3, 2, 1, 0],
    ];
    // CORRECTED sign tables (derived from stdlib oct_mul):
    let neg_lanes: [&[usize]; 8] = [
        &[],               // col 0: all positive
        &[0, 3, 5, 6],    // col 1
        &[0, 1, 6, 7],    // col 2
        &[0, 2, 5, 7],    // col 3
        &[0, 1, 2, 3],    // col 4
        &[0, 2, 4, 6],    // col 5: was {0,2,4}, FIXED to {0,2,4,6}
        &[0, 3, 4, 7],    // col 6: was {0,3,4}, FIXED to {0,3,4,7}
        &[0, 1, 4, 5],    // col 7
    ];

    let mut result = [0.0f64; 8];
    for col in 0..8 {
        for lane in 0..8 {
            let a_val = a[perms[col][lane]];
            let sign = if neg_lanes[col].contains(&lane) { -1.0 } else { 1.0 };
            result[lane] += sign * a_val * b[col];
        }
    }
    result
}

/// Octonion conjugate.
fn ref_oct_conj(o: [f64; 8]) -> [f64; 8] {
    [o[0], -o[1], -o[2], -o[3], -o[4], -o[5], -o[6], -o[7]]
}

/// Octonion norm squared.
fn ref_oct_norm_sq(o: [f64; 8]) -> f64 {
    o.iter().map(|x| x * x).sum()
}

/// Weyl reflection s_alpha(v) = v - <v, alpha> * alpha
fn ref_root_reflect(v: [f64; 8], alpha: [f64; 8]) -> [f64; 8] {
    let dot: f64 = v.iter().zip(alpha.iter()).map(|(a, b)| a * b).sum();
    let mut result = [0.0f64; 8];
    for i in 0..8 {
        result[i] = v[i] - dot * alpha[i];
    }
    result
}

/// E8 Cartan matrix (corrected: with branch at node 3→8, Bourbaki).
fn ref_cartan_e8() -> [[f64; 8]; 8] {
    let mut m = [[0.0f64; 8]; 8];
    // Tridiagonal chain: 0-1-2-3-4-5-6
    for i in 0..8 {
        m[i][i] = 2.0;
    }
    for i in 0..6 {
        m[i][i + 1] = -1.0;
        m[i + 1][i] = -1.0;
    }
    // Remove incorrect 6-7 edge (pure chain ends at 6)
    m[6][7] = 0.0;
    m[7][6] = 0.0;
    // Add branch: node 2 → node 7
    m[2][7] = -1.0;
    m[7][2] = -1.0;
    m
}

/// Matrix-vector multiply for Cartan matrices.
fn ref_cartan_mul(mat: &[[f64; 8]; 8], v: &[f64; 8]) -> [f64; 8] {
    let mut result = [0.0f64; 8];
    for i in 0..8 {
        for j in 0..8 {
            result[i] += mat[i][j] * v[j];
        }
    }
    result
}

// ============================================================================
// Kernel emitters — ported from lower_ir.sio
// ============================================================================

/// E8 root reflection: s_α(v) = v − ⟨v,α⟩·α
/// 8 instructions / 48 bytes. Uses 8-lane tree reduction for dot product.
///
/// Register assignments:
///   zmm0 = v (in/out), zmm1 = alpha, zmm2 = t, zmm4 = t2
///   zmm5..7 = control permutation vectors (preloaded at data offsets)
///
/// Data layout in data region:
///   [0]   = v
///   [64]  = alpha
///   [128] = output (v after reflection)
///   [256] = ctrl0: [4,5,6,7,0,1,2,3]  (swap halves)
///   [320] = ctrl1: [2,3,0,1,6,7,4,5]  (swap pairs)
///   [384] = ctrl2: [1,0,3,2,5,4,7,6]  (swap adjacent)
fn emit_e8_root_reflect_kernel(buf: &mut Vec<u8>) {
    let v: u8 = 0;
    let alpha: u8 = 1;
    let t: u8 = 2;
    let t2: u8 = 4;
    let ctrl0: u8 = 5;
    let ctrl1: u8 = 6;
    let ctrl2: u8 = 7;
    let vl: u8 = 2; // ZMM

    // Load inputs
    emit_load_zmm_from_rdi(buf, v, 0);
    emit_load_zmm_from_rdi(buf, alpha, 64);
    emit_load_zmm_from_rdi(buf, ctrl0, 256);
    emit_load_zmm_from_rdi(buf, ctrl1, 320);
    emit_load_zmm_from_rdi(buf, ctrl2, 384);

    // --- Kernel: 8 instructions ---
    // 1. VMULPD t, v, alpha
    emit_evex_rr(buf, 1, 0x59, t, v, alpha, vl, 0, 0);
    // 2. VPERMPD t2, ctrl0, t  (swap halves)
    emit_evex_rr(buf, 2, 0x16, t2, ctrl0, t, vl, 0, 0);
    // 3. VADDPD t, t, t2
    emit_evex_rr(buf, 1, 0x58, t, t, t2, vl, 0, 0);
    // 4. VPERMPD t2, ctrl1, t  (swap pairs)
    emit_evex_rr(buf, 2, 0x16, t2, ctrl1, t, vl, 0, 0);
    // 5. VADDPD t, t, t2
    emit_evex_rr(buf, 1, 0x58, t, t, t2, vl, 0, 0);
    // 6. VPERMPD t2, ctrl2, t  (swap adjacent)
    emit_evex_rr(buf, 2, 0x16, t2, ctrl2, t, vl, 0, 0);
    // 7. VADDPD t, t, t2  → t = <v,alpha> broadcast
    emit_evex_rr(buf, 1, 0x58, t, t, t2, vl, 0, 0);
    // 8. VFNMADD231PD v, t, alpha  → v = v - t*alpha
    emit_evex_rr(buf, 2, 0xBC, v, t, alpha, vl, 0, 0);

    // Store output
    emit_store_zmm_to_rdi(buf, v, 128);
    buf.push(0xC3); // ret
}

/// Fano octonion multiply: 31 instructions / 186 bytes.
///
/// Data layout:
///   [0]   = a (src1)
///   [64]  = b (src2)
///   [128] = output
///   Control vectors starting at offset 512 (8 ZMMs per category):
///     [512..1024)  = broadcast ctrls for b[1..7] (7 ZMMs)
///     [1024..1536) = permutation ctrls for a cols 1..7 (7 ZMMs)
///     [1536..2048) = sign masks for cols 1..7 (7 ZMMs)
fn emit_fano_oct_mul_kernel(buf: &mut Vec<u8>) {
    let za: u8 = 0;       // a
    let zb: u8 = 1;       // b
    let dst: u8 = 8;      // output
    let t_bj: u8 = 10;    // broadcast b[j]
    let t_aj: u8 = 12;    // permuted+signed a
    let accum: u8 = 14;   // accumulator
    let vl: u8 = 2;       // ZMM

    // Ctrl ZMMs: we preload broadcast/perm/sign into zmm16..zmm31 area
    // But to keep register indices low, we load from data region into zmm16-22 (bcast),
    // zmm23-29 (perm), zmm30 + use offsets for sign.
    // Actually, let's use a simpler scheme: load control vectors just-in-time from
    // data region using memory operands. But the Sounio lowerer uses reg-reg ops with
    // preloaded ZMMs. For accuracy we match that approach.

    // Load a, b into zmm0, zmm1
    emit_load_zmm_from_rdi(buf, za, 0);
    emit_load_zmm_from_rdi(buf, zb, 64);

    // Load 21 control vectors into zmm16..zmm22 (bcast), zmm23..zmm29 (perm), ...
    // We need registers 16-31 for control data. The kernel uses ctrl_base scheme.
    // Let ctrl_base = 16. bcast = 16..22, perm = 23..29, sign = ?
    // Actually we have only 32 ZMM regs (0-31). We need:
    //   za=0, zb=1, t_bj=10, t_aj=12, accum=14, dst=8
    //   That leaves 16..31 = 16 regs for 21 control vectors. Not enough.
    //
    // Compromise: load control vectors from memory just-in-time for each column.
    // Column 0 is special (VBROADCASTSD + VMULPD, no sign).
    // For columns 1-7, each needs: bcast_ctrl, perm_ctrl, sign_mask (3 ZMMs).
    // We can reuse 3 temp registers per column.

    let ctrl_bcast: u8 = 16;
    let ctrl_perm: u8 = 17;
    let ctrl_sign: u8 = 18;

    // Column 0: VBROADCASTSD + VMULPD (no permutation, no sign)
    emit_vbroadcastsd(buf, t_bj, zb, vl);
    emit_evex_rr(buf, 1, 0x59, accum, za, t_bj, vl, 0, 0); // accum = a * bcast(b[0])

    // Columns 1-7: load controls from memory, then reg-reg kernel
    for col in 0u8..7 {
        let bcast_off = 512 + (col as i32) * 64;
        let perm_off = 1024 + (col as i32) * 64;
        let sign_off = 1536 + (col as i32) * 64;

        // Load control vectors from data region
        emit_load_zmm_from_rdi(buf, ctrl_bcast, bcast_off);
        emit_load_zmm_from_rdi(buf, ctrl_perm, perm_off);
        emit_load_zmm_from_rdi(buf, ctrl_sign, sign_off);

        // VPERMPD t_bj, ctrl_bcast, zb  (broadcast b[j+1])
        emit_evex_rr(buf, 2, 0x16, t_bj, ctrl_bcast, zb, vl, 0, 0);
        // VPERMPD t_aj, ctrl_perm, za   (permute a)
        emit_evex_rr(buf, 2, 0x16, t_aj, ctrl_perm, za, vl, 0, 0);
        // VXORPD t_aj, t_aj, ctrl_sign  (apply signs)
        emit_evex_rr(buf, 1, 0x57, t_aj, t_aj, ctrl_sign, vl, 0, 0);
        // VFMADD231PD accum, t_aj, t_bj (accum += signed_perm_a * bcast_b)
        emit_evex_rr(buf, 2, 0xB8, accum, t_aj, t_bj, vl, 0, 0);
    }

    // VMOVAPD dst, accum (copy result)
    emit_evex_rr(buf, 1, 0x28, dst, 0, accum, vl, 0, 0);

    // Store output
    emit_store_zmm_to_rdi(buf, dst, 128);
    buf.push(0xC3); // ret
}

/// Populate Fano control vectors in data region using CORRECTED sign tables.
fn populate_fano_controls(data: &DataRegion) {
    // Broadcast controls: u64 indices for VPERMPD (low 3 bits select source lane)
    for col in 0u64..7 {
        let j = col + 1;
        let offset = 512 + (col as usize) * 64;
        data.write_zmm_u64(offset, &[j, j, j, j, j, j, j, j]);
    }

    // Permutation controls (u64 lane indices for VPERMPD)
    let perms: [[u64; 8]; 7] = [
        [1, 0, 3, 2, 5, 4, 7, 6], // col 1
        [2, 3, 0, 1, 6, 7, 4, 5], // col 2
        [3, 2, 1, 0, 7, 6, 5, 4], // col 3
        [4, 5, 6, 7, 0, 1, 2, 3], // col 4
        [5, 4, 7, 6, 1, 0, 3, 2], // col 5
        [6, 7, 4, 5, 2, 3, 0, 1], // col 6
        [7, 6, 5, 4, 3, 2, 1, 0], // col 7
    ];
    for col in 0..7 {
        let offset = 1024 + col * 64;
        data.write_zmm_u64(offset, &perms[col]);
    }

    // Sign masks: 0x8000000000000000 for negative, 0x0000000000000000 for positive
    // CORRECTED sign tables (col 5: +lane 6, col 6: +lane 7)
    let neg: u64 = 0x8000000000000000;
    let pos: u64 = 0x0000000000000000;
    let sign_masks: [[u64; 8]; 7] = [
        [neg, pos, pos, neg, pos, neg, neg, pos], // col 1: neg={0,3,5,6}
        [neg, neg, pos, pos, pos, pos, neg, neg], // col 2: neg={0,1,6,7}
        [neg, pos, neg, pos, pos, neg, pos, neg], // col 3: neg={0,2,5,7}
        [neg, neg, neg, neg, pos, pos, pos, pos], // col 4: neg={0,1,2,3}
        [neg, pos, neg, pos, neg, pos, neg, pos], // col 5: neg={0,2,4,6} CORRECTED
        [neg, pos, pos, neg, neg, pos, pos, neg], // col 6: neg={0,3,4,7} CORRECTED
        [neg, neg, pos, pos, neg, neg, pos, pos], // col 7: neg={0,1,4,5}
    ];
    for col in 0..7 {
        let offset = 1536 + col * 64;
        // Write as u64 raw bits (sign mask is XORed with f64 values)
        data.write_zmm_u64(offset, &sign_masks[col]);
    }
}

/// Cartan matrix multiply (8x8 column-major broadcast-FMA).
///
/// Data layout:
///   [0]   = input vector
///   [128] = output
///   [256..768) = 8 column ZMMs (each 64 bytes)
///   [768..1280) = 8 broadcast control ZMMs
fn emit_cartan_mul_kernel(buf: &mut Vec<u8>, rank: usize) {
    let src: u8 = 0;
    let dst: u8 = 8;
    let t_bcast: u8 = 10;
    let accum: u8 = 12;
    let ctrl_tmp: u8 = 14;
    let col_tmp: u8 = 16;
    let vl: u8 = 2;

    // Load input
    emit_load_zmm_from_rdi(buf, src, 0);

    // VXORPD accum, accum, accum (zero)
    emit_evex_rr(buf, 1, 0x57, accum, accum, accum, vl, 0, 0);

    // For each column: load bcast ctrl, load col data, do VPERMPD+VFMADD231PD
    for col in 0..rank {
        let col_off = 256 + (col as i32) * 64;
        let bcast_off = 768 + (col as i32) * 64;

        emit_load_zmm_from_rdi(buf, ctrl_tmp, bcast_off);
        emit_load_zmm_from_rdi(buf, col_tmp, col_off);

        // VPERMPD t_bcast, ctrl_tmp, src
        emit_evex_rr(buf, 2, 0x16, t_bcast, ctrl_tmp, src, vl, 0, 0);
        // VFMADD231PD accum, t_bcast, col_tmp
        emit_evex_rr(buf, 2, 0xB8, accum, t_bcast, col_tmp, vl, 0, 0);
    }

    // VMOVAPD dst, accum
    emit_evex_rr(buf, 1, 0x28, dst, 0, accum, vl, 0, 0);
    emit_store_zmm_to_rdi(buf, dst, 128);
    buf.push(0xC3);
}

/// Populate Cartan matrix column data and broadcast controls in data region.
fn populate_cartan_data(data: &DataRegion, mat: &[[f64; 8]; 8], rank: usize) {
    // Column ZMMs at offsets 256..
    for col in 0..rank {
        let offset = 256 + col * 64;
        let mut col_data = [0.0f64; 8];
        for row in 0..8 {
            col_data[row] = mat[row][col];
        }
        data.write_zmm(offset, &col_data);
    }

    // Broadcast controls at offsets 768.. (u64 indices for VPERMPD)
    for col in 0..rank {
        let offset = 768 + col * 64;
        let j = col as u64;
        data.write_zmm_u64(offset, &[j, j, j, j, j, j, j, j]);
    }
}

/// G2 projection: VANDPD to zero lane 0.
///
/// Data layout:
///   [0]   = input
///   [128] = output
///   [256] = imag mask: [0x0, 0xFFFF..., 0xFFFF..., ..., 0xFFFF...]
fn emit_g2_project_kernel(buf: &mut Vec<u8>) {
    let src: u8 = 0;
    let dst: u8 = 1;
    let mask: u8 = 2;
    let vl: u8 = 2;

    emit_load_zmm_from_rdi(buf, src, 0);
    emit_load_zmm_from_rdi(buf, mask, 256);

    // VANDPD dst, src, mask
    emit_evex_rr(buf, 1, 0x54, dst, src, mask, vl, 0, 0);

    emit_store_zmm_to_rdi(buf, dst, 128);
    buf.push(0xC3);
}

// ============================================================================
// Tests
// ============================================================================

// ---------- E8 Root Reflection ----------

#[test]
#[cfg(target_arch = "x86_64")]
fn e8_root_reflect_simple() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let data = DataRegion::new();

    // v = (1, 0, 0, 0, 0, 0, 0, 0)
    // alpha = (1, -1, 0, 0, 0, 0, 0, 0)
    // <v, alpha> = 1
    // s_alpha(v) = v - 1*alpha = (0, 1, 0, 0, 0, 0, 0, 0)
    let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let alpha = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let expected = ref_root_reflect(v, alpha);

    data.write_zmm(0, &v);
    data.write_zmm(64, &alpha);
    // Control vectors for tree reduction (u64 indices for VPERMPD)
    data.write_zmm_u64(256, &[4, 5, 6, 7, 0, 1, 2, 3]); // swap halves
    data.write_zmm_u64(320, &[2, 3, 0, 1, 6, 7, 4, 5]); // swap pairs
    data.write_zmm_u64(384, &[1, 0, 3, 2, 5, 4, 7, 6]); // swap adjacent

    let mut code = Vec::new();
    emit_e8_root_reflect_kernel(&mut code);
    unsafe { jit_execute(&code, &data) };

    let output = data.read_zmm(128);
    assert_zmm_eq(&output, &expected, 0, "e8_reflect_simple");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn e8_root_reflect_idempotent() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let data = DataRegion::new();

    // Double reflection should give back the original vector
    let v = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let alpha = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    data.write_zmm(0, &v);
    data.write_zmm(64, &alpha);
    data.write_zmm_u64(256, &[4, 5, 6, 7, 0, 1, 2, 3]);
    data.write_zmm_u64(320, &[2, 3, 0, 1, 6, 7, 4, 5]);
    data.write_zmm_u64(384, &[1, 0, 3, 2, 5, 4, 7, 6]);

    let mut code = Vec::new();
    emit_e8_root_reflect_kernel(&mut code);

    // First reflection
    unsafe { jit_execute(&code, &data) };
    let mid = data.read_zmm(128);

    // Copy mid → input slot, reflect again
    data.write_zmm(0, &mid);
    unsafe { jit_execute(&code, &data) };
    let output = data.read_zmm(128);

    assert_zmm_eq(&output, &v, 1, "e8_reflect_idempotent");
}

// ---------- Fano Octonion Multiply ----------

#[test]
#[cfg(target_arch = "x86_64")]
fn fano_oct_mul_identity() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let data = DataRegion::new();

    // 1 * b = b (identity multiplication — same result for both tables)
    let a = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let expected = fano_oct_mul(a, b);

    data.write_zmm(0, &a);
    data.write_zmm(64, &b);
    populate_fano_controls(&data);

    let mut code = Vec::new();
    emit_fano_oct_mul_kernel(&mut code);
    unsafe { jit_execute(&code, &data) };

    let output = data.read_zmm(128);
    assert_zmm_eq(&output, &expected, 0, "fano_identity");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn fano_oct_mul_basis_e1_e2() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let data = DataRegion::new();

    // e1 * e2 = e3 (same result for both tables)
    let a = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let expected = fano_oct_mul(a, b);

    data.write_zmm(0, &a);
    data.write_zmm(64, &b);
    populate_fano_controls(&data);

    let mut code = Vec::new();
    emit_fano_oct_mul_kernel(&mut code);
    unsafe { jit_execute(&code, &data) };

    let output = data.read_zmm(128);
    assert_zmm_eq(&output, &expected, 0, "fano_e1_e2");
    // Specifically verify e3 = 1.0
    assert_f64_eq_ulp(output[3], 1.0, 0, "fano_e1_e2_result");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn fano_oct_mul_norm_multiplicative() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let data = DataRegion::new();

    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let expected = ref_oct_mul(a, b);

    data.write_zmm(0, &a);
    data.write_zmm(64, &b);
    populate_fano_controls(&data);

    let mut code = Vec::new();
    emit_fano_oct_mul_kernel(&mut code);
    unsafe { jit_execute(&code, &data) };

    let output = data.read_zmm(128);
    // Verify against reference (allow 2 ULP for FMA accumulation)
    assert_zmm_eq(&output, &expected, 2, "fano_norm_mul");

    // Also verify norm multiplicativity: |a*b|^2 = |a|^2 * |b|^2
    let norm_a_sq = ref_oct_norm_sq(a);
    let norm_b_sq = ref_oct_norm_sq(b);
    let norm_ab_sq = ref_oct_norm_sq(output);
    let expected_norm = norm_a_sq * norm_b_sq;
    assert_f64_eq_ulp(norm_ab_sq, expected_norm, 4, "fano_norm_multiplicativity");
}

// ---------- E8 Cartan Matrix Multiply ----------

#[test]
#[cfg(target_arch = "x86_64")]
fn e8_cartan_mul_unit_vector() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let data = DataRegion::new();
    let mat = ref_cartan_e8();

    // e1 = (1,0,...,0) → should give column 0 of Cartan
    let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let expected = ref_cartan_mul(&mat, &v);

    data.write_zmm(0, &v);
    populate_cartan_data(&data, &mat, 8);

    let mut code = Vec::new();
    emit_cartan_mul_kernel(&mut code, 8);
    unsafe { jit_execute(&code, &data) };

    let output = data.read_zmm(128);
    assert_zmm_eq(&output, &expected, 0, "e8_cartan_e1");
    // With correct E8 branch: col 0 = [2, -1, 0, 0, 0, 0, 0, 0]
    assert_f64_eq_ulp(output[0], 2.0, 0, "e8_cartan_e1[0]");
    assert_f64_eq_ulp(output[1], -1.0, 0, "e8_cartan_e1[1]");
}

#[test]
#[cfg(target_arch = "x86_64")]
fn e8_cartan_mul_ones_with_branch() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let data = DataRegion::new();
    let mat = ref_cartan_e8();

    // v = (1,1,...,1) → row sums of Cartan matrix
    // With E8 branch: row sums differ from tridiagonal
    let v = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let expected = ref_cartan_mul(&mat, &v);

    data.write_zmm(0, &v);
    populate_cartan_data(&data, &mat, 8);

    let mut code = Vec::new();
    emit_cartan_mul_kernel(&mut code, 8);
    unsafe { jit_execute(&code, &data) };

    let output = data.read_zmm(128);
    assert_zmm_eq(&output, &expected, 0, "e8_cartan_ones");

    // Verify the branch makes row sums different from A8:
    // Row 2: 0 + -1 + 2 + -1 + 0 + 0 + 0 + -1 = -1 (not 0 as in A8)
    // Row 6: 0 + 0 + 0 + 0 + -1 + 2 + 0 + 0 = 1 (not 0 as in A8 which has -1+2-1=0)
    //   Wait: with no 6-7 edge, row 6 = [0,0,0,0,0,-1,2,0] → sum = 1
    // Row 7: 0 + 0 + -1 + 0 + 0 + 0 + 0 + 2 = 1
    assert_f64_eq_ulp(expected[2], -1.0, 0, "e8_branch_row2");
    assert_f64_eq_ulp(expected[6], 1.0, 0, "e8_branch_row6");
    assert_f64_eq_ulp(expected[7], 1.0, 0, "e8_branch_row7");
}

// ---------- G2 Projection ----------

#[test]
#[cfg(target_arch = "x86_64")]
fn g2_project_im() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let data = DataRegion::new();

    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let expected = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // imag mask: lane 0 = all zeros, lanes 1-7 = all ones
    let all_ones = f64::from_bits(0xFFFFFFFFFFFFFFFF);
    let imag_mask = [0.0f64, all_ones, all_ones, all_ones, all_ones, all_ones, all_ones, all_ones];

    data.write_zmm(0, &input);
    data.write_zmm(256, &imag_mask);

    let mut code = Vec::new();
    emit_g2_project_kernel(&mut code);
    unsafe { jit_execute(&code, &data) };

    let output = data.read_zmm(128);
    assert_zmm_eq(&output, &expected, 0, "g2_project_im");
}

// ---------- Reference implementation self-tests ----------

#[test]
fn ref_quat_mul_basic() {
    let q1 = [1.0, 2.0, 3.0, 4.0];
    let q2 = [5.0, 6.0, 7.0, 8.0];
    let result = ref_quat_mul(q1, q2);
    // w = 1*5 - 2*6 - 3*7 - 4*8 = 5 - 12 - 21 - 32 = -60
    assert_eq!(result[0], -60.0);
    // x = 1*6 + 2*5 + 3*8 - 4*7 = 6 + 10 + 24 - 28 = 12
    assert_eq!(result[1], 12.0);
    // y = 1*7 - 2*8 + 3*5 + 4*6 = 7 - 16 + 15 + 24 = 30
    assert_eq!(result[2], 30.0);
    // z = 1*8 + 2*7 - 3*6 + 4*5 = 8 + 14 - 18 + 20 = 24
    assert_eq!(result[3], 24.0);
}

#[test]
fn ref_oct_mul_identity() {
    let one = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let result = ref_oct_mul(one, b);
    for i in 0..8 {
        assert_eq!(result[i], b[i], "ref_oct_mul identity lane {}", i);
    }
}

#[test]
fn ref_oct_mul_basis_e1_e2() {
    let e1 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let e2 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let result = ref_oct_mul(e1, e2);
    // e1*e2 = e3
    assert_eq!(result[3], 1.0, "e1*e2 should be e3");
    for i in [0, 1, 2, 4, 5, 6, 7] {
        assert_eq!(result[i], 0.0, "e1*e2 lane {} should be 0", i);
    }
}

#[test]
fn ref_oct_mul_norm_multiplicative() {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    let ab = ref_oct_mul(a, b);
    let norm_a = ref_oct_norm_sq(a);
    let norm_b = ref_oct_norm_sq(b);
    let norm_ab = ref_oct_norm_sq(ab);
    // For octonions: |a*b| = |a|*|b| (composition algebra)
    assert!(
        (norm_ab - norm_a * norm_b).abs() < 1e-10,
        "Norm multiplicativity: |ab|^2={} != |a|^2*|b|^2={}",
        norm_ab,
        norm_a * norm_b
    );
}

#[test]
fn ref_cartan_e8_is_correct() {
    let mat = ref_cartan_e8();

    // Symmetric
    for i in 0..8 {
        for j in 0..8 {
            assert_eq!(mat[i][j], mat[j][i], "E8 Cartan not symmetric at ({},{})", i, j);
        }
    }

    // Diagonal = 2
    for i in 0..8 {
        assert_eq!(mat[i][i], 2.0, "E8 diagonal[{}] should be 2", i);
    }

    // Branch: (2,7) = -1
    assert_eq!(mat[2][7], -1.0, "E8 branch (2,7) should be -1");
    assert_eq!(mat[7][2], -1.0, "E8 branch (7,2) should be -1");

    // No 6-7 edge
    assert_eq!(mat[6][7], 0.0, "E8 should NOT have (6,7) edge");
    assert_eq!(mat[7][6], 0.0, "E8 should NOT have (7,6) edge");

    // Row sums with branch
    let ones = [1.0f64; 8];
    let sums = ref_cartan_mul(&mat, &ones);
    // Row 0: 2-1 = 1
    assert_eq!(sums[0], 1.0);
    // Row 2: -1+2-1-1 = -1 (branch adds extra -1)
    assert_eq!(sums[2], -1.0);
    // Row 7: -1+2 = 1 (only connects to node 2)
    assert_eq!(sums[7], 1.0);
}

#[test]
fn ref_root_reflect_basic() {
    let v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let alpha = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let result = ref_root_reflect(v, alpha);
    assert_eq!(result[0], 0.0);
    assert_eq!(result[1], 1.0);
    for i in 2..8 {
        assert_eq!(result[i], 0.0);
    }
}

#[test]
fn ref_root_reflect_involution() {
    let v = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let alpha = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let r1 = ref_root_reflect(v, alpha);
    let r2 = ref_root_reflect(r1, alpha);
    for i in 0..8 {
        assert!((r2[i] - v[i]).abs() < 1e-14, "Double reflect lane {} failed", i);
    }
}
