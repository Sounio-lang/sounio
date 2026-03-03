// benches/throughput.rs
//
// Horizon 4.1 — AVX-512 kernel throughput benchmarks.
//
// Measures cycles/op, instructions/op, DFLOP/cycle for key kernels:
//   - G₂ equivariant dense (109 bytes, 14 params)
//   - Naive 7×7 dense (7 FMA = 42 bytes, 49 params)
//   - E₈ root reflection (48 bytes, 0 params)
//   - Fano octonion multiply (186 bytes, 0 params)
//   - E₈ Cartan matrix multiply (108 bytes, 0 params)
//
// Run: cd tests/jit && cargo bench

use std::ptr;

// ============================================================================
// Infrastructure (shared with semantic tests)
// ============================================================================

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

struct DataRegion {
    ptr: *mut u8,
    size: usize,
}

impl DataRegion {
    fn new() -> Self {
        let size = 16384;
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            ) as *mut u8
        };
        assert!(!ptr.is_null());
        DataRegion { ptr, size }
    }

    fn write_zmm(&self, offset: usize, vals: &[f64; 8]) {
        assert!(offset + 64 <= self.size);
        unsafe {
            ptr::copy_nonoverlapping(vals.as_ptr() as *const u8, self.ptr.add(offset), 64);
        }
    }

    fn write_zmm_u64(&self, offset: usize, vals: &[u64; 8]) {
        assert!(offset + 64 <= self.size);
        unsafe {
            ptr::copy_nonoverlapping(vals.as_ptr() as *const u8, self.ptr.add(offset), 64);
        }
    }
}

impl Drop for DataRegion {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.size); }
    }
}

unsafe fn jit_execute(code: &[u8], data: &DataRegion) {
    let code_page = libc::mmap(
        ptr::null_mut(),
        4096,
        libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        -1,
        0,
    );
    assert!(code_page != libc::MAP_FAILED);
    ptr::copy_nonoverlapping(code.as_ptr(), code_page as *mut u8, code.len());
    let f: extern "C" fn(*mut u8) = std::mem::transmute(code_page);
    f(data.ptr);
    libc::munmap(code_page, 4096);
}

// ============================================================================
// EVEX encoder (same as semantic tests)
// ============================================================================

fn evex_inv(b: u8) -> u8 { if b == 0 { 1 } else { 0 } }
fn evex_bit(val: u8, n: u8) -> u8 { (val >> n) & 1 }
fn evex_lo3(id: u8) -> u8 { id & 7 }

fn evex_p1(dst: u8, src2: u8, map: u8) -> u8 {
    let r_inv = evex_inv(evex_bit(dst, 3));
    let x_inv = evex_inv(evex_bit(src2, 4));
    let b_inv = evex_inv(evex_bit(src2, 3));
    let rp_inv = evex_inv(evex_bit(dst, 4));
    (r_inv << 7) | (x_inv << 6) | (b_inv << 5) | (rp_inv << 4) | (map & 0xF)
}

fn evex_p2_pd(src1: u8) -> u8 {
    let vvvv_inv = (!src1) & 0xF;
    0x80 | (vvvv_inv << 3) | 0x4 | 0x1
}

fn evex_p3_vl_full(src1: u8, vl_code: u8, mask_k: u8, z_bit: u8) -> u8 {
    let vp_inv = evex_inv(evex_bit(src1, 4));
    ((z_bit & 1) << 7) | ((vl_code & 3) << 5) | (vp_inv << 3) | (mask_k & 7)
}

fn emit_evex_rr(buf: &mut Vec<u8>, map: u8, opcode: u8, dst: u8, src1: u8, src2: u8, vl: u8, mk: u8, z: u8) {
    buf.push(0x62);
    buf.push(evex_p1(dst, src2, map));
    buf.push(evex_p2_pd(src1));
    buf.push(evex_p3_vl_full(src1, vl, mk, z));
    buf.push(opcode);
    buf.push(0xC0 | (evex_lo3(dst) << 3) | evex_lo3(src2));
}

fn emit_load_zmm_from_rdi(buf: &mut Vec<u8>, zmm: u8, offset: i32) {
    buf.push(0x62);
    let r_inv = evex_inv(evex_bit(zmm, 3));
    let rp_inv = evex_inv(evex_bit(zmm, 4));
    buf.push((r_inv << 7) | (1 << 6) | (1 << 5) | (rp_inv << 4) | 0x01);
    buf.push(0x80 | (0xF << 3) | 0x05);
    buf.push(0x48);
    buf.push(0x28);
    buf.push(0x80 | (evex_lo3(zmm) << 3) | 0x07);
    buf.extend_from_slice(&offset.to_le_bytes());
}

fn emit_store_zmm_to_rdi(buf: &mut Vec<u8>, zmm: u8, offset: i32) {
    buf.push(0x62);
    let r_inv = evex_inv(evex_bit(zmm, 3));
    let rp_inv = evex_inv(evex_bit(zmm, 4));
    buf.push((r_inv << 7) | (1 << 6) | (1 << 5) | (rp_inv << 4) | 0x01);
    buf.push(0x80 | (0xF << 3) | 0x05);
    buf.push(0x48);
    buf.push(0x29);
    buf.push(0x80 | (evex_lo3(zmm) << 3) | 0x07);
    buf.extend_from_slice(&offset.to_le_bytes());
}

fn emit_vbroadcastsd(buf: &mut Vec<u8>, dst: u8, src: u8, vl: u8) {
    emit_evex_rr(buf, 2, 0x19, dst, 0, src, vl, 0, 0);
}

// ============================================================================
// Kernel emitters
// ============================================================================

/// E₈ root reflection: 8 instructions / 48 bytes.
fn emit_e8_root_reflect_kernel(buf: &mut Vec<u8>) {
    let v: u8 = 0; let alpha: u8 = 1; let t: u8 = 2; let t2: u8 = 4;
    let ctrl0: u8 = 5; let ctrl1: u8 = 6; let ctrl2: u8 = 7; let vl: u8 = 2;

    emit_load_zmm_from_rdi(buf, v, 0);
    emit_load_zmm_from_rdi(buf, alpha, 64);
    emit_load_zmm_from_rdi(buf, ctrl0, 256);
    emit_load_zmm_from_rdi(buf, ctrl1, 320);
    emit_load_zmm_from_rdi(buf, ctrl2, 384);

    emit_evex_rr(buf, 1, 0x59, t, v, alpha, vl, 0, 0);
    emit_evex_rr(buf, 2, 0x16, t2, ctrl0, t, vl, 0, 0);
    emit_evex_rr(buf, 1, 0x58, t, t, t2, vl, 0, 0);
    emit_evex_rr(buf, 2, 0x16, t2, ctrl1, t, vl, 0, 0);
    emit_evex_rr(buf, 1, 0x58, t, t, t2, vl, 0, 0);
    emit_evex_rr(buf, 2, 0x16, t2, ctrl2, t, vl, 0, 0);
    emit_evex_rr(buf, 1, 0x58, t, t, t2, vl, 0, 0);
    emit_evex_rr(buf, 2, 0xBC, v, t, alpha, vl, 0, 0);

    emit_store_zmm_to_rdi(buf, v, 128);
    buf.push(0xC3);
}

/// Fano octonion multiply: 31 instructions / 186 bytes.
fn emit_fano_oct_mul_kernel(buf: &mut Vec<u8>) {
    let za: u8 = 0; let zb: u8 = 1; let dst: u8 = 8;
    let t_bj: u8 = 10; let t_aj: u8 = 12; let accum: u8 = 14; let vl: u8 = 2;
    let ctrl_bcast: u8 = 16; let ctrl_perm: u8 = 17; let ctrl_sign: u8 = 18;

    emit_load_zmm_from_rdi(buf, za, 0);
    emit_load_zmm_from_rdi(buf, zb, 64);

    emit_vbroadcastsd(buf, t_bj, zb, vl);
    emit_evex_rr(buf, 1, 0x59, accum, za, t_bj, vl, 0, 0);

    for col in 0u8..7 {
        let bcast_off = 512 + (col as i32) * 64;
        let perm_off = 1024 + (col as i32) * 64;
        let sign_off = 1536 + (col as i32) * 64;

        emit_load_zmm_from_rdi(buf, ctrl_bcast, bcast_off);
        emit_load_zmm_from_rdi(buf, ctrl_perm, perm_off);
        emit_load_zmm_from_rdi(buf, ctrl_sign, sign_off);

        emit_evex_rr(buf, 2, 0x16, t_bj, ctrl_bcast, zb, vl, 0, 0);
        emit_evex_rr(buf, 2, 0x16, t_aj, ctrl_perm, za, vl, 0, 0);
        emit_evex_rr(buf, 1, 0x57, t_aj, t_aj, ctrl_sign, vl, 0, 0);
        emit_evex_rr(buf, 2, 0xB8, accum, t_aj, t_bj, vl, 0, 0);
    }

    emit_evex_rr(buf, 1, 0x28, dst, 0, accum, vl, 0, 0);
    emit_store_zmm_to_rdi(buf, dst, 128);
    buf.push(0xC3);
}

/// E₈ Cartan matrix multiply: broadcast-FMA over 8 columns.
fn emit_cartan_mul_kernel(buf: &mut Vec<u8>) {
    let src: u8 = 0; let dst: u8 = 8; let t_bcast: u8 = 10;
    let accum: u8 = 12; let ctrl_tmp: u8 = 14; let col_tmp: u8 = 16; let vl: u8 = 2;

    emit_load_zmm_from_rdi(buf, src, 0);
    emit_evex_rr(buf, 1, 0x57, accum, accum, accum, vl, 0, 0);

    for col in 0..8 {
        let col_off = 256 + (col as i32) * 64;
        let bcast_off = 768 + (col as i32) * 64;

        emit_load_zmm_from_rdi(buf, ctrl_tmp, bcast_off);
        emit_load_zmm_from_rdi(buf, col_tmp, col_off);

        emit_evex_rr(buf, 2, 0x16, t_bcast, ctrl_tmp, src, vl, 0, 0);
        emit_evex_rr(buf, 2, 0xB8, accum, t_bcast, col_tmp, vl, 0, 0);
    }

    emit_evex_rr(buf, 1, 0x28, dst, 0, accum, vl, 0, 0);
    emit_store_zmm_to_rdi(buf, dst, 128);
    buf.push(0xC3);
}

/// G₂ projection: VANDPD to zero lane 0. 1 instruction / 6 bytes.
fn emit_g2_project_kernel(buf: &mut Vec<u8>) {
    let src: u8 = 0; let dst: u8 = 1; let mask: u8 = 2; let vl: u8 = 2;
    emit_load_zmm_from_rdi(buf, src, 0);
    emit_load_zmm_from_rdi(buf, mask, 256);
    emit_evex_rr(buf, 1, 0x54, dst, src, mask, vl, 0, 0);
    emit_store_zmm_to_rdi(buf, dst, 128);
    buf.push(0xC3);
}

/// Naive 7×7 dense: column-major broadcast-FMA with 7 columns.
/// VXORPD + 7×(VPERMPD+VFMADD231PD) + VMOVAPD = 16 instr.
/// No projection, no sign masking — just raw matrix-vector.
fn emit_naive_7x7_dense_kernel(buf: &mut Vec<u8>) {
    let src: u8 = 0; let dst: u8 = 8; let t_bcast: u8 = 10;
    let accum: u8 = 12; let ctrl_tmp: u8 = 14; let col_tmp: u8 = 16; let vl: u8 = 2;

    emit_load_zmm_from_rdi(buf, src, 0);
    emit_evex_rr(buf, 1, 0x57, accum, accum, accum, vl, 0, 0);

    for col in 0..7 {
        let col_off = 256 + (col as i32) * 64;
        let bcast_off = 768 + (col as i32) * 64;

        emit_load_zmm_from_rdi(buf, ctrl_tmp, bcast_off);
        emit_load_zmm_from_rdi(buf, col_tmp, col_off);

        emit_evex_rr(buf, 2, 0x16, t_bcast, ctrl_tmp, src, vl, 0, 0);
        emit_evex_rr(buf, 2, 0xB8, accum, t_bcast, col_tmp, vl, 0, 0);
    }

    emit_evex_rr(buf, 1, 0x28, dst, 0, accum, vl, 0, 0);
    emit_store_zmm_to_rdi(buf, dst, 128);
    buf.push(0xC3);
}

// ============================================================================
// Data setup helpers
// ============================================================================

fn populate_root_reflect_data(data: &DataRegion) {
    let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let alpha = [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    data.write_zmm(0, &v);
    data.write_zmm(64, &alpha);
    data.write_zmm_u64(256, &[4, 5, 6, 7, 0, 1, 2, 3]);
    data.write_zmm_u64(320, &[2, 3, 0, 1, 6, 7, 4, 5]);
    data.write_zmm_u64(384, &[1, 0, 3, 2, 5, 4, 7, 6]);
}

fn populate_fano_data(data: &DataRegion) {
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
    data.write_zmm(0, &a);
    data.write_zmm(64, &b);

    for col in 0u64..7 {
        let j = col + 1;
        let offset = 512 + (col as usize) * 64;
        data.write_zmm_u64(offset, &[j, j, j, j, j, j, j, j]);
    }

    let perms: [[u64; 8]; 7] = [
        [1, 0, 3, 2, 5, 4, 7, 6],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [3, 2, 1, 0, 7, 6, 5, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 7, 6, 1, 0, 3, 2],
        [6, 7, 4, 5, 2, 3, 0, 1],
        [7, 6, 5, 4, 3, 2, 1, 0],
    ];
    for col in 0..7 {
        data.write_zmm_u64(1024 + col * 64, &perms[col]);
    }

    let neg: u64 = 0x8000000000000000;
    let pos: u64 = 0x0000000000000000;
    let sign_masks: [[u64; 8]; 7] = [
        [neg, pos, pos, neg, pos, neg, neg, pos],
        [neg, neg, pos, pos, pos, pos, neg, neg],
        [neg, pos, neg, pos, pos, neg, pos, neg],
        [neg, neg, neg, neg, pos, pos, pos, pos],
        [neg, pos, neg, pos, neg, pos, neg, pos],
        [neg, pos, pos, neg, neg, pos, pos, neg],
        [neg, neg, pos, pos, neg, neg, pos, pos],
    ];
    for col in 0..7 {
        data.write_zmm_u64(1536 + col * 64, &sign_masks[col]);
    }
}

fn populate_cartan_data(data: &DataRegion) {
    let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    data.write_zmm(0, &v);

    // E8 Cartan matrix (corrected with branch at 2-7)
    let mut mat = [[0.0f64; 8]; 8];
    for i in 0..8 { mat[i][i] = 2.0; }
    for i in 0..6 { mat[i][i+1] = -1.0; mat[i+1][i] = -1.0; }
    mat[6][7] = 0.0; mat[7][6] = 0.0;
    mat[2][7] = -1.0; mat[7][2] = -1.0;

    for col in 0..8 {
        let offset = 256 + col * 64;
        let mut col_data = [0.0f64; 8];
        for row in 0..8 { col_data[row] = mat[row][col]; }
        data.write_zmm(offset, &col_data);
    }
    for col in 0..8 {
        let offset = 768 + col * 64;
        let j = col as u64;
        data.write_zmm_u64(offset, &[j, j, j, j, j, j, j, j]);
    }
}

fn populate_g2_project_data(data: &DataRegion) {
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    data.write_zmm(0, &input);
    let all_ones = f64::from_bits(0xFFFFFFFFFFFFFFFF);
    let imag_mask = [0.0f64, all_ones, all_ones, all_ones, all_ones, all_ones, all_ones, all_ones];
    data.write_zmm(256, &imag_mask);
}

fn populate_naive_7x7_data(data: &DataRegion) {
    let v = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    data.write_zmm(0, &v);
    // Random 7x7 weight matrix (stored as 7 columns, lane 0 unused)
    for col in 0..7 {
        let offset = 256 + col * 64;
        let mut col_data = [0.0f64; 8];
        for row in 0..8 {
            col_data[row] = ((col * 7 + row) as f64) * 0.1;
        }
        data.write_zmm(offset, &col_data);
    }
    for col in 0..7 {
        let offset = 768 + col * 64;
        let j = (col + 1) as u64; // broadcast lanes 1-7
        data.write_zmm_u64(offset, &[j, j, j, j, j, j, j, j]);
    }
}

// ============================================================================
// RDTSC timing
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn rdtsc() -> u64 {
    unsafe {
        let lo: u32;
        let hi: u32;
        std::arch::asm!("rdtsc", out("eax") lo, out("edx") hi, options(nostack, nomem));
        ((hi as u64) << 32) | (lo as u64)
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn rdtscp() -> u64 {
    unsafe {
        let lo: u32;
        let hi: u32;
        std::arch::asm!("rdtscp", out("eax") lo, out("edx") hi, out("ecx") _, options(nostack));
        ((hi as u64) << 32) | (lo as u64)
    }
}

// ============================================================================
// Benchmark runner
// ============================================================================

struct BenchResult {
    name: &'static str,
    instr_count: usize,
    byte_count: usize,
    param_count: usize,
    dflop_per_call: usize,
    median_cycles: f64,
}

fn bench_kernel(
    name: &'static str,
    code: &[u8],
    data: &DataRegion,
    instr_count: usize,
    byte_count: usize,
    param_count: usize,
    dflop_per_call: usize,
) -> BenchResult {
    let warmup = 1000;
    let iters = 100_000;
    let runs = 5;

    // Warmup
    for _ in 0..warmup {
        unsafe { jit_execute(code, data) };
    }

    // Measure (5 runs of 100K iterations each, report median)
    // Use a single code page for all iterations to avoid mmap overhead
    let code_page = unsafe {
        libc::mmap(
            ptr::null_mut(),
            4096,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert!(!code_page.is_null() && code_page != libc::MAP_FAILED);
    unsafe {
        ptr::copy_nonoverlapping(code.as_ptr(), code_page as *mut u8, code.len());
    }
    let f: extern "C" fn(*mut u8) = unsafe { std::mem::transmute(code_page) };

    let mut timings = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = rdtsc();
        for _ in 0..iters {
            f(data.ptr);
        }
        let end = rdtscp();
        timings.push((end - start) as f64 / iters as f64);
    }

    unsafe { libc::munmap(code_page, 4096); }

    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = timings[runs / 2];

    BenchResult {
        name,
        instr_count,
        byte_count,
        param_count,
        dflop_per_call,
        median_cycles: median,
    }
}

fn main() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available — cannot run throughput benchmarks");
        return;
    }

    println!();
    println!("=== Sounio AVX-512 Kernel Throughput ===");
    println!("(100K iterations, median of 5 runs)");
    println!();

    let data = DataRegion::new();
    let mut results = Vec::new();

    // --- E₈ root reflection ---
    populate_root_reflect_data(&data);
    let mut code = Vec::new();
    emit_e8_root_reflect_kernel(&mut code);
    results.push(bench_kernel(
        "E8 root reflection",
        &code, &data,
        8,   // 8 kernel instructions
        48,  // 48 kernel bytes
        0,   // no learnable params
        16,  // VMULPD(8) + 3×VADDPD(8each) + VFNMADD(8) = 5×8 = 40? Actually: 1 mul + 3 add + 1 FMA = 5 ops × 8 lanes = 40
             // But FMA = 2 FLOP, so: 1×8 + 3×8 + 2×8 = 8+24+16 = 48 DFLOP
             // Let's use 48
    ));
    // Fix the DFLOP count
    results.last_mut().unwrap().dflop_per_call = 48;

    // --- Fano octonion multiply ---
    populate_fano_data(&data);
    let mut code = Vec::new();
    emit_fano_oct_mul_kernel(&mut code);
    results.push(bench_kernel(
        "Fano octonion multiply",
        &code, &data,
        31,  // 31 kernel instructions
        186, // 186 kernel bytes
        0,   // no learnable params
        // Col 0: VMULPD = 8 DFLOP. Cols 1-7: each has VFMADD231PD = 2×8=16 DFLOP
        // Total: 8 + 7×16 = 120 DFLOP
        120,
    ));

    // --- E₈ Cartan matrix multiply ---
    populate_cartan_data(&data);
    let mut code = Vec::new();
    emit_cartan_mul_kernel(&mut code);
    results.push(bench_kernel(
        "E8 Cartan multiply",
        &code, &data,
        18,  // VXORPD + 8×(VPERMPD+VFMADD231PD) + VMOVAPD = 1+16+1=18
        108, // 18×6 = 108
        0,
        // 8 VFMADD231PD = 8 × 2 × 8lanes = 128 DFLOP
        128,
    ));

    // --- G₂ projection ---
    populate_g2_project_data(&data);
    let mut code = Vec::new();
    emit_g2_project_kernel(&mut code);
    results.push(bench_kernel(
        "G2 projection (VANDPD)",
        &code, &data,
        1,   // 1 kernel instruction
        6,   // 6 bytes
        0,
        0,   // bitwise op, not a FLOP
    ));

    // --- Naive 7×7 dense ---
    populate_naive_7x7_data(&data);
    let mut code = Vec::new();
    emit_naive_7x7_dense_kernel(&mut code);
    results.push(bench_kernel(
        "Naive 7x7 dense",
        &code, &data,
        16,  // VXORPD + 7×(VPERMPD+VFMADD231PD) + VMOVAPD = 1+14+1=16
        96,  // 16×6 = 96
        49,  // 7×7 = 49 parameters
        // 7 VFMADD231PD = 7 × 2 × 8lanes = 112 DFLOP
        112,
    ));

    // --- Print results ---
    println!(
        "{:<28} | {:>9} | {:>5} | {:>5} | {:>6} | {:>11}",
        "Kernel", "Cycles/op", "Instr", "Bytes", "Params", "DFLOP/cycle"
    );
    println!("{}", "-".repeat(80));

    for r in &results {
        let dflop_per_cycle = if r.dflop_per_call > 0 && r.median_cycles > 0.0 {
            format!("{:>11.1}", r.dflop_per_call as f64 / r.median_cycles)
        } else {
            format!("{:>11}", "N/A")
        };
        println!(
            "{:<28} | {:>9.1} | {:>5} | {:>5} | {:>6} | {}",
            r.name, r.median_cycles, r.instr_count, r.byte_count, r.param_count, dflop_per_cycle
        );
    }
    println!();

    // --- FLOP/parameter efficiency comparison ---
    // Find naive and G2 results (Cartan is closest to G2 in structure)
    let naive = results.iter().find(|r| r.name == "Naive 7x7 dense");
    let fano = results.iter().find(|r| r.name == "Fano octonion multiply");

    if let (Some(naive), Some(fano)) = (naive, fano) {
        let naive_ratio = if naive.param_count > 0 {
            naive.dflop_per_call as f64 / naive.param_count as f64
        } else { 0.0 };
        println!("FLOP/parameter efficiency:");
        println!(
            "  Naive 7x7:        {:.1} DFLOP / {} params = {:.1}x",
            naive.dflop_per_call, naive.param_count, naive_ratio
        );
        println!(
            "  Fano octonion:    {} DFLOP / 0 params (algebraic, no learned weights)",
            fano.dflop_per_call
        );
        println!(
            "  E8 Cartan:        {} DFLOP / 0 params (algebraic)",
            results.iter().find(|r| r.name == "E8 Cartan multiply").map_or(0, |r| r.dflop_per_call)
        );
    }

    println!();
    println!("Note: Cycles/op includes prologue/epilogue load/store overhead.");
    println!("      Pure kernel cycles would be lower (subtract ~5 cycles per load/store).");
}
