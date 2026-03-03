// tests/jit/tests/native_e2e.rs
//
// Horizon 4.2 — End-to-end native compilation: standalone ELF programs.
//
// Builds complete statically-linked x86-64 Linux executables that:
//   1. Embed input data + control vectors in the binary
//   2. Load data into ZMM registers
//   3. Execute the AVX-512 exceptional Lie kernel
//   4. Compare results against expected values
//   5. Exit with code 0 (pass) or non-zero (which lane failed)
//
// These are REAL programs, not function calls in a test harness.
// Requires AVX-512F; tests skip on non-AVX-512 hardware.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

// Serialize ELF write+execute to avoid ETXTBSY when multiple threads
// race to execute binaries in /tmp on the same filesystem.
static ELF_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// ============================================================================
// CPUID check (same as semantic tests)
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

// ============================================================================
// EVEX encoder (same as semantic_avx512.rs)
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

/// VMOVAPD zmm, [rdi+disp32]
fn emit_load_zmm(buf: &mut Vec<u8>, zmm: u8, offset: i32) {
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

/// VMOVAPD [rdi+disp32], zmm
fn emit_store_zmm(buf: &mut Vec<u8>, zmm: u8, offset: i32) {
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

/// VBROADCASTSD zmm, xmm (reg-reg form)
fn emit_vbroadcastsd(buf: &mut Vec<u8>, dst: u8, src: u8, vl: u8) {
    emit_evex_rr(buf, 2, 0x19, dst, 0, src, vl, 0, 0);
}

// ============================================================================
// x86-64 scalar instruction helpers
// ============================================================================

/// LEA rdi, [rip + disp32]
fn emit_lea_rdi_rip(buf: &mut Vec<u8>, disp32: i32) {
    buf.extend_from_slice(&[0x48, 0x8D, 0x3D]); // REX.W LEA rdi, [rip+disp32]
    buf.extend_from_slice(&disp32.to_le_bytes());
}

/// MOV r64, imm64
fn emit_mov_r64_imm64(buf: &mut Vec<u8>, reg: u8, val: u64) {
    buf.push(0x48 | ((reg >> 3) & 1)); // REX.W (+ REX.B if reg >= 8)
    buf.push(0xB8 + (reg & 7));
    buf.extend_from_slice(&val.to_le_bytes());
}

/// CMP rax, r64
fn emit_cmp_rax_reg(buf: &mut Vec<u8>, reg: u8) {
    buf.push(0x48 | ((reg >> 3) << 2)); // REX.W + REX.R if reg >= 8
    buf.push(0x39);
    buf.push(0xC0 | ((reg & 7) << 3)); // ModRM: mod=11, reg=reg, rm=rax
}

/// XOR edi, edi (2 bytes — sets rdi = 0)
fn emit_xor_edi_edi(buf: &mut Vec<u8>) {
    buf.push(0x31);
    buf.push(0xFF);
}

/// MOV edi, imm32
fn emit_mov_edi_imm32(buf: &mut Vec<u8>, val: u32) {
    buf.push(0xBF);
    buf.extend_from_slice(&val.to_le_bytes());
}

/// MOV eax, 60; SYSCALL (exit)
fn emit_exit_syscall(buf: &mut Vec<u8>) {
    buf.push(0xB8); // mov eax, 60
    buf.extend_from_slice(&60u32.to_le_bytes());
    buf.push(0x0F); // syscall
    buf.push(0x05);
}

// ============================================================================
// ELF64 builder
// ============================================================================

const ELF_BASE: u64 = 0x400000;
const CODE_OFFSET: u64 = 0x1000; // .text starts at file offset 0x1000
const CODE_VADDR: u64 = ELF_BASE + CODE_OFFSET;

fn build_elf(code_and_data: &[u8]) -> Vec<u8> {
    let mut elf = Vec::new();
    let file_size = CODE_OFFSET as usize + code_and_data.len();
    let mem_size = file_size;

    // --- ELF header (64 bytes) ---
    elf.extend_from_slice(&[0x7F, b'E', b'L', b'F']); // e_ident magic
    elf.push(2);    // EI_CLASS: ELFCLASS64
    elf.push(1);    // EI_DATA: ELFDATA2LSB
    elf.push(1);    // EI_VERSION: EV_CURRENT
    elf.push(0);    // EI_OSABI: ELFOSABI_NONE
    elf.extend_from_slice(&[0; 8]); // EI_ABIVERSION + padding
    elf.extend_from_slice(&2u16.to_le_bytes()); // e_type: ET_EXEC
    elf.extend_from_slice(&0x3Eu16.to_le_bytes()); // e_machine: EM_X86_64
    elf.extend_from_slice(&1u32.to_le_bytes()); // e_version
    elf.extend_from_slice(&CODE_VADDR.to_le_bytes()); // e_entry
    elf.extend_from_slice(&64u64.to_le_bytes()); // e_phoff (phdr right after ehdr)
    elf.extend_from_slice(&0u64.to_le_bytes()); // e_shoff (no sections)
    elf.extend_from_slice(&0u32.to_le_bytes()); // e_flags
    elf.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
    elf.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
    elf.extend_from_slice(&1u16.to_le_bytes());  // e_phnum (1 PT_LOAD)
    elf.extend_from_slice(&64u16.to_le_bytes()); // e_shentsize
    elf.extend_from_slice(&0u16.to_le_bytes());  // e_shnum
    elf.extend_from_slice(&0u16.to_le_bytes());  // e_shstrndx
    assert_eq!(elf.len(), 64);

    // --- Program header (56 bytes) ---
    elf.extend_from_slice(&1u32.to_le_bytes()); // p_type: PT_LOAD
    elf.extend_from_slice(&7u32.to_le_bytes()); // p_flags: PF_R | PF_W | PF_X
    elf.extend_from_slice(&0u64.to_le_bytes()); // p_offset
    elf.extend_from_slice(&ELF_BASE.to_le_bytes()); // p_vaddr
    elf.extend_from_slice(&ELF_BASE.to_le_bytes()); // p_paddr
    elf.extend_from_slice(&(file_size as u64).to_le_bytes()); // p_filesz
    elf.extend_from_slice(&(mem_size as u64).to_le_bytes()); // p_memsz
    elf.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align
    assert_eq!(elf.len(), 120);

    // --- Padding to CODE_OFFSET ---
    elf.resize(CODE_OFFSET as usize, 0);

    // --- Code + embedded data ---
    elf.extend_from_slice(code_and_data);

    elf
}

// ============================================================================
// Self-verifying program builders
// ============================================================================

/// MOV rax, [rdi+disp32] — scalar 8-byte load
fn emit_mov_rax_mem_rdi(buf: &mut Vec<u8>, offset: i32) {
    buf.extend_from_slice(&[0x48, 0x8B, 0x87]); // REX.W MOV rax, [rdi+disp32]
    buf.extend_from_slice(&offset.to_le_bytes());
}

/// Build a self-verifying program that:
/// 1. Points rdi at embedded data (via LEA rdi, [rip+data_offset])
/// 2. Runs the kernel_fn (which loads from [rdi], computes, stores to [rdi+128])
/// 3. Compares each lane of the output against expected values (scalar memory loads)
/// 4. Exits 0 on success, lane+1 on first mismatch
fn build_self_verifying_program(
    kernel_fn: fn(&mut Vec<u8>),
    data: &[u8],  // 64-byte aligned data block (inputs + controls at known offsets)
    expected: &[f64; 8], // expected output (at data offset 128)
) -> Vec<u8> {
    let mut code = Vec::new();

    // Phase 1: LEA rdi, [rip + data_offset]
    let lea_pos = code.len();
    emit_lea_rdi_rip(&mut code, 0); // placeholder, 7 bytes

    // Phase 2: Kernel (loads from [rdi], computes, stores result to [rdi+128])
    kernel_fn(&mut code);

    // Phase 3: Verify each lane via scalar memory loads
    // The result is at [rdi+128]. Each f64 lane is 8 bytes apart.
    for lane in 0u8..8 {
        let expected_bits = expected[lane as usize].to_bits();
        let mem_offset = 128 + (lane as i32) * 8;

        // MOV rax, [rdi+128+lane*8]
        emit_mov_rax_mem_rdi(&mut code, mem_offset);

        // MOV rcx, expected_bits
        emit_mov_r64_imm64(&mut code, 1, expected_bits);

        // CMP rax, rcx
        emit_cmp_rax_reg(&mut code, 1);

        // JE skip_fail
        let fail_block_size: i8 = 5 + 7; // MOV edi,imm32 (5) + exit syscall (7=mov_eax_5+syscall_2)
        buf_push_je_rel8(&mut code, fail_block_size);

        // Fail: exit(lane+1)
        emit_mov_edi_imm32(&mut code, (lane as u32) + 1);
        emit_exit_syscall(&mut code);
    }

    // Phase 4: All lanes matched — exit(0)
    emit_xor_edi_edi(&mut code);
    emit_exit_syscall(&mut code);

    // Phase 5: Append data section (64-byte aligned)
    let code_len = code.len();
    let data_padding = (64 - (code_len % 64)) % 64;
    code.extend(std::iter::repeat(0x90u8).take(data_padding)); // NOP padding
    let data_offset = code.len();

    code.extend_from_slice(data);

    // Fix up the LEA rdi, [rip+data_offset]
    // RIP at time of LEA = position after LEA instruction = lea_pos + 7
    let disp32 = (data_offset as i32) - ((lea_pos as i32) + 7);
    code[lea_pos + 3..lea_pos + 7].copy_from_slice(&disp32.to_le_bytes());

    code
}

/// JE rel8 (2 bytes)
fn buf_push_je_rel8(buf: &mut Vec<u8>, rel8: i8) {
    buf.push(0x74);
    buf.push(rel8 as u8);
}

// ============================================================================
// E₈ Root Reflection program
// ============================================================================

fn build_e8_reflect_program() -> Vec<u8> {
    // Data layout (same as semantic_avx512.rs):
    //   [0]   = v = (1, 0, 0, 0, 0, 0, 0, 0)
    //   [64]  = alpha = (1, -1, 0, 0, 0, 0, 0, 0)
    //   [128] = output (written by kernel)
    //   [256] = ctrl0: [4,5,6,7,0,1,2,3]
    //   [320] = ctrl1: [2,3,0,1,6,7,4,5]
    //   [384] = ctrl2: [1,0,3,2,5,4,7,6]
    let mut data = vec![0u8; 512];

    // v = (1, 0, ...)
    write_f64s(&mut data, 0, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    // alpha = (1, -1, 0, ...)
    write_f64s(&mut data, 64, &[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    // Control vectors (u64 indices)
    write_u64s(&mut data, 256, &[4, 5, 6, 7, 0, 1, 2, 3]);
    write_u64s(&mut data, 320, &[2, 3, 0, 1, 6, 7, 4, 5]);
    write_u64s(&mut data, 384, &[1, 0, 3, 2, 5, 4, 7, 6]);

    // Expected: s_alpha(v) = v - <v,alpha>*alpha = (1,0,...) - 1*(1,-1,...) = (0,1,0,...)
    let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let code_and_data = build_self_verifying_program(
        emit_e8_reflect_kernel,
        &data,
        &expected,
    );
    build_elf(&code_and_data)
}

fn emit_e8_reflect_kernel(buf: &mut Vec<u8>) {
    let v: u8 = 0; let alpha: u8 = 1; let t: u8 = 2; let t2: u8 = 4;
    let ctrl0: u8 = 5; let ctrl1: u8 = 6; let ctrl2: u8 = 7; let vl: u8 = 2;

    emit_load_zmm(buf, v, 0);
    emit_load_zmm(buf, alpha, 64);
    emit_load_zmm(buf, ctrl0, 256);
    emit_load_zmm(buf, ctrl1, 320);
    emit_load_zmm(buf, ctrl2, 384);

    emit_evex_rr(buf, 1, 0x59, t, v, alpha, vl, 0, 0);     // VMULPD
    emit_evex_rr(buf, 2, 0x16, t2, ctrl0, t, vl, 0, 0);    // VPERMPD
    emit_evex_rr(buf, 1, 0x58, t, t, t2, vl, 0, 0);        // VADDPD
    emit_evex_rr(buf, 2, 0x16, t2, ctrl1, t, vl, 0, 0);
    emit_evex_rr(buf, 1, 0x58, t, t, t2, vl, 0, 0);
    emit_evex_rr(buf, 2, 0x16, t2, ctrl2, t, vl, 0, 0);
    emit_evex_rr(buf, 1, 0x58, t, t, t2, vl, 0, 0);
    emit_evex_rr(buf, 2, 0xBC, v, t, alpha, vl, 0, 0);     // VFNMADD231PD

    emit_store_zmm(buf, v, 128);
}

// ============================================================================
// Fano Octonion Multiply program (e1 * e2 = e3)
// ============================================================================

fn build_fano_e1e2_program() -> Vec<u8> {
    let mut data = vec![0u8; 2112]; // 2048 + 64 spare

    // a = e1 = (0, 1, 0, 0, 0, 0, 0, 0)
    write_f64s(&mut data, 0, &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    // b = e2 = (0, 0, 1, 0, 0, 0, 0, 0)
    write_f64s(&mut data, 64, &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    // Control vectors at offsets 512, 1024, 1536 (same as semantic test)
    populate_fano_data(&mut data);

    // Expected: e1*e2 = e3 = (0, 0, 0, 1, 0, 0, 0, 0)
    let expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

    let code_and_data = build_self_verifying_program(
        emit_fano_kernel,
        &data,
        &expected,
    );
    build_elf(&code_and_data)
}

fn emit_fano_kernel(buf: &mut Vec<u8>) {
    let za: u8 = 0; let zb: u8 = 1; let dst: u8 = 8;
    let t_bj: u8 = 10; let t_aj: u8 = 12; let accum: u8 = 14; let vl: u8 = 2;
    let ctrl_bcast: u8 = 16; let ctrl_perm: u8 = 17; let ctrl_sign: u8 = 18;

    emit_load_zmm(buf, za, 0);
    emit_load_zmm(buf, zb, 64);

    emit_vbroadcastsd(buf, t_bj, zb, vl);
    emit_evex_rr(buf, 1, 0x59, accum, za, t_bj, vl, 0, 0);

    for col in 0u8..7 {
        let bcast_off = 512 + (col as i32) * 64;
        let perm_off = 1024 + (col as i32) * 64;
        let sign_off = 1536 + (col as i32) * 64;

        emit_load_zmm(buf, ctrl_bcast, bcast_off);
        emit_load_zmm(buf, ctrl_perm, perm_off);
        emit_load_zmm(buf, ctrl_sign, sign_off);

        emit_evex_rr(buf, 2, 0x16, t_bj, ctrl_bcast, zb, vl, 0, 0);
        emit_evex_rr(buf, 2, 0x16, t_aj, ctrl_perm, za, vl, 0, 0);
        emit_evex_rr(buf, 1, 0x57, t_aj, t_aj, ctrl_sign, vl, 0, 0);
        emit_evex_rr(buf, 2, 0xB8, accum, t_aj, t_bj, vl, 0, 0);
    }

    emit_evex_rr(buf, 1, 0x28, dst, 0, accum, vl, 0, 0);
    emit_store_zmm(buf, dst, 128);
}

// ============================================================================
// E₈ Cartan Multiply program (e1 → column 0 with branch)
// ============================================================================

fn build_cartan_e1_program() -> Vec<u8> {
    let mut data = vec![0u8; 1344]; // enough for cols + bcast controls

    // v = e1 = (1, 0, 0, 0, 0, 0, 0, 0)
    write_f64s(&mut data, 0, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    populate_cartan_e8_data(&mut data);

    // Expected: Cartan * e1 = column 0 = (2, -1, 0, 0, 0, 0, 0, 0)
    // Branch is 2-7, not 0-7; mat[7][0] = 0
    let expected = [2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let code_and_data = build_self_verifying_program(
        emit_cartan_kernel,
        &data,
        &expected,
    );
    build_elf(&code_and_data)
}

fn emit_cartan_kernel(buf: &mut Vec<u8>) {
    let src: u8 = 0; let dst: u8 = 8; let t_bcast: u8 = 10;
    let accum: u8 = 12; let ctrl_tmp: u8 = 14; let col_tmp: u8 = 16; let vl: u8 = 2;

    emit_load_zmm(buf, src, 0);
    emit_evex_rr(buf, 1, 0x57, accum, accum, accum, vl, 0, 0); // VXORPD

    for col in 0..8 {
        let col_off = 256 + (col as i32) * 64;
        let bcast_off = 768 + (col as i32) * 64;

        emit_load_zmm(buf, ctrl_tmp, bcast_off);
        emit_load_zmm(buf, col_tmp, col_off);

        emit_evex_rr(buf, 2, 0x16, t_bcast, ctrl_tmp, src, vl, 0, 0);
        emit_evex_rr(buf, 2, 0xB8, accum, t_bcast, col_tmp, vl, 0, 0);
    }

    emit_evex_rr(buf, 1, 0x28, dst, 0, accum, vl, 0, 0);
    emit_store_zmm(buf, dst, 128);
}

// ============================================================================
// Data helpers
// ============================================================================

fn write_f64s(data: &mut [u8], offset: usize, vals: &[f64; 8]) {
    for (i, v) in vals.iter().enumerate() {
        let bytes = v.to_le_bytes();
        data[offset + i * 8..offset + i * 8 + 8].copy_from_slice(&bytes);
    }
}

fn write_u64s(data: &mut [u8], offset: usize, vals: &[u64; 8]) {
    for (i, v) in vals.iter().enumerate() {
        let bytes = v.to_le_bytes();
        data[offset + i * 8..offset + i * 8 + 8].copy_from_slice(&bytes);
    }
}

fn populate_fano_data(data: &mut [u8]) {
    // Broadcast controls at 512..
    for col in 0u64..7 {
        let j = col + 1;
        write_u64s(data, 512 + (col as usize) * 64, &[j, j, j, j, j, j, j, j]);
    }

    // Permutation controls at 1024..
    let perms: [[u64; 8]; 7] = [
        [1, 0, 3, 2, 5, 4, 7, 6],
        [2, 3, 0, 1, 6, 7, 4, 5],
        [3, 2, 1, 0, 7, 6, 5, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 7, 6, 1, 0, 3, 2],
        [6, 7, 4, 5, 2, 3, 0, 1],
        [7, 6, 5, 4, 3, 2, 1, 0],
    ];
    for (col, perm) in perms.iter().enumerate() {
        write_u64s(data, 1024 + col * 64, perm);
    }

    // Sign masks at 1536.. (CORRECTED)
    let neg: u64 = 0x8000000000000000;
    let pos: u64 = 0;
    let signs: [[u64; 8]; 7] = [
        [neg, pos, pos, neg, pos, neg, neg, pos], // col 1
        [neg, neg, pos, pos, pos, pos, neg, neg], // col 2
        [neg, pos, neg, pos, pos, neg, pos, neg], // col 3
        [neg, neg, neg, neg, pos, pos, pos, pos], // col 4
        [neg, pos, neg, pos, neg, pos, neg, pos], // col 5 CORRECTED
        [neg, pos, pos, neg, neg, pos, pos, neg], // col 6 CORRECTED
        [neg, neg, pos, pos, neg, neg, pos, pos], // col 7
    ];
    for (col, sign) in signs.iter().enumerate() {
        write_u64s(data, 1536 + col * 64, sign);
    }
}

fn populate_cartan_e8_data(data: &mut [u8]) {
    // E8 Cartan matrix (corrected with branch 2-7)
    let mut mat = [[0.0f64; 8]; 8];
    for i in 0..8 { mat[i][i] = 2.0; }
    for i in 0..6 { mat[i][i + 1] = -1.0; mat[i + 1][i] = -1.0; }
    mat[6][7] = 0.0; mat[7][6] = 0.0;
    mat[2][7] = -1.0; mat[7][2] = -1.0;

    // Column data at 256..
    for col in 0..8 {
        let offset = 256 + col * 64;
        let mut col_data = [0.0f64; 8];
        for row in 0..8 { col_data[row] = mat[row][col]; }
        write_f64s(data, offset, &col_data);
    }

    // Broadcast controls at 768..
    for col in 0..8u64 {
        write_u64s(data, 768 + (col as usize) * 64, &[col, col, col, col, col, col, col, col]);
    }
}

// ============================================================================
// Test runner
// ============================================================================

fn write_and_execute(elf_bytes: &[u8], name: &str, expected_exit: i32) {
    let _lock = ELF_LOCK.lock().unwrap();
    let seq = FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let path = format!("/tmp/sounio_e2e_{name}_{pid}_{seq}");
    fs::write(&path, elf_bytes).expect("Failed to write ELF");
    fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("Failed to chmod");
    let output = Command::new(&path)
        .output()
        .expect("Failed to execute ELF");
    let _ = fs::remove_file(&path);
    let actual_exit = output.status.code().unwrap_or(-1);
    assert_eq!(
        actual_exit, expected_exit,
        "ELF '{}' exited with {} (expected {})\nstdout: {}\nstderr: {}",
        name,
        actual_exit,
        expected_exit,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

// ============================================================================
// Tests
// ============================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn e2e_e8_root_reflect() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let elf = build_e8_reflect_program();
    write_and_execute(&elf, "e8_reflect", 0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn e2e_fano_e1_e2() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let elf = build_fano_e1e2_program();
    write_and_execute(&elf, "fano_e1e2", 0);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn e2e_cartan_e1() {
    if !has_avx512f() {
        eprintln!("SKIP: AVX-512F not available");
        return;
    }
    let elf = build_cartan_e1_program();
    write_and_execute(&elf, "cartan_e1", 0);
}

// ============================================================================
// Simple (non-AVX) ELF tests — proves the ELF builder fundamentally works
// ============================================================================

/// Minimal: exit(42)
fn build_exit42_program() -> Vec<u8> {
    let mut code = Vec::new();
    emit_mov_edi_imm32(&mut code, 42);
    emit_exit_syscall(&mut code);
    build_elf(&code)
}

/// Arithmetic in GP registers: exit(21 + 21) = exit(42)
fn build_gp_arithmetic_program() -> Vec<u8> {
    let mut code = Vec::new();
    // MOV edi, 21
    emit_mov_edi_imm32(&mut code, 21);
    // ADD edi, 21
    code.extend_from_slice(&[0x81, 0xC7]); // ADD edi, imm32
    code.extend_from_slice(&21u32.to_le_bytes());
    emit_exit_syscall(&mut code);
    build_elf(&code)
}

#[test]
fn e2e_exit42() {
    let elf = build_exit42_program();
    write_and_execute(&elf, "exit42", 42);
}

#[test]
fn e2e_gp_arithmetic() {
    let elf = build_gp_arithmetic_program();
    write_and_execute(&elf, "gp_arith", 42);
}

// ============================================================================
// Self-hosted pipeline type-check tests
// ============================================================================

fn souc_check(sio_path: &str) {
    let souc = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/debug/souc");
    let project_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..");
    let output = Command::new(&souc)
        .arg("check")
        .arg(sio_path)
        .current_dir(&project_root)
        .output()
        .expect("Failed to run souc");
    assert!(
        output.status.success(),
        "souc check {} failed:\nstdout: {}\nstderr: {}",
        sio_path,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

#[test]
fn self_hosted_test_native_execution_typechecks() {
    souc_check("self-hosted/native/test_native_execution.sio");
}

#[test]
fn self_hosted_test_exceptional_typechecks() {
    souc_check("self-hosted/native/test_exceptional.sio");
}

#[test]
fn self_hosted_test_semantic_execution_typechecks() {
    souc_check("self-hosted/native/test_semantic_execution.sio");
}
