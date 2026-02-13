//! Minimal native executable emission for bootstrap.
//!
//! On Linux we emit a minimal ELF64 executable with PT_LOAD segments.
//! On macOS we emit a minimal Mach-O executable (implemented behind `#[cfg]`).
//!
//! This is not a full linker. It exists to unblock stage-0 native emission.

use std::collections::HashMap;

#[derive(Debug, Default, Clone)]
pub struct NativeEmitter {
    /// Machine code bytes (.text)
    pub code: Vec<u8>,
    /// Read-only data bytes (.rodata / const)
    pub data: Vec<u8>,
    /// Uninitialized data size (BSS)
    pub bss_size: usize,
    /// Optional labels for debugging / future relocations
    pub labels: HashMap<String, u32>,
}

impl NativeEmitter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn emit_code(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    pub fn add_string(&mut self, s: &str) -> u32 {
        let off = self.data.len() as u32;
        self.data.extend_from_slice(s.as_bytes());
        off
    }

    pub fn add_bss(&mut self, size: usize) -> u32 {
        let off = self.bss_size as u32;
        self.bss_size = self.bss_size.saturating_add(size);
        off
    }

    pub fn finalize(&self, entry_offset: usize) -> Vec<u8> {
        #[cfg(target_os = "linux")]
        {
            return finalize_elf64(self, entry_offset);
        }
        #[cfg(target_os = "macos")]
        {
            return finalize_macho64(self, entry_offset);
        }
        #[allow(unreachable_code)]
        {
            panic!("native emitter not supported on this platform");
        }
    }

    pub fn write_executable(&self, path: &str, entry_offset: usize) -> std::io::Result<()> {
        let binary = self.finalize(entry_offset);
        std::fs::write(path, &binary)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))?;
        }
        Ok(())
    }
}

#[cfg(target_os = "linux")]
const ELF_PAGE: usize = 0x1000;

#[cfg(target_os = "linux")]
const ELF_BASE: u64 = 0x400000;

#[cfg(target_os = "linux")]
const ELF_HDR_SIZE: usize = 64;

#[cfg(target_os = "linux")]
const ELF_PHDR_SIZE: usize = 56;

#[cfg(target_os = "linux")]
fn align_up(n: usize, align: usize) -> usize {
    if align == 0 {
        return n;
    }
    (n + align - 1) & !(align - 1)
}

#[cfg(target_os = "linux")]
#[derive(Debug, Clone, Copy)]
struct ElfLayout {
    phnum: u16,
    header_and_ph_size: usize,
    code_offset: usize,
    rodata_offset: usize,
    bss_offset: usize,
    file_size: usize,
}

#[cfg(target_os = "linux")]
fn elf_layout(code_len: usize, rodata_len: usize, bss_size: usize) -> ElfLayout {
    let phnum = if bss_size > 0 { 3 } else { 2 };
    let header_and_ph_size = ELF_HDR_SIZE + (ELF_PHDR_SIZE * phnum);
    let code_offset = align_up(header_and_ph_size, ELF_PAGE);
    let rodata_offset = align_up(code_offset + code_len, ELF_PAGE);
    let bss_offset = align_up(rodata_offset + rodata_len, ELF_PAGE);
    let file_size = if bss_size > 0 {
        bss_offset
    } else {
        rodata_offset + rodata_len
    };
    ElfLayout {
        phnum: phnum as u16,
        header_and_ph_size,
        code_offset,
        rodata_offset,
        bss_offset,
        file_size,
    }
}

#[cfg(target_os = "linux")]
fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[cfg(target_os = "linux")]
fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[cfg(target_os = "linux")]
fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[cfg(target_os = "linux")]
fn finalize_elf64(e: &NativeEmitter, entry_offset: usize) -> Vec<u8> {
    // Minimal, statically-linked ET_EXEC with PT_LOAD program headers.
    let layout = elf_layout(e.code.len(), e.data.len(), e.bss_size);
    let entry = ELF_BASE + (layout.code_offset as u64) + (entry_offset as u64);

    #[cfg(target_arch = "x86_64")]
    let e_machine: u16 = 62; // EM_X86_64
    #[cfg(target_arch = "aarch64")]
    let e_machine: u16 = 183; // EM_AARCH64

    let mut out = Vec::with_capacity(layout.file_size.max(layout.code_offset));

    // ---------------------------------------------------------------------
    // ELF header (64 bytes)
    // ---------------------------------------------------------------------
    // e_ident
    out.extend_from_slice(&[
        0x7f, b'E', b'L', b'F', // magic
        2,    // EI_CLASS = ELFCLASS64
        1,    // EI_DATA  = ELFDATA2LSB
        1,    // EI_VERSION
        0,    // EI_OSABI (System V)
        0,    // EI_ABIVERSION
        0, 0, 0, 0, 0, 0, 0, // EI_PAD
    ]);
    push_u16(&mut out, 2); // e_type = ET_EXEC
    push_u16(&mut out, e_machine);
    push_u32(&mut out, 1); // e_version
    push_u64(&mut out, entry);
    push_u64(&mut out, ELF_HDR_SIZE as u64); // e_phoff
    push_u64(&mut out, 0); // e_shoff
    push_u32(&mut out, 0); // e_flags
    push_u16(&mut out, ELF_HDR_SIZE as u16); // e_ehsize
    push_u16(&mut out, ELF_PHDR_SIZE as u16); // e_phentsize
    push_u16(&mut out, layout.phnum); // e_phnum
    push_u16(&mut out, 0); // e_shentsize
    push_u16(&mut out, 0); // e_shnum
    push_u16(&mut out, 0); // e_shstrndx
    debug_assert_eq!(out.len(), ELF_HDR_SIZE);

    // ---------------------------------------------------------------------
    // Program headers
    // ---------------------------------------------------------------------
    // p_type, p_flags, p_offset, p_vaddr, p_paddr, p_filesz, p_memsz, p_align
    const PT_LOAD: u32 = 1;
    const PF_X: u32 = 1;
    const PF_W: u32 = 2;
    const PF_R: u32 = 4;

    // Segment 0: headers + padding + code (RX), ends at start of rodata.
    push_u32(&mut out, PT_LOAD);
    push_u32(&mut out, PF_R | PF_X);
    push_u64(&mut out, 0);
    push_u64(&mut out, ELF_BASE);
    push_u64(&mut out, ELF_BASE);
    push_u64(&mut out, layout.rodata_offset as u64);
    push_u64(&mut out, layout.rodata_offset as u64);
    push_u64(&mut out, ELF_PAGE as u64);

    // Segment 1: rodata (R)
    push_u32(&mut out, PT_LOAD);
    push_u32(&mut out, PF_R);
    push_u64(&mut out, layout.rodata_offset as u64);
    push_u64(&mut out, ELF_BASE + layout.rodata_offset as u64);
    push_u64(&mut out, ELF_BASE + layout.rodata_offset as u64);
    push_u64(&mut out, e.data.len() as u64);
    push_u64(&mut out, e.data.len() as u64);
    push_u64(&mut out, ELF_PAGE as u64);

    // Segment 2: bss (RW) (optional)
    if e.bss_size > 0 {
        push_u32(&mut out, PT_LOAD);
        push_u32(&mut out, PF_R | PF_W);
        push_u64(&mut out, layout.bss_offset as u64);
        push_u64(&mut out, ELF_BASE + layout.bss_offset as u64);
        push_u64(&mut out, ELF_BASE + layout.bss_offset as u64);
        push_u64(&mut out, 0);
        push_u64(&mut out, e.bss_size as u64);
        push_u64(&mut out, ELF_PAGE as u64);
    }

    debug_assert_eq!(out.len(), layout.header_and_ph_size);

    // ---------------------------------------------------------------------
    // Segment contents
    // ---------------------------------------------------------------------
    out.resize(layout.code_offset, 0);
    out.extend_from_slice(&e.code);
    out.resize(layout.rodata_offset, 0);
    out.extend_from_slice(&e.data);
    if e.bss_size > 0 {
        out.resize(layout.bss_offset, 0);
    }

    out
}

#[cfg(target_os = "macos")]
fn align_up_macho(n: usize, align: usize) -> usize {
    if align == 0 {
        return n;
    }
    (n + align - 1) & !(align - 1)
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
struct MachoLayout {
    ncmds: u32,
    sizeofcmds: u32,
    code_offset: usize,
    data_offset: usize,
    text_filesize: usize,
    text_vmsize: usize,
    entry_pc: u64,
}

#[cfg(target_os = "macos")]
fn macho_layout(
    code_len: usize,
    rodata_len: usize,
    bss_size: usize,
    entry_offset: usize,
) -> MachoLayout {
    const MACH_HDR_SIZE: usize = 32;
    const LC_SEGMENT_64_SIZE: usize = 72;

    #[cfg(target_arch = "aarch64")]
    const LC_UNIXTHREAD_SIZE: usize = 288;
    #[cfg(target_arch = "x86_64")]
    const LC_UNIXTHREAD_SIZE: usize = 184;

    const PAGE: usize = 0x1000;
    const TEXT_BASE: u64 = 0x100000000;

    let mut ncmds: u32 = 3; // __PAGEZERO + __TEXT + LC_UNIXTHREAD
    let mut sizeofcmds: usize = (LC_SEGMENT_64_SIZE * 2) + LC_UNIXTHREAD_SIZE;

    if bss_size > 0 {
        ncmds += 1; // __DATA
        sizeofcmds += LC_SEGMENT_64_SIZE;
    }

    let code_offset = align_up_macho(MACH_HDR_SIZE + sizeofcmds, 16);
    let data_offset = align_up_macho(code_offset + code_len, 16);
    let file_end = data_offset + rodata_len;

    let text_filesize = align_up_macho(file_end, PAGE);
    let text_vmsize = text_filesize;
    let entry_pc = TEXT_BASE + (code_offset as u64) + (entry_offset as u64);

    MachoLayout {
        ncmds,
        sizeofcmds: sizeofcmds as u32,
        code_offset,
        data_offset,
        text_filesize,
        text_vmsize,
        entry_pc,
    }
}

#[cfg(target_os = "macos")]
fn push_i32(out: &mut Vec<u8>, v: i32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[cfg(target_os = "macos")]
fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[cfg(target_os = "macos")]
fn push_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

#[cfg(target_os = "macos")]
fn push_padded_str16(out: &mut Vec<u8>, s: &str) {
    let mut buf = [0u8; 16];
    let bytes = s.as_bytes();
    let n = bytes.len().min(buf.len());
    buf[..n].copy_from_slice(&bytes[..n]);
    out.extend_from_slice(&buf);
}

#[cfg(target_os = "macos")]
fn finalize_macho64(e: &NativeEmitter, entry_offset: usize) -> Vec<u8> {
    // Minimal Mach-O 64-bit executable that does not rely on dyld.
    // Layout: mach_header_64 + load commands + code + rodata, padded to 4KiB.
    const MH_MAGIC_64: u32 = 0xfeedfacf;
    const MH_EXECUTE: u32 = 2;
    const LC_SEGMENT_64: u32 = 0x19;
    const LC_UNIXTHREAD: u32 = 0x5;

    const VM_PROT_READ: i32 = 1;
    const VM_PROT_WRITE: i32 = 2;
    const VM_PROT_EXECUTE: i32 = 4;

    const PAGE: usize = 0x1000;
    const TEXT_BASE: u64 = 0x100000000;

    #[cfg(target_arch = "aarch64")]
    let cputype: i32 = 0x0100_000c; // CPU_TYPE_ARM64
    #[cfg(target_arch = "aarch64")]
    let cpusubtype: i32 = 0; // CPU_SUBTYPE_ARM64_ALL

    #[cfg(target_arch = "x86_64")]
    let cputype: i32 = 0x0100_0007; // CPU_TYPE_X86_64
    #[cfg(target_arch = "x86_64")]
    let cpusubtype: i32 = 0x8000_0003_u32 as i32; // CPU_SUBTYPE_X86_64_ALL | CPU_SUBTYPE_LIB64

    let layout = macho_layout(e.code.len(), e.data.len(), e.bss_size, entry_offset);

    let mut out = Vec::with_capacity(layout.text_filesize.max(layout.code_offset));

    // mach_header_64
    push_u32(&mut out, MH_MAGIC_64);
    push_i32(&mut out, cputype);
    push_i32(&mut out, cpusubtype);
    push_u32(&mut out, MH_EXECUTE);
    push_u32(&mut out, layout.ncmds);
    push_u32(&mut out, layout.sizeofcmds);
    push_u32(&mut out, 0); // flags (non-PIE)
    push_u32(&mut out, 0); // reserved

    // LC_SEGMENT_64 __PAGEZERO
    push_u32(&mut out, LC_SEGMENT_64);
    push_u32(&mut out, 72);
    push_padded_str16(&mut out, "__PAGEZERO");
    push_u64(&mut out, 0);
    push_u64(&mut out, TEXT_BASE); // vmsize (4GiB)
    push_u64(&mut out, 0);
    push_u64(&mut out, 0);
    push_i32(&mut out, 0);
    push_i32(&mut out, 0);
    push_u32(&mut out, 0);
    push_u32(&mut out, 0);

    // LC_SEGMENT_64 __TEXT
    push_u32(&mut out, LC_SEGMENT_64);
    push_u32(&mut out, 72);
    push_padded_str16(&mut out, "__TEXT");
    push_u64(&mut out, TEXT_BASE);
    push_u64(&mut out, layout.text_vmsize as u64);
    push_u64(&mut out, 0);
    push_u64(&mut out, layout.text_filesize as u64);
    push_i32(&mut out, VM_PROT_READ | VM_PROT_EXECUTE);
    push_i32(&mut out, VM_PROT_READ | VM_PROT_EXECUTE);
    push_u32(&mut out, 0);
    push_u32(&mut out, 0);

    // Optional LC_SEGMENT_64 __DATA (BSS only)
    if e.bss_size > 0 {
        let data_vmsize = align_up_macho(e.bss_size, PAGE);
        push_u32(&mut out, LC_SEGMENT_64);
        push_u32(&mut out, 72);
        push_padded_str16(&mut out, "__DATA");
        push_u64(&mut out, TEXT_BASE + layout.text_vmsize as u64);
        push_u64(&mut out, data_vmsize as u64);
        push_u64(&mut out, layout.text_filesize as u64);
        push_u64(&mut out, 0);
        push_i32(&mut out, VM_PROT_READ | VM_PROT_WRITE);
        push_i32(&mut out, VM_PROT_READ | VM_PROT_WRITE);
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
    }

    // LC_UNIXTHREAD
    #[cfg(target_arch = "aarch64")]
    {
        // cmdsize = 16 (header) + 272 (arm_thread_state64_t) = 288
        push_u32(&mut out, LC_UNIXTHREAD);
        push_u32(&mut out, 288);
        push_u32(&mut out, 6); // ARM_THREAD_STATE64
        push_u32(&mut out, 68); // ARM_THREAD_STATE64_COUNT (272 / 4)

        // x0-x28
        for _ in 0..29 {
            push_u64(&mut out, 0);
        }
        // fp, lr, sp, pc
        push_u64(&mut out, 0);
        push_u64(&mut out, 0);
        push_u64(&mut out, 0);
        push_u64(&mut out, layout.entry_pc);
        // cpsr, pad
        push_u32(&mut out, 0);
        push_u32(&mut out, 0);
    }

    #[cfg(target_arch = "x86_64")]
    {
        // cmdsize = 16 (header) + 168 (x86_thread_state64_t) = 184
        push_u32(&mut out, LC_UNIXTHREAD);
        push_u32(&mut out, 184);
        push_u32(&mut out, 4); // x86_THREAD_STATE64
        push_u32(&mut out, 42); // x86_THREAD_STATE64_COUNT (168 / 4)

        // rax..r15
        for _ in 0..16 {
            push_u64(&mut out, 0);
        }
        // rip
        push_u64(&mut out, layout.entry_pc);
        // rflags, cs, fs, gs
        push_u64(&mut out, 0);
        push_u64(&mut out, 0);
        push_u64(&mut out, 0);
        push_u64(&mut out, 0);
    }

    debug_assert_eq!(out.len(), 32 + (layout.sizeofcmds as usize));

    out.resize(layout.code_offset, 0);
    out.extend_from_slice(&e.code);
    out.resize(layout.data_offset, 0);
    out.extend_from_slice(&e.data);
    out.resize(layout.text_filesize, 0);

    out
}

#[cfg(all(test, target_os = "linux", target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn assemble_hello_world_x86_64_abs(msg_addr: u64, msg_len: usize) -> Vec<u8> {
        // mov rax, 1
        // mov rdi, 1
        // mov rsi, msg_addr
        // mov rdx, msg_len
        // syscall
        // mov rax, 60
        // xor rdi, rdi
        // syscall
        let mut code = vec![
            0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
            0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00, // mov rdi, 1
            0x48, 0xbe, 0, 0, 0, 0, 0, 0, 0, 0, // mov rsi, imm64 (patched)
            0x48, 0xc7, 0xc2, 0, 0, 0, 0, // mov rdx, imm32 (patched)
            0x0f, 0x05, // syscall
            0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00, // mov rax, 60
            0x48, 0x31, 0xff, // xor rdi, rdi
            0x0f, 0x05, // syscall
        ];

        // Patch message address (imm64) at offset 16.
        let addr_off = 16;
        code[addr_off..addr_off + 8].copy_from_slice(&msg_addr.to_le_bytes());

        // Patch length (imm32) at offset 30.
        // mov rdx, imm32 starts at offset 24, immediate begins after 3-byte opcode.
        let len_off = 27;
        let len_u32: u32 = msg_len.try_into().expect("msg_len fits in u32");
        code[len_off..len_off + 4].copy_from_slice(&len_u32.to_le_bytes());

        code
    }

    #[test]
    fn test_emit_hello_world() {
        let mut e = NativeEmitter::new();
        let msg = "Hello from Sounio!\n";
        let msg_offset = e.add_string(msg);

        // Assemble with a placeholder address first to compute layout.
        let placeholder = assemble_hello_world_x86_64_abs(0, msg.len());
        let layout = elf_layout(placeholder.len(), e.data.len(), e.bss_size);
        let msg_addr = ELF_BASE + (layout.rodata_offset as u64) + (msg_offset as u64);

        let code = assemble_hello_world_x86_64_abs(msg_addr, msg.len());
        e.emit_code(&code);

        let path = "/tmp/sounio_hello";
        let _ = std::fs::remove_file(path);
        e.write_executable(path, 0).unwrap();

        let output = std::process::Command::new(path).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&output.stdout), msg);
        assert_eq!(output.status.code(), Some(0));

        let _ = std::fs::remove_file(path);
    }
}

#[cfg(all(test, target_os = "macos", target_arch = "aarch64"))]
mod tests_macos_arm64 {
    use super::*;

    fn encode_adr(rd: u32, imm: i64) -> u32 {
        // ADR (literal): imm is a signed 21-bit byte offset from the instruction address.
        let immlo = (imm & 0x3) as u32;
        let immhi = ((imm >> 2) & 0x7ffff) as u32;
        0x1000_0000u32 | (immlo << 29) | (immhi << 5) | (rd & 0x1f)
    }

    fn assemble_hello_world_arm64(msg_imm: i64, msg_len: usize) -> Vec<u8> {
        // movz x0, #1
        // adr  x1, msg
        // movz x2, #len
        // movz x16, #4
        // movk x16, #0x0200, lsl #16
        // svc  #0x80
        // movz x0, #0
        // movz x16, #1
        // movk x16, #0x0200, lsl #16
        // svc  #0x80
        let mut code = Vec::with_capacity(40);
        code.extend_from_slice(&0xd280_0020u32.to_le_bytes()); // movz x0, #1

        let adr_off = code.len();
        code.extend_from_slice(&0u32.to_le_bytes()); // patched ADR

        let len_u32: u32 = msg_len.try_into().expect("msg_len fits in u32");
        let mov_x2 = 0xd280_0000u32 + (len_u32 << 5) + 2; // movz x2, #len
        code.extend_from_slice(&mov_x2.to_le_bytes());

        code.extend_from_slice(&0xd280_0090u32.to_le_bytes()); // movz x16, #4
        code.extend_from_slice(&0xf2a0_4010u32.to_le_bytes()); // movk x16, #0x0200, lsl #16
        code.extend_from_slice(&0xd400_1001u32.to_le_bytes()); // svc #0x80

        code.extend_from_slice(&0xd280_0000u32.to_le_bytes()); // movz x0, #0
        code.extend_from_slice(&0xd280_0030u32.to_le_bytes()); // movz x16, #1
        code.extend_from_slice(&0xf2a0_4010u32.to_le_bytes()); // movk x16, #0x0200, lsl #16
        code.extend_from_slice(&0xd400_1001u32.to_le_bytes()); // svc #0x80

        let adr = encode_adr(1, msg_imm);
        code[adr_off..adr_off + 4].copy_from_slice(&adr.to_le_bytes());
        code
    }

    #[test]
    fn test_emit_hello_world() {
        let mut e = NativeEmitter::new();
        let msg = "Hello from Sounio!\n";
        let msg_offset = e.add_string(msg);

        // Compute Mach-O layout and patch the ADR immediate.
        let placeholder = assemble_hello_world_arm64(0, msg.len());
        let layout = macho_layout(placeholder.len(), e.data.len(), e.bss_size, 0);

        let msg_file_off = (layout.data_offset as u64) + (msg_offset as u64);
        let adr_pc = 0x100000000_u64 + (layout.code_offset as u64) + 4;
        let msg_addr = 0x100000000_u64 + msg_file_off;
        let msg_imm = (msg_addr as i64) - (adr_pc as i64);

        let code = assemble_hello_world_arm64(msg_imm, msg.len());
        e.emit_code(&code);

        let path = "/tmp/sounio_hello_macho";
        let _ = std::fs::remove_file(path);
        e.write_executable(path, 0).unwrap();

        let output = std::process::Command::new(path).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&output.stdout), msg);
        assert_eq!(output.status.code(), Some(0));

        let _ = std::fs::remove_file(path);
    }
}

#[cfg(all(test, target_os = "macos", target_arch = "x86_64"))]
mod tests_macos_x86_64 {
    use super::*;

    fn assemble_hello_world_x86_64_abs(msg_addr: u64, msg_len: usize) -> Vec<u8> {
        // Same as Linux, but Darwin uses a 0x2000000-prefixed syscall number.
        let mut code = vec![
            0x48, 0xc7, 0xc0, 0x04, 0x00, 0x00, 0x02, // mov rax, 0x2000004 (write)
            0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00, // mov rdi, 1
            0x48, 0xbe, 0, 0, 0, 0, 0, 0, 0, 0, // mov rsi, imm64 (patched)
            0x48, 0xc7, 0xc2, 0, 0, 0, 0, // mov rdx, imm32 (patched)
            0x0f, 0x05, // syscall
            0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x02, // mov rax, 0x2000001 (exit)
            0x48, 0x31, 0xff, // xor rdi, rdi
            0x0f, 0x05, // syscall
        ];

        let addr_off = 16;
        code[addr_off..addr_off + 8].copy_from_slice(&msg_addr.to_le_bytes());

        let len_off = 27;
        let len_u32: u32 = msg_len.try_into().expect("msg_len fits in u32");
        code[len_off..len_off + 4].copy_from_slice(&len_u32.to_le_bytes());

        code
    }

    #[test]
    fn test_emit_hello_world() {
        let mut e = NativeEmitter::new();
        let msg = "Hello from Sounio!\n";
        let msg_offset = e.add_string(msg);

        let placeholder = assemble_hello_world_x86_64_abs(0, msg.len());
        let layout = macho_layout(placeholder.len(), e.data.len(), e.bss_size, 0);
        let msg_addr = 0x100000000_u64 + (layout.data_offset as u64) + (msg_offset as u64);

        let code = assemble_hello_world_x86_64_abs(msg_addr, msg.len());
        e.emit_code(&code);

        let path = "/tmp/sounio_hello_macho";
        let _ = std::fs::remove_file(path);
        e.write_executable(path, 0).unwrap();

        let output = std::process::Command::new(path).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&output.stdout), msg);
        assert_eq!(output.status.code(), Some(0));

        let _ = std::fs::remove_file(path);
    }
}
