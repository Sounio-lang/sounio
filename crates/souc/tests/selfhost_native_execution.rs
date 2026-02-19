//! Phase 6A: Self-Hosted Native Backend Execution Tests
//!
//! Validates that:
//! 1. ELF binaries can be written to disk
//! 2. ELF binaries can be executed on Linux x86-64
//! 3. Exit codes are correct
//!
//! Full integration with compile_to_elf() is deferred to Phase 6B.

use std::os::unix::fs::PermissionsExt;
use std::process::Command;
/// Minimal working ELF that returns exit code 42
/// Code path: main() { return 42; }
/// Assembly:
///   _start:
///     mov rax, 42      # rax = exit code
///     mov rdi, rax     # rdi = arg1 = exit code
///     mov rax, 60      # rax = syscall(sys_exit)
///     syscall          # exit(42)
fn create_minimal_elf_return42() -> Vec<u8> {
    // Pre-built minimal ELF header + code
    // This is equivalent to what compile_to_elf() should produce
    let mut elf = vec![0u8; 0x1020]; // Allocate space for ELF header + padding + code

    // ELF Header (64 bytes)
    elf[0..4].copy_from_slice(b"\x7fELF"); // Magic
    elf[4] = 2; // EI_CLASS: 64-bit
    elf[5] = 1; // EI_DATA: little-endian
    elf[6] = 1; // EI_VERSION: current

    // e_type: ET_EXEC (2) at offset 16
    elf[16] = 2;

    // e_machine: EM_X86_64 (62) at offset 18
    elf[18] = 62;

    // e_version at offset 20
    elf[20] = 1;

    // e_entry: 0x400000 at offset 24
    elf[24..32].copy_from_slice(&0x400000u64.to_le_bytes());

    // e_phoff: 64 at offset 32
    elf[32..40].copy_from_slice(&64u64.to_le_bytes());

    // e_shoff: 0 at offset 40 (no section headers)

    // e_flags at offset 48 (0)

    // e_ehsize: 64 at offset 52
    elf[52..54].copy_from_slice(&64u16.to_le_bytes());

    // e_phentsize: 56 at offset 54
    elf[54..56].copy_from_slice(&56u16.to_le_bytes());

    // e_phnum: 1 at offset 56
    elf[56..58].copy_from_slice(&1u16.to_le_bytes());

    // Program header (.text, PT_LOAD) at offset 64
    // ELF64 Phdr structure (56 bytes):
    // uint32_t p_type        (0)
    // uint32_t p_flags       (4)
    // uint64_t p_offset      (8)
    // uint64_t p_vaddr       (16)
    // uint64_t p_paddr       (24)
    // uint64_t p_filesz      (32)
    // uint64_t p_memsz       (40)
    // uint64_t p_align       (48)
    let ph_off = 64;

    // p_type: PT_LOAD (1) at offset 0
    elf[ph_off..ph_off + 4].copy_from_slice(&1u32.to_le_bytes());

    // p_flags: PF_X | PF_R (5 = 4+1) at offset 4
    elf[ph_off + 4..ph_off + 8].copy_from_slice(&5u32.to_le_bytes());

    // p_offset: 0x1000 at offset 8
    elf[ph_off + 8..ph_off + 16].copy_from_slice(&0x1000u64.to_le_bytes());

    // p_vaddr: 0x400000 at offset 16
    elf[ph_off + 16..ph_off + 24].copy_from_slice(&0x400000u64.to_le_bytes());

    // p_paddr: 0x400000 at offset 24
    elf[ph_off + 24..ph_off + 32].copy_from_slice(&0x400000u64.to_le_bytes());

    // p_filesz: 19 bytes (code size) at offset 32
    elf[ph_off + 32..ph_off + 40].copy_from_slice(&19u64.to_le_bytes());

    // p_memsz: 19 bytes at offset 40
    elf[ph_off + 40..ph_off + 48].copy_from_slice(&19u64.to_le_bytes());

    // p_align: 0x1000 at offset 48
    elf[ph_off + 48..ph_off + 56].copy_from_slice(&0x1000u64.to_le_bytes());

    // Code at 0x1000
    let code_off = 0x1000;
    let code = [
        0x48, 0xc7, 0xc0, 0x2a, 0x00, 0x00, 0x00, // mov rax, 42
        0x48, 0x89, 0xc7, // mov rdi, rax
        0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00, // mov rax, 60
        0x0f, 0x05, // syscall
    ];
    elf[code_off..code_off + code.len()].copy_from_slice(&code);

    elf
}

#[test]
#[cfg(target_os = "linux")]
fn phase6a_minimal_elf_executes() {
    let elf_bytes = create_minimal_elf_return42();

    // Verify ELF magic
    assert_eq!(elf_bytes[0], 0x7f, "ELF magic byte 0 invalid");
    assert_eq!(elf_bytes[1], b'E', "ELF magic byte 1 invalid");
    assert_eq!(elf_bytes[2], b'L', "ELF magic byte 2 invalid");
    assert_eq!(elf_bytes[3], b'F', "ELF magic byte 3 invalid");

    // Verify ELF class (64-bit)
    assert_eq!(elf_bytes[4], 2, "ELF class should be 64-bit");

    // Verify ELF data (little-endian)
    assert_eq!(elf_bytes[5], 1, "ELF data should be little-endian");

    // Write to disk
    let out_path = "/tmp/sounio_phase6a_test.elf";
    std::fs::write(out_path, &elf_bytes).expect("Failed to write ELF to disk");

    // Make executable
    std::fs::set_permissions(out_path, std::fs::Permissions::from_mode(0o755))
        .expect("Failed to set executable permissions");

    // Execute and verify exit code
    let output = Command::new(out_path)
        .output()
        .expect("Failed to execute ELF binary");

    let exit_code = output.status.code();
    assert_eq!(
        exit_code,
        Some(42),
        "Expected exit code 42, got {:?}",
        exit_code
    );

    println!("✓ Phase 6A: Minimal ELF executed successfully with exit code 42");
}

#[test]
#[cfg(target_os = "linux")]
fn phase6a_elf_structure_valid() {
    let elf_bytes = create_minimal_elf_return42();

    // Verify minimum size
    assert!(elf_bytes.len() >= 0x1013, "ELF too small");

    // Verify header
    assert_eq!(&elf_bytes[0..4], b"\x7fELF", "Invalid ELF magic");

    // Verify e_type (ET_EXEC = 2)
    let e_type = u16::from_le_bytes([elf_bytes[16], elf_bytes[17]]);
    assert_eq!(e_type, 2, "e_type should be ET_EXEC (2)");

    // Verify e_machine (EM_X86_64 = 62)
    let e_machine = u16::from_le_bytes([elf_bytes[18], elf_bytes[19]]);
    assert_eq!(e_machine, 62, "e_machine should be EM_X86_64 (62)");

    // Verify entry point (0x400000)
    let entry = u64::from_le_bytes([
        elf_bytes[24],
        elf_bytes[25],
        elf_bytes[26],
        elf_bytes[27],
        elf_bytes[28],
        elf_bytes[29],
        elf_bytes[30],
        elf_bytes[31],
    ]);
    assert_eq!(entry, 0x400000, "Entry point should be 0x400000");

    println!("✓ Phase 6A: ELF structure is valid");
}

#[test]
#[cfg(target_os = "linux")]
fn phase6a_elf_cleanup() {
    // Clean up temporary files
    let _ = std::fs::remove_file("/tmp/sounio_phase6a_test.elf");
    let _ = std::fs::remove_file("/tmp/sounio_test_return42.elf");
    let _ = std::fs::remove_file("/tmp/sounio_minimal_return42.elf");
}
