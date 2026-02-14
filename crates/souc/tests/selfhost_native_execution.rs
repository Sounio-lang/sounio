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
    vec![
        // ELF Header (64 bytes)
        0x7f, b'E', b'L', b'F',  // Magic: 0x7f 'E' 'L' 'F'
        2,     // EI_CLASS: 64-bit
        1,     // EI_DATA: little-endian
        1,     // EI_VERSION: current
        0,     // EI_OSABI
        0,     // EI_ABIVERSION
        0, 0, 0, 0, 0, 0, 0,     // Padding (7 bytes)
        2, 0,  // e_type: ET_EXEC (2)
        62, 0, // e_machine: EM_X86_64 (62)
        1, 0, 0, 0,              // e_version
        0x00, 0x40, 0, 0, 0, 0, 0, 0,     // e_entry: 0x400000
        64, 0, 0, 0, 0, 0, 0, 0,         // e_phoff: 64
        0, 0, 0, 0, 0, 0, 0, 0,          // e_shoff: 0
        0, 0, 0, 0,                      // e_flags
        64, 0,                           // e_ehsize: 64
        56, 0,                           // e_phentsize: 56
        1, 0,                            // e_phnum: 1 (only .text)
        0, 0,                            // e_shentsize: 0
        0, 0,                            // e_shnum: 0
        0, 0,                            // e_shstrndx: 0
        // Program header (.text, PT_LOAD)
        1, 0, 0, 0,                      // p_type: PT_LOAD
        0x00, 0x10, 0, 0, 0, 0, 0, 0,   // p_offset: 0x1000
        0x00, 0x40, 0, 0, 0, 0, 0, 0,   // p_vaddr: 0x400000
        0x00, 0x40, 0, 0, 0, 0, 0, 0,   // p_paddr: 0x400000
        0x13, 0, 0, 0, 0, 0, 0, 0,      // p_filesz: 19 bytes (code size)
        0x13, 0, 0, 0, 0, 0, 0, 0,      // p_memsz: 19 bytes
        5, 0, 0, 0, 0, 0, 0, 0,         // p_flags: PF_X | PF_R
        0x00, 0x10, 0, 0, 0, 0, 0, 0,   // p_align: 0x1000
        // Padding to reach 0x1000
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Code at 0x1000
        // mov rax, 42 (0x2a)
        0x48, 0xc7, 0xc0, 0x2a, 0x00, 0x00, 0x00,
        // mov rdi, rax
        0x48, 0x89, 0xc7,
        // mov rax, 60 (sys_exit)
        0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00,
        // syscall
        0x0f, 0x05,
    ]
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
        elf_bytes[32],
        elf_bytes[33],
        elf_bytes[34],
        elf_bytes[35],
        elf_bytes[36],
        elf_bytes[37],
        elf_bytes[38],
        elf_bytes[39],
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
