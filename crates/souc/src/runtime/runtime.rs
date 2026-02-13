//! Native runtime code stubs for bootstrap.
//!
//! This module is the bridge between builtin function names and native machine code
//! implementations (via raw syscalls). The native backend will eventually emit calls
//! to these routines instead of relying on Rust builtins.

use crate::runtime::elf::NativeEmitter;
use std::collections::HashMap;

pub struct RuntimeCodegen {
    emitter: NativeEmitter,
    builtin_offsets: HashMap<String, usize>,
}

impl RuntimeCodegen {
    pub fn new() -> Self {
        Self {
            emitter: NativeEmitter::new(),
            builtin_offsets: HashMap::new(),
        }
    }

    pub fn emitter(&self) -> &NativeEmitter {
        &self.emitter
    }

    pub fn emitter_mut(&mut self) -> &mut NativeEmitter {
        &mut self.emitter
    }

    pub fn into_emitter(self) -> NativeEmitter {
        self.emitter
    }

    pub fn emit_runtime(&mut self) {
        // These names mirror the interpreter builtins and are the initial target
        // surface area for stage-0 native emission.
        self.emit_print();
        self.emit_exit();

        // TODO: Implement the rest as the native calling convention and String layout
        // are finalized.
        self.emit_trap_stub("print_int");
        self.emit_trap_stub("print_char");
        self.emit_trap_stub("read_byte");
        self.emit_trap_stub("read_file");
        self.emit_trap_stub("read_file_prefix");
        self.emit_trap_stub("file_size");
        self.emit_trap_stub("write_bytes");
        self.emit_trap_stub("arg_count");
        self.emit_trap_stub("get_arg");
        self.emit_trap_stub("str_len");
        self.emit_trap_stub("str_eq");
        self.emit_trap_stub("str_concat");
        self.emit_trap_stub("str_slice");
    }

    pub fn builtin_offset(&self, name: &str) -> Option<usize> {
        self.builtin_offsets.get(name).copied()
    }

    fn mark(&mut self, name: &str) -> usize {
        let off = self.emitter.code.len();
        self.builtin_offsets.insert(name.to_string(), off);
        off
    }

    fn emit_trap_stub(&mut self, name: &str) {
        self.mark(name);
        emit_trap_then_ret(&mut self.emitter);
    }

    fn emit_print(&mut self) {
        self.mark("print");
        emit_print_syswrite(&mut self.emitter);
    }

    fn emit_exit(&mut self) {
        self.mark("exit");
        emit_exit_syscall(&mut self.emitter);
    }
}

fn emit_trap_then_ret(emitter: &mut NativeEmitter) {
    #[cfg(target_arch = "x86_64")]
    emitter.emit_code(&[0x0f, 0x0b, 0xc3]); // ud2; ret

    #[cfg(target_arch = "aarch64")]
    emitter.emit_code(&[
        0x00, 0x00, 0x20, 0xd4, // brk #0
        0xc0, 0x03, 0x5f, 0xd6, // ret
    ]);
}

fn emit_print_syswrite(emitter: &mut NativeEmitter) {
    // Assumption for the initial bootstrap runtime:
    // - System V ABI style args: rdi = ptr, rsi = len (x86_64) / x0 = ptr, x1 = len (aarch64).
    // - Implements: write(1, ptr, len); return.
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    emitter.emit_code(&[
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1 (SYS_write)
        0x48, 0x89, 0xf2, // mov rdx, rsi (len)
        0x48, 0x89, 0xfe, // mov rsi, rdi (buf)
        0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00, // mov rdi, 1 (stdout)
        0x0f, 0x05, // syscall
        0xc3, // ret
    ]);

    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    emitter.emit_code(&[
        0x48, 0xc7, 0xc0, 0x04, 0x00, 0x00, 0x02, // mov rax, 0x2000004 (SYS_write)
        0x48, 0x89, 0xf2, // mov rdx, rsi (len)
        0x48, 0x89, 0xfe, // mov rsi, rdi (buf)
        0x48, 0xc7, 0xc7, 0x01, 0x00, 0x00, 0x00, // mov rdi, 1 (stdout)
        0x0f, 0x05, // syscall
        0xc3, // ret
    ]);

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    emitter.emit_code(&[
        0xe2, 0x03, 0x01, 0xaa, // mov x2, x1 (len)
        0xe1, 0x03, 0x00, 0xaa, // mov x1, x0 (buf)
        0x20, 0x00, 0x80, 0xd2, // movz x0, #1 (stdout)
        0x08, 0x08, 0x80, 0xd2, // movz x8, #64 (SYS_write)
        0x01, 0x00, 0x00, 0xd4, // svc #0
        0xc0, 0x03, 0x5f, 0xd6, // ret
    ]);

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    emitter.emit_code(&[
        0xe2, 0x03, 0x01, 0xaa, // mov x2, x1 (len)
        0xe1, 0x03, 0x00, 0xaa, // mov x1, x0 (buf)
        0x20, 0x00, 0x80, 0xd2, // movz x0, #1 (stdout)
        0x90, 0x00, 0x80, 0xd2, // movz x16, #4 (write)
        0x10, 0x40, 0xa0, 0xf2, // movk x16, #0x0200, lsl #16
        0x01, 0x10, 0x00, 0xd4, // svc #0x80
        0xc0, 0x03, 0x5f, 0xd6, // ret
    ]);
}

fn emit_exit_syscall(emitter: &mut NativeEmitter) {
    // exit(code) where code is already in rdi (x86_64) / x0 (aarch64).
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    emitter.emit_code(&[
        0x48, 0xc7, 0xc0, 0x3c, 0x00, 0x00, 0x00, // mov rax, 60 (SYS_exit)
        0x0f, 0x05, // syscall
    ]);

    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    emitter.emit_code(&[
        0x48, 0xc7, 0xc0, 0x01, 0x00, 0x00, 0x02, // mov rax, 0x2000001 (SYS_exit)
        0x0f, 0x05, // syscall
    ]);

    #[cfg(all(target_arch = "aarch64", target_os = "linux"))]
    emitter.emit_code(&[
        0xa8, 0x0b, 0x80, 0xd2, // movz x8, #93 (SYS_exit)
        0x01, 0x00, 0x00, 0xd4, // svc #0
    ]);

    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    emitter.emit_code(&[
        0x30, 0x00, 0x80, 0xd2, // movz x16, #1 (exit)
        0x10, 0x40, 0xa0, 0xf2, // movk x16, #0x0200, lsl #16
        0x01, 0x10, 0x00, 0xd4, // svc #0x80
    ]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_codegen_registers_offsets() {
        let mut rt = RuntimeCodegen::new();
        rt.emit_runtime();
        assert!(rt.builtin_offset("print").is_some());
        assert!(rt.builtin_offset("exit").is_some());
        assert!(rt.emitter().code.len() > 0);
    }
}
