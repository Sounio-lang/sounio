//! # Sounio Compiler: Native Runtime
//!
//! Minimal runtime support for standalone Sounio executables on Linux x86-64.
//!
//! ## Components
//!
//! 1. **Entry point** (`_start`) - Sets up stack and calls main
//! 2. **Exit** - Syscall wrapper for program termination
//! 3. **Write** - Syscall wrapper for stdout/stderr output
//! 4. **Panic handler** - Epistemic panic with confidence reporting
//!
//! ## Usage
//!
//! The runtime is automatically linked when producing executables:
//!
//! ```bash
//! souc build --backend=native --output=exec main.dm -o myprogram
//! ./myprogram
//! ```
//!
//! ## Syscall Numbers (x86-64 Linux)
//!
//! - read:  0
//! - write: 1
//! - exit:  60
//!
//! ## Author
//!
//! Demetrios Chiuratto Agourakis <demetrios@chiuratto.ai>

// ============================================================================
// SYSCALL NUMBERS (x86-64 Linux)
// ============================================================================

const SYS_READ: u64 = 0;
const SYS_WRITE: u64 = 1;
const SYS_OPEN: u64 = 2;
const SYS_CLOSE: u64 = 3;
const SYS_MMAP: u64 = 9;
const SYS_MUNMAP: u64 = 11;
const SYS_BRK: u64 = 12;
const SYS_EXIT: u64 = 60;
const SYS_EXIT_GROUP: u64 = 231;

// File descriptors
const STDIN: u64 = 0;
const STDOUT: u64 = 1;
const STDERR: u64 = 2;

// ============================================================================
// RUNTIME ASSEMBLY CODE GENERATION
// ============================================================================

/// Generate the _start entry point assembly
pub fn generate_start_asm() -> String {
    r#"
# Demetrios Runtime: Entry Point
# x86-64 Linux System V ABI

.section .text
.globl _start
.type _start, @function
_start:
    # Clear frame pointer (ABI requirement)
    xorq %rbp, %rbp

    # Stack layout at _start:
    # (%rsp)     = argc
    # 8(%rsp)    = argv[0]
    # ...        = argv[argc-1]
    # 0          = NULL
    # ...        = envp

    # Get argc
    movq (%rsp), %rdi

    # Get argv (pointer to argv[0])
    leaq 8(%rsp), %rsi

    # Calculate envp: argv + 8*(argc+1)
    movq %rdi, %rdx
    incq %rdx
    shlq $3, %rdx
    addq %rsi, %rdx

    # Align stack to 16 bytes (required by ABI)
    andq $-16, %rsp

    # Call main(argc, argv, envp)
    # If main is not defined, link will fail
    call main

    # main returned - exit with return value
    movq %rax, %rdi
    call _demetrios_exit

    # Should never reach here
    ud2

.size _start, .-_start
"#.to_string()
}

/// Generate exit syscall wrapper
pub fn generate_exit_asm() -> String {
    r#"
# Demetrios Runtime: Exit
.section .text
.globl _demetrios_exit
.type _demetrios_exit, @function
_demetrios_exit:
    # rdi = exit code
    movq $60, %rax      # SYS_exit
    syscall
    # Never returns
    ud2
.size _demetrios_exit, .-_demetrios_exit

.globl _demetrios_exit_group
.type _demetrios_exit_group, @function
_demetrios_exit_group:
    # rdi = exit code
    movq $231, %rax     # SYS_exit_group
    syscall
    ud2
.size _demetrios_exit_group, .-_demetrios_exit_group
"#.to_string()
}

/// Generate write syscall wrapper
pub fn generate_write_asm() -> String {
    r#"
# Demetrios Runtime: Write
.section .text
.globl _demetrios_write
.type _demetrios_write, @function
_demetrios_write:
    # rdi = fd
    # rsi = buf
    # rdx = count
    movq $1, %rax       # SYS_write
    syscall
    retq
.size _demetrios_write, .-_demetrios_write

# Convenience: print to stdout
.globl _demetrios_print
.type _demetrios_print, @function
_demetrios_print:
    # rdi = buf
    # rsi = count
    movq %rsi, %rdx     # count -> rdx
    movq %rdi, %rsi     # buf -> rsi
    movq $1, %rdi       # fd = stdout
    movq $1, %rax       # SYS_write
    syscall
    retq
.size _demetrios_print, .-_demetrios_print

# Print to stderr
.globl _demetrios_eprint
.type _demetrios_eprint, @function
_demetrios_eprint:
    # rdi = buf
    # rsi = count
    movq %rsi, %rdx     # count -> rdx
    movq %rdi, %rsi     # buf -> rsi
    movq $2, %rdi       # fd = stderr
    movq $1, %rax       # SYS_write
    syscall
    retq
.size _demetrios_eprint, .-_demetrios_eprint
"#.to_string()
}

/// Generate read syscall wrapper
pub fn generate_read_asm() -> String {
    r#"
# Demetrios Runtime: Read
.section .text
.globl _demetrios_read
.type _demetrios_read, @function
_demetrios_read:
    # rdi = fd
    # rsi = buf
    # rdx = count
    movq $0, %rax       # SYS_read
    syscall
    retq
.size _demetrios_read, .-_demetrios_read
"#.to_string()
}

/// Generate epistemic panic handler
pub fn generate_panic_asm() -> String {
    r#"
# Demetrios Runtime: Epistemic Panic
# Called when confidence drops below threshold
.section .text
.globl _demetrios_epistemic_panic
.type _demetrios_epistemic_panic, @function
_demetrios_epistemic_panic:
    # rdi = confidence (as u64 bits of f64)
    # rsi = threshold (as u64 bits of f64)
    
    # Save confidence for error message
    pushq %rdi
    
    # Print error message
    leaq .Lepistemic_panic_msg(%rip), %rdi
    movq $.Lepistemic_panic_msg_len, %rsi
    call _demetrios_eprint
    
    # Exit with error code 42 (epistemic failure)
    movq $42, %rdi
    call _demetrios_exit

.section .rodata
.Lepistemic_panic_msg:
    .ascii "EPISTEMIC PANIC: Confidence below threshold\n"
.Lepistemic_panic_msg_len = . - .Lepistemic_panic_msg

.section .text
.size _demetrios_epistemic_panic, .-_demetrios_epistemic_panic

# Panic with message
.globl _demetrios_panic
.type _demetrios_panic, @function
_demetrios_panic:
    # rdi = message ptr
    # rsi = message len
    
    # Print "PANIC: "
    pushq %rdi
    pushq %rsi
    leaq .Lpanic_prefix(%rip), %rdi
    movq $.Lpanic_prefix_len, %rsi
    call _demetrios_eprint
    
    # Print message
    popq %rsi
    popq %rdi
    call _demetrios_eprint
    
    # Print newline
    leaq .Lnewline(%rip), %rdi
    movq $1, %rsi
    call _demetrios_eprint
    
    # Exit with code 1
    movq $1, %rdi
    call _demetrios_exit

.section .rodata
.Lpanic_prefix:
    .ascii "PANIC: "
.Lpanic_prefix_len = . - .Lpanic_prefix
.Lnewline:
    .ascii "\n"

.section .text
.size _demetrios_panic, .-_demetrios_panic
"#.to_string()
}

/// Generate memory allocation stubs
pub fn generate_memory_asm() -> String {
    r#"
# Demetrios Runtime: Memory Management
# Simple bump allocator using brk()

.section .data
.align 8
_heap_start:
    .quad 0
_heap_current:
    .quad 0
_heap_end:
    .quad 0

.section .text

# Initialize heap
.globl _demetrios_heap_init
.type _demetrios_heap_init, @function
_demetrios_heap_init:
    # Get current brk
    xorq %rdi, %rdi
    movq $12, %rax      # SYS_brk
    syscall
    
    # Store as heap start and current
    movq %rax, _heap_start(%rip)
    movq %rax, _heap_current(%rip)
    
    # Request initial heap (64KB)
    addq $0x10000, %rax
    movq %rax, %rdi
    movq $12, %rax      # SYS_brk
    syscall
    
    # Store as heap end
    movq %rax, _heap_end(%rip)
    retq
.size _demetrios_heap_init, .-_demetrios_heap_init

# Allocate memory (simple bump allocator)
# rdi = size
# returns: pointer in rax, or 0 on failure
.globl _demetrios_alloc
.type _demetrios_alloc, @function
_demetrios_alloc:
    # Align size to 16 bytes
    addq $15, %rdi
    andq $-16, %rdi
    
    # Load current pointer
    movq _heap_current(%rip), %rax
    
    # Calculate new pointer
    movq %rax, %rcx
    addq %rdi, %rcx
    
    # Check if we have space
    cmpq _heap_end(%rip), %rcx
    ja .Lalloc_grow
    
    # Update current and return old pointer
    movq %rcx, _heap_current(%rip)
    retq

.Lalloc_grow:
    # Need to grow heap
    pushq %rdi          # Save size
    pushq %rax          # Save old pointer
    
    # Request more space (at least 64KB or requested size)
    movq _heap_end(%rip), %rdi
    addq $0x10000, %rdi
    movq $12, %rax      # SYS_brk
    syscall
    
    # Update heap end
    movq %rax, _heap_end(%rip)
    
    popq %rax           # Restore old pointer
    popq %rdi           # Restore size
    
    # Try allocation again
    movq %rax, %rcx
    addq %rdi, %rcx
    
    cmpq _heap_end(%rip), %rcx
    ja .Lalloc_fail
    
    movq %rcx, _heap_current(%rip)
    retq

.Lalloc_fail:
    xorq %rax, %rax     # Return NULL
    retq
.size _demetrios_alloc, .-_demetrios_alloc

# Free memory (no-op for bump allocator)
.globl _demetrios_free
.type _demetrios_free, @function
_demetrios_free:
    # Bump allocator doesn't support free
    retq
.size _demetrios_free, .-_demetrios_free
"#.to_string()
}

/// Generate floating-point helpers
pub fn generate_float_asm() -> String {
    r#"
# Demetrios Runtime: Floating-Point Helpers

.section .text

# Square root (wrapper for sqrtsd)
.globl _demetrios_sqrt
.type _demetrios_sqrt, @function
_demetrios_sqrt:
    # xmm0 = input
    sqrtsd %xmm0, %xmm0
    retq
.size _demetrios_sqrt, .-_demetrios_sqrt

# Natural logarithm (x87 fyl2x * ln(2))
.globl _demetrios_ln
.type _demetrios_ln, @function
_demetrios_ln:
    # xmm0 = input
    subq $16, %rsp
    movsd %xmm0, (%rsp)
    fldln2              # Load ln(2)
    fldl (%rsp)         # Load x
    fyl2x               # ln(2) * log2(x) = ln(x)
    fstpl (%rsp)
    movsd (%rsp), %xmm0
    addq $16, %rsp
    retq
.size _demetrios_ln, .-_demetrios_ln

# Exponential (e^x using x87 f2xm1)
.globl _demetrios_exp
.type _demetrios_exp, @function
_demetrios_exp:
    # xmm0 = input x
    subq $16, %rsp
    movsd %xmm0, (%rsp)
    fldl (%rsp)         # x
    fldl2e              # log2(e)
    fmulp               # x * log2(e)
    fld %st(0)          # duplicate
    frndint             # integer part
    fxch %st(1)         # swap
    fsub %st(1), %st(0) # fractional part
    f2xm1               # 2^frac - 1
    fld1
    faddp               # 2^frac
    fscale              # 2^frac * 2^int
    fstp %st(1)         # cleanup
    fstpl (%rsp)
    movsd (%rsp), %xmm0
    addq $16, %rsp
    retq
.size _demetrios_exp, .-_demetrios_exp

# Power function (x^y = e^(y*ln(x)))
.globl _demetrios_pow
.type _demetrios_pow, @function
_demetrios_pow:
    # xmm0 = x, xmm1 = y
    # Result = exp(y * ln(x))
    subq $32, %rsp
    
    # Handle special cases
    xorpd %xmm2, %xmm2
    ucomisd %xmm2, %xmm0
    jbe .Lpow_special    # x <= 0
    
    # ln(x)
    movsd %xmm0, (%rsp)
    movsd %xmm1, 8(%rsp)
    call _demetrios_ln
    
    # y * ln(x)
    mulsd 8(%rsp), %xmm0
    
    # exp(y * ln(x))
    call _demetrios_exp
    
    addq $32, %rsp
    retq

.Lpow_special:
    # Handle x <= 0 (return NaN for now)
    movq $0x7ff8000000000000, %rax  # NaN
    movq %rax, %xmm0
    addq $32, %rsp
    retq
.size _demetrios_pow, .-_demetrios_pow
"#.to_string()
}

/// Generate complete runtime assembly
pub fn generate_runtime_asm() -> String {
    let mut asm = String::new();
    
    asm.push_str("# Demetrios Runtime Library\n");
    asm.push_str("# Auto-generated - do not edit\n\n");
    
    asm.push_str(&generate_start_asm());
    asm.push_str("\n");
    asm.push_str(&generate_exit_asm());
    asm.push_str("\n");
    asm.push_str(&generate_write_asm());
    asm.push_str("\n");
    asm.push_str(&generate_read_asm());
    asm.push_str("\n");
    asm.push_str(&generate_panic_asm());
    asm.push_str("\n");
    asm.push_str(&generate_memory_asm());
    asm.push_str("\n");
    asm.push_str(&generate_float_asm());
    
    asm
}

/// Write runtime to file
pub fn write_runtime_asm(path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    let asm = generate_runtime_asm();
    std::fs::write(path, asm)
}

/// Build runtime object file
pub fn build_runtime_object(output_dir: impl AsRef<std::path::Path>) -> Result<std::path::PathBuf, String> {
    let output_dir = output_dir.as_ref();
    let asm_path = output_dir.join("demetrios_runtime.s");
    let obj_path = output_dir.join("demetrios_runtime.o");
    
    // Write assembly
    write_runtime_asm(&asm_path)
        .map_err(|e| format!("Failed to write runtime assembly: {}", e))?;
    
    // Assemble
    let output = std::process::Command::new("as")
        .arg("-o")
        .arg(&obj_path)
        .arg(&asm_path)
        .output()
        .map_err(|e| format!("Failed to run assembler: {}", e))?;
    
    if !output.status.success() {
        return Err(format!(
            "Assembler failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    
    Ok(obj_path)
}

// ============================================================================
// RUNTIME SYMBOLS
// ============================================================================

/// List of runtime symbols that can be called from user code
pub const RUNTIME_SYMBOLS: &[&str] = &[
    "_demetrios_exit",
    "_demetrios_exit_group",
    "_demetrios_write",
    "_demetrios_print",
    "_demetrios_eprint",
    "_demetrios_read",
    "_demetrios_epistemic_panic",
    "_demetrios_panic",
    "_demetrios_heap_init",
    "_demetrios_alloc",
    "_demetrios_free",
    "_demetrios_sqrt",
    "_demetrios_ln",
    "_demetrios_exp",
    "_demetrios_pow",
];

/// Check if a symbol is provided by the runtime
pub fn is_runtime_symbol(name: &str) -> bool {
    RUNTIME_SYMBOLS.contains(&name)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_runtime() {
        let asm = generate_runtime_asm();
        
        // Check for key components
        assert!(asm.contains("_start:"));
        assert!(asm.contains("_demetrios_exit:"));
        assert!(asm.contains("_demetrios_write:"));
        assert!(asm.contains("_demetrios_epistemic_panic:"));
        assert!(asm.contains("_demetrios_sqrt:"));
    }

    #[test]
    fn test_runtime_symbols() {
        assert!(is_runtime_symbol("_demetrios_exit"));
        assert!(is_runtime_symbol("_demetrios_alloc"));
        assert!(!is_runtime_symbol("main"));
        assert!(!is_runtime_symbol("random_func"));
    }

    #[test]
    fn test_start_asm() {
        let asm = generate_start_asm();
        assert!(asm.contains("_start:"));
        assert!(asm.contains("call main"));
        assert!(asm.contains("call _demetrios_exit"));
    }
}

// ============================================================================
// MODULE EXPORTS
// ============================================================================

pub mod prelude {
    pub use super::{
        generate_runtime_asm,
        write_runtime_asm,
        build_runtime_object,
        is_runtime_symbol,
        RUNTIME_SYMBOLS,
    };
}
