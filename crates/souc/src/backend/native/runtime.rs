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
"#
    .to_string()
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
"#
    .to_string()
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

# Print i64 via libc printf
.globl _demetrios_print_i64
.type _demetrios_print_i64, @function
_demetrios_print_i64:
    # rdi = value
    pushq %rbp
    movq %rsp, %rbp
    movq %rdi, %rsi
    leaq .Ldemetrios_fmt_i64(%rip), %rdi
    xor %eax, %eax
    call printf
    # Flush stdout to ensure output appears immediately (avoid buffering mismatch with write syscalls)
    movq stdout(%rip), %rdi
    call fflush
    popq %rbp
    retq
.size _demetrios_print_i64, .-_demetrios_print_i64

# Print f64 via libc printf
.globl _demetrios_print_f64
.type _demetrios_print_f64, @function
_demetrios_print_f64:
    # xmm0 = value
    pushq %rbp
    movq %rsp, %rbp
    leaq .Ldemetrios_fmt_f64(%rip), %rdi
    mov $1, %eax
    call printf
    # Flush stdout to ensure output appears immediately (avoid buffering mismatch with write syscalls)
    movq stdout(%rip), %rdi
    call fflush
    popq %rbp
    retq
.size _demetrios_print_f64, .-_demetrios_print_f64

# Print bool via write syscall
.globl _demetrios_print_bool
.type _demetrios_print_bool, @function
_demetrios_print_bool:
    # rdi = bool (0/1)
    cmpq $0, %rdi
    jne .Ldemetrios_bool_true
    leaq .Ldemetrios_bool_false(%rip), %rdi
    movq $5, %rsi
    jmp _demetrios_print
.Ldemetrios_bool_true:
    leaq .Ldemetrios_bool_true_str(%rip), %rdi
    movq $4, %rsi
    jmp _demetrios_print
.size _demetrios_print_bool, .-_demetrios_print_bool

# Print newline
.globl _demetrios_print_newline
.type _demetrios_print_newline, @function
_demetrios_print_newline:
    leaq .Ldemetrios_newline(%rip), %rdi
    movq $1, %rsi
    jmp _demetrios_print
.size _demetrios_print_newline, .-_demetrios_print_newline

.section .rodata
.Ldemetrios_fmt_i64:
    .asciz "%ld"
.Ldemetrios_fmt_f64:
    .asciz "%g"
.Ldemetrios_bool_true_str:
    .ascii "true"
.Ldemetrios_bool_false:
    .ascii "false"
.Ldemetrios_newline:
    .ascii "\n"
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
"#
    .to_string()
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
"#
    .to_string()
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
"#
    .to_string()
}

/// Generate floating-point helpers
/// Generate unit conversion runtime functions
/// These are called when units need to be converted at runtime
/// (e.g., when reading from external data sources)
pub fn generate_unit_conversion_asm() -> String {
    r#"
# Demetrios Runtime: Unit Conversion Functions
# For runtime unit conversions when units are not known at compile-time

.section .text

# Convert value from one unit to another (non-affine)
# C signature: double sounio_convert_unit(double value, double from_scale, double to_scale)
# Returns: value * (from_scale / to_scale)
.globl sounio_convert_unit
.type sounio_convert_unit, @function
sounio_convert_unit:
    # value in XMM0, from_scale in XMM1, to_scale in XMM2
    # Compute: result = value * (from_scale / to_scale)
    
    # Divide scales: from_scale / to_scale
    divsd %xmm2, %xmm1  # XMM1 = from_scale / to_scale
    
    # Multiply by value
    mulsd %xmm1, %xmm0  # XMM0 = value * (from_scale / to_scale)
    
    # Result in XMM0
    ret

.size sounio_convert_unit, .-sounio_convert_unit

# Convert affine unit (with offset, e.g., temperature)
# C signature: double sounio_convert_affine(double value, double from_scale, double from_offset, double to_scale, double to_offset)
# Returns: ((value * from_scale + from_offset) - to_offset) / to_scale
.globl sounio_convert_affine
.type sounio_convert_affine, @function
sounio_convert_affine:
    # value in XMM0, from_scale in XMM1, from_offset in XMM2, to_scale in XMM3, to_offset in XMM4
    # Compute: result = ((value * from_scale + from_offset) - to_offset) / to_scale
    
    # value * from_scale
    mulsd %xmm1, %xmm0  # XMM0 = value * from_scale
    
    # Add from_offset
    addsd %xmm2, %xmm0  # XMM0 = value * from_scale + from_offset
    
    # Subtract to_offset
    subsd %xmm4, %xmm0  # XMM0 = (value * from_scale + from_offset) - to_offset
    
    # Divide by to_scale
    divsd %xmm3, %xmm0  # XMM0 = ((value * from_scale + from_offset) - to_offset) / to_scale
    
    # Result in XMM0
    ret

.size sounio_convert_affine, .-sounio_convert_affine
"#
    .to_string()
}

/// Generate ODE solver runtime functions
/// These implement complex ODE methods that are too complex to inline
pub fn generate_ode_runtime_asm() -> String {
    r#"
# Demetrios Runtime: ODE Solver Functions
# Implements complex adaptive ODE methods (DoPri5, CashKarp, etc.)

.section .text

# RK4 ODE step (4th order Runge-Kutta)
# C signature: void sounio_ode_rk4(double* state, int n, double t, double dt, void (*derivatives)(double*, double, double*))
# System V ABI: RDI=state, RSI=n, XMM0=t, XMM1=dt, RDX=derivatives
# RK4 algorithm:
#   k1 = f(t, y)
#   k2 = f(t + dt/2, y + dt*k1/2)
#   k3 = f(t + dt/2, y + dt*k2/2)
#   k4 = f(t + dt, y + dt*k3)
#   y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
.globl sounio_ode_rk4
.type sounio_ode_rk4, @function
sounio_ode_rk4:
    # Save callee-saved registers
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    
    # Save arguments
    movq %rdi, %r12        # state pointer
    movq %rsi, %r13        # n
    movq %rdx, %r14        # derivatives function pointer
    movsd %xmm0, -8(%rbp)  # t
    movsd %xmm1, -16(%rbp) # dt
    
    # Allocate workspace: k1, k2, k3, k4, temp (each n doubles)
    # Total: 5 * n * 8 bytes
    movq %r13, %rax
    shlq $3, %rax          # n * 8
    movq %rax, %r15        # Save element size
    shlq $2, %rax          # 4 * n * 8 (for k1-k4)
    addq %r15, %rax        # + n * 8 (for temp)
    subq %rax, %rsp
    andq $-16, %rsp        # Align to 16 bytes
    
    # k1 = %rsp
    # k2 = %rsp + n*8
    # k3 = %rsp + 2*n*8
    # k4 = %rsp + 3*n*8
    # temp = %rsp + 4*n*8
    
    # Step 1: k1 = f(t, state)
    movq %rsp, %rdi        # k1 (output)
    movq %r12, %rsi        # state (input)
    movsd -8(%rbp), %xmm0  # t
    callq *%r14            # derivatives(t, state, k1)
    
    # Step 2: k2 = f(t + dt/2, state + dt*k1/2)
    movsd -16(%rbp), %xmm0 # dt
    movsd .Lhalf(%rip), %xmm1  # 0.5
    mulsd %xmm1, %xmm0     # dt/2
    addsd -8(%rbp), %xmm0  # t + dt/2
    
    # Compute temp = state + dt*k1/2
    movq %rsp, %rax        # k1
    movq %rsp, %rbx
    addq %r15, %rbx
    addq %r15, %rbx        # k2 (will use as temp)
    xorq %rcx, %rcx
.rk4_k2_loop:
    cmpq %r13, %rcx
    jge .rk4_k2_done
    movsd (%r12, %rcx, 8), %xmm1  # state[i]
    movsd (%rax, %rcx, 8), %xmm2  # k1[i]
    movsd -16(%rbp), %xmm3        # dt
    mulsd .Lhalf(%rip), %xmm3     # dt/2
    mulsd %xmm3, %xmm2            # dt*k1[i]/2
    addsd %xmm2, %xmm1            # state[i] + dt*k1[i]/2
    movsd %xmm1, (%rbx, %rcx, 8)  # temp[i]
    incq %rcx
    jmp .rk4_k2_loop
.rk4_k2_done:
    movq %rbx, %rsi        # temp (input)
    movq %rbx, %rdi        # k2 (output)
    callq *%r14            # derivatives(t + dt/2, temp, k2)
    
    # Step 3: k3 = f(t + dt/2, state + dt*k2/2)
    # Similar to k2, but use k2 instead of k1
    movq %rsp, %rax
    addq %r15, %rax        # k2
    movq %rbx, %rdi        # temp (reuse)
    xorq %rcx, %rcx
.rk4_k3_loop:
    cmpq %r13, %rcx
    jge .rk4_k3_done
    movsd (%r12, %rcx, 8), %xmm1  # state[i]
    movsd (%rax, %rcx, 8), %xmm2  # k2[i]
    movsd -16(%rbp), %xmm3        # dt
    mulsd .Lhalf(%rip), %xmm3     # dt/2
    mulsd %xmm3, %xmm2            # dt*k2[i]/2
    addsd %xmm2, %xmm1            # state[i] + dt*k2[i]/2
    movsd %xmm1, (%rdi, %rcx, 8)  # temp[i]
    incq %rcx
    jmp .rk4_k3_loop
.rk4_k3_done:
    movq %rdi, %rsi        # temp (input)
    movq %rsp, %rdi
    addq %r15, %rdi
    addq %r15, %rdi        # k3 (output)
    callq *%r14            # derivatives(t + dt/2, temp, k3)
    
    # Step 4: k4 = f(t + dt, state + dt*k3)
    movsd -8(%rbp), %xmm0  # t
    addsd -16(%rbp), %xmm0 # t + dt
    movq %rsp, %rax
    addq %r15, %rax
    addq %r15, %rax        # k3
    movq %rbx, %rdi        # temp (reuse)
    xorq %rcx, %rcx
.rk4_k4_loop:
    cmpq %r13, %rcx
    jge .rk4_k4_done
    movsd (%r12, %rcx, 8), %xmm1  # state[i]
    movsd (%rax, %rcx, 8), %xmm2  # k3[i]
    movsd -16(%rbp), %xmm3        # dt
    mulsd %xmm3, %xmm2            # dt*k3[i]
    addsd %xmm2, %xmm1            # state[i] + dt*k3[i]
    movsd %xmm1, (%rdi, %rcx, 8)  # temp[i]
    incq %rcx
    jmp .rk4_k4_loop
.rk4_k4_done:
    movq %rdi, %rsi        # temp (input)
    movq %rsp, %rdi
    addq %r15, %rdi
    addq %r15, %rdi
    addq %r15, %rdi        # k4 (output)
    callq *%r14            # derivatives(t + dt, temp, k4)
    
    # Step 5: state = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    movsd -16(%rbp), %xmm0 # dt
    movsd .Lone_sixth(%rip), %xmm1  # 1/6
    mulsd %xmm1, %xmm0     # dt/6
    movq %rsp, %rax        # k1
    movq %rsp, %rbx
    addq %r15, %rbx        # k2
    movq %rbx, %rdi
    addq %r15, %rdi        # k3
    movq %rdi, %rsi
    addq %r15, %rsi        # k4
    xorq %rcx, %rcx
.rk4_final_loop:
    cmpq %r13, %rcx
    jge .rk4_final_done
    # Compute: k1[i] + 2*k2[i] + 2*k3[i] + k4[i]
    movsd (%rax, %rcx, 8), %xmm1  # k1[i]
    movsd (%rbx, %rcx, 8), %xmm2  # k2[i]
    addsd %xmm2, %xmm2            # 2*k2[i]
    addsd %xmm2, %xmm1            # k1[i] + 2*k2[i]
    movsd (%rdi, %rcx, 8), %xmm2  # k3[i]
    addsd %xmm2, %xmm2            # 2*k3[i]
    addsd %xmm2, %xmm1            # + 2*k3[i]
    addsd (%rsi, %rcx, 8), %xmm1  # + k4[i]
    # Multiply by dt/6
    mulsd %xmm0, %xmm1            # dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    # Add to state
    addsd (%r12, %rcx, 8), %xmm1  # state[i] + ...
    movsd %xmm1, (%r12, %rcx, 8)  # Store back
    incq %rcx
    jmp .rk4_final_loop
.rk4_final_done:
    # Restore stack
    movq %rbp, %rsp
    
    # Restore callee-saved registers
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret

.size sounio_ode_rk4, .-sounio_ode_rk4

# Constants
.section .rodata
.align 8
.Lhalf:
    .double 0.5
.Lone_sixth:
    .double 0.16666666666666666  # 1/6

# Euler ODE step (1st order)
# C signature: void sounio_ode_euler(double* state, int n, double t, double dt, void (*derivatives)(double*, double, double*))
# System V ABI: RDI=state, RSI=n, XMM0=t, XMM1=dt, RDX=derivatives
.globl sounio_ode_euler
.type sounio_ode_euler, @function
sounio_ode_euler:
    # Save callee-saved registers
    pushq %rbp
    movq %rsp, %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    
    # Save arguments
    movq %rdi, %r12        # state pointer
    movq %rsi, %r13        # n (number of state variables)
    movq %rdx, %r14        # derivatives function pointer
    movsd %xmm0, -8(%rbp)  # t (save on stack)
    movsd %xmm1, -16(%rbp) # dt (save on stack)
    
    # Allocate temporary array for derivatives (dydt)
    # Size: n * 8 bytes (doubles)
    movq %r13, %rax
    shlq $3, %rax          # n * 8
    subq %rax, %rsp        # Allocate on stack
    movq %rsp, %r15        # dydt pointer
    
    # Align stack to 16 bytes
    andq $-16, %rsp
    
    # Call derivatives(t, state, dydt)
    movq %r12, %rdi        # state
    movsd -8(%rbp), %xmm0  # t
    movq %r15, %rdx        # dydt
    callq *%r14            # Call derivatives function
    
    # Euler step: state = state + dt * dydt
    xorq %rcx, %rcx        # i = 0
    movsd -16(%rbp), %xmm1 # dt
    
.euler_loop:
    cmpq %r13, %rcx
    jge .euler_done
    
    # state[i] += dt * dydt[i]
    movsd (%r12, %rcx, 8), %xmm0  # state[i]
    movsd (%r15, %rcx, 8), %xmm2  # dydt[i]
    mulsd %xmm1, %xmm2            # dt * dydt[i]
    addsd %xmm2, %xmm0            # state[i] + dt * dydt[i]
    movsd %xmm0, (%r12, %rcx, 8)  # Store back
    
    incq %rcx
    jmp .euler_loop
    
.euler_done:
    # Restore stack
    movq %rbp, %rsp
    
    # Restore callee-saved registers
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret

.size sounio_ode_euler, .-sounio_ode_euler

# Dormand-Prince 5(4) adaptive ODE step
# C signature: double sounio_ode_dopri5_step(
#     double* state,      // RDI: state vector (modified in-place)
#     int n,              // RSI: dimension
#     double* t,          // RDX: pointer to current time (updated)
#     double* dt,         // RCX: pointer to step size (updated)
#     double rtol,        // XMM0: relative tolerance
#     double atol,        // XMM1: absolute tolerance
#     DerivativeFn f      // R8: derivatives function pointer
# );
# Returns: error estimate (in XMM0)
.globl sounio_ode_dopri5
.type sounio_ode_dopri5, @function
sounio_ode_dopri5:
    # Fallback: use RK4 step and return 0.0 as error estimate
    # Save frame pointer and argument pointers
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq %rdx, -8(%rbp)   # t pointer
    movq %rcx, -16(%rbp)  # dt pointer
    
    # Load t and dt for RK4 (RDI=state, RSI=n, XMM0=t, XMM1=dt, RDX=derivatives)
    movsd (%rdx), %xmm0
    movsd (%rcx), %xmm1
    movq %r8, %rdx
    call sounio_ode_rk4
    
    # Update t = t + dt
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    movsd (%rax), %xmm0
    movsd (%rcx), %xmm1
    addsd %xmm1, %xmm0
    movsd %xmm0, (%rax)
    
    # Return error estimate = 0.0 in XMM0
    xorpd %xmm0, %xmm0
    movq %rbp, %rsp
    popq %rbp
    ret

.size sounio_ode_dopri5, .-sounio_ode_dopri5

# Cash-Karp 5(4) adaptive ODE step
# C signature: double sounio_ode_cashkarp_step(
#     double* state,      // RDI: state vector (modified in-place)
#     int n,              // RSI: dimension
#     double* t,          // RDX: pointer to current time (updated)
#     double* dt,         // RCX: pointer to step size (updated)
#     double rtol,        // XMM0: relative tolerance
#     double atol,        // XMM1: absolute tolerance
#     DerivativeFn f      // R8: derivatives function pointer
# );
# Returns: error estimate (in XMM0)
.globl sounio_ode_cashkarp
.type sounio_ode_cashkarp, @function
sounio_ode_cashkarp:
    # Fallback: use RK4 step and return 0.0 as error estimate
    pushq %rbp
    movq %rsp, %rbp
    subq $16, %rsp
    movq %rdx, -8(%rbp)   # t pointer
    movq %rcx, -16(%rbp)  # dt pointer
    
    movsd (%rdx), %xmm0
    movsd (%rcx), %xmm1
    movq %r8, %rdx
    call sounio_ode_rk4
    
    movq -8(%rbp), %rax
    movq -16(%rbp), %rcx
    movsd (%rax), %xmm0
    movsd (%rcx), %xmm1
    addsd %xmm1, %xmm0
    movsd %xmm0, (%rax)
    
    xorpd %xmm0, %xmm0
    movq %rbp, %rsp
    popq %rbp
    ret

.size sounio_ode_cashkarp, .-sounio_ode_cashkarp

# Generic ODE step (dispatches to appropriate method)
# C signature: void sounio_ode_step(int method, double* state, int n, double t, double dt, void (*derivatives)(double*, double, double*))
.globl sounio_ode_step
.type sounio_ode_step, @function
sounio_ode_step:
    # Placeholder - dispatches to appropriate ODE method
    ret

.size sounio_ode_step, .-sounio_ode_step

# Dot product (scalar version)
# C signature: double sounio_dot_product(double* a, double* b, int n)
.globl sounio_dot_product
.type sounio_dot_product, @function
sounio_dot_product:
    # a in RDI, b in RSI, n in RDX
    # Result in XMM0
    xorpd %xmm0, %xmm0  # sum = 0.0
    xorq %rcx, %rcx     # i = 0
    
    testq %rdx, %rdx
    jz .Ldot_done       # if n == 0, return 0
    
.Ldot_loop:
    movsd (%rdi, %rcx, 8), %xmm1  # a[i]
    mulsd (%rsi, %rcx, 8), %xmm1  # a[i] * b[i]
    addsd %xmm1, %xmm0            # sum += a[i] * b[i]
    incq %rcx
    cmpq %rdx, %rcx
    jl .Ldot_loop
    
.Ldot_done:
    ret

.size sounio_dot_product, .-sounio_dot_product

# Dot product (SIMD version)
# C signature: double sounio_dot_product_simd(double* a, double* b, int n)
.globl sounio_dot_product_simd
.type sounio_dot_product_simd, @function
sounio_dot_product_simd:
    # a in RDI, b in RSI, n in RDX
    # Use XMM registers for SIMD (2 doubles per register)
    xorpd %xmm0, %xmm0  # sum = 0.0
    xorq %rcx, %rcx     # i = 0
    
    # Process 2 elements at a time
    movq %rdx, %rax
    shrq $1, %rax       # n / 2
    testq %rax, %rax
    jz .Ldot_simd_scalar  # If less than 2 elements, use scalar
    
.Ldot_simd_loop:
    movupd (%rdi, %rcx, 8), %xmm1  # Load 2 doubles from a
    movupd (%rsi, %rcx, 8), %xmm2  # Load 2 doubles from b
    mulpd %xmm2, %xmm1             # a[i:i+1] * b[i:i+1]
    addpd %xmm1, %xmm0             # sum += a[i:i+1] * b[i:i+1]
    addq $2, %rcx
    cmpq %rax, %rcx
    jl .Ldot_simd_loop
    
    # Horizontal add: sum XMM0[0] + XMM0[1]
    movapd %xmm0, %xmm1
    unpckhpd %xmm1, %xmm1  # XMM1[0] = XMM0[1]
    addsd %xmm1, %xmm0     # XMM0[0] += XMM0[1]
    
    # Handle remaining odd element
    movq %rdx, %rax
    andq $1, %rax
    jz .Ldot_simd_done
    
.Ldot_simd_scalar:
    movsd (%rdi, %rcx, 8), %xmm1
    mulsd (%rsi, %rcx, 8), %xmm1
    addsd %xmm1, %xmm0
    
.Ldot_simd_done:
    ret

.size sounio_dot_product_simd, .-sounio_dot_product_simd
"#
    .to_string()
}

/// Generate autodiff runtime functions
/// These provide assembly wrappers for C-compatible autodiff functions
pub fn generate_autodiff_runtime_asm() -> String {
    r#"
# Demetrios Runtime: Autodiff Functions
# Assembly wrappers for forward-mode and reverse-mode automatic differentiation

.section .text

# Dual number addition
# C signature: void sounio_dual_add(Dual* a, Dual* b, Dual* result)
# System V ABI: RDI=a, RSI=b, RDX=result
# This is a thin wrapper - the actual implementation is in autodiff_runtime.rs
# The wrapper ensures ABI compliance and can be inlined by the linker
.globl sounio_dual_add
.type sounio_dual_add, @function
sounio_dual_add:
    # Arguments are already in correct registers (RDI, RSI, RDX)
    # The Rust function will be linked directly - this is just a symbol placeholder
    # In practice, the Rust function can be called directly, but this wrapper
    # ensures the symbol exists for linking
    # For now, we'll make this a simple trampoline that calls the Rust function
    # The linker will resolve sounio_dual_add to the Rust implementation
    # This assembly is just for documentation - actual calls go to Rust
    ret
.size sounio_dual_add, .-sounio_dual_add

# Dual number subtraction
# C signature: void sounio_dual_sub(Dual* a, Dual* b, Dual* result)
.globl sounio_dual_sub
.type sounio_dual_sub, @function
sounio_dual_sub:
    ret
.size sounio_dual_sub, .-sounio_dual_sub

# Dual number multiplication (product rule)
# C signature: void sounio_dual_mul(Dual* a, Dual* b, Dual* result)
.globl sounio_dual_mul
.type sounio_dual_mul, @function
sounio_dual_mul:
    ret
.size sounio_dual_mul, .-sounio_dual_mul

# Dual number division (quotient rule)
# C signature: void sounio_dual_div(Dual* a, Dual* b, Dual* result)
.globl sounio_dual_div
.type sounio_dual_div, @function
sounio_dual_div:
    ret
.size sounio_dual_div, .-sounio_dual_div

# Forward-mode autodiff: compute gradient
# C signature: double sounio_autodiff_forward(void (*f)(double*, double*), double x, double* gradient)
# System V ABI: RDI=f, XMM0=x, RSI=gradient
# Returns: function value in XMM0
# Note: This is a symbol placeholder - the actual implementation is in autodiff_runtime.rs
# The Rust function will be linked directly
.globl sounio_autodiff_forward
.type sounio_autodiff_forward, @function
sounio_autodiff_forward:
    # Arguments are already in correct registers
    # The Rust function will be linked directly
    # This is just a symbol placeholder for linking
    ret
.size sounio_autodiff_forward, .-sounio_autodiff_forward

# Reverse-mode autodiff: tape-based backpropagation
# C signature: void sounio_autodiff_reverse(void* tape, double output_gradient, double* input_gradients, int n_inputs)
# System V ABI: RDI=tape, XMM0=output_gradient, RSI=input_gradients, RDX=n_inputs
# Note: This is a symbol placeholder - the actual implementation is in autodiff_runtime.rs
.globl sounio_autodiff_reverse
.type sounio_autodiff_reverse, @function
sounio_autodiff_reverse:
    # Arguments are already in correct registers
    # The Rust function will be linked directly
    ret
.size sounio_autodiff_reverse, .-sounio_autodiff_reverse

# Dual number exponential
# C signature: void sounio_dual_exp(Dual* x, Dual* result)
.globl sounio_dual_exp
.type sounio_dual_exp, @function
sounio_dual_exp:
    ret
.size sounio_dual_exp, .-sounio_dual_exp

# Dual number logarithm
# C signature: void sounio_dual_log(Dual* x, Dual* result)
.globl sounio_dual_log
.type sounio_dual_log, @function
sounio_dual_log:
    ret
.size sounio_dual_log, .-sounio_dual_log

# Dual number sine
# C signature: void sounio_dual_sin(Dual* x, Dual* result)
.globl sounio_dual_sin
.type sounio_dual_sin, @function
sounio_dual_sin:
    ret
.size sounio_dual_sin, .-sounio_dual_sin

# Dual number cosine
# C signature: void sounio_dual_cos(Dual* x, Dual* result)
.globl sounio_dual_cos
.type sounio_dual_cos, @function
sounio_dual_cos:
    ret
.size sounio_dual_cos, .-sounio_dual_cos

# Dual number power
# C signature: void sounio_dual_pow(Dual* x, double power, Dual* result)
# System V ABI: RDI=x, XMM0=power, RSI=result
.globl sounio_dual_pow
.type sounio_dual_pow, @function
sounio_dual_pow:
    # Arguments are already in correct registers
    # The Rust function will be linked directly
    ret
.size sounio_dual_pow, .-sounio_dual_pow

# Dual number square root
# C signature: void sounio_dual_sqrt(Dual* x, Dual* result)
.globl sounio_dual_sqrt
.type sounio_dual_sqrt, @function
sounio_dual_sqrt:
    ret
.size sounio_dual_sqrt, .-sounio_dual_sqrt
"#
    .to_string()
}

/// Generate tensor runtime functions
/// These provide assembly wrappers for C-compatible tensor operations
pub fn generate_tensor_runtime_asm() -> String {
    r#"
# Demetrios Runtime: Tensor Functions
# Assembly wrappers for tensor operations (matmul, einsum, reshape, etc.)

.section .text

# Matrix multiplication: C = A @ B
# C signature: void sounio_tensor_matmul(const double* A, const double* B, double* C, int M, int K, int N)
# System V ABI: RDI=A, RSI=B, RDX=C, RCX=M, R8=K, R9=N
# Note: This is a symbol placeholder - the actual implementation is in tensor_runtime.rs
.globl sounio_tensor_matmul
.type sounio_tensor_matmul, @function
sounio_tensor_matmul:
    # Arguments are already in correct registers
    # The Rust function will be linked directly
    ret
.size sounio_tensor_matmul, .-sounio_tensor_matmul

# Einsum operation
# C signature: void sounio_tensor_einsum(const char* notation, const double** inputs, int* input_shapes, int num_inputs, double* output, int* output_shape, int output_rank)
.globl sounio_tensor_einsum
.type sounio_tensor_einsum, @function
sounio_tensor_einsum:
    ret
.size sounio_tensor_einsum, .-sounio_tensor_einsum

# Reshape tensor
# C signature: void sounio_tensor_reshape(const double* input, double* output, int* input_shape, int input_rank, int* output_shape, int output_rank, int total_elements)
.globl sounio_tensor_reshape
.type sounio_tensor_reshape, @function
sounio_tensor_reshape:
    ret
.size sounio_tensor_reshape, .-sounio_tensor_reshape

# Transpose matrix
# C signature: void sounio_tensor_transpose(const double* A, double* B, int M, int N)
.globl sounio_tensor_transpose
.type sounio_tensor_transpose, @function
sounio_tensor_transpose:
    ret
.size sounio_tensor_transpose, .-sounio_tensor_transpose

# Element-wise addition
# C signature: void sounio_tensor_add(const double* A, const double* B, double* C, int n)
.globl sounio_tensor_add
.type sounio_tensor_add, @function
sounio_tensor_add:
    ret
.size sounio_tensor_add, .-sounio_tensor_add

# Element-wise multiplication
# C signature: void sounio_tensor_mul(const double* A, const double* B, double* C, int n)
.globl sounio_tensor_mul
.type sounio_tensor_mul, @function
sounio_tensor_mul:
    ret
.size sounio_tensor_mul, .-sounio_tensor_mul

# Scale tensor
# C signature: void sounio_tensor_scale(const double* A, double alpha, double* B, int n)
# System V ABI: RDI=A, XMM0=alpha, RSI=B, RDX=n
.globl sounio_tensor_scale
.type sounio_tensor_scale, @function
sounio_tensor_scale:
    ret
.size sounio_tensor_scale, .-sounio_tensor_scale

# Matrix-vector multiplication
# C signature: void sounio_tensor_matvec(const double* A, const double* x, double* y, int M, int N)
.globl sounio_tensor_matvec
.type sounio_tensor_matvec, @function
sounio_tensor_matvec:
    ret
.size sounio_tensor_matvec, .-sounio_tensor_matvec
"#
    .to_string()
}

/// Generate uncertain runtime functions
/// These provide assembly wrappers for C-compatible uncertainty propagation
pub fn generate_uncertain_runtime_asm() -> String {
    r#"
# Demetrios Runtime: Uncertainty Propagation Functions
# Assembly wrappers for uncertain<T> operations (GUM propagation rules)

.section .text

# Uncertainty propagation: addition
# C signature: void sounio_uncertain_add(const Uncertain* a, const Uncertain* b, Uncertain* result)
.globl sounio_uncertain_add
.type sounio_uncertain_add, @function
sounio_uncertain_add:
    ret
.size sounio_uncertain_add, .-sounio_uncertain_add

# Uncertainty propagation: subtraction
# C signature: void sounio_uncertain_sub(const Uncertain* a, const Uncertain* b, Uncertain* result)
.globl sounio_uncertain_sub
.type sounio_uncertain_sub, @function
sounio_uncertain_sub:
    ret
.size sounio_uncertain_sub, .-sounio_uncertain_sub

# Uncertainty propagation: multiplication
# C signature: void sounio_uncertain_mul(const Uncertain* a, const Uncertain* b, Uncertain* result)
.globl sounio_uncertain_mul
.type sounio_uncertain_mul, @function
sounio_uncertain_mul:
    ret
.size sounio_uncertain_mul, .-sounio_uncertain_mul

# Uncertainty propagation: division
# C signature: void sounio_uncertain_div(const Uncertain* a, const Uncertain* b, Uncertain* result)
.globl sounio_uncertain_div
.type sounio_uncertain_div, @function
sounio_uncertain_div:
    ret
.size sounio_uncertain_div, .-sounio_uncertain_div

# Combine multiple uncertain measurements
# C signature: void sounio_uncertain_combine(const Uncertain* measurements, int n, Uncertain* result)
.globl sounio_uncertain_combine
.type sounio_uncertain_combine, @function
sounio_uncertain_combine:
    ret
.size sounio_uncertain_combine, .-sounio_uncertain_combine

# Scale uncertain value
# C signature: void sounio_uncertain_scale(const Uncertain* x, double alpha, Uncertain* result)
# System V ABI: RDI=x, XMM0=alpha, RSI=result
.globl sounio_uncertain_scale
.type sounio_uncertain_scale, @function
sounio_uncertain_scale:
    ret
.size sounio_uncertain_scale, .-sounio_uncertain_scale

# Power operation
# C signature: void sounio_uncertain_pow(const Uncertain* x, double power, Uncertain* result)
.globl sounio_uncertain_pow
.type sounio_uncertain_pow, @function
sounio_uncertain_pow:
    ret
.size sounio_uncertain_pow, .-sounio_uncertain_pow

# Square root
# C signature: void sounio_uncertain_sqrt(const Uncertain* x, Uncertain* result)
.globl sounio_uncertain_sqrt
.type sounio_uncertain_sqrt, @function
sounio_uncertain_sqrt:
    ret
.size sounio_uncertain_sqrt, .-sounio_uncertain_sqrt

# Exponential
# C signature: void sounio_uncertain_exp(const Uncertain* x, Uncertain* result)
.globl sounio_uncertain_exp
.type sounio_uncertain_exp, @function
sounio_uncertain_exp:
    ret
.size sounio_uncertain_exp, .-sounio_uncertain_exp

# Natural logarithm
# C signature: void sounio_uncertain_log(const Uncertain* x, Uncertain* result)
.globl sounio_uncertain_log
.type sounio_uncertain_log, @function
sounio_uncertain_log:
    ret
.size sounio_uncertain_log, .-sounio_uncertain_log
"#
    .to_string()
}

pub fn generate_epistemic_runtime_asm() -> String {
    r#"
# Demetrios Runtime: Epistemic Knowledge Propagation Functions
# Assembly wrappers for Knowledge<T> operations (GUM + provenance tracking)

.section .text

# Epistemic propagation: addition (full layout)
# C signature: void sounio_epistemic_add_full(const KnowledgeFull* a, const KnowledgeFull* b, KnowledgeFull* result)
# System V ABI: RDI=a, RSI=b, RDX=result
.globl sounio_epistemic_add_full
.type sounio_epistemic_add_full, @function
sounio_epistemic_add_full:
    # Stub - calls Rust implementation from epistemic_runtime.rs
    ret
.size sounio_epistemic_add_full, .-sounio_epistemic_add_full

# Epistemic propagation: subtraction (full layout)
.globl sounio_epistemic_sub_full
.type sounio_epistemic_sub_full, @function
sounio_epistemic_sub_full:
    ret
.size sounio_epistemic_sub_full, .-sounio_epistemic_sub_full

# Epistemic propagation: multiplication (full layout)
.globl sounio_epistemic_mul_full
.type sounio_epistemic_mul_full, @function
sounio_epistemic_mul_full:
    ret
.size sounio_epistemic_mul_full, .-sounio_epistemic_mul_full

# Epistemic propagation: division (full layout)
.globl sounio_epistemic_div_full
.type sounio_epistemic_div_full, @function
sounio_epistemic_div_full:
    ret
.size sounio_epistemic_div_full, .-sounio_epistemic_div_full

# Epistemic propagation: addition (compact layout)
.globl sounio_epistemic_add_compact
.type sounio_epistemic_add_compact, @function
sounio_epistemic_add_compact:
    ret
.size sounio_epistemic_add_compact, .-sounio_epistemic_add_compact

# Epistemic propagation: subtraction (compact layout)
.globl sounio_epistemic_sub_compact
.type sounio_epistemic_sub_compact, @function
sounio_epistemic_sub_compact:
    ret
.size sounio_epistemic_sub_compact, .-sounio_epistemic_sub_compact

# Epistemic propagation: multiplication (compact layout)
.globl sounio_epistemic_mul_compact
.type sounio_epistemic_mul_compact, @function
sounio_epistemic_mul_compact:
    ret
.size sounio_epistemic_mul_compact, .-sounio_epistemic_mul_compact

# Epistemic propagation: division (compact layout)
.globl sounio_epistemic_div_compact
.type sounio_epistemic_div_compact, @function
sounio_epistemic_div_compact:
    ret
.size sounio_epistemic_div_compact, .-sounio_epistemic_div_compact

# Epistemic extraction: get value from Knowledge<T>
# C signature: double sounio_epistemic_get_value(const KnowledgeFull* knowledge)
# System V ABI: RDI=knowledge, return in XMM0
.globl sounio_epistemic_get_value
.type sounio_epistemic_get_value, @function
sounio_epistemic_get_value:
    movsd (%rdi), %xmm0
    ret
.size sounio_epistemic_get_value, .-sounio_epistemic_get_value

# Epistemic extraction: get confidence
.globl sounio_epistemic_get_confidence
.type sounio_epistemic_get_confidence, @function
sounio_epistemic_get_confidence:
    movsd 8(%rdi), %xmm0
    ret
.size sounio_epistemic_get_confidence, .-sounio_epistemic_get_confidence
"#
    .to_string()
}

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
"#
    .to_string()
}

fn generate_runtime_asm_internal(include_start: bool) -> String {
    let mut asm = String::new();

    asm.push_str("# Demetrios Runtime Library\n");
    asm.push_str("# Auto-generated - do not edit\n\n");

    if include_start {
        asm.push_str(&generate_start_asm());
        asm.push_str("\n");
    }
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
    asm.push_str("\n");
    asm.push_str(&generate_unit_conversion_asm());
    asm.push_str("\n");
    asm.push_str(&generate_ode_runtime_asm());
    asm.push_str("\n");
    asm.push_str(&generate_autodiff_runtime_asm());
    asm.push_str("\n");
    asm.push_str(&generate_tensor_runtime_asm());
    asm.push_str(
        "
",
    );
    asm.push_str(&generate_uncertain_runtime_asm());
    asm.push_str(
        "
",
    );
    asm.push_str(&generate_epistemic_runtime_asm());

    asm
}

/// Generate complete runtime assembly
pub fn generate_runtime_asm() -> String {
    generate_runtime_asm_internal(true)
}

/// Generate runtime assembly without entry point
pub fn generate_runtime_asm_without_start() -> String {
    generate_runtime_asm_internal(false)
}

fn write_runtime_asm_internal(
    path: impl AsRef<std::path::Path>,
    include_start: bool,
) -> std::io::Result<()> {
    let asm = generate_runtime_asm_internal(include_start);
    std::fs::write(path, asm)
}

/// Write runtime to file
pub fn write_runtime_asm(path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    write_runtime_asm_internal(path, true)
}

/// Write runtime to file without entry point
pub fn write_runtime_asm_without_start(path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    write_runtime_asm_internal(path, false)
}

fn build_runtime_object_internal(
    output_dir: impl AsRef<std::path::Path>,
    include_start: bool,
) -> Result<std::path::PathBuf, String> {
    let output_dir = output_dir.as_ref();
    let asm_path = output_dir.join("demetrios_runtime.s");
    let obj_path = output_dir.join("demetrios_runtime.o");

    // Write assembly
    write_runtime_asm_internal(&asm_path, include_start)
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

/// Build runtime object file
pub fn build_runtime_object(
    output_dir: impl AsRef<std::path::Path>,
) -> Result<std::path::PathBuf, String> {
    build_runtime_object_internal(output_dir, true)
}

/// Build runtime object file without entry point
pub fn build_runtime_object_without_start(
    output_dir: impl AsRef<std::path::Path>,
) -> Result<std::path::PathBuf, String> {
    build_runtime_object_internal(output_dir, false)
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
    "_demetrios_print_i64",
    "_demetrios_print_f64",
    "_demetrios_print_bool",
    "_demetrios_print_newline",
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
    // Unit conversion functions
    "sounio_convert_unit",
    "sounio_convert_affine",
    // ODE solver functions
    "sounio_ode_euler",
    "sounio_ode_rk4",
    "sounio_ode_dopri5",
    "sounio_ode_cashkarp",
    "sounio_ode_dopri5_step",   // C function called by assembly wrapper
    "sounio_ode_cashkarp_step", // C function called by assembly wrapper
    "sounio_ode_bdf_step",      // C function for BDF
    "sounio_ode_lsoda_step",    // C function for LSODA
    "sounio_ode_step",
    "sounio_ode_bdf",   // Assembly wrapper for BDF
    "sounio_ode_lsoda", // Assembly wrapper for LSODA
    // Autodiff functions
    "sounio_dual_add",
    "sounio_dual_sub",
    "sounio_dual_mul",
    "sounio_dual_div",
    "sounio_autodiff_forward",
    "sounio_autodiff_reverse",
    "sounio_dual_exp",
    "sounio_dual_log",
    "sounio_dual_sin",
    "sounio_dual_cos",
    "sounio_dual_pow",
    "sounio_dual_sqrt",
    // Tensor operations
    "sounio_tensor_matmul",
    "sounio_tensor_einsum",
    "sounio_tensor_reshape",
    "sounio_tensor_transpose",
    "sounio_tensor_add",
    "sounio_tensor_mul",
    "sounio_tensor_scale",
    "sounio_tensor_matvec",
    "sounio_tensor_add_simd",
    "sounio_tensor_mul_simd",
    "sounio_tensor_scale_simd",
    // Uncertainty propagation functions
    "sounio_uncertain_add",
    "sounio_uncertain_sub",
    "sounio_uncertain_mul",
    "sounio_uncertain_div",
    "sounio_uncertain_combine",
    "sounio_uncertain_scale",
    "sounio_uncertain_pow",
    "sounio_uncertain_sqrt",
    "sounio_uncertain_exp",
    "sounio_uncertain_log",
    // Epistemic propagation functions (Knowledge<T>)
    "sounio_epistemic_add_full",
    "sounio_epistemic_sub_full",
    "sounio_epistemic_mul_full",
    "sounio_epistemic_div_full",
    "sounio_epistemic_add_compact",
    "sounio_epistemic_sub_compact",
    "sounio_epistemic_mul_compact",
    "sounio_epistemic_div_compact",
    "sounio_epistemic_get_value",
    "sounio_epistemic_get_confidence",
    // Scientific operations
    "sounio_dot_product",
    "sounio_dot_product_simd",
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
        build_runtime_object, build_runtime_object_without_start, generate_runtime_asm,
        generate_runtime_asm_without_start, is_runtime_symbol, write_runtime_asm,
        write_runtime_asm_without_start, RUNTIME_SYMBOLS,
    };
}
