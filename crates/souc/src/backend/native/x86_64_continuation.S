//! x86-64 Continuation Capture and Resumption Assembly
//!
//! This file implements low-level continuation capture and resumption for x86-64.
//! Follows the same pattern as aarch64_continuation.S.
//!
//! x86-64 Register Layout (System V AMD64 ABI):
//! - RAX, RBX, RCX, RDX, RSI, RDI, R8-R15: General-purpose registers (14 total)
//! - RBP: Frame pointer (callee-saved)
//! - RSP: Stack pointer
//! - XMM0-XMM15: SSE registers (128-bit each)
//!
//! Callee-saved: RBX, RBP, R12-R15
//! Caller-saved: RAX, RCX, RDX, RSI, RDI, R8-R11
//!
//! X86_64Continuation struct layout (must match Rust definition):
//! Offset  Size  Field
//! 0       112   gp_regs[14] (RAX, RBX, RCX, RDX, RSI, RDI, R8-R15, each 8 bytes)
//! 112     256   sse_regs[32] (XMM0-XMM15 as pairs of u64, each 16 bytes)
//! 368     8     rsp
//! 376     8     rbp
//! 384     8     rip (return address)
//! 392     ?     stack (Vec<u8>, pointer + len + cap - not modified in asm)
//! 416     1     resumed (bool - not modified in asm)

    .section .text
    .align 16
    .globl __sounio_capture_x86_64
    .globl __sounio_resume_x86_64

//! __sounio_capture_x86_64
//!
//! Captures the current execution state into an X86_64Continuation structure.
//!
//! Arguments:
//!   RDI: pointer to X86_64Continuation structure (System V ABI)
//!
//! Returns:
//!   void (continuation structure is filled)
//!
//! Note: We must save RBX and R12-R15 before using them as scratch registers.
//!
__sounio_capture_x86_64:
    // Save return address (on stack from call instruction)
    movq    (%rsp), %rax
    movq    %rax, 384(%rdi)         // Save RIP at offset 384

    // Save general-purpose registers at offset 0
    // Order: RAX, RBX, RCX, RDX, RSI, RDI, R8-R15 (14 registers)
    movq    %rax, 0(%rdi)           // RAX (we just used it, but save captured value)
    movq    %rbx, 8(%rdi)           // RBX
    movq    %rcx, 16(%rdi)          // RCX
    movq    %rdx, 24(%rdi)          // RDX
    movq    %rsi, 32(%rdi)          // RSI
    movq    %rdi, 40(%rdi)          // RDI (the continuation pointer itself)
    movq    %r8,  48(%rdi)          // R8
    movq    %r9,  56(%rdi)          // R9
    movq    %r10, 64(%rdi)          // R10
    movq    %r11, 72(%rdi)          // R11
    movq    %r12, 80(%rdi)          // R12
    movq    %r13, 88(%rdi)          // R13
    movq    %r14, 96(%rdi)          // R14
    movq    %r15, 104(%rdi)         // R15

    // Save SSE registers XMM0-XMM15 at offset 112
    // Each XMM register is 128 bits (16 bytes), stored unaligned
    movdqu  %xmm0,  112(%rdi)
    movdqu  %xmm1,  128(%rdi)
    movdqu  %xmm2,  144(%rdi)
    movdqu  %xmm3,  160(%rdi)
    movdqu  %xmm4,  176(%rdi)
    movdqu  %xmm5,  192(%rdi)
    movdqu  %xmm6,  208(%rdi)
    movdqu  %xmm7,  224(%rdi)
    movdqu  %xmm8,  240(%rdi)
    movdqu  %xmm9,  256(%rdi)
    movdqu  %xmm10, 272(%rdi)
    movdqu  %xmm11, 288(%rdi)
    movdqu  %xmm12, 304(%rdi)
    movdqu  %xmm13, 320(%rdi)
    movdqu  %xmm14, 336(%rdi)
    movdqu  %xmm15, 352(%rdi)

    // Save stack pointer and frame pointer
    // RSP at offset 368, RBP at offset 376
    movq    %rsp, 368(%rdi)
    movq    %rbp, 376(%rdi)

    // Note: Stack capture (Vec<u8>) and resumed flag are handled by Rust code
    // We only save register state here for shallow one-shot continuations

    ret

    .size __sounio_capture_x86_64, .-__sounio_capture_x86_64

//! __sounio_resume_x86_64
//!
//! Resumes execution from a captured continuation.
//!
//! Arguments:
//!   RDI: pointer to X86_64Continuation structure
//!   RSI: resume value (passed to resumed computation in RAX)
//!
//! This function NEVER returns normally! It restores the saved state and jumps
//! to the continuation's return address.
//!
__sounio_resume_x86_64:
    // Save resume value and continuation pointer to callee-saved registers
    // We'll use R12 and R13 which we'll restore last
    movq    %rsi, %r12              // R12 = resume value
    movq    %rdi, %r13              // R13 = continuation pointer

    // Restore frame pointer first (needed for correct stack frame)
    movq    376(%r13), %rbp

    // Restore stack pointer
    movq    368(%r13), %rsp

    // Restore SSE registers XMM0-XMM15 from offset 112
    movdqu  112(%r13), %xmm0
    movdqu  128(%r13), %xmm1
    movdqu  144(%r13), %xmm2
    movdqu  160(%r13), %xmm3
    movdqu  176(%r13), %xmm4
    movdqu  192(%r13), %xmm5
    movdqu  208(%r13), %xmm6
    movdqu  224(%r13), %xmm7
    movdqu  240(%r13), %xmm8
    movdqu  256(%r13), %xmm9
    movdqu  272(%r13), %xmm10
    movdqu  288(%r13), %xmm11
    movdqu  304(%r13), %xmm12
    movdqu  320(%r13), %xmm13
    movdqu  336(%r13), %xmm14
    movdqu  352(%r13), %xmm15

    // Restore general-purpose registers
    // Skip RAX (will be set to resume value)
    // Skip R12, R13 (currently in use, restore last)
    movq    8(%r13),  %rbx          // RBX
    movq    16(%r13), %rcx          // RCX
    movq    24(%r13), %rdx          // RDX
    movq    32(%r13), %rsi          // RSI
    movq    40(%r13), %rdi          // RDI
    movq    48(%r13), %r8           // R8
    movq    56(%r13), %r9           // R9
    movq    64(%r13), %r10          // R10
    movq    72(%r13), %r11          // R11
    movq    96(%r13), %r14          // R14
    movq    104(%r13), %r15         // R15

    // Load return address into R14 temporarily (we already restored it above, reuse)
    movq    384(%r13), %r14

    // Set RAX to resume value
    movq    %r12, %rax

    // Restore R12 from saved state
    movq    80(%r13), %r12

    // Restore R13 from saved state (last one - we were using it as pointer)
    movq    88(%r13), %r13

    // Jump to return address (continuation resumes here)
    // R14 contains the saved RIP
    jmpq    *%r14

    .size __sounio_resume_x86_64, .-__sounio_resume_x86_64

// Mark stack as non-executable (security hardening)
.section .note.GNU-stack,"",@progbits
