// x86-64 Assembly Stubs for Continuation Capture/Resume
//
// These functions provide low-level register saving and restoring for
// continuation capture on x86-64 architecture (System V AMD64 ABI).
//
// Key differences from AArch64:
// 1. x86-64 has 16 GP registers (vs 31 in AArch64), 16 XMM registers (vs 32 SIMD)
// 2. Return address is on stack (not in link register)
// 3. Stack grows down; rsp is decremented for pushes
// 4. Uses AT&T syntax by default (src, dst) vs Intel syntax (dst, src)
// 5. XMM registers are 128-bit (saved as pairs of doubles)
//
// Calling Convention (System V AMD64):
// - Arguments: rdi, rsi, rdx, rcx, r8, r9, then stack
// - Return value: rax (rdx for 128-bit returns)
// - Callee-saved: rbx, rbp, r12-r15
// - Caller-saved: rax, rcx, rdx, rsi, rdi, r8-r11
// - Stack pointer: rsp (16-byte aligned before call)
// - Frame pointer: rbp
//
// - XMM0-XMM7: Argument/result registers (FP/SIMD)
// - XMM8-XMM15: Temporary registers
// - All XMM registers are caller-saved

.global __sounio_capture_continuation_asm
.global __sounio_resume_continuation_asm
.global __sounio_get_return_address

.text
.align 16

// Capture current continuation state
//
// C signature: void* __sounio_capture_continuation_asm(void* cont_ptr)
//
// Arguments:
//   rdi = pointer to NativeContinuation struct
//
// Returns:
//   rax = same pointer (for chaining)
//
// This function saves all registers into the continuation structure:
// - Offset 0: id (16 bytes)
// - Offset 16: return_address (8 bytes)
// - Offset 24: gp_registers[32] (256 bytes) - we use first 16 entries
// - Offset 280: fp_registers[32] (256 bytes) - we use first 16 entries (as f64 pairs)
// - Offset 536: stack_pointer (8 bytes)
// - Offset 544: frame_pointer (8 bytes)
__sounio_capture_continuation_asm:
    // Save continuation pointer (rdi) to callee-saved register
    movq    %rdi, %r12

    // Get return address from stack (caller's return address)
    // The call instruction pushed it onto the stack
    movq    (%rsp), %rax
    movq    %rax, 16(%r12)      # Save at offset 16

    // Save general-purpose registers at offset 24
    // We save all 16 x86-64 GP registers sequentially
    // Order: rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8-r15

    movq    %rax, 24(%r12)      # gp_registers[0] = rax
    movq    %rbx, 32(%r12)      # gp_registers[1] = rbx
    movq    %rcx, 40(%r12)      # gp_registers[2] = rcx
    movq    %rdx, 48(%r12)      # gp_registers[3] = rdx
    movq    %rsi, 56(%r12)      # gp_registers[4] = rsi
    movq    %rdi, 64(%r12)      # gp_registers[5] = rdi
    movq    %rbp, 72(%r12)      # gp_registers[6] = rbp
    movq    %rsp, 80(%r12)      # gp_registers[7] = rsp
    movq    %r8,  88(%r12)      # gp_registers[8] = r8
    movq    %r9,  96(%r12)      # gp_registers[9] = r9
    movq    %r10, 104(%r12)     # gp_registers[10] = r10
    movq    %r11, 112(%r12)     # gp_registers[11] = r11
    movq    %r12, 120(%r12)     # gp_registers[12] = r12
    movq    %r13, 128(%r12)     # gp_registers[13] = r13
    movq    %r14, 136(%r12)     # gp_registers[14] = r14
    movq    %r15, 144(%r12)     # gp_registers[15] = r15

    // Save XMM registers (128-bit each) at offset 280
    // The fp_registers array is [f64; 32], so each XMM register occupies 16 bytes (2 f64s)
    // We save xmm0-xmm15 (all 16 SSE2 registers)

    movdqu  %xmm0,  280(%r12)   # fp_registers[0-1] = xmm0
    movdqu  %xmm1,  296(%r12)   # fp_registers[2-3] = xmm1
    movdqu  %xmm2,  312(%r12)   # fp_registers[4-5] = xmm2
    movdqu  %xmm3,  328(%r12)   # fp_registers[6-7] = xmm3
    movdqu  %xmm4,  344(%r12)   # fp_registers[8-9] = xmm4
    movdqu  %xmm5,  360(%r12)   # fp_registers[10-11] = xmm5
    movdqu  %xmm6,  376(%r12)   # fp_registers[12-13] = xmm6
    movdqu  %xmm7,  392(%r12)   # fp_registers[14-15] = xmm7
    movdqu  %xmm8,  408(%r12)   # fp_registers[16-17] = xmm8
    movdqu  %xmm9,  424(%r12)   # fp_registers[18-19] = xmm9
    movdqu  %xmm10, 440(%r12)   # fp_registers[20-21] = xmm10
    movdqu  %xmm11, 456(%r12)   # fp_registers[22-23] = xmm11
    movdqu  %xmm12, 472(%r12)   # fp_registers[24-25] = xmm12
    movdqu  %xmm13, 488(%r12)   # fp_registers[26-27] = xmm13
    movdqu  %xmm14, 504(%r12)   # fp_registers[28-29] = xmm14
    movdqu  %xmm15, 520(%r12)   # fp_registers[30-31] = xmm15

    // Save stack pointer at offset 536
    movq    %rsp, 536(%r12)

    // Save frame pointer at offset 544
    movq    %rbp, 544(%r12)

    // Return continuation pointer in rax
    movq    %r12, %rax
    ret

// Resume a continuation
//
// C signature: void __sounio_resume_continuation_asm(void* cont_ptr, u64 value) __attribute__((noreturn))
//
// Arguments:
//   rdi = pointer to NativeContinuation struct
//   rsi = resume value (will be placed in rax after restoration)
//
// This function restores all registers from the continuation structure
// and jumps to the saved return address. It never returns.
__sounio_resume_continuation_asm:
    // Save resume value and continuation pointer to non-restored registers temporarily
    movq    %rsi, %r12          # r12 = resume value
    movq    %rdi, %r13          # r13 = continuation pointer

    // Restore frame pointer first (offset 544)
    movq    544(%r13), %rbp

    // Restore stack pointer (offset 536)
    movq    536(%r13), %rsp

    // Restore XMM registers from offset 280
    movdqu  280(%r13),  %xmm0
    movdqu  296(%r13),  %xmm1
    movdqu  312(%r13),  %xmm2
    movdqu  328(%r13),  %xmm3
    movdqu  344(%r13),  %xmm4
    movdqu  360(%r13),  %xmm5
    movdqu  376(%r13),  %xmm6
    movdqu  392(%r13),  %xmm7
    movdqu  408(%r13),  %xmm8
    movdqu  424(%r13),  %xmm9
    movdqu  440(%r13),  %xmm10
    movdqu  456(%r13),  %xmm11
    movdqu  472(%r13),  %xmm12
    movdqu  488(%r13),  %xmm13
    movdqu  504(%r13),  %xmm14
    movdqu  520(%r13),  %xmm15

    // Restore general-purpose registers from offset 24
    // Skip rax (will be set to resume value)
    // Skip r12, r13 (currently in use)

    movq    32(%r13),  %rbx     # rbx = gp_registers[1]
    movq    40(%r13),  %rcx     # rcx = gp_registers[2]
    movq    48(%r13),  %rdx     # rdx = gp_registers[3]
    movq    56(%r13),  %rsi     # rsi = gp_registers[4]
    movq    64(%r13),  %rdi     # rdi = gp_registers[5]
    # rbp already restored above
    # rsp already restored above
    movq    88(%r13),  %r8      # r8 = gp_registers[8]
    movq    96(%r13),  %r9      # r9 = gp_registers[9]
    movq    104(%r13), %r10     # r10 = gp_registers[10]
    movq    112(%r13), %r11     # r11 = gp_registers[11]
    # r12 will be restored after we're done using it
    movq    128(%r13), %r14     # r14 = gp_registers[13]
    movq    136(%r13), %r15     # r15 = gp_registers[14]

    // Load return address into r14 temporarily
    movq    16(%r13), %r14

    // Now restore r12 (we saved the resume value there)
    // Set rax to resume value first
    movq    %r12, %rax

    // Restore r12 from saved state
    movq    120(%r13), %r12

    // Restore r13 last
    movq    128(%r13), %r13

    // Jump to return address (continuation resumes here)
    // Note: r14 contains the return address
    jmpq    *%r14

// Get the return address of the caller
//
// C signature: usize __sounio_get_return_address()
//
// Returns:
//   rax = return address (from stack)
__sounio_get_return_address:
    // Return address is at top of stack (pushed by call instruction)
    movq    (%rsp), %rax
    ret

.size __sounio_capture_continuation_asm, . - __sounio_capture_continuation_asm
.size __sounio_resume_continuation_asm, . - __sounio_resume_continuation_asm
.size __sounio_get_return_address, . - __sounio_get_return_address

// Mark stack as non-executable (security hardening)
.section .note.GNU-stack,"",@progbits
