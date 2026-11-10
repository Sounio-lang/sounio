        .intel_syntax noprefix
        .text
# SIO-396 core: exact decimal expansion of a finite f64 with |x| >= 2^63.
# in : rax = IEEE bits of |x| (sign already cleared, >= 0x43E0000000000000, finite)
# out: rsi = pointer to first digit, rdx = length ("<digits>.000000")
# clobbers rax rbx rcx rdx rsi rdi r8-r15; lowers rsp by 1536 (caller restores via rbp).
        sub     rsp, 1536
        mov     rcx, rax
        shr     rcx, 52                 # biased exponent
        mov     r8, 0x000FFFFFFFFFFFFF
        and     rax, r8
        mov     r8, 0x0010000000000000
        or      rax, r8                 # 53-bit mantissa m
        sub     rcx, 1075               # e = E - 1075 >= 11
        mov     r10, 1000000000
        xor     rdx, rdx
        div     r10
        mov     qword ptr [rsp], rdx
        xor     rdx, rdx
        div     r10
        mov     qword ptr [rsp+8], rdx
        mov     qword ptr [rsp+16], rax
        mov     r9, 3                   # n limbs (little endian, base 1e9)
1:      test    rcx, rcx
        jz      6f
        mov     rbx, 30
        cmp     rcx, rbx
        jae     2f
        mov     rbx, rcx
2:      sub     rcx, rbx                # remaining shift
        mov     r8, 1
        push    rcx
        mov     rcx, rbx
        shl     r8, cl                  # r8 = 2^k, k = rbx <= 30
        pop     rcx
        xor     r11, r11                # carry
        xor     rdi, rdi                # i
3:      cmp     rdi, r9
        jae     4f
        mov     rax, qword ptr [rsp+rdi*8]
        mul     r8                      # rdx:rax = limb*2^k (rdx = 0)
        add     rax, r11
        xor     rdx, rdx
        div     r10
        mov     qword ptr [rsp+rdi*8], rdx
        mov     r11, rax
        inc     rdi
        jmp     3b
4:      test    r11, r11
        jz      1b
        mov     rax, r11
        xor     rdx, rdx
        div     r10
        mov     qword ptr [rsp+r9*8], rdx
        inc     r9
        mov     r11, rax
        jmp     4b
6:      # print: 9 digits per limb from the top, then strip leading zeros
        lea     r12, [rsp+512]          # write pointer
        mov     rdi, r9
7:      dec     rdi
        js      9f
        mov     rax, qword ptr [rsp+rdi*8]
        lea     r13, [r12+8]            # last digit position of this chunk
        mov     rbx, 10
        mov     rcx, 9
8:      xor     rdx, rdx
        div     rbx
        add     dl, 48
        mov     byte ptr [r13], dl
        dec     r13
        dec     rcx
        jnz     8b
        add     r12, 9
        jmp     7b
9:      lea     rsi, [rsp+512]
10:     cmp     byte ptr [rsi], 48
        jne     11f
        inc     rsi
        jmp     10b
11:     mov     byte ptr [r12], 46      # '.'
        mov     rax, 0x3030303030303030 # "000000" + spare
        mov     qword ptr [r12+1], rax
        add     r12, 7
        mov     rdx, r12
        sub     rdx, rsi
