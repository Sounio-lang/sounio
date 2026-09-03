set pagination off
set logging file /tmp/sounio_pure_1/r2_3_compiler_tuple_bug/gdb_session/discriminator.log
set logging overwrite on
set logging on

# Break right BEFORE main pushes the f64 arg to println
# (offset 0x18ef in raw view = 0x4018ef virtual)
break *0x4018ef

run

# We're now stopped just before the f64 print sequence begins.
printf "=== AT 0x4018ef (just before push f64 arg) ===\n"
info reg rbp rsp r12 r13 r14 rbx
printf "\nMain RBP = %#lx\n", $rbp
printf "SRET base   (rbp - 0x90) = %#lx, value = %#lx\n", $rbp - 0x90, *((unsigned long*)($rbp - 0x90))
printf "SRET +8     (rbp - 0x88) = %#lx, value = %#lx\n", $rbp - 0x88, *((unsigned long*)($rbp - 0x88))
printf "SRET +16    (rbp - 0x80) = %#lx, value = %#lx\n", $rbp - 0x80, *((unsigned long*)($rbp - 0x80))
printf "SRET +24    (rbp - 0x78) = %#lx, value = %#lx (the UNTOUCHED slot)\n", $rbp - 0x78, *((unsigned long*)($rbp - 0x78))
printf "SRET +32    (rbp - 0x70) = %#lx, f64 value bits = %#lx\n", $rbp - 0x70, *((unsigned long*)($rbp - 0x70))
printf "Saved SRET-ptr (rbp - 0x98) = %#lx, value = %#lx\n", $rbp - 0x98, *((unsigned long*)($rbp - 0x98))

# Set hardware watchpoints. gdb supports up to 4 hw watchpoints simultaneously.
# Priority: (1) the f64 zeroing, (2) any of the 3-cluster writes
printf "\n=== Setting hardware watchpoints ===\n"
watch *((unsigned long*)($rbp - 0x90))
watch *((unsigned long*)($rbp - 0x78))
watch *((unsigned long*)($rbp - 0x70))

# Also: detect %rbp clobber (hypothesis b discriminator)
# Save current rbp; if it ever changes mid-println, hypothesis (b)
set $main_rbp = $rbp
printf "Main RBP frozen for comparison: %#lx\n", $main_rbp

# Continue. Each watchpoint hit gives PC + register state.
printf "\n=== Continuing through f64 println ===\n"
continue
