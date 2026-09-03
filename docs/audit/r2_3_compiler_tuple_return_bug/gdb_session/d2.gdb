set pagination off
set logging file /tmp/sounio_pure_1/r2_3_compiler_tuple_bug/gdb_session/d2.log
set logging overwrite on
set logging enabled on

# Stop right BEFORE the f64 push (= just before the corrupting call sequence)
break *0x4018ef
run

printf "\n=== STATE AT 0x4018ef (pre-corruption) ===\n"
info reg rbp rsp r12

# The REAL SRET buffer address = *(rbp - 0x98)
set $sret = *((unsigned long*)($rbp - 0x98))
printf "Saved SRET ptr = %#lx\n", $sret
printf "  [+0]  = %#lx (r1.0.a expected 340518492412)\n", *((unsigned long*)($sret))
printf "  [+8]  = %#lx (r1.0.b)\n", *((unsigned long*)($sret + 8))
printf "  [+16] = %#lx (r1.0.c)\n", *((unsigned long*)($sret + 16))
printf "  [+24] = %#lx (r1.0.d — UNTOUCHED witness)\n", *((unsigned long*)($sret + 24))
printf "  [+32] = %#lx (r1.1 f64 bits)\n", *((unsigned long*)($sret + 32))

# Set hardware watchpoints on the REAL SRET buffer addresses
printf "\n=== Setting hw watches on real SRET buffer ===\n"
watch *((unsigned long*)($sret))
watch *((unsigned long*)($sret + 24))
watch *((unsigned long*)($sret + 32))

printf "\n=== Continuing through f64 println ===\n"
continue
