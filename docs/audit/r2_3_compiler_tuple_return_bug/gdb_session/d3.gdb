set pagination off
break *0x4018ef
run
# Set up watch on the f64 slot
set $sret = *((unsigned long*)($rbp - 0x98))
watch *((unsigned long*)($sret + 32))
watch *((unsigned long*)($sret))
watch *((unsigned long*)($sret + 24))
continue
# At write point — print state
printf "\n=== WRITER STATE at PC %#lx ===\n", $pc
info reg rax rbx rcx rdx rsi rdi rbp rsp r8 r9 r10 r11 r12 r13 r14 r15
printf "\nDisasm around writer:\n"
x/5i $pc - 12
printf "\n[+0] now = %#lx\n", *((unsigned long*)($sret))
printf "[+24] now = %#lx\n", *((unsigned long*)($sret + 24))
printf "[+32] now = %#lx\n", *((unsigned long*)($sret + 32))
