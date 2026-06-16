#!/usr/bin/env python3
# Grow the fixed 512-byte prologue frame to 8192 in a backup-Madaros-emitted reproducer ELF.
# Anchor on the full prologue (push rbp; mov rbp,rsp; sub rsp,0x200) to hit only real prologues.
# End-to-end validation of the frame-overflow fix without rebuilding the compiler.
import sys
A = bytes.fromhex("554889e54881ec00020000")   # ...; sub rsp,0x200
R = bytes.fromhex("554889e54881ec00200000")   # ...; sub rsp,0x2000
b = open(sys.argv[1], "rb").read()
open(sys.argv[2], "wb").write(b.replace(A, R))
print(f"patched {b.count(A)} prologues, size {len(b)} unchanged")
