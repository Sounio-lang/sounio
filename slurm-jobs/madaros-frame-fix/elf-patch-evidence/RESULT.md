# End-to-end validation via ELF frame-patch (2026-06-16)

Backup Madaros emits the reproducer with a fixed 512-byte frame. Patching only the
prologue frame immediate 0x200 -> 0x2000 (8192) and re-running, same worker, back-to-back:

| ELF | unpatched (512B) | patched (8192B) |
|---|---|---|
| bk_2 (N=2) | pass=1 trail=1 conflict=1  (WRONG) | pass=1 trail=5 conflict=1  (PASS) |
| bk_5 (N=5) | SIGSEGV (rc=139)                  | pass=1 trail=5 conflict=1  (PASS) |

Enlarging only the frame flips N=2 wrong->correct and N=5 segv->correct => the bug is the
fixed-512 frame overflow; sizing the frame to need (align16(reg_count*8) ~= 1280B for this
reproducer) fixes both. Files: bk_*.elf (unpatched) / bk_*_patched.elf (patched); patch_frame.py.
