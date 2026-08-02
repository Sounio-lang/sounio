#!/usr/bin/env bash
# Bootstrap Chain Gate -- proves Sounio compiles Sounio natively (no JIT)
set -eo pipefail

TMPDIR=/tmp/bootstrap_gate_$$
mkdir -p "$TMPDIR"
pass=0; fail=0; not_run=0

# ── S1: stage0.c compiles ────────────────────────────────────────────────────
if cc -O2 -Wno-unused-function -Wno-unused-variable -o "$TMPDIR/stage0" bootstrap/stage0.c 2>/dev/null; then
    echo "PASS  S1  stage0.c compiles with cc"
    pass=$((pass+1))
else
    echo "FAIL  S1  stage0.c compilation failed"
    fail=$((fail+1))
fi

# ── S2: stage0 compiles boot1.sio ────────────────────────────────────────────
if "$TMPDIR/stage0" bootstrap/boot1.sio "$TMPDIR/boot1.elf" 2>/dev/null; then
    echo "PASS  S2  stage0 compiles boot1.sio → boot1.elf"
    pass=$((pass+1))
else
    echo "FAIL  S2  stage0 failed to compile boot1.sio"
    fail=$((fail+1))
fi

# ── S3: boot1.elf compiles hello.sio ─────────────────────────────────────────
cat > "$TMPDIR/hello.sio" << 'EOF'
fn main() -> i64 {
    print("BOOTSTRAP OK\n")
    0
}
EOF

chmod +x "$TMPDIR/boot1.elf" 2>/dev/null
if "$TMPDIR/boot1.elf" "$TMPDIR/hello.sio" "$TMPDIR/hello.elf" 2>/dev/null; then
    echo "PASS  S3  boot1.elf compiles hello.sio → hello.elf"
    pass=$((pass+1))
else
    echo "FAIL  S3  boot1.elf failed to compile hello.sio"
    fail=$((fail+1))
fi

# ── S4: hello.elf runs correctly ─────────────────────────────────────────────
chmod +x "$TMPDIR/hello.elf" 2>/dev/null
actual=$("$TMPDIR/hello.elf" 2>/dev/null || true)
if echo "$actual" | grep -qF "BOOTSTRAP OK"; then
    echo "PASS  S4  hello.elf → 'BOOTSTRAP OK'"
    pass=$((pass+1))
else
    echo "FAIL  S4  wrong output: '$actual'"
    fail=$((fail+1))
fi

# ── S5: boot1.elf compiles itself (produces ELF) ────────────────────────────
if "$TMPDIR/boot1.elf" bootstrap/boot1.sio "$TMPDIR/boot1_stage2.elf" 2>/dev/null; then
    echo "PASS  S5  boot1.elf compiles boot1.sio (self-compile produces ELF)"
    pass=$((pass+1))
else
    echo "FAIL  S5  boot1.elf cannot compile boot1.sio"
    fail=$((fail+1))
fi

# ── S6: Memory usage < 100MB ────────────────────────────────────────────────
# Capture ru_maxrss directly so the gate does not depend on GNU /usr/bin/time.
cat > "$TMPDIR/maxrss.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    pid_t pid = fork();
    if (pid < 0) return 3;
    if (pid == 0) {
        execv(argv[1], &argv[1]);
        _exit(127);
    }
    int status = 0;
    struct rusage usage;
    if (wait4(pid, &status, 0, &usage) < 0) return 4;
    long maxrss_kb = usage.ru_maxrss;
#if defined(__APPLE__)
    maxrss_kb /= 1024;
#endif
    printf("%ld\n", maxrss_kb);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return 5;
}
EOF

mem_kb=""
if cc -O2 -o "$TMPDIR/maxrss" "$TMPDIR/maxrss.c" 2>/dev/null; then
    set +e
    mem_kb=$("$TMPDIR/maxrss" "$TMPDIR/stage0" bootstrap/boot1.sio "$TMPDIR/boot1_mem.elf" 2>/dev/null)
    mem_rc=$?
    set -e
else
    mem_rc=1
fi

if [ "$mem_rc" -eq 0 ] && [ -n "$mem_kb" ] && [ "$mem_kb" -lt 102400 ]; then
    echo "PASS  S6  stage0 peak RSS ${mem_kb}KB < 100MB"
    pass=$((pass+1))
else
    echo "FAIL  S6  stage0 peak RSS '${mem_kb}'KB (runner exit $mem_rc)"
    fail=$((fail+1))
fi

# ── S7: stage0.c structural checks ──────────────────────────────────────────
if grep -q "GLOBAL_SAVED_RSP" bootstrap/stage0.c && \
   grep -q "emit_print_int_builtin" bootstrap/stage0.c && \
   grep -q "is_byte_index" bootstrap/stage0.c; then
    echo "PASS  S7  stage0.c has required functions"
    pass=$((pass+1))
else
    echo "FAIL  S7  stage0.c missing required functions"
    fail=$((fail+1))
fi

# ── S8: boot1.sio structural checks ─────────────────────────────────────────
if grep -q "fn main" bootstrap/boot1.sio && \
   grep -q "write_file\|io_write" bootstrap/boot1.sio; then
    echo "PASS  S8  boot1.sio has main and file output"
    pass=$((pass+1))
else
    echo "FAIL  S8  boot1.sio missing main or file output"
    fail=$((fail+1))
fi

# Cleanup
rm -rf "$TMPDIR"

echo ""
echo "=== Bootstrap Chain Gate: PASS=$pass FAIL=$fail NOT_RUN=$not_run / total=$((pass+fail+not_run)) ==="
[ $fail -eq 0 ] || exit 1
