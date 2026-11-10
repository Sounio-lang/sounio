#!/usr/bin/env bash
# Rerun a crashing Madaros-compiled program under gdb and print where it died.
#
# Madaros-compiled programs carry no debug info. Frames keep the classic rbp
# chain, so a backtrace is the faulting rip plus the return address of every
# frame, named through the symbol map that SOUNIO_NV2_SYMBOL_MAP wrote at build
# time (`<hex vaddr> <function>`). madaros_fixed_point_gate.sh calls this when
# gen3 dies on a signal, so the artifact names the failing function instead of
# only an exit code (#2966): the gen3 crash cannot be reproduced on a machine
# with less memory than the CI runner.
#
# Usage: madaros_crash_backtrace.sh <binary> <symbol-map> -- <args...>
# Prints the stop reason, registers and a symbolized frame list on stdout.
set -uo pipefail
BIN="$1"; MAP="$2"; shift 2
[[ "${1:-}" == "--" ]] && shift
command -v gdb >/dev/null 2>&1 || { echo "crash-backtrace: gdb not installed"; exit 0; }
[[ -s "$MAP" ]] || { echo "crash-backtrace: no symbol map at $MAP"; exit 0; }
WORK="$(mktemp -d "${TMPDIR:-/tmp}/madaros-bt.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

cat >"$WORK/bt.gdb" <<'GDB'
set pagination off
set confirm off
handle SIGSEGV stop print
handle SIGILL stop print
handle SIGBUS stop print
handle SIGFPE stop print
handle SIGABRT stop print
run
info registers rip rsp rbp rax rbx rcx rdx rsi rdi
x/4i $rip
python
import gdb
try:
    rbp = int(gdb.parse_and_eval("$rbp"))
    pcs = [int(gdb.parse_and_eval("$rip"))]
    for _ in range(80):
        try:
            ret = int(gdb.parse_and_eval("*(unsigned long*)(%d+8)" % rbp))
            nrbp = int(gdb.parse_and_eval("*(unsigned long*)%d" % rbp))
        except Exception:
            break
        pcs.append(ret)
        if nrbp <= rbp:
            break
        rbp = nrbp
    print("PCS " + " ".join("%x" % p for p in pcs))
except Exception as e:
    print("PCS-ERROR %s" % e)
end
kill
quit
GDB

gdb -q -batch -x "$WORK/bt.gdb" --args "$BIN" "$@" >"$WORK/gdb.out" 2>&1
grep -vE '^\[(New|Thread|Inferior)' "$WORK/gdb.out" | grep -E 'signal|^rip|^rsp|^rbp|^rax|^rbx|^rcx|^rdx|^rsi|^rdi|=>|^   0x|PCS' | grep -v '^PCS ' | head -30

while read -r a n; do printf '%d %s\n' "$((16#$a))" "$n"; done <"$MAP" | sort -n >"$WORK/map.dec"
pcs="$(grep -m1 '^PCS ' "$WORK/gdb.out" | cut -d' ' -f2-)"
[[ -n "$pcs" ]] || { echo "crash-backtrace: no frames (the program did not stop on a signal)"; tail -n 5 "$WORK/gdb.out"; exit 0; }
echo "frames:"
for pc in $pcs; do
  awk -v a="$((16#$pc))" '$1 <= a { best = $1; name = $2 } END { printf "  %x %s+0x%x\n", a, name, a - best }' "$WORK/map.dec"
done
