#!/usr/bin/env bash
# A = fixed compiler, markers ON   (baseline)
# B = fixed compiler, markers OFF  (the claim: bits alone suffice)
# C = pre-fix compiler, markers OFF (control: the corpus must be able to tell)
cd /workspace/wt-ir-soa-phase0 || { echo "DURABLE_FAIL: worktree missing -- refusing to scan a wrong cwd" >&2; exit 1; }
mkdir -p /tmp/cl3
: > /tmp/cl3/durable.tsv
n=0
for t in $(grep -l 'f64' tests/run-pass/*.sio 2>/dev/null); do
  n=$((n+1))
  run() { # $1=compiler $2=nomarkers
    local elf="/tmp/cl3/d_$2_$$.elf"
    rm -f "$elf"
    if ! env SOUNIO_FLOAT_NO_MARKERS="$2" timeout 60 "/tmp/$1" --native-v2-compile "$t" -O -o "$elf" >/dev/null 2>&1; then echo "COMPILE_FAIL"; return; fi
    [ -f "$elf" ] || { echo "NO_ELF"; return; }
    chmod +x "$elf" 2>/dev/null
    # Capture the ELF's own exit status; bound the output for the digest
    # afterwards. The old `out=$(... | head -c 300); rc=$?` read head's status
    # through the pipe (always 0) and could EPIPE the writer, so every digest
    # said rc0 no matter how the program actually exited -- an rc-3 program
    # and an rc-0 program with the same stdout digested identically.
    local out rc=0
    out="$(timeout 30 "$elf" 2>&1)" || rc=$?
    out="${out:0:300}"
    printf 'rc%s|%s' "$rc" "$out"
  }
  a=$(run mad_ord2 0); b=$(run mad_ord2 1); c=$(run mad_final 1)
  printf '%s\t%s\t%s\t%s\n' "$(basename $t)" "$(echo "$a"|md5sum|cut -c1-10)" "$(echo "$b"|md5sum|cut -c1-10)" "$(echo "$c"|md5sum|cut -c1-10)" >> /tmp/cl3/durable.tsv
done
# n=0 is a dead scan (bad cd, marker drift, moved corpus), not "all durable".
if [ "$n" -lt 1 ]; then
  echo "DURABLE_FAIL: corpus scan answered nothing (n=0)" >&2
  exit 1
fi
echo "DURABLE_DONE n=$n"
