#!/usr/bin/env bash
# A = fixed compiler, markers ON   (baseline)
# B = fixed compiler, markers OFF  (the claim: bits alone suffice)
# C = pre-fix compiler, markers OFF (control: the corpus must be able to tell)
cd /workspace/wt-ir-soa-phase0
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
    local out; out=$(timeout 30 "$elf" 2>&1 | head -c 300); local rc=$?
    printf 'rc%s|%s' "$rc" "$out"
  }
  a=$(run mad_ord2 0); b=$(run mad_ord2 1); c=$(run mad_final 1)
  printf '%s\t%s\t%s\t%s\n' "$(basename $t)" "$(echo "$a"|md5sum|cut -c1-10)" "$(echo "$b"|md5sum|cut -c1-10)" "$(echo "$c"|md5sum|cut -c1-10)" >> /tmp/cl3/durable.tsv
done
echo "DURABLE_DONE n=$n"
