#!/usr/bin/env bash
cd /workspace/wt-guard
done_list=$(cut -f1 /tmp/cl4/final.tsv)
for t in $(grep -l 'f64' tests/run-pass/*.sio 2>/dev/null); do
  b=$(basename "$t")
  echo "$done_list" | grep -qx "$b" && continue
  r() { local e="/tmp/cl4/r_$1_$2.elf"; rm -f "$e"
        timeout -k 5 45 "/tmp/$1" --native-v2-compile "$t" $2 -o "$e" >/dev/null 2>&1 || { echo CF; return; }
        [ -f "$e" ] || { echo NOELF; return; }; chmod +x "$e"
        timeout -k 5 20 "$e" >/tmp/cl4/ro_$1.txt 2>&1; printf 'rc%s|%s' "$?" "$(head -c 200 /tmp/cl4/ro_$1.txt)"; }
  s0=$(r mad_cur ""); s1=$(r mad_cur "-O"); b0=$(r mad_bss ""); b1=$(r mad_bss "-O")
  printf '%s\t%s\t%s\t%s\t%s\n' "$b" "$(echo "$s0"|md5sum|cut -c1-8)" "$(echo "$s1"|md5sum|cut -c1-8)" "$(echo "$b0"|md5sum|cut -c1-8)" "$(echo "$b1"|md5sum|cut -c1-8)" >> /tmp/cl4/final.tsv
done
echo RESUME_DONE
