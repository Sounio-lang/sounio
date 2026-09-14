#!/usr/bin/env bash
# Usage (repo root): e035_sites.sh OUT_DIR SIG_TSV PERFILE_OUT
# Per LEAN_ONLY non-stub file with lean E035, count reported sites by kind, one awk pass per file:
#   localvar (var declared in the enclosing fn) | global (top-level [pub] var) | nonlocal_ident | fieldstore | iocall | call | other
O="$1"; S="$2"; PF="$3"; : > "$PF"; LF="$(dirname "$PF")/.e035_lines"
awk -F'\t' '$2=="LEAN_ONLY" && $3 ~ /E035\]: effect not declared in function signature at line/ {print $1}' "$S" |
while read -r f; do
  grep -q "Aspirational example preserved below" "$f" && continue
  grep -oE "E035\]: effect not declared in function signature at line [0-9]+" "$O/lean_only.$(echo "$f" | tr / _).log" | grep -oE "[0-9]+$" | sort -un > "$LF"
  [ -s "$LF" ] || continue
  awk -v file="$f" '
    FNR == 1 { fileno++ }
    fileno == 1 { want[$1] = 1; next }
    fileno == 2 { if ($0 ~ /^(pub[[:space:]]+)?(static[[:space:]]+)?var[[:space:]]+[A-Za-z_]/) { g = $0; sub(/^(pub[[:space:]]+)?(static[[:space:]]+)?var[[:space:]]+/, "", g); sub(/[^A-Za-z0-9_].*$/, "", g); glob[g] = 1 } next }
    /^[[:space:]]*(pub[[:space:]]+)?(kernel[[:space:]]+)?fn[[:space:]]/ { sig = $0; insig = ($0 !~ /\{/); split("", vars) }
    insig && !/^[[:space:]]*(pub[[:space:]]+)?(kernel[[:space:]]+)?fn[[:space:]]/ { sig = sig " " $0; if ($0 ~ /\{/) insig = 0 }
    $0 !~ /^(pub[[:space:]]+)?(static[[:space:]]+)?var[[:space:]]/ {
      line = $0
      while (match(line, /var[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/)) { v = substr(line, RSTART, RLENGTH); sub(/^var[[:space:]]+/, "", v); vars[v] = 1; line = substr(line, RSTART + RLENGTH) }
    }
    (FNR in want) {
      s = $0; kind = "other"
      if (match(s, /(^|[[:space:]{;])[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])?[[:space:]]*=[^=]/)) {
        t = substr(s, RSTART, RLENGTH); sub(/^[[:space:]{;]+/, "", t); sub(/[^A-Za-z0-9_].*$/, "", t)
        kind = (t in vars) ? "localvar" : ((t in glob) ? "global" : "nonlocal_ident")
      } else if (s ~ /[A-Za-z_][A-Za-z0-9_]*(\[[^]]*\])?\.[A-Za-z_][A-Za-z0-9_.]*[[:space:]]*=([^=]|$)/) { kind = "fieldstore"
      } else if (s ~ /(^|[^A-Za-z_])(println|print|print_int|print_i64|print_f64|print_char|eprintln|write_file|read_file|syscall[0-9]*)[[:space:]]*\(/) { kind = "iocall"
      } else if (s ~ /[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\(/) { kind = "call" }
      c[kind]++
    }
    END { out = ""; for (k in c) out = out k "=" c[k] ";"; print file "\t" out }' "$LF" "$f" "$f" >> "$PF"
done
