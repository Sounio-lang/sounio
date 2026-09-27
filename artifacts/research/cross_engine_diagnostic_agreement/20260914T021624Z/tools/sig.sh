#!/usr/bin/env bash
# Usage: sig.sh OUT_DIR  -> TSV: file \t verdict \t signature (first diagnostic of the rejecting engine)
O="$1"
awk -F'\t' 'NR>1 && ($4=="LEAN_ONLY" || $4=="MADAROS_ONLY") {print $1"\t"$4}' "$O/agreement.tsv" |
while IFS=$'\t' read -r f v; do
  k=$([ "$v" = LEAN_ONLY ] && echo lean_only || echo madaros_only)
  log="$O/$k.$(echo "$f" | tr '/' '_').log"
  l=$(grep -m1 -E '(^|[[:space:]])error(\[|:)|^(lex|type|parse) error|^panic|parse error|run_check_mode: (module failed|AST closure)|Segmentation fault|Illegal instruction|could not read' "$log" 2>/dev/null)
  [ -z "$l" ] && l="<no diagnostic line; last: $(grep -v '^[[:space:]]*$' "$log" 2>/dev/null | tail -1 | cut -c1-60)>"
  s=$(printf '%s' "$l" | sed -E \
    -e 's/^.*line [0-9]+: +[0-9]+ Segmentation fault.*/Segmentation fault (lean_single crashed)/' \
    -e 's/at [A-Za-z0-9_\/.-]+\.sio:[0-9]+ \(bundle line [0-9]+\)/at FILE:N/g' \
    -e 's/<main>:[0-9]+( \(bundle line [0-9]+\))?/<main>:N/g' \
    -e 's/(at )?line [0-9]+(:[0-9]+)?/\1line N/g' \
    -e 's/at [0-9]+\.\.[0-9]+/at A..B/g' \
    -e 's/ in [A-Za-z0-9_\/.:-]+::[A-Za-z0-9_]+ / in M::F /g' \
    -e 's/`[^`]*`/`X`/g' \
    -e 's/\[struct=[A-Za-z0-9_]+\]/[struct=X]/g' \
    -e 's/nodes=[0-9]+ unresolved=[0-9]+/nodes=N unresolved=N/g' \
    -e 's/module failed to parse: .*/module failed to parse: PATH/' \
    -e 's/unreadable import: .*/unreadable import: PATH/' \
    -e 's/expected=-?[0-9]+ actual=-?[0-9]+/expected=T actual=T/g' | cut -c1-140)
  printf '%s\t%s\t%s\n' "$f" "$v" "$s"
done
